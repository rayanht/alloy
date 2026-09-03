"""Metal runtime — thin Python wrapper over C++ function API.

No Metal objects cross the Python/C++ boundary. All Metal state (device,
pipelines, buffers) lives in C++ globals. Python passes numpy arrays
and gets results via shared memory.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import hashlib
import re
import subprocess
import tempfile
from alloy.log import get_logger
from alloy._runtime import _metal_ext

__all__ = [
    "MetalDevice",
    "CompiledKernel",
    "MetalDispatcher",
    "default_device",
    "default_dispatcher",
]


logger = get_logger("alloy.runtime")


# --- MetalDevice (reads from C++ global) ---


class MetalDevice:
    _info = None

    @staticmethod
    def default() -> "MetalDevice":
        if MetalDevice._info is None:
            MetalDevice._info = _metal_ext.device_info()
            logger.info(
                "device_initialized",
                device_name=MetalDevice._info["name"],
                gpu_family=MetalDevice._info["gpu_family"],
                max_threads_per_tg=MetalDevice._info["max_threads_per_threadgroup"],
                max_tg_memory=MetalDevice._info["max_threadgroup_memory_length"],
                has_bfloat16=MetalDevice._info["has_bfloat16"],
            )
        return MetalDevice()

    @property
    def name(self) -> str:
        return self._get_info()["name"]

    @property
    def gpu_family(self) -> str:
        return self._get_info()["gpu_family"]

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._get_info()["max_threads_per_threadgroup"]

    @property
    def max_threadgroup_memory_length(self) -> int:
        return self._get_info()["max_threadgroup_memory_length"]

    @property
    def has_bfloat16(self) -> bool:
        return self._get_info()["has_bfloat16"]

    @property
    def recommended_max_working_set_size(self) -> int:
        """OS-recommended GPU working-set budget in bytes (Metal's
        ``recommendedMaxWorkingSetSize``). On Apple Silicon's unified memory
        this is the figure to size KV-cache fill against."""
        return self._get_info()["recommended_max_working_set_size"]

    def _get_info(self):
        if MetalDevice._info is None:
            MetalDevice._info = _metal_ext.device_info()
        return MetalDevice._info

    def __repr__(self) -> str:
        return f"MetalDevice({self.name!r}, family={self.gpu_family})"


# --- CompiledKernel (holds opaque int64 handle to C++ pipeline) ---


# Async copy binary patching: placeholder asm labels → real AIR intrinsic names.
# The Metal frontend compiler rejects dots in asm labels, so we use underscores
# as placeholders and binary-patch the compiled AIR bitcode.
_ASYNC_COPY_PLACEHOLDER = "air_simdgroup_async_copy_2d_p3i8_p1i8"


# pso_handle (int) → (msl_source, function_name). Lets the dispatch path
# look up the MSL when only an integer handle is available (e.g. the
# cached-fusion-dispatch hot path that bypasses CompiledKernel). Needed
# to serialise compiled plans (L5): every RecordedDispatch must carry
# enough info to recompile in a different process.
_pso_source_registry: dict[int, tuple[str, str]] = {}


def _register_pso_source(handle: int, source: str, function_name: str) -> None:
    _pso_source_registry[handle] = (source, function_name)


def pso_source(handle: int) -> tuple[str, str] | None:
    """Look up (msl_source, function_name) for a compiled pso_handle.
    Returns None when the handle wasn't routed through CompiledKernel.from_msl."""
    return _pso_source_registry.get(handle)
_ASYNC_COPY_PATCHES = [
    (b"air_simdgroup_async_copy_2d_p3i8_p1i8", b"air.simdgroup_async_copy_2d.p3i8.p1i8"),
    (b"air_wait_simdgroup_events", b"air.wait_simdgroup_events"),
]


class CompiledKernel:
    def __init__(self, handle: int, function_name: str, msl_source: str = "") -> None:
        self._handle = handle
        self._function_name = function_name
        # For the L5 plan cache: serialise the MSL alongside the plan so a
        # stale-process load can recompile to a fresh pso_handle via from_msl().
        # Empty when constructed from an already-cached pso_handle.
        self._msl_source = msl_source

    @classmethod
    def from_msl(cls, device: MetalDevice, source: str, function_name: str) -> "CompiledKernel":
        # Route through AIR patching if MSL contains async copy placeholders.
        # Compile failures here are not logged: this entry point is called
        # speculatively by the fusion engine, which catches and falls back to
        # per-op dispatch. The exception still propagates with the full Metal
        # diagnostic in its message.
        if _ASYNC_COPY_PLACEHOLDER in source:
            kernel = cls._from_msl_patched(device, source, function_name)
            kernel._msl_source = source
            _register_pso_source(kernel._handle, source, function_name)
            return kernel
        handle = _metal_ext.compile_msl(source, function_name)
        _register_pso_source(handle, source, function_name)
        return cls(handle, function_name, source)

    @classmethod
    def _from_msl_patched(
        cls, device: MetalDevice, source: str, function_name: str
    ) -> "CompiledKernel":
        """Compile MSL with async copy intrinsics via AIR round-trip patching.

        Pipeline: MSL → AIR → text IR → patch intrinsics + optimize → AIR → metallib
        Replaces simdgroup_load with per-lane insertelement, fixes async_copy
        attributes, and removes dead code.
        """

        cache_dir = os.path.join(tempfile.gettempdir(), "alloy_async_cache")
        os.makedirs(cache_dir, exist_ok=True)

        h = hashlib.sha256(source.encode()).hexdigest()[:16]
        metallib_path = os.path.join(cache_dir, f"{function_name}_{h}.metallib")

        if not os.path.exists(metallib_path):
            msl_path = os.path.join(cache_dir, f"{function_name}_{h}.metal")
            air_path = os.path.join(cache_dir, f"{function_name}_{h}.air")

            with open(msl_path, "w") as f:
                f.write(source)

            # MSL → AIR (with full Metal compiler optimizations)
            result = subprocess.run(
                ["xcrun", "metal", "-c", msl_path, "-o", air_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Metal compilation failed: {result.stderr}")

            # AIR → text IR
            result = subprocess.run(
                ["xcrun", "metal-objdump", "-d", air_path],
                capture_output=True,
                text=True,
            )
            ir_lines = result.stdout.split("\n")
            ir_start = next(
                (
                    i
                    for i, l in enumerate(ir_lines)
                    if "source_filename" in l or l.startswith("; ModuleID")
                ),
                0,
            )
            ir = "\n".join(ir_lines[ir_start:])

            # Patch async_copy intrinsic names
            for old_b, new_b in _ASYNC_COPY_PATCHES:
                ir = ir.replace(old_b.decode(), new_b.decode())

            # Fix async_copy declaration attributes (nocapture writeonly/readonly)
            ir = re.sub(
                r"declare %struct\._simdgroup_event_t\* @air\.simdgroup_async_copy_2d\.p3i8\.p1i8\([^)]+\)[^\n]+",
                "declare %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8("
                "i64, i64, i8 addrspace(3)* nocapture writeonly, i64, i64, "
                "<2 x i64>, i8 addrspace(1)* nocapture readonly, i64, i64, "
                "<2 x i64>, <2 x i64>, i32) local_unnamed_addr #2",
                ir,
            )
            ir = re.sub(
                r"declare void @air\.wait_simdgroup_events\([^)]+\)[^\n]+",
                "declare void @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t** nocapture) local_unnamed_addr #2",
                ir,
            )

            # Replace simdgroup_load with per-lane insertelement
            # (Apple GPU hardware lane mapping: row=(lane%8)/2+(lane/16)*4, col=(lane%2)*2+((lane/8)%2)*4)
            # Find simd_lane arg index from AIR metadata (not hardcoded — varies with buffer count)
            has_sgload = "simdgroup_matrix_8x8_load" in ir
            if has_sgload:
                lane_arg = "%5"  # default
                lane_match = re.search(r'!\{i32 (\d+), !"air\.thread_index_in_simdgroup"', ir)
                if lane_match:
                    lane_arg = f"%{lane_match.group(1)}"
                ir = ir.replace(
                    "local_unnamed_addr #0 {\n",
                    "local_unnamed_addr #0 {\n"
                    f"  %_hw_l8=and i32 {lane_arg},7\n"
                    f"  %_hw_rlo=lshr i32 %_hw_l8,1\n"
                    f"  %_hw_l16=lshr i32 {lane_arg},4\n"
                    f"  %_hw_rhi=shl nuw nsw i32 %_hw_l16,2\n"
                    f"  %_hw_row=add nuw nsw i32 %_hw_rlo,%_hw_rhi\n"
                    f"  %_hw_clo=shl nuw nsw i32 {lane_arg},1\n"
                    f"  %_hw_clo2=and i32 %_hw_clo,2\n"
                    f"  %_hw_l8b=lshr i32 {lane_arg},3\n"
                    f"  %_hw_chi=shl nuw nsw i32 %_hw_l8b,2\n"
                    f"  %_hw_chi2=and i32 %_hw_chi,4\n"
                    f"  %_hw_col=add nuw nsw i32 %_hw_clo2,%_hw_chi2\n",
                    1,
                )

                _ie_counter = [0]

                def _make_sgload_replacer(ir_type, vec_type):
                    """Create a replacer for simdgroup_load of a given type (float or half)."""

                    def _replace(m):
                        _ie_counter[0] += 1
                        n = _ie_counter[0]
                        rv, pv, s = m.group(1), m.group(2), int(m.group(3))
                        return (
                            f"  %_ld_off_{n}=mul nuw nsw i32 %_hw_row,{s}\n"
                            f"  %_ld_off2_{n}=add nuw nsw i32 %_ld_off_{n},%_hw_col\n"
                            f"  %_ld_off64_{n}=zext i32 %_ld_off2_{n} to i64\n"
                            f"  %_ld_p0_{n}=getelementptr inbounds {ir_type},{ir_type} addrspace(3)* {pv},i64 %_ld_off64_{n}\n"
                            f"  %_ld_v0_{n}=load {ir_type},{ir_type} addrspace(3)* %_ld_p0_{n}\n"
                            f"  %_ld_p1_{n}=getelementptr inbounds {ir_type},{ir_type} addrspace(3)* %_ld_p0_{n},i64 1\n"
                            f"  %_ld_v1_{n}=load {ir_type},{ir_type} addrspace(3)* %_ld_p1_{n}\n"
                            f"  %_ld_t_{n}=insertelement <64 x {ir_type}> zeroinitializer,{ir_type} %_ld_v0_{n},i64 0\n"
                            f"  {rv}=insertelement <64 x {ir_type}> %_ld_t_{n},{ir_type} %_ld_v1_{n},i64 1"
                        )

                    return _replace

                # Replace f32 simdgroup_load
                ir = re.sub(
                    r"(%\d+) = (?:tail )?call fast <64 x float> @air\.simdgroup_matrix_8x8_load\.v64f32\.p3f32\("
                    r"float addrspace\(3\)\* nocapture readonly (%\d+), "
                    r"<2 x i64> <i64 (\d+), i64 8>, <2 x i64> <i64 1, i64 \d+>, "
                    r"<2 x i64> zeroinitializer\) (#\d+)",
                    _make_sgload_replacer("float", "v64f32"),
                    ir,
                )
                # Replace f16 simdgroup_load
                ir = re.sub(
                    r"(%\d+) = (?:tail )?call fast <64 x half> @air\.simdgroup_matrix_8x8_load\.v64f16\.p3f16\("
                    r"half addrspace\(3\)\* nocapture readonly (%\d+), "
                    r"<2 x i64> <i64 (\d+), i64 8>, <2 x i64> <i64 1, i64 \d+>, "
                    r"<2 x i64> zeroinitializer\) (#\d+)",
                    _make_sgload_replacer("half", "v64f16"),
                    ir,
                )
                # Only remove declarations if no call sites remain
                calls_remain = bool(re.search(r"call.*@air\.simdgroup_matrix_8x8_load", ir))
                if not calls_remain:
                    ir = re.sub(r"declare.*simdgroup_matrix_8x8_load.*\n", "", ir)

            # Remove null-check guards on async events (events are never null)
            ir = re.sub(
                r"icmp eq %struct\._simdgroup_event_t\* (%\d+), null",
                r"icmp eq i32 0, 1",
                ir,
            )

            ll_path = os.path.join(cache_dir, f"{function_name}_{h}.ll")
            patched_air = os.path.join(cache_dir, f"{function_name}_{h}_opt.air")
            with open(ll_path, "w") as f:
                f.write(ir)

            # Text IR → AIR
            result = subprocess.run(
                ["xcrun", "metal-as", ll_path, "-o", patched_air],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"metal-as failed: {result.stderr}")

            # AIR → metallib
            result = subprocess.run(
                ["xcrun", "metallib", patched_air, "-o", metallib_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Metallib linking failed: {result.stderr}")

        handle = _metal_ext.compile_metallib(metallib_path, function_name)
        return cls(handle, function_name)

    @property
    def max_total_threads_per_threadgroup(self) -> int:
        return _metal_ext.pipeline_max_threads(self._handle)

    @property
    def thread_execution_width(self) -> int:
        return _metal_ext.pipeline_thread_width(self._handle)

    @property
    def function_name(self) -> str:
        return self._function_name

    def __repr__(self) -> str:
        return f"CompiledKernel({self._function_name!r})"


# --- MetalDispatcher ---


class MetalDispatcher:
    def __init__(self, device: MetalDevice) -> None:
        self._device = device
        self.dispatch_count = 0

    def __repr__(self) -> str:
        return f"MetalDispatcher(device={self._device.name!r})"


# --- Module-level singletons ---

_default_device: Optional[MetalDevice] = None
_default_dispatcher: Optional[MetalDispatcher] = None
_lock = threading.Lock()


def default_device() -> MetalDevice:
    global _default_device
    if _default_device is None:
        with _lock:
            if _default_device is None:
                _default_device = MetalDevice.default()
    return _default_device


def default_dispatcher() -> MetalDispatcher:
    global _default_dispatcher
    if _default_dispatcher is None:
        device = default_device()  # resolve outside _lock to avoid deadlock
        with _lock:
            if _default_dispatcher is None:
                _default_dispatcher = MetalDispatcher(device)
    return _default_dispatcher
