"""Emit MSL source code from tile IR.

Two emission paths:
  1D (per-thread): one thread per element, mask guard, scalar operations.
  2D (GEMM):       cooperative threadgroup loads, simdgroup MMA, barrier sync.

The emitter inspects the tile IR structure to decide which path to use.
"""

from __future__ import annotations

import re

from alloy._compiler.dtypes import from_ir
from alloy._compiler.tile_ir import (
    Atomic,
    Barrier,
    BinOp,
    Cast,
    Compare,
    Constant,
    CoopLoad,
    Copy,
    Copy4,
    DebugPrint,
    Dot,
    Dot4,
    AsChar4,
    Unpack4,
    ExpandDims,
    FlowControl,
    ForLoop,
    FusedElementwise,
    IfElse,
    IndexLoad,
    IndexStore,
    InterleaveVec4,
    Load,
    Load4Vec,
    LoadWide,
    LocalAlloc,
    MakeRange,
    NumPrograms,
    ProgramId,
    Reduce,
    RowPass,
    Select,
    SharedAlloc,
    SimdMatrixOp,
    SimdOp,
    SimdReduce,
    Splat,
    Store,
    TernaryOp,
    ThreadId,
    TileFunction,
    TileKernelPlan,
    TileOp,
    TileValue,
    UnaryOp,
    WhileLoop,
    Zeros,
    walk_ops,
)
from alloy._compiler.tile_opt import optimize_tile_ir
from alloy._compiler.tile_plan import pick_dot_reg, plan_tile_kernel
from alloy._compiler.msl.control import ControlEmitterMixin
from alloy._compiler.msl.context import (
    ADDRESS,
    MMA,
    PER_THREAD,
    PERSISTENT_MMA,
    ValLoc,
)
from alloy._compiler.msl.reductions import ReductionEmitterMixin
from alloy._compiler.msl.mma import MmaEmitterMixin
from alloy._compiler.msl.memory import MemoryEmitterMixin
from alloy._compiler.msl.elementwise import ElementwiseEmitterMixin
from alloy._compiler.msl.transforms import TransformEmitterMixin
from alloy._compiler.msl.math import (
    I64_HELPERS,
    eval_expr_chain,
    format_scalar_op,
)


def emit_msl_from_tile_ir(func: TileFunction, debug: bool = False) -> str:
    """Emit MSL from tile IR. All IR goes through the unified TileCompiler."""
    return _compile_composable(func, debug=debug)


def _compile_composable(func: TileFunction, debug: bool = False) -> str:
    """Compile a composable tile kernel through TileCompiler.

    Pipeline: optimize IR → plan → emit.
    """
    optimize_tile_ir(func)
    plan = plan_tile_kernel(func)
    compiler = TileCompiler(func, plan=plan)
    compiler._debug = debug
    return compiler.compile()


# ===================================================================
# Tile compiler — unified emitter for all tile IR (1D and 2D)
# ===================================================================


class TileCompiler(
    ReductionEmitterMixin,
    ControlEmitterMixin,
    MmaEmitterMixin,
    TransformEmitterMixin,
    MemoryEmitterMixin,
    ElementwiseEmitterMixin,
):
    """General-purpose tile IR → MSL compiler.

    Handles any composition of tile ops: 2D Loads (cooperative shared
    memory loads), Dot (simdgroup MMA), Reduce (per-row reductions),
    BinOp/UnaryOp (per-thread scalar/row ops), Barrier, ForLoop.

    All 2D tile data flows through threadgroup shared memory. Layout
    transitions (MMA → shared → per-thread) are managed automatically.
    """

    def __init__(
        self,
        func: TileFunction,
        plan: "TileKernelPlan | None" = None,
        shmem_plan: dict | None = None,
        config: dict | None = None,
        cfg: dict | None = None,
    ):
        self.func = func
        # Lazily-built value→users index for `_find_users` (IR is frozen during
        # codegen, so one walk serves all lookups). None until first use.
        self._users_index_cache: dict[str, list] | None = None

        # Accept plan or legacy config/shmem_plan dicts
        if plan is not None:
            self._plan = plan
            self._config = cfg or {}  # optional callbacks alongside plan
        else:
            # Legacy path: build plan from config dict
            cfg = config or {}
            self._plan = TileKernelPlan(
                threads=cfg.get("threads", 256),
                tpr=cfg.get("tpr", 1),
                dtype=cfg.get("dtype", "float"),
                acc_dtype=cfg.get("dtype", "float"),
                shmem_dtype=cfg.get("dtype", "float"),
                reg_m=cfg.get("reg", 2),
                reg_n=cfg.get("reg", 2),
                pad=cfg.get("pad", 4),
                block_n=cfg.get("block_n"),
                col_tiled=cfg.get("block_n") is not None,
                register_resident=cfg.get("register_resident", False),
                shmem_plan=shmem_plan or {},
                row_bound="M" if "M" in func.constexpr_values else None,
            )
            # Keep raw config for callback-style hooks (attention path)
            self._config = cfg

        p = self._plan
        self._shmem_plan = p.shmem_plan
        # Loads marked to stream from device into the MMA (no shmem staging).
        # `_device_operands` is filled by the Load emitter (val → addressing)
        # and consumed by _emit_mma.
        self._device_direct_loads = p.device_direct_loads
        self._device_operands: dict = {}
        self._reg = p.reg_m
        # User-supplied `_reg` constexpr override (1, 2, 4) — passed to
        # `pick_dot_reg` so per-dot reg picks honor it. 0 means auto.
        self._reg_override = int(func.constexpr_values.get("_reg", 0))
        # Per-kernel n_sg target: planner-computed max n_sg across dots.
        # Plumbed to every `pick_dot_reg` call so emitter's per-dot reg
        # decisions match the planner's.
        self._n_sg_target = p.n_sg_target
        # Per-buffer shmem dtype overrides. Defaults to plan.shmem_dtype.
        # Set by `_pass_shmem_and_column_tiling` for buffers that can be kept
        # narrower than the kernel-global pick (e.g. bf16 input loads under
        # HIGH_PRECISION=1, where the global is f32 but Q/K/V/dO loads can
        # stay bf16 since their MMA pair partner is also bf16).
        self._shmem_buf_dtype = dict(p.shmem_buf_dtype or {})
        self._pad = p.pad
        self._dtype = p.dtype
        self._acc_dtype = p.acc_dtype
        self._shmem_dtype = p.shmem_dtype
        self._vec_width = 4
        self._vec_type = f"{self._shmem_dtype}4"
        self._buffer_dtypes = p.buffer_dtypes  # per-buffer dtypes for mixed-precision
        self._deferred_cast: dict[str, tuple[TileValue, str]] = {}

        self._lines: list[str] = []
        self._indent = 1

        self._buffer_params: list[str] = []
        self._outputs: set[str] = set()

        # Scope tracking: indent level at which each variable was declared.
        # When a variable is referenced at a shallower level, it must be
        # hoisted (re-declared) to avoid MSL scoping errors.
        self._decl_scope: dict[str, int] = {}  # name → indent when declared

        # Value expression tracking: name → MSL expression string
        self._exprs: dict[str, str] = {}
        # Value location: name → ValLoc
        self._val_loc: dict[str, ValLoc] = {}
        # Shared buffer declarations: buf_name → (total_elems, declared)
        self._shmem_decls: dict[str, tuple[int, bool]] = {}
        self._threads = p.threads
        self._tpr = p.tpr
        self._register_resident = p.register_resident
        self._local_arrays: dict[str, int] = {}  # name → size (D)
        # 1D broadcast loads consumed only by flat-threaded fused chains:
        # name → "{dtype}({ptr}[uint({base}) + {{col}}])" template. The
        # broadcast loop + per-thread float[D] storage gets dropped; the
        # `_resolve` closure in `_emit_fused_elementwise` substitutes the
        # chain's `_c` and inlines a single scalar load per element.
        self._flat_loads_inline: dict[str, str] = {}

        # Row bounds for composable tile — guards cooperative loads/stores
        self._row_bound = p.row_bound

        # Column tiling: when N is too large for shared memory
        self._block_n = p.block_n
        self._col_tiled = p.col_tiled
        self._col_offset = None  # set to "_cn" inside column loops
        self._N = func.constexpr_values.get("N", 0)

        # Op index: value_name → producing op (for address tracing)
        self._op_map: dict[str, TileOp] = {}
        # MakeRange info: value_name → (start, end)
        self._make_ranges: dict[str, tuple[int, int]] = {}

        # ForLoop context for cooperative loads inside K-loops
        self._loop_var: str | None = None  # loop variable name (e.g., "k")
        self._loop_step: int | None = None  # loop step (e.g., BLOCK_K)
        self._carried_inits: set[str] = set()  # names of carried init values

        # Double-buffering: per-buffer offset expressions for shared memory
        self._double_buffer = p.double_buffer
        self._db_shmem_offsets: dict[str, str] | None = None  # buf_name → offset expr
        # Async copy: use simdgroup_async_copy_2d for device→threadgroup loads.
        # Supported for float32 and float16. Disabled when any Load has a
        # fusion transform — the post-copy transform pass negates the DMA benefit.
        has_load_transform = any(isinstance(op, Load) and op.transform for op in walk_ops(func.ops))
        # Async copy has no bounds checking — only safe when tile dimensions
        # evenly divide the matrix dimensions (M % BLOCK_M == 0, N % BLOCK_N == 0).
        M_val = func.constexpr_values.get("M", 0)
        N_val = func.constexpr_values.get("N", 0)
        block_m = func.constexpr_values.get("BLOCK_M", 0)
        block_n = func.constexpr_values.get("BLOCK_N", 0)
        tiles_aligned = (
            M_val
            and block_m
            and M_val % block_m == 0
            and N_val
            and block_n
            and N_val % block_n == 0
        )
        self._async_copy = (
            bool(func.options.get("async_copy", 0))
            and p.shmem_dtype in ("float", "half")
            and not has_load_transform
            and tiles_aligned
        )

        # Single IR scan pass for all init-time analysis
        self._pmma_acc_names: set[str] = set()
        self._scalar_pmma_acc_names: set[str] = set()
        self._scalar_dot_result = None  # (acc_name, M, N) for scalar dot store
        self._reduce_counter = 0
        self._has_1d_reduce = False
        self._mask_expr: str | None = None
        self._per_thread = True  # assume per-thread until proven otherwise
        self._needs_num_programs = False
        self._debug = False
        self._atomic_int_buffers: set[str] = set()
        self._atomic_float_buffers: set[str] = set()
        self._cas_counter = 0
        self._coop_load_counter = 0
        self._pmma_store_counter = 0
        self._pending_async_events: list[str] = []  # deferred async copy event names
        self._pending_tg_barrier = False  # deferred barrier after cooperative loads
        self._coop_constant_trip = False  # set per coop loop; True = constant-trip form
        # Shmem buffers WRITTEN since the last barrier. An incoming op flushes
        # the deferred barrier when its read/write conflicts with one of these:
        # reader (RAW) or writer (WAW). Consecutive coop-loads to distinct
        # buffers skip the inter-op barrier; the eventual reader absorbs it.
        self._buffers_since_barrier: set[str] = set()
        # Shmem buffers READ since the last barrier. A subsequent WRITE to one
        # of these forces a flush (WAR). Reads are fine (RAR), so tracking reads
        # separately avoids spurious flushes between two ops that both read the
        # same buffer.
        self._buffers_read: set[str] = set()
        # Set when a scalar op (BinOp/UnaryOp/Select/Ternary) touches shmem.
        # Since scalar ops typically run only on the first BLOCK_M rows (masked
        # via `if _row < BLOCK_M`), simdgroups beyond that row range are idle
        # and may race ahead to the next cooperative load/MMA before active
        # threads finish their shmem accesses.  Cleared on barrier emission.
        self._scalar_shmem_dirty = False
        # Tile swizzle disabled by default — at small M (decode-shape GEMMs)
        # the grid is too small for Z-order L2 benefit.
        self._tile_swizzle = False

        # Use-count tracking for copy-on-write: how many times each IR value
        # is referenced as an operand. When a shared-memory value is about to
        # be overwritten but has remaining consumers, we save it to a register.
        self._use_counts: dict[str, int] = {}
        self._cow_counter = 0
        # For COW: for each op, the set of value names read by any strictly-later
        # op. Lets us skip COW for values that won't be read again.
        self._future_reads: dict[int, frozenset[str]] = {}

        for op in walk_ops(func.ops):
            if isinstance(op, Dot):
                self._per_thread = False
                if op.acc:
                    self._pmma_acc_names.add(op.acc.name)
            elif isinstance(op, Load) and op.result and len(op.result.shape) == 2:
                self._per_thread = False
            elif isinstance(op, Reduce):
                if op.input and len(op.input.shape) <= 1:
                    self._has_1d_reduce = True
            elif isinstance(op, NumPrograms):
                self._needs_num_programs = True
            elif isinstance(op, Atomic) and op.ptr:
                if op.op.endswith("_float"):
                    self._atomic_float_buffers.add(op.ptr.name)
                else:
                    self._atomic_int_buffers.add(op.ptr.name)
            elif isinstance(op, Store) and op.reduce == "add" and op.ptr:
                # Atomic scatter-accumulate store (MoE grouped-down combine): the
                # destination must be declared atomic<float>* for atomic_fetch_add.
                self._atomic_float_buffers.add(op.ptr.name)
            for v in op.operand_values():
                self._use_counts[v.name] = self._use_counts.get(v.name, 0) + 1

        # Snapshot total operand reference counts before COW emission mutates
        # _use_counts. Used to decide scalar CSE: a value referenced more than
        # once (e.g. a lane index reused across many load addresses) is
        # materialized into a named temp instead of re-inlined at each use.
        self._scalar_uses: dict[str, int] = dict(self._use_counts)

        # Opaque-vector values: load4_vec / interleave / as_char4 results carry
        # shape () but ARE vec4 (uchar4/float4). BinOp/Cast/Unary over them stay
        # vec4 and must NOT be CSE-materialized as scalars (a `float t = v4`
        # mis-declares the type). unpack4/dot4 consume a vec and yield a true
        # scalar, breaking the chain.
        self._opaque_vec: set[str] = set()
        for op in walk_ops(func.ops):
            r = op.result
            if r is None:
                continue
            if isinstance(op, (Load4Vec, InterleaveVec4, AsChar4)):
                self._opaque_vec.add(r.name)
            elif isinstance(op, (Unpack4, Dot4)):
                pass  # vec → scalar
            elif isinstance(op, (BinOp, UnaryOp, Select, TernaryOp, Compare, Cast)):
                if any(v.name in self._opaque_vec for v in op.operand_values()):
                    self._opaque_vec.add(r.name)

        # Build future-reads map: iterate ops in order, carry forward the set
        # of names read by any later op so lookups are O(1) per query.
        flat_ops = list(walk_ops(func.ops))
        future: set[str] = set()
        for op in reversed(flat_ops):
            self._future_reads[id(op)] = frozenset(future)
            for v in op.operand_values():
                future.add(v.name)

    def _has_future_use(self, vname: str, current_op) -> bool:
        """Check if vname is read by any op strictly after current_op."""
        return vname in self._future_reads.get(id(current_op), frozenset())

    def _flush_tg_barrier(self):
        """Emit deferred threadgroup barrier if one is pending."""
        if self._pending_tg_barrier:
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            self._pending_tg_barrier = False
            # A full barrier syncs all simdgroups, subsuming scalar-shmem dirty.
            self._scalar_shmem_dirty = False
            self._buffers_since_barrier.clear()
            self._buffers_read.clear()

    def _check_flush_for(self, reads=frozenset(), writes=frozenset()):
        """Flush the deferred barrier if `reads`/`writes` conflict with
        pending state: RAW (read of a written buf), WAW (write of a written
        buf), or WAR (write of a read buf). RAR is silent."""
        if (
            (reads & self._buffers_since_barrier)
            or (writes & self._buffers_since_barrier)
            or (writes & self._buffers_read)
        ):
            self._flush_tg_barrier()

    def compile(self) -> str:
        # Use plan's buffer/output classification if available, else compute
        if self._plan.buffer_params:
            self._buffer_params = list(self._plan.buffer_params)
            self._outputs = set(self._plan.outputs)
        else:
            self._classify_params()
        self._build_op_map()
        self._count_refs()
        self._setup_shmem()
        lines = self._generate()
        return "\n".join(lines)

    def _emit(self, line: str = ""):
        self._lines.append("    " * self._indent + line if line else "")

    def _eval_transform(
        self,
        transform,
        source_expr,
        store_offs=None,
        extra_transforms=None,
        chain_source_name=None,
        operand_exprs=None,
    ):
        """Evaluate a transform chain — delegates to standalone eval_expr_chain."""
        return eval_expr_chain(
            transform,
            source_expr,
            store_offs=store_offs,
            extra_transforms=extra_transforms,
            chain_source_name=chain_source_name,
            operand_exprs=operand_exprs,
        )

    def _find_users(self, val_name: str) -> list:
        """Ops in the IR that reference `val_name` as an operand."""
        idx = self._users_index_cache
        if idx is None:
            idx = {}
            for op in walk_ops(self.func.ops):
                seen: set[str] = set()
                for v in op.operand_values():
                    if v.name not in seen:
                        seen.add(v.name)
                        idx.setdefault(v.name, []).append(op)
            self._users_index_cache = idx
        return idx.get(val_name, [])

    def _all_consumers_flat_threaded(self, val_name: str, D: int) -> bool:
        """True iff every op that references `val_name` (or its passthrough
        aliases — chained `ExpandDims` ops) lives inside a FusedElementwise
        marked `flat_threaded=True` whose column dimension matches `D`. Used
        by `_emit_scalar_load` to decide whether the per-thread broadcast
        array can be replaced with a per-element inline scalar load.

        A non-fused user, or a fused user in a chain with mismatched cols
        or row-iter form, blocks the optimization.
        """
        # Collect ExpandDims aliases of `val_name`. Chain operands typically
        # reference an ExpandDims-of-load (shape (1, D)) rather than the bare
        # 1D load — `_resolve` later folds the alias back to `<load>[_c]`.
        aliases: set[str] = {val_name}
        for op in walk_ops(self.func.ops):
            if op.result is None or not isinstance(op, ExpandDims):
                continue
            inp = op.input
            if inp is not None and inp.name in aliases:
                aliases.add(op.result.name)

        consumer_count = 0
        for top in walk_ops(self.func.ops):
            if not isinstance(top, FusedElementwise):
                continue
            uses = False
            for inner in top.ops:
                if any(v.name in aliases for v in inner.operand_values()):
                    uses = True
                    break
            if not uses:
                continue
            consumer_count += 1
            if not top.flat_threaded:
                return False
            if top.result is None or len(top.result.shape) != 2:
                return False
            if self._eff_cols(top.result.shape[1]) != D:
                return False

        if consumer_count == 0:
            return False

        # No non-fused users of any alias allowed. `_find_users` walks
        # recursively (including FusedElementwise inner ops); we ignore the
        # inner ops (already accounted for above) and only flag external
        # consumers. ExpandDims aliases themselves are transparent.
        fused_inner_ids: set[int] = set()
        for top in walk_ops(self.func.ops):
            if isinstance(top, FusedElementwise):
                for inner in top.ops:
                    fused_inner_ids.add(id(inner))
        for alias in aliases:
            for user in self._find_users(alias):
                if id(user) in fused_inner_ids:
                    continue
                if isinstance(user, (FusedElementwise, ExpandDims)):
                    continue
                return False
        return True

    def _classify_params(self):
        for p in self.func.params:
            if not p.is_constexpr:
                self._buffer_params.append(p.name)
        self._scan_stores(self.func.ops)

    def _scan_stores(self, ops):
        for op in walk_ops(ops):
            if isinstance(op, Store) and op.ptr:
                self._outputs.add(op.ptr.name)
            elif isinstance(op, SimdMatrixOp) and op.op == "store":
                # simd_store writes to a device buffer
                if len(op.args) >= 2 and op.args[1]:
                    self._outputs.add(op.args[1].name)
            elif isinstance(op, IndexStore) and op.base:
                self._outputs.add(op.base.name)

    def _build_op_map(self):
        """Build index from value name → producing op."""
        for op in walk_ops(self.func.ops):
            if op.result:
                self._op_map[op.result.name] = op

    def _count_refs(self):
        """Count how many times each value is referenced as an operand.

        Used to detect multi-use values that must not be mutated in-place
        by the local_array BinOp/UnaryOp handlers.
        """
        self._remaining_refs: dict[str, int] = {}
        for op in walk_ops(self.func.ops):
            for v in op.operand_values():
                self._remaining_refs[v.name] = self._remaining_refs.get(v.name, 0) + 1
        self._next_tmp = 0

    def _consume_ref(self, val_name: str):
        """Decrement remaining reference count for a value."""
        if val_name in self._remaining_refs:
            self._remaining_refs[val_name] -= 1

    def _local_arr_can_mutate(self, val_name: str) -> bool:
        """Check if a local_array value can be safely mutated in-place.

        Safe only if this is the last remaining reference to the value.
        Loop-carried values are always mutable (accumulated in-place across iterations).
        """
        if val_name in self._carried_inits:
            return True
        return self._remaining_refs.get(val_name, 0) <= 1

    def _alloc_local_copy(self, src_arr: str, D: int) -> str:
        """Allocate a new local array and copy from an existing one."""
        new_arr = f"_la{self._next_tmp}"
        self._next_tmp += 1
        self._emit(f"float {new_arr}[{D}];")
        self._emit(f"for (uint _d = 0; _d < {D}u; _d++) {new_arr}[_d] = {src_arr}[_d];")
        return new_arr

    def _setup_shmem(self):
        for val_name, (buf_name, rows, cols, stride) in self._shmem_plan.items():
            total = rows * stride
            if self._double_buffer:
                total *= 2  # Two copies for ping-pong
            if buf_name not in self._shmem_decls or total > self._shmem_decls[buf_name][0]:
                self._shmem_decls[buf_name] = (total, False)
        for op in walk_ops(self.func.ops):
            if not isinstance(op, Dot) or op.result is None:
                continue
            if op.transpose_lhs:
                _, m_dim = op.lhs.shape
            else:
                m_dim, _ = op.lhs.shape
            if m_dim >= 8:
                continue
            if op.transpose_rhs:
                n_dim = op.rhs.shape[0]
            else:
                _, n_dim = op.rhs.shape
            stride = self._eff_cols(n_dim)
            acc_value = op.acc.name if op.acc is not None else op.result.name
            buf_name = f"_sacc_{acc_value}"
            total = m_dim * stride
            if buf_name not in self._shmem_decls or total > self._shmem_decls[buf_name][0]:
                self._shmem_decls[buf_name] = (total, False)
        # FA-2 forward persistent o needs a small per-row alpha broadcast slot.
        for op in walk_ops(self.func.ops):
            if isinstance(op, Dot) and op.acc is not None and op.acc_pre_scale is not None:
                buf_name = self._acc_pre_scale_buf_name(op)
                rows = op.acc.shape[0]
                if buf_name not in self._shmem_decls or rows > self._shmem_decls[buf_name][0]:
                    self._shmem_decls[buf_name] = (rows, False)
        # FA-2 forward post-loop normalization: per-row (1/l) broadcast slot.
        for op in walk_ops(self.func.ops):
            if isinstance(op, Store) and op.acc_post_scale is not None:
                # The producing Dot is op.value's source; look up its acc.
                producer = self._op_map.get(op.value.name) if op.value else None
                if isinstance(producer, Dot) and producer.acc is not None:
                    buf_name = self._acc_post_scale_buf_name(producer)
                    rows = producer.acc.shape[0]
                    if buf_name not in self._shmem_decls or rows > self._shmem_decls[buf_name][0]:
                        self._shmem_decls[buf_name] = (rows, False)

    def _generate(self) -> list[str]:
        header = []
        header.append("#include <metal_stdlib>")
        header.append("#include <metal_simdgroup_matrix>")
        header.append("using namespace metal;")
        header.append("")
        header.extend(I64_HELPERS.strip().splitlines())
        header.append("")
        if self._async_copy:
            header.append("// simdgroup_async_copy — accessed via AIR binary patching")
            header.append("struct _simdgroup_event_t;")
            header.append(
                "thread _simdgroup_event_t* _alloy_async_copy_2d("
                "ulong,ulong,threadgroup void*,ulong,ulong,ulong2,"
                "const device void*,ulong,ulong,ulong2,long2,int"
                ') __asm("air_simdgroup_async_copy_2d_p3i8_p1i8");'
            )
            header.append(
                "void _alloy_wait_events(int,thread _simdgroup_event_t**)"
                ' __asm("air_wait_simdgroup_events");'
            )
            header.append("")

        buf_params = []
        for i, name in enumerate(self._buffer_params):
            if name in self._atomic_int_buffers:
                buf_params.append(f"    device atomic_int* {name} [[buffer({i})]]")
            elif name in self._atomic_float_buffers:
                buf_params.append(f"    device atomic<float>* {name} [[buffer({i})]]")
            else:
                const = "const " if name not in self._outputs else ""
                bdt = self._buffer_dtypes.get(name, self._shmem_dtype)
                buf_params.append(f"    device {const}{bdt}* {name} [[buffer({i})]]")
        header.append(f"kernel void {self.func.name}(")
        sig_params = buf_params + [
            "    uint tid [[thread_index_in_threadgroup]]",
            "    uint simd_gid [[simdgroup_index_in_threadgroup]]",
            "    uint simd_lane [[thread_index_in_simdgroup]]",
            "    uint3 gid [[threadgroup_position_in_grid]]",
        ]
        if self._needs_num_programs:
            sig_params.append("    uint3 _num_programs [[threadgroups_per_grid]]")
        header.append(",\n".join(sig_params))
        header.append(") {")

        # Generate ops first to determine which shmem slots are actually used
        self._lines = []
        if self._tpr > 1:
            self._emit(f"uint _row = tid / {self._tpr}u;")
            self._emit(f"uint _lane = tid % {self._tpr}u;")
        else:
            self._emit("uint _row = tid;")
        # Tile swizzle preamble: remap gid for L2 cache-friendly dispatch
        if self._tile_swizzle:
            # Blocked walk SX=2, SY=4: dispatch 2x4 tile groups together so
            # adjacent simdgroups share A and B rows in L2.
            self._emit("// Tile swizzle for L2 cache locality")
            self._emit("uint _swiz_linear = gid.x + gid.y * num_programs_x;")
            self._emit("uint _swiz_block = _swiz_linear / 8u;")  # 2*4=8 tiles per block
            self._emit("uint _swiz_local = _swiz_linear % 8u;")
            self._emit("uint _swiz_blocks_per_row = (num_programs_x + 1u) / 2u;")
            self._emit("uint _swiz_block_y = _swiz_block / _swiz_blocks_per_row;")
            self._emit("uint _swiz_block_x = _swiz_block % _swiz_blocks_per_row;")
            self._emit("uint _swiz_x = _swiz_block_x * 2u + _swiz_local % 2u;")
            self._emit("uint _swiz_y = _swiz_block_y * 4u + _swiz_local / 2u;")
            # Clamp to grid bounds (in case grid isn't evenly divisible)
            self._emit("_swiz_x = min(_swiz_x, num_programs_x - 1u);")
            self._emit("_swiz_y = min(_swiz_y, num_programs_y - 1u);")
        for line in self._config.get("preamble", []):
            self._emit(line)
        if self._col_tiled:
            self._emit_ops_column_tiled(self.func.ops)
        else:
            self._emit_ops(self.func.ops)
        op_lines = self._lines

        # Now emit declarations — only include shmem slots referenced in ops
        self._lines = []
        self._emit_constants()
        if self._needs_num_programs:
            for comp in ("x", "y", "z"):
                self._emit(f"uint num_programs_{comp} = _num_programs.{comp};")
        op_text = "\n".join(op_lines)
        self._emit_shmem_declarations(referenced_code=op_text)

        footer = ["}"]
        return header + self._lines + op_lines + footer

    def _emit_constants(self):
        ce = self.func.constexpr_values
        for key, val in ce.items():
            if key.startswith("_") and not key.startswith("_bufsize_"):
                continue
            if isinstance(val, (int, float)):
                ctype = "float" if isinstance(val, float) else "uint"
                if key.startswith("_bufsize_"):
                    ctype = "int"  # bufsize is int for signed comparison
                self._emit(f"const {ctype} {key} = {val};")
                self._exprs[key] = key

    def _buf_shmem_dtype(self, buf_name: str) -> str:
        """Per-buffer shmem dtype, falling back to the kernel-global default."""
        return self._shmem_buf_dtype.get(buf_name, self._shmem_dtype)

    def _val_shmem_dtype(self, val_name: str) -> str:
        """Per-value shmem dtype: looks up the value's slot's per-buffer dtype.

        For 2D shmem-resident values (Load/Dot/BinOp/UnaryOp results that the
        planner allocated a slot for), returns that slot's dtype. Otherwise
        falls back to the kernel-global shmem dtype.
        """
        slot = self._shmem_plan.get(val_name) if self._shmem_plan else None
        if not slot:
            return self._shmem_dtype
        buf_name = slot[0]
        return self._buf_shmem_dtype(buf_name)

    def _emit_shmem_declarations(self, referenced_code: str = ""):
        any_decl = False
        for buf_name, (total, _) in self._shmem_decls.items():
            # Only declare slots actually referenced in the generated code
            if referenced_code and buf_name not in referenced_code:
                continue
            buf_dt = self._buf_shmem_dtype(buf_name)
            self._emit(f"threadgroup {buf_dt} {buf_name}[{total}];")
            any_decl = True
        # Shared memory for 1D cross-simdgroup reduction
        if self._has_1d_reduce:
            n_sg = self._threads // 32
            self._emit(f"threadgroup float _red[{n_sg}];")
            any_decl = True
        if any_decl:
            self._emit()

    def _flush_async_events(self):
        """Wait for all pending async copy events before reading shared memory."""
        if not self._pending_async_events:
            return
        for ev_name in self._pending_async_events:
            self._emit(f"if (_ev_{ev_name}) {{")
            self._emit(f"    thread _simdgroup_event_t* _evs_{ev_name}[1] = {{_ev_{ev_name}}};")
            self._emit(f"    _alloy_wait_events(1, _evs_{ev_name});")
            self._emit("}")
        # Barrier ensures all SGs' DMA writes are visible before MMA reads
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self._pending_async_events.clear()

    def _emit_ops(self, ops: list[TileOp]):
        for op in ops:
            # Flush pending async copy events before ops that read shared memory
            if self._pending_async_events and isinstance(op, (Dot, Reduce, Store)):
                self._flush_async_events()
            self._emit_op(op)

    def _eff_cols(self, shape_cols: int) -> int:
        """Effective column count — BLOCK_N when inside a column-tiled loop."""
        if self._col_tiled and self._col_offset is not None:
            return self._block_n
        return shape_cols

    def _emit_ops_column_tiled(self, ops: list[TileOp]):
        """Emit ops with column tiling: multiple passes over BLOCK_N chunks."""
        block_n = self._block_n

        # Step 1: find first 2D Load — everything before it is address setup
        first_load_idx = len(ops)
        for i, op in enumerate(ops):
            if isinstance(op, Load) and op.result and len(op.result.shape) == 2:
                first_load_idx = i
                break

        # Emit address setup ops (before first 2D load)
        for op in ops[:first_load_idx]:
            self._emit_op(op)
        data_ops = list(ops[first_load_idx:])

        # Save baseline state (address expressions + constants)
        baseline_exprs = dict(self._exprs)
        baseline_val_loc = dict(self._val_loc)

        # Find 2D reduce ops among data_ops
        reduce_indices = []
        for i, op in enumerate(data_ops):
            if isinstance(op, Reduce) and op.input and len(op.input.shape) == 2:
                reduce_indices.append(i)

        n_passes = len(reduce_indices) + 1
        finalized_reduces: set[str] = set()

        for pass_idx in range(n_passes):
            is_final = pass_idx == len(reduce_indices)

            # Restore baseline state for this pass
            self._exprs = dict(baseline_exprs)
            self._val_loc = dict(baseline_val_loc)
            for rname in finalized_reduces:
                self._exprs[rname] = rname
                self._val_loc[rname] = PER_THREAD

            if not is_final:
                # Declare accumulator for this pass's reduce
                reduce_op = data_ops[reduce_indices[pass_idx]]
                rname = reduce_op.result.name
                identity = self._reduce_identity(reduce_op.op)
                self._emit(f"{self._acc_dtype} {rname} = {identity};")

            # Column loop
            self._emit(f"for (uint _cn = 0; _cn < N; _cn += {block_n}u) {{")
            self._indent += 1
            self._col_offset = "_cn"

            # Determine which ops to execute in this pass
            if not is_final:
                end_idx = reduce_indices[pass_idx] + 1
            else:
                end_idx = len(data_ops)

            for i in range(end_idx):
                op = data_ops[i]
                if isinstance(op, Reduce) and op.result.name in finalized_reduces:
                    continue
                if isinstance(op, Reduce) and not is_final and i == reduce_indices[pass_idx]:
                    self._emit_reduce_partial(op)
                    continue
                if isinstance(op, Store):
                    if is_final:
                        self._emit_op(op)
                    continue
                self._emit_op(op)

            self._col_offset = None
            self._indent -= 1
            self._emit("}")

            if not is_final:
                finalized_reduces.add(reduce_op.result.name)

    def _emit_op(self, op: TileOp):
        # Persistent MMA accumulators live in simdgroup_matrix registers and
        # can only be consumed directly by Dot (chained MMA, handled in
        # _emit_mma) or Store (which routes to _emit_persistent_mma_store).
        # For any other op kind, spill the persistent acc to shmem once so the
        # consumer reads it through the regular shared-memory path.
        # `_materialize_mma_to_shmem` is a no-op for non-persistent MMA results.
        if not isinstance(op, (Dot, Store)):
            for v in op.operand_values():
                loc = self._val_loc.get(v.name)
                if loc == PERSISTENT_MMA or loc == MMA:
                    self._materialize_mma_to_shmem(v)
        if isinstance(op, ProgramId):
            if self._tile_swizzle and op.axis < 2:
                # Remap (gid.x, gid.y) through the blocked walk so adjacent
                # tiles share L2 lines (linearize then delinearize with
                # swizzled strides).
                self._exprs[op.result.name] = f"_swiz_{'xy'[op.axis]}"
            else:
                self._exprs[op.result.name] = f"gid.{'xyz'[op.axis]}"
            self._val_loc[op.result.name] = ADDRESS
        elif isinstance(op, Constant):
            self._exprs[op.result.name] = format_scalar_op(op, self._get)
            self._val_loc[op.result.name] = PER_THREAD
        elif isinstance(op, MakeRange):
            # Record for address tracing; emit as tid for per-thread eval
            self._make_ranges[op.result.name] = (op.start, op.end)
            # Row-range (size = BLOCK_M) maps to _row when tpr > 1,
            # since _row = tid / tpr is the actual row index
            block_m = self.func.constexpr_values.get("BLOCK_M", 0)
            if self._tpr > 1 and (op.end - op.start) == block_m:
                self._exprs[op.result.name] = "_row"
            else:
                self._exprs[op.result.name] = "tid"
            self._val_loc[op.result.name] = ADDRESS
        elif isinstance(op, ExpandDims):
            self._exprs[op.result.name] = self._get(op.input)
            self._val_loc[op.result.name] = ADDRESS
        elif isinstance(op, Zeros):
            name = op.result.name
            shape = op.shape
            if name in self._pmma_acc_names:
                M_dim, N_dim = shape
                if M_dim < 8:
                    # Scalar dot path: keep the persistent accumulator in
                    # threadgroup memory. Multiple SIMD groups own different
                    # output columns, so a per-thread local array is not a
                    # coherent accumulator.
                    acc_name = f"_sacc_{name}"
                    stride = self._eff_cols(N_dim)
                    self._emit(f"// Persistent scalar dot accumulator: {M_dim}×{N_dim}")
                    self._emit(
                        f"for (uint _i = tid; _i < {M_dim * stride}u; _i += NUM_THREADS) "
                        f"{acc_name}[_i] = 0;"
                    )
                    self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    self._val_loc[name] = ValLoc("shared", acc_name, stride)
                    self._exprs[name] = acc_name
                    self._scalar_pmma_acc_names.add(name)
                    self._scalar_dot_result = (acc_name, M_dim, N_dim)
                    return
                # Persistent MMA: declare simdgroup accumulators. Use the
                # producing Dot's reg pick (matches `_emit_mma`) — the
                # accumulators here MUST match the layout the Dot will write
                # into. The Zeros tile shape (M_dim, N_dim) IS the Dot's
                # output, so we can reuse pick_dot_reg directly.
                reg = pick_dot_reg(
                    M_dim, N_dim, override=self._reg_override, target_n_sg=self._n_sg_target
                )
                sg_cols = N_dim // (reg * 8)
                acc_pfx = f"_acc_{name}"
                for i in range(reg):
                    for j in range(reg):
                        self._emit(
                            f"simdgroup_matrix<{self._acc_dtype}, 8, 8> {acc_pfx}_{i}_{j}(0);"
                        )
                # Per-accumulator sg variables (unique names to support multiple
                # persistent MMAs with different geometries, e.g. chained GEMM)
                sg_m_var = f"_sg_m_{name}"
                sg_n_var = f"_sg_n_{name}"
                self._emit(f"const uint {sg_m_var} = simd_gid / {sg_cols}u;")
                self._emit(f"const uint {sg_n_var} = simd_gid % {sg_cols}u;")
                self._val_loc[name] = PERSISTENT_MMA
                return
            if len(shape) == 2:
                D = shape[1]
                self._local_arrays[name] = D
                self._emit(f"{self._acc_dtype} {name}[{D}];")
                self._emit(f"for (uint _d = 0; _d < {D}u; _d++) {name}[_d] = 0.0f;")
                self._val_loc[name] = ValLoc("local_array")
            elif len(shape) == 1:
                # If already declared at an outer scope (hoisted for cross-loop
                # carry), emit assignment instead of re-declaration to avoid shadowing.
                if name in self._decl_scope and self._decl_scope[name] < self._indent:
                    self._emit(f"{name} = 0.0f;")
                else:
                    self._emit(f"{self._acc_dtype} {name} = 0.0f;")
                    self._decl_scope[name] = self._indent
                self._exprs[name] = name
                self._val_loc[name] = PER_THREAD
            else:
                self._exprs[name] = "0.0f"
                self._val_loc[name] = PER_THREAD
        elif isinstance(op, Load):
            if op.result and len(op.result.shape) == 2:
                self._emit_coop_load(op)
            else:
                self._emit_scalar_load(op)
        elif isinstance(op, Store):
            val_loc = self._val_loc.get(op.value.name, PER_THREAD) if op.value else PER_THREAD
            if val_loc == PERSISTENT_MMA:
                # Scalar dot path stores differently from MMA
                if self._scalar_dot_result is not None:
                    acc_name, M_dim, N_dim = self._scalar_dot_result
                    row_start, row_stride, col_start, base_offset = self._resolve_2d_addr(op)
                    addr_base = f"({base_offset}) + " if base_offset != "0" else ""
                    ptr_name = op.ptr.name if op.ptr else "C"
                    out_dt = self._buffer_dtypes.get(ptr_name, self._acc_dtype)
                    total_out = M_dim * N_dim
                    self._emit(f"for (uint _mn = tid; _mn < {total_out}u; _mn += NUM_THREADS) {{")
                    self._indent += 1
                    self._emit(f"uint _m = _mn / {N_dim}u;")
                    self._emit(f"uint _n = _mn % {N_dim}u;")
                    self._emit(f"uint _gm = uint({row_start}) + _m;")
                    self._emit(f"uint _gn = uint({col_start}) + _n;")
                    self._emit(
                        f"{ptr_name}[{addr_base}_gm * {row_stride}u + _gn] = {out_dt}({acc_name}[_m][_n]);"
                    )
                    self._indent -= 1
                    self._emit("}")
                    self._scalar_dot_result = None
                    return
                self._emit_persistent_mma_store(op)
                return
            if op.offsets and len(op.offsets.shape) == 2:
                self._emit_coop_store(op)
            elif op.value and len(op.value.shape) == 2 and val_loc.kind != "per_thread":
                self._emit_coop_store(op)
            else:
                self._emit_store(op)
        elif isinstance(op, Dot):
            # MMA requires 8×8 tiles — fall back to scalar dot for BLOCK_M < 8
            M_dim = op.lhs.shape[1] if op.transpose_lhs else op.lhs.shape[0]
            if M_dim < 8:
                self._emit_scalar_dot(op)
            else:
                self._emit_mma(op)
        elif isinstance(op, Reduce):
            self._flush_tg_barrier()
            self._emit_reduce(op)
        elif isinstance(op, SimdReduce):
            self._emit_simd_reduce(op)
        elif isinstance(op, Load4Vec):
            self._emit_load4_vec(op)
        elif isinstance(op, LoadWide):
            self._emit_load_wide(op)
        elif isinstance(op, Dot4):
            self._emit_dot4(op)
        elif isinstance(op, Unpack4):
            self._emit_unpack4(op)
        elif isinstance(op, AsChar4):
            self._emit_as_char4(op)
        elif isinstance(op, InterleaveVec4):
            self._emit_interleave_vec4(op)
        elif isinstance(op, BinOp):
            self._emit_binop(op)
        elif isinstance(op, UnaryOp):
            self._emit_unaryop(op)
        elif isinstance(op, TernaryOp):
            self._emit_ternaryop(op)
        elif isinstance(op, Compare):
            self._emit_compare(op)
        elif isinstance(op, Select):
            self._emit_select(op)
        elif isinstance(op, Splat):
            self._exprs[op.result.name] = self._get(op.value)
            self._val_loc[op.result.name] = self._val_loc.get(op.value.name, PER_THREAD)
        elif isinstance(op, Barrier):
            self._flush_tg_barrier()
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
            self._scalar_shmem_dirty = False
            self._pending_tg_barrier = False
        elif isinstance(op, ForLoop):
            self._emit_for_loop(op)
        elif isinstance(op, WhileLoop):
            self._emit_while_loop(op)
        elif isinstance(op, IfElse):
            self._emit_if_else(op)
        elif isinstance(op, FlowControl):
            self._emit_flow_control(op)
        elif isinstance(op, ThreadId):
            self._exprs[op.result.name] = "tid"
            self._val_loc[op.result.name] = PER_THREAD
        elif isinstance(op, SharedAlloc):
            self._emit(f"threadgroup float {op.result.name}[{op.size}];")
            self._exprs[op.result.name] = op.result.name
            self._val_loc[op.result.name] = ValLoc("shared")
        elif isinstance(op, LocalAlloc):
            self._emit(f"float {op.result.name}[{op.size}];")
            self._exprs[op.result.name] = op.result.name
            self._val_loc[op.result.name] = ValLoc("local_array")
        elif isinstance(op, IndexLoad):
            base_expr = self._get(op.base)
            idx_expr = self._get(op.index)
            self._exprs[op.result.name] = f"{base_expr}[{idx_expr}]"
            self._val_loc[op.result.name] = PER_THREAD
        elif isinstance(op, IndexStore):
            base_expr = self._get(op.base)
            idx_expr = self._get(op.index)
            val_expr = self._get(op.value)
            self._emit(f"{base_expr}[{idx_expr}] = {val_expr};")
        elif isinstance(op, Atomic):
            self._emit_atomic(op)
        elif isinstance(op, SimdOp):
            self._emit_simd_op(op)
        elif isinstance(op, SimdMatrixOp):
            self._emit_simd_matrix_op(op)
        elif isinstance(op, NumPrograms):
            self._needs_num_programs = True
            comp = ["x", "y", "z"][op.axis]
            self._exprs[op.result.name] = f"num_programs_{comp}"
            self._val_loc[op.result.name] = PER_THREAD
        elif isinstance(op, DebugPrint):
            self._emit_debug_print(op)
        elif isinstance(op, Cast):
            self._emit_cast(op)
        elif isinstance(op, CoopLoad):
            self._emit_coop_load_op(op)
        elif isinstance(op, Copy):
            self._emit_copy(op)
        elif isinstance(op, Copy4):
            self._emit_copy4(op)
        elif isinstance(op, FusedElementwise):
            self._emit_fused_elementwise(op)
        elif isinstance(op, RowPass):
            self._emit_row_pass(op)

    # --- Fused elementwise emission ---

    def _emit_fused_elementwise(self, fused: FusedElementwise):
        """Emit a single column loop for a FusedElementwise node.

        The IR pass (tile_opt) already decided which ops to fuse and which
        results need shmem writeback. This method just translates to MSL.
        """
        # Conditional flush: only if our chain reads/writes intersect with
        # pending state. RAR-only (a previous persistent MMA read the same
        # input) doesn't need a barrier between the two ops.
        _chain_reads: set[str] = set()
        _chain_writes: set[str] = set()
        _chain_internal = {op.result.name for op in fused.ops if op.result}
        for inner in fused.ops:
            for v in inner.operand_values():
                if v.name in _chain_internal:
                    continue
                loc = self._val_loc.get(v.name, PER_THREAD)
                if loc.kind == "shared":
                    _chain_reads.add(loc.name)
        for wb_name in fused.writeback:
            wb_plan = self._shmem_plan.get(wb_name)
            if wb_plan:
                _chain_writes.add(wb_plan[0])
        self._check_flush_for(reads=_chain_reads, writes=_chain_writes)
        rows = fused.result.shape[0]
        cols = self._eff_cols(fused.result.shape[1])

        # Find the source shared buffer from the first op's inputs
        default_buf, default_stride = "_s0", cols
        for v in fused.ops[0].operand_values():
            loc = self._val_loc.get(v.name, PER_THREAD)
            if loc.kind == "shared":
                default_buf, default_stride = loc.name, loc.stride
                break

        # Column variable and expression map. `use_flat` is captured by the
        # `_resolve` closure below so (M, 1) row-broadcast operands whose
        # cached expression contains `tid` get rewritten to use `_row` —
        # under flat threading kernel-scope `tid` ≠ this element's row.
        col_var = "_c"
        exprs: dict[str, str] = {}
        use_flat = fused.flat_threaded and self._tpr == 1

        def _resolve(val):
            if val is None:
                return "0"
            if val.name in exprs:
                return exprs[val.name]
            loc = self._val_loc.get(val.name, PER_THREAD)
            if loc.kind == "shared":
                return f"{loc.name}[_row * {loc.stride}u + {col_var}]"
            # 1D broadcast load whose entire fanout is flat-threaded chains:
            # the broadcast array was elided in `_emit_scalar_load`; emit a
            # single inline device read at this element's column index.
            if loc.kind == "flat_load_inline":
                return self._flat_loads_inline[val.name].format(col=col_var)
            # 1D local array (broadcast load like LSE/Delta): index by
            # column inside the chain loop. Without this the chain references
            # the array name as a scalar, which Metal rejects in a binary op.
            if loc.kind == "local_array":
                arr = self._exprs.get(val.name, val.name)
                return f"{arr}[{col_var}]"
            # ADDRESS values from ExpandDims of a local_array: the
            # expression IS the array name. Index by `_row` for
            # (M, 1) row vectors, by `col_var` for (1, N) column vectors.
            inner = self._exprs.get(val.name)
            if inner and inner in self._local_arrays:
                if len(val.shape) == 2 and val.shape[1] == 1:
                    return f"{inner}[_row]"
                return f"{inner}[{col_var}]"
            # ADDRESS values from ExpandDims of a flat_load_inline source:
            # there's no per-thread storage at all — emit the device load
            # inline at this element's column index. (M, 1) row-broadcast
            # of an inline source is absent because 1D loads here are always
            # (1, D) column broadcasts of LSE/Delta.
            if inner and inner in self._flat_loads_inline:
                return self._flat_loads_inline[inner].format(col=col_var)
            expr = self._get(val)
            # Broadcast operands carry a cached `tid`-based expression
            # (from MakeRange / bound checks). Substitute the right
            # element index for the broadcast direction:
            #   (1, N) column broadcast → tid → col_var
            #   (M, 1) row broadcast    → tid → _row (flat threading only;
            #                              row-iter form already has
            #                              `_row = tid` so no rewrite needed)
            if "tid" in expr and val.shape and len(val.shape) == 2:
                if val.shape[0] == 1 and val.shape[1] > 1:
                    expr = expr.replace("tid", col_var)
                elif use_flat and val.shape[1] == 1 and val.shape[0] > 1:
                    expr = expr.replace("tid", "_row")
            return expr

        # Emit the loop. Flat-threaded form: one element per thread, strided
        # across NUM_THREADS for tiles bigger than the threadgroup. Row-iter
        # form is the fallback for tpr > 1 (softmax-style) or rejected chains.
        if use_flat:
            total = rows * cols
            self._emit(f"for (uint _flat = tid; _flat < {total}u; _flat += {self._threads}u) {{")
            self._indent += 1
            self._emit(f"uint _row = _flat / {cols}u;")
            self._emit(f"uint {col_var} = _flat % {cols}u;")
            close_n = 1
        else:
            self._emit(f"if (_row < {rows}u) {{")
            self._indent += 1
            if self._tpr > 1:
                self._emit(
                    f"for (uint {col_var} = _lane; {col_var} < {cols}u; {col_var} += {self._tpr}u) {{"
                )
            else:
                self._emit(f"for (uint {col_var} = 0; {col_var} < {cols}u; {col_var}++) {{")
            self._indent += 1
            close_n = 2

        reg_idx = 0
        for op in fused.ops:
            expr = format_scalar_op(op, _resolve)
            if expr is None:
                continue
            name = op.result.name
            if name in fused.writeback:
                reg = f"_fv{reg_idx}"
                reg_idx += 1
                self._emit(f"{self._acc_dtype} {reg} = {expr};")
                plan = self._shmem_plan.get(name)
                if plan:
                    out_buf, _, _, out_stride = plan
                else:
                    out_buf, out_stride = default_buf, default_stride
                self._emit(
                    f"{out_buf}[_row * {out_stride}u + {col_var}] = {self._shmem_cast(reg, out_buf)};"
                )
                exprs[name] = reg
                self._val_loc[name] = ValLoc("shared", out_buf, out_stride)
            else:
                exprs[name] = f"({expr})"

        for _ in range(close_n):
            self._indent -= 1
            self._emit("}")

        # A row-guarded scalar write to shmem is visible only to the writing
        # thread; subsequent cross-thread reads (simdgroup_load, cross-thread
        # reduce, another fused chain) need a barrier first, otherwise neighbour
        # lanes observe stale shmem.
        self._scalar_shmem_dirty = True
        # Buffer-aware tracking lets the flush logic catch WAR/RAW conflicts
        # even when `_scalar_shmem_dirty` is cleared by an intervening barrier.
        self._buffers_read |= _chain_reads
        self._buffers_since_barrier |= _chain_writes
        if _chain_writes or _chain_reads:
            self._pending_tg_barrier = True

        # Non-writeback intermediates become per-thread
        for op in fused.ops:
            if op.result and op.result.name not in fused.writeback:
                self._exprs[op.result.name] = op.result.name
                self._val_loc[op.result.name] = PER_THREAD

    # --- Row-pass fusion emission ---

    def _emit_row_pass(self, rp: RowPass):
        """Lower a RowPass into one or more phases separated by cross-lane
        reductions. The RowPass model assumes one row per threadgroup (row
        selected externally via program_id); 1D ops inside are held as
        per-thread scalars (tid indexes the column), so phases emit as
        straight-line per-thread statements rather than explicit column
        loops. Axis=0 reductions fold the 1D tile to a per-row scalar.

        Phase boundaries are inferred by dataflow: a phase closes before
        any op that transitively depends on a Reduce produced in that
        phase. At the close, the emitter issues a cross-lane reduction
        (hardware butterfly + cross-simdgroup shmem spill when threads
        span >1 simdgroup) so the scalar becomes visible to every lane.

        Contract:
            * `rp.ops` contains only BinOp/UnaryOp/Select/Compare/TernaryOp
              with 1D results, and Reduce(axis=0) ops.
            * 1D operands are per-thread scalars (e.g. from Load); no
              shared-memory tile indexing is performed by this emitter.
            * RowPass writeback is not implemented; downstream consumers read the
              per-thread result expressions directly.
        """
        if rp.writeback:
            raise NotImplementedError("RowPass writeback to shared memory is not yet implemented")
        self._flush_tg_barrier()

        # ---- Dataflow partitioning: flat ops → ordered phases ----
        phases: list[list[TileOp]] = [[]]
        pending_reduces: set[str] = set()
        for op in rp.ops:
            op_reads = {v.name for v in op.operand_values()}
            if pending_reduces and (op_reads & pending_reduces):
                phases.append([])
                pending_reduces = set()
            phases[-1].append(op)
            if isinstance(op, Reduce) and op.result is not None:
                pending_reduces.add(op.result.name)

        # ---- Per-op expression cache: maps TileValue.name → MSL expression ----
        local_exprs: dict[str, str] = {}

        def _resolve(val):
            if val is None:
                return "0"
            if val.name in local_exprs:
                return local_exprs[val.name]
            return self._get(val)

        n_sg = self._threads // 32

        for phase_ops in phases:
            # Emit ops in the order they appear so scalars that feed 1D
            # expressions are declared first. Reductions are emitted AFTER
            # their input producers (which are regular ops scheduled here).
            reduces_in_phase: list[Reduce] = []
            for op in phase_ops:
                if isinstance(op, Reduce):
                    reduces_in_phase.append(op)
                    continue
                if op.result is None:
                    continue
                shape = op.result.shape
                if len(shape) == 1 and shape[0] > 1:
                    # 1D elementwise — materialise into a named per-thread scalar
                    # so downstream phases can reference it after butterflies.
                    expr = format_scalar_op(op, _resolve)
                    if expr is None:
                        continue
                    name = op.result.name
                    self._emit(f"{self._acc_dtype} {name} = {expr};")
                    local_exprs[name] = name
                    self._exprs[name] = name
                    self._val_loc[name] = PER_THREAD
                else:
                    # 0-D / REPLICATED scalar (constant, scalar BinOp, etc.).
                    # Route through the normal op emitter.
                    self._emit_op(op)
                    if op.result is not None:
                        local_exprs[op.result.name] = self._get(op.result)

            # Reductions: each accumulates from a single 1D value, then
            # butterflies across lanes (plus cross-simdgroup via shmem when
            # the threadgroup spans more than one simdgroup).
            for red in reduces_in_phase:
                if red.axis != 0:
                    raise RuntimeError(
                        f"RowPass only supports axis=0 reductions, got axis={red.axis}"
                    )
                name = red.result.name
                inp_expr = _resolve(red.input)
                combine = {"max": "max", "min": "min", "sum": "+"}.get(red.op, "+")
                identity = self._reduce_identity(red.op)
                # Seed the per-thread accumulator from this lane's input.
                self._emit(f"{self._acc_dtype} {name} = {inp_expr};")
                # Cross-lane butterfly (32-wide).
                self._emit_simd_butterfly(name, combine)
                # Cross-simdgroup reduction via shmem when needed.
                if n_sg > 1:
                    uid = self._reduce_counter
                    self._reduce_counter += 1
                    sg = f"_sg{uid}"
                    ln = f"_ln{uid}"
                    self._emit(f"uint {sg} = tid / 32, {ln} = tid % 32;")
                    if uid > 0:
                        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    self._emit(f"if ({ln} == 0) _red[{sg}] = {name};")
                    self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    self._emit(f"{name} = ({sg} == 0 && {ln} < {n_sg}u) ? _red[{ln}] : {identity};")
                    self._emit(f"if ({sg} == 0) {{")
                    self._indent += 1
                    self._emit_simd_butterfly(name, combine)
                    self._indent -= 1
                    self._emit("}")
                    self._emit(f"if ({sg} == 0 && {ln} == 0) _red[0] = {name};")
                    self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
                    self._emit(f"{name} = _red[0];")
                local_exprs[name] = name
                self._exprs[name] = name
                self._val_loc[name] = PER_THREAD

    def _emit_atomic(self, op: Atomic):
        """Emit atomic memory operation."""
        # Device memory fence before atomics ensures all prior device writes
        # are visible to other threadgroups that observe this atomic.
        self._emit("threadgroup_barrier(mem_flags::mem_device);")
        ptr = self._get(op.ptr)
        idx = self._get(op.index)
        val = self._get(op.value)
        result_name = op.result.name if op.result else None

        if op.op == "cas":
            self._cas_counter += 1
            tmp = f"_cas_expected_{self._cas_counter}"
            expected = self._get(op.expected)
            self._emit(f"int {tmp} = (int)({expected});")
            self._emit(
                f"atomic_compare_exchange_weak_explicit("
                f"&{ptr}[{idx}], &{tmp}, (int)({val}), "
                f"memory_order_relaxed, memory_order_relaxed);"
            )
            if result_name:
                self._exprs[result_name] = tmp
                self._val_loc[result_name] = PER_THREAD
        elif op.op == "add_float":
            expr = f"atomic_fetch_add_explicit(&{ptr}[{idx}], ({val}), memory_order_relaxed)"
            if result_name:
                self._emit(f"float {result_name} = {expr};")
                self._exprs[result_name] = result_name
                self._val_loc[result_name] = PER_THREAD
            else:
                self._emit(f"{expr};")
        elif op.op in ("max_float", "min_float"):
            self._cas_counter += 1
            n = self._cas_counter
            cmp = ">" if op.op == "max_float" else "<"
            self._emit(f"{{ float _af_new_{n} = ({val});")
            self._emit(
                f"  float _af_old_{n} = atomic_load_explicit(&{ptr}[{idx}], memory_order_relaxed);"
            )
            self._emit(f"  while (_af_new_{n} {cmp} _af_old_{n}) {{")
            self._emit(
                f"    if (atomic_compare_exchange_weak_explicit(&{ptr}[{idx}], &_af_old_{n}, _af_new_{n}, memory_order_relaxed, memory_order_relaxed)) break;"
            )
            self._emit("  }")
            if result_name:
                self._emit(f"  float {result_name} = _af_old_{n};")
                self._exprs[result_name] = result_name
                self._val_loc[result_name] = PER_THREAD
            self._emit("}")
        else:
            # Integer atomics: add, max, min, xchg, and, or, xor
            msl_ops = {
                "add": "atomic_fetch_add_explicit",
                "max": "atomic_fetch_max_explicit",
                "min": "atomic_fetch_min_explicit",
                "xchg": "atomic_exchange_explicit",
                "and": "atomic_fetch_and_explicit",
                "or": "atomic_fetch_or_explicit",
                "xor": "atomic_fetch_xor_explicit",
            }
            msl_op = msl_ops[op.op]
            expr = f"{msl_op}(&{ptr}[{idx}], (int)({val}), memory_order_relaxed)"
            if result_name:
                self._emit(f"auto {result_name} = {expr};")
                self._exprs[result_name] = result_name
                self._val_loc[result_name] = PER_THREAD
            else:
                self._emit(f"{expr};")

    def _emit_debug_print(self, op: DebugPrint):
        """Emit printf for debug_print."""
        if not self._debug:
            return

        def _convert_fmt(m):
            spec = m.group(1) or ""
            if not spec:
                return "%f"
            if spec.startswith(":"):
                spec = spec[1:]
            if "d" in spec:
                return "%d"
            if "f" in spec:
                return "%" + spec
            return "%f"

        c_fmt = re.sub(r"\{([^}]*)\}", _convert_fmt, op.fmt)
        if not c_fmt.endswith("\\n"):
            c_fmt += "\\n"
        args_str = ", ".join(self._get(a) for a in op.args)
        if args_str:
            self._emit(f'printf("{c_fmt}", {args_str});')
        else:
            self._emit(f'printf("{c_fmt}");')

    def _emit_cast(self, op: Cast):
        """Emit type cast (uses _alloy_cast_<type> to work on scalar or vec).

        For 2D inputs that live in shared/local memory, the cast is recorded
        as deferred (input ref + target dtype) so `_elem_access` applies it at
        the element-access point. Otherwise the cast references the load var
        name, which only exists inside the cooperative-load loop scope, and
        downstream BinOp/Reduce reading it outside that scope hit an undeclared
        reference. The deferred cast emits the cast against the shmem element
        access instead.
        """
        msl_type = from_ir(op.target_dtype).msl
        # Inherit the input's location so element access on the cast falls
        # through to the underlying shared/local storage.
        input_loc = self._val_loc.get(op.input.name, PER_THREAD)
        if (
            input_loc.kind in ("shared", "local_array")
            and op.result.shape
            and len(op.result.shape) == 2
        ):
            self._deferred_cast[op.result.name] = (op.input, msl_type)
            self._val_loc[op.result.name] = input_loc
            self._exprs[op.result.name] = self._exprs.get(op.input.name, op.input.name)
            return
        src = self._get(op.input)
        self._exprs[op.result.name] = f"_alloy_cast_{msl_type}({src})"
        self._val_loc[op.result.name] = PER_THREAD

    def _emit_coop_load_op(self, op: CoopLoad):
        """Emit cooperative threadgroup load with auto-barrier."""
        dst = self._get(op.dst)
        src = self._get(op.src)
        count = self._get(op.count)
        self._coop_load_counter += 1
        loop_var = f"_cl{self._coop_load_counter}"
        threads = self._threads
        self._emit(
            f"for (uint {loop_var} = tid; {loop_var} < uint({count}); {loop_var} += {threads}u) {{"
        )
        self._indent += 1
        self._emit(f"{dst}[{loop_var}] = {src}[{loop_var}];")
        self._indent -= 1
        self._emit("}")
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")

    def _emit_copy(self, op: Copy):
        """Emit identity copy — creates a distinct MSL variable.

        Always uses float (not half) to avoid precision loss on pointer offsets.
        half loses exact integer representation above 2048, corrupting address
        arithmetic for any buffer larger than 4KB.
        """
        src_expr = self._get(op.source)
        name = op.result.name
        ctype = "float" if self._acc_dtype == "half" else self._acc_dtype
        self._emit(f"{ctype} {name} = {src_expr};")
        self._exprs[name] = name

    def _emit_copy4(self, op: Copy4):
        """Emit vec4 memory copy: dst[offset] ← src_ptr[offset] as float4."""
        dst = self._get(op.dst)
        dst_off = self._get(op.dst_offset)
        src = self._get(op.src_ptr)
        src_off = self._get(op.src_offset)
        self._emit(
            f"*(threadgroup float4*)(&{dst}[{dst_off}]) = "
            f"*(device const float4*)(&{src}[{src_off}]);"
        )

    def _emit_simd_op(self, op: SimdOp):
        """Emit SIMD group operation."""
        # Consuming a comparison as simd predicate — not a mask for stores
        self._mask_expr = None
        args_str = ", ".join(self._get(a) for a in op.args)
        result_name = op.result.name if op.result else None
        # Map op name to MSL function
        msl_map = {
            "shuffle_xor": "simd_shuffle_xor",
            "shuffle": "simd_shuffle",
            "shuffle_up": "simd_shuffle_up",
            "shuffle_down": "simd_shuffle_down",
            "prefix_exclusive_sum": "simd_prefix_exclusive_sum",
            "prefix_inclusive_sum": "simd_prefix_inclusive_sum",
            "all": "simd_all",
            "any": "simd_any",
        }
        if op.op == "id":
            if result_name:
                self._exprs[result_name] = "simd_gid"
                self._val_loc[result_name] = PER_THREAD
            return
        if op.op == "lane_id":
            if result_name:
                self._exprs[result_name] = "simd_lane"
                self._val_loc[result_name] = PER_THREAD
            return
        msl_fn = msl_map.get(op.op, f"simd_{op.op}")
        expr = f"{msl_fn}({args_str})"
        if result_name:
            # Materialize shuffle/prefix results to named variables — they're
            # collective ops that must execute at definition point, not lazily
            # inlined at the use site (which may be inside a divergent branch).
            self._emit(f"{self._acc_dtype} {result_name} = {expr};")
            self._exprs[result_name] = result_name
            self._val_loc[result_name] = PER_THREAD

    def _emit_simd_matrix_op(self, op: SimdMatrixOp):
        """Emit SIMD group matrix operation."""
        result_name = op.result.name if op.result else None
        if op.op == "create":
            self._emit(f"simdgroup_matrix<float, 8, 8> {result_name}(0);")
            self._exprs[result_name] = result_name
            self._val_loc[result_name] = PER_THREAD
        elif op.op == "load":
            # args: [shared, offset, stride]
            shared = self._get(op.args[0])
            offset = self._get(op.args[1])
            stride = self._get(op.args[2]) if len(op.args) > 2 else str(op.stride)
            if result_name:
                self._emit(f"simdgroup_matrix<float, 8, 8> {result_name};")
                self._exprs[result_name] = result_name
                self._val_loc[result_name] = PER_THREAD
            target = result_name or self._get(op.args[0])
            if op.transpose:
                self._emit(
                    f"simdgroup_load({target}, &{shared}[{offset}], {stride}, ulong2(0,0), true);"
                )
            else:
                self._emit(f"simdgroup_load({target}, &{shared}[{offset}], {stride});")
        elif op.op == "mma":
            # args: [acc, a, b]
            # MMA accumulates in-place: result aliases the accumulator
            acc = self._get(op.args[0])
            a = self._get(op.args[1])
            b = self._get(op.args[2])
            self._emit(f"simdgroup_multiply_accumulate({acc}, {a}, {b}, {acc});")
            if result_name:
                self._exprs[result_name] = acc
                self._val_loc[result_name] = PER_THREAD
        elif op.op == "store":
            # args: [mat, ptr, offset, stride]
            mat = self._get(op.args[0])
            ptr = self._get(op.args[1])
            offset = self._get(op.args[2])
            stride = self._get(op.args[3]) if len(op.args) > 3 else str(op.stride)
            self._emit(f"simdgroup_store({mat}, &{ptr}[{offset}], {stride});")

    # --- Helpers ---

    def _get(self, val: TileValue | None) -> str:
        if val is None:
            return "0"
        return self._exprs.get(val.name, val.name)
