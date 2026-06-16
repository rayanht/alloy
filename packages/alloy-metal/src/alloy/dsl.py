"""DSL primitives — the `al.load`, `al.store`, `al.arange`, etc. stubs.

Each stub dispatches to the trace module during kernel tracing and raises
RuntimeError if called at runtime outside a kernel.
"""

from typing import TYPE_CHECKING

from alloy._compiler import trace as trace_mod
from alloy._compiler.trace import _active as trace_active

if TYPE_CHECKING:
    from alloy._runtime.alloy_buffer import AlloyBuffer


class ConstExprType(int):
    """Marker for kernel parameters that are compile-time constants.

    Used as a type annotation: ``def my_kernel(N: al.constexpr)``.
    The compiler recognises annotated params and substitutes their values
    directly into the generated MSL as ``const`` declarations.
    """

    pass


constexpr = ConstExprType


if TYPE_CHECKING:
    output = AlloyBuffer
else:

    class output:
        """Marker for kernel parameters written by the kernel."""

        pass


def _no_runtime(name: str) -> None:
    raise RuntimeError(f"alloy.{name} can only be used inside @al.kernel functions")


def _make_dsl_op(name: str, trace_name: str | None = None):
    """Generate a DSL stub that dispatches to trace_mod.trace_{name} during tracing."""
    tn = trace_name or f"trace_{name}"
    trace_handler = trace_mod.__dict__[tn]

    def op(*args, **kwargs):
        if trace_active():
            return trace_handler(*args, **kwargs)
        _no_runtime(name)

    op.__name__ = name
    op.__qualname__ = name
    return op


def _make_dsl_noop(name: str):
    """Generate a DSL stub for ops that only exist in MSL (no trace implementation)."""

    def op(*args, **kwargs):
        _no_runtime(name)

    op.__name__ = name
    op.__qualname__ = name
    return op


def unroll(iterable):
    """Marker for the loop AST-rewriter: `for x in al.unroll(range(...))` is left
    as native Python and UNROLLED at trace (x is a real int), not lowered to a
    runtime ForLoop. Use it for constexpr-bounded loops whose body indexes Python
    containers — e.g. a carried accumulator array, so one kernel covers every
    register-blocking split instead of hand-flattening o0..oN per case:

        o = [0.0] * PER_LANE
        for j in range(start, end):                # runtime, traced; carries `o`
            for d in al.unroll(range(PER_LANE)):   # unrolled; d is an int
                o[d] = o[d] * alpha + p * v[d]

    At runtime it is the identity, so the loop iterates the underlying range."""
    return iterable


# --- DSL primitives (generated) ---
_DSL_OPS: list[str] = [
    # Core
    "program_id",
    "arange",
    "load",
    "store",
    "thread_id",
    "shared",
    "local",
    "barrier",
    "debug_print",
    "cast",
    "bitcast",
    "where",
    "num_programs",
    "fma",
    "clamp",
    "copy4",
    "coop_load",
    # Simdgroup
    "simd_shuffle_xor",
    "simd_shuffle",
    "simd_shuffle_up",
    "simd_shuffle_down",
    "simd_prefix_exclusive_sum",
    "simd_prefix_inclusive_sum",
    "simd_all",
    "simd_any",
    "simd_id",
    "simd_lane_id",
    "simd_matrix",
    "simd_load",
    "simd_store",
    "simd_mma",
    # Atomics
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_cas",
    "atomic_xchg",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_add_float",
    "atomic_max_float",
    "atomic_min_float",
    # Math
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "tanh",
    "erf",
    "sin",
    "cos",
    "abs",
    "ceil",
    "floor",
    "round",
    "exp2",
    "log2",
    "sigmoid",
    "relu",
    "gelu",
    "gelu_tanh",
    # Binary
    "maximum",
    "minimum",
    # Reductions
    "sum",
    "max",
    "min",
    # Tile
    "zeros",
]

_DSL_OPS_CUSTOM: dict[str, str] = {
    "tile_dot": "trace_dot",
    "simd_reduce": "trace_simd_reduce",
    "load4_vec": "trace_load4_vec",
    "load_wide": "trace_load_wide",
    "dot4": "trace_dot4",
    "unpack4": "trace_unpack4",
    "as_char4": "trace_as_char4",
    "interleave_vec4": "trace_interleave_vec4",
}

_DSL_NOOP: list[str] = ["scale4", "fma4", "store4_vec"]

# Generate all stubs as module globals
for _name in _DSL_OPS:
    globals()[_name] = _make_dsl_op(_name)
for _name, _tn in _DSL_OPS_CUSTOM.items():
    globals()[_name] = _make_dsl_op(_name, _tn)
for _name in _DSL_NOOP:
    globals()[_name] = _make_dsl_noop(_name)
del _name, _tn

# Dtype sentinels for zeros()
float32 = "float32"
float16 = "float16"
bfloat16 = "bfloat16"
int64 = "int64"
int32 = "int32"
int16 = "int16"
int8 = "int8"
uint64 = "uint64"
uint32 = "uint32"
uint16 = "uint16"
uint8 = "uint8"
