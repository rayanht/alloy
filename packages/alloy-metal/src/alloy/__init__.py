"""Alloy: GPU kernels and torch.compile inference on Apple Silicon."""

from alloy._compiler.trace import trace_if as if_  # noqa: F401

# --- Observability (structlog wrapper, configured on first import) ---
from alloy.log import configure_logging, get_logger  # noqa: F401

# --- DSL primitives (from _dsl.py) ---
from alloy.dsl import (  # noqa: F401
    abs as abs,
    arange as arange,
    atomic_add as atomic_add,
    atomic_add_float as atomic_add_float,
    atomic_and as atomic_and,
    atomic_cas as atomic_cas,
    atomic_max as atomic_max,
    atomic_max_float as atomic_max_float,
    atomic_min as atomic_min,
    atomic_min_float as atomic_min_float,
    atomic_or as atomic_or,
    atomic_xchg as atomic_xchg,
    atomic_xor as atomic_xor,
    barrier as barrier,
    bfloat16 as bfloat16,
    bitcast as bitcast,
    cast as cast,
    ceil as ceil,
    clamp as clamp,
    constexpr as constexpr,
    as_char4 as as_char4,
    coop_load as coop_load,
    copy4 as copy4,
    cos as cos,
    debug_print as debug_print,
    dot4 as dot4,
    exp as exp,
    exp2 as exp2,
    float16 as float16,
    float32 as float32,
    floor as floor,
    fma as fma,
    fma4 as fma4,
    gelu as gelu,
    gelu_tanh as gelu_tanh,
    int8 as int8,
    int16 as int16,
    int32 as int32,
    int64 as int64,
    uint8 as uint8,
    uint16 as uint16,
    uint32 as uint32,
    uint64 as uint64,
    interleave_vec4 as interleave_vec4,
    load as load,
    load4_vec as load4_vec,
    load_wide as load_wide,
    local as local,
    unroll as unroll,
    log as log,
    log2 as log2,
    max as max,
    maximum as maximum,
    min as min,
    minimum as minimum,
    num_programs as num_programs,
    output as output,
    program_id as program_id,
    relu as relu,
    round as round,
    rsqrt as rsqrt,
    scale4 as scale4,
    shared as shared,
    sigmoid as sigmoid,
    simd_all as simd_all,
    simd_any as simd_any,
    simd_id as simd_id,
    simd_lane_id as simd_lane_id,
    simd_load as simd_load,
    simd_matrix as simd_matrix,
    simd_mma as simd_mma,
    simd_prefix_exclusive_sum as simd_prefix_exclusive_sum,
    simd_prefix_inclusive_sum as simd_prefix_inclusive_sum,
    simd_reduce as simd_reduce,
    simd_shuffle as simd_shuffle,
    simd_shuffle_down as simd_shuffle_down,
    simd_shuffle_up as simd_shuffle_up,
    simd_shuffle_xor as simd_shuffle_xor,
    simd_store as simd_store,
    sin as sin,
    sqrt as sqrt,
    store as store,
    store4_vec as store4_vec,
    sum as sum,
    unpack4 as unpack4,
    tanh as tanh,
    thread_id as thread_id,
    tile_dot as tile_dot,
    where as where,
    zeros as zeros,
)

# --- Buffer utilities (from _dispatch/_buf_utils.py) ---
from alloy._dispatch.buf_utils import (  # noqa: F401
    get_debug as get_debug,
    set_debug as set_debug,
)

# --- Kernel decorators ---
from alloy._dispatch.kernel import kernel, tunable  # noqa: F401
from alloy._dispatch.dispatch import _engine

from alloy._debug.inspect import inspect as inspect
from alloy._debug.visualize import visualize as visualize
from alloy._runtime.tune import tune as tune
from alloy._runtime.tune import tune_report as tune_report
from alloy._runtime.profile import (
    profile_json as profile_json,
    profile_reset as profile_reset,
    profile_summary as profile_summary,
    set_profile as set_profile,
)

# --- Standard library kernels (imported after core DSL is available) ---
from alloy.std import (
    attention as attention,
    cross_entropy as cross_entropy,
    dot as dot,
    dot_transpose_rhs as dot_transpose_rhs,
    dot_transpose_lhs as dot_transpose_lhs,
    layernorm as layernorm,
    mean as mean,
    reduce_max as reduce_max,
    reduce_min as reduce_min,
    reduce_sum as reduce_sum,
    rms_norm as rms_norm,
    rope as rope,
    softmax as softmax,
)

# --- Buffer operator dispatch (after std kernels are available) ---
import alloy._runtime.buffer_ops  # noqa: F401  populates _buf_ops


def sync() -> None:
    """Wait for pending GPU work submitted through Alloy."""
    _engine.gpu_sync()

__version__ = "0.68.0"
