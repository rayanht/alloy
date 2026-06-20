"""Tile IR — lightweight SSA IR for tile-level kernel compilation.

Every value carries a shape and layout annotation describing how elements
are distributed across threads.  Lowering passes transform tile ops into
thread-level Metal code.

Three layouts:
  Blocked   — each thread owns a contiguous chunk (cooperative loads)
  MMA       — 8x8 fragments in simdgroup matrix registers
  Replicated — every thread has the full value (scalars, broadcasts)
"""

from __future__ import annotations
import dataclasses
import hashlib
from functools import cached_property
from alloy._compiler.dispatch_spec import DispatchContract
from alloy._compiler.fusion_transforms import IndexTransform

from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import Any, ClassVar, Iterator

# --- Layouts ---


class Layout(Enum):
    REPLICATED = auto()  # every thread has the value
    BLOCKED = auto()  # elements distributed across threads in contiguous chunks
    MMA = auto()  # 8x8 simdgroup matrix register fragments


# --- Tile values — every value in the IR has a shape and layout ---


@dataclass
class TileValue:
    """A value in the tile IR with shape and layout metadata."""

    name: str
    shape: tuple[int, ...]  # () for scalar, (N,) for 1D, (M, N) for 2D
    layout: Layout
    dtype: str = "f32"  # "f32", "f16", "bf16", "i32"

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        r = 1
        for s in self.shape:
            r *= s
        return r

    @property
    def is_scalar(self) -> bool:
        return self.rank == 0


# --- Operations ---


@dataclass
class TileOp:
    """Base class for tile IR operations.

    Each subclass declares `_operand_fields` — the attribute names holding
    TileValue operands (directly, or nested in list / list-of-tuple fields).
    `operand_values()` and `remap()` are built on that. Subclasses that leave
    `_operand_fields` as the inherited `None` fall back to dataclass
    introspection (slower).
    """

    result: TileValue | None = None  # None for ops with no result (store)

    # Subclass contract. None = generic fallback walker.
    _operand_fields: ClassVar[tuple[str, ...] | None] = None

    def operand_values(self) -> list[TileValue]:
        """All TileValue operands of this op (excluding result)."""
        of = self.__class__._operand_fields
        if of is None:
            return self._operand_values_generic()
        out: list[TileValue] = []
        for name in of:
            v = self.__dict__.get(name)
            if v is None:
                continue
            if isinstance(v, TileValue):
                out.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, TileValue):
                        out.append(item)
                    elif isinstance(item, tuple):
                        for tv in item:
                            if isinstance(tv, TileValue):
                                out.append(tv)
        return out

    def remap(self, mapping: dict[str, TileValue]) -> None:
        """Rewrite TileValue operand refs in-place by name.

        Replaces every operand whose name appears in `mapping` with the
        mapped TileValue. Handles scalar, list, and list-of-tuple slots.
        """
        of = self.__class__._operand_fields
        if of is None:
            self._remap_generic(mapping)
            return
        for name in of:
            v = self.__dict__.get(name)
            if v is None:
                continue
            if isinstance(v, TileValue):
                if v.name in mapping:
                    self.__dict__[name] = mapping[v.name]
            elif isinstance(v, list):
                new_v = _remap_list_copy_on_write(v, mapping)
                if new_v is not v:
                    self.__dict__[name] = new_v

    # --- Fallback generic walker (used when _operand_fields is None) ---

    def _operand_values_generic(self) -> list[TileValue]:
        vals: list[TileValue] = []
        for f in fields(self):
            if f.name == "result":
                continue
            v = self.__dict__[f.name]
            if isinstance(v, TileValue):
                vals.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, TileValue):
                        vals.append(item)
                    elif isinstance(item, tuple):
                        for tv in item:
                            if isinstance(tv, TileValue):
                                vals.append(tv)
        return vals

    def _remap_generic(self, mapping: dict[str, TileValue]) -> None:
        for f in fields(self):
            if f.name == "result":
                continue
            v = self.__dict__[f.name]
            if isinstance(v, TileValue):
                if v.name in mapping:
                    self.__dict__[f.name] = mapping[v.name]
            elif isinstance(v, list):
                new_v = _remap_list_copy_on_write(v, mapping)
                if new_v is not v:
                    self.__dict__[f.name] = new_v


def _remap_list_copy_on_write(v: list, mapping: dict[str, "TileValue"]) -> list:
    """Remap TileValue refs in list items. Returns the same list if nothing
    changed, else a fresh list — so a shallow-copied parent op sharing its
    list with the original doesn't leak mutation back to the cached IR."""
    out: list | None = None
    for i, item in enumerate(v):
        new_item = item
        if isinstance(item, TileValue):
            if item.name in mapping:
                new_item = mapping[item.name]
        elif isinstance(item, tuple):
            if any(isinstance(tv, TileValue) and tv.name in mapping for tv in item):
                new_item = tuple(
                    mapping[tv.name]
                    if isinstance(tv, TileValue) and tv.name in mapping
                    else tv
                    for tv in item
                )
        if new_item is not item:
            if out is None:
                out = list(v)
            out[i] = new_item
    return out if out is not None else v


@dataclass
class MakeRange(TileOp):
    """Create a 1D index tile: [start, start+1, ..., end-1]."""

    start: int = 0
    end: int = 0
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class Splat(TileOp):
    """Broadcast a scalar to a tile shape."""

    value: TileValue | None = None
    shape: tuple[int, ...] = ()
    _operand_fields: ClassVar[tuple[str, ...]] = ("value",)


@dataclass
class Zeros(TileOp):
    """Zero-initialized tile."""

    shape: tuple[int, ...] = ()
    dtype: str = "f32"
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class ExpandDims(TileOp):
    """Add a dimension: (N,) -> (N, 1) or (1, N)."""

    input: TileValue | None = None
    axis: int = 0
    _operand_fields: ClassVar[tuple[str, ...]] = ("input",)


@dataclass
class BinOp(TileOp):
    """Elementwise binary operation."""

    op: str = ""  # "add", "sub", "mul", "div", "mod"
    lhs: TileValue | None = None
    rhs: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("lhs", "rhs")


@dataclass
class UnaryOp(TileOp):
    """Elementwise unary operation."""

    op: str = ""  # "neg", "exp", "log", "sqrt", "tanh", etc.
    input: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("input",)


@dataclass
class TernaryOp(TileOp):
    """Elementwise ternary function call (fma, clamp)."""

    op: str = ""  # "fma", "clamp"
    a: TileValue | None = None
    b: TileValue | None = None
    c: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("a", "b", "c")


@dataclass
class Compare(TileOp):
    """Elementwise comparison."""

    op: str = ""  # "lt", "le", "gt", "ge", "eq", "ne"
    lhs: TileValue | None = None
    rhs: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("lhs", "rhs")


@dataclass
class BoolOp(TileOp):
    """Elementwise boolean operation (and, or)."""

    op: str = ""  # "and", "or"
    lhs: TileValue | None = None
    rhs: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("lhs", "rhs")


@dataclass
class Select(TileOp):
    """Conditional select: result = cond ? true_val : false_val."""

    cond: TileValue | None = None
    true_val: TileValue | None = None
    false_val: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("cond", "true_val", "false_val")


@dataclass
class Load(TileOp):
    """Load from global memory.

    For 1D: ptr + offsets, each thread loads one element.
    For 2D: ptr + row_indices[:, None] * row_stride + col_indices[None, :],
            cooperative threadgroup load into shared memory.

    Semantic 2D addressing (populated by AST lowering):
        row_indices: 1D row index values (before expand_dims)
        col_indices: 1D col index values (before expand_dims)
        row_stride:  elements per row (the stride constant, e.g. K or N)
    When these are set, the emitter uses them directly instead of
    reverse-engineering the 2D structure from flat offsets.

    transform: list of elementwise tile IR ops to apply to the loaded value
    before writing to shared memory. Set by prologue fusion.
    """

    ptr: TileValue | None = None  # base pointer (scalar, Replicated)
    offsets: TileValue | None = None  # flat index tile (1D or 2D) — fallback
    mask: TileValue | None = None  # optional bool mask
    other: float = 0.0  # value for masked-out elements
    transform: list["TileOp"] = field(default_factory=list)
    transform_extras: dict[str, IndexTransform] = field(default_factory=dict)
    transform_source_name: str | None = None
    # Semantic 2D addressing (set by AST lowering for 2D patterns)
    row_indices: TileValue | None = None
    col_indices: TileValue | None = None
    row_stride: int | None = None
    base_offset: TileValue | None = None  # scalar offset (e.g., head_off in attention)
    addr_transposed: bool = False  # stride is on the [None,:] dim — coop load must scatter-store
    # Packed format: when set, the cooperative load divides the column index
    # by pack_factor, loads from the packed address, and extracts the element
    # via nibble/byte extraction. The result shape uses the unpacked column count.
    pack_factor: int = 0  # 0 = not packed. 2 = INT4 (2 per byte), etc.
    pack_bits: int = 0  # bits per element (4 for INT4, 8 for INT8)
    # Inline dequant: (val - zero_point) * scale + bias applied during packed load
    dequant_scale_ptr: TileValue | None = None
    dequant_bias_ptr: TileValue | None = None
    dequant_zero_point: float = 0.0  # scalar constexpr, not a buffer
    dequant_n_groups: int = 0
    # Q5_0 5th-bit side buffer: same packed layout as `ptr`, but each nibble
    # position holds bit 0 = the high (5th) bit of that element. When set, the
    # packed-load reconstructs `nibble | (high_bit << pack_bits)` before the
    # (val - zero_point) * scale dequant. No q4_k/q8_0/q6_k path uses it.
    dequant_high_ptr: TileValue | None = None
    # Block-structured dequant format that can't be expressed as flat
    # pack_factor + per-group scale. "q6_k" → 210-byte blocks with QL/QH/
    # scale/d sub-arrays; the emitter walks the layout per element.
    dequant_format: str = ""

    # `transform` holds nested TileOps (the prologue chain), not operands of
    # THIS Load — readers that need to walk the chain do it via walk_ops().
    _operand_fields: ClassVar[tuple[str, ...]] = (
        "ptr",
        "offsets",
        "mask",
        "row_indices",
        "col_indices",
        "base_offset",
        "dequant_scale_ptr",
        "dequant_bias_ptr",
        "dequant_high_ptr",
    )


@dataclass(frozen=True, slots=True)
class ColumnSliceInfo:
    """Column-slice epilogue bounds — restricts a Store to a column range.

    Set by the fusion engine when an epilogue op reads a column slice of the
    anchor output. The emitter generates a column guard and remapped addressing.
    """

    col_start: int
    col_end: int
    out_stride: int  # row stride for the slice output (= slice width)


@dataclass
class Store(TileOp):
    """Store to global memory.

    Semantic 2D addressing mirrors Load: when row_indices/col_indices/row_stride
    are set, the emitter uses them directly for cooperative scatter.

    transform: list of elementwise tile IR ops to apply to the value before
    writing to global memory. Set by epilogue fusion.
    """

    ptr: TileValue | None = None
    offsets: TileValue | None = None
    value: TileValue | None = None
    mask: TileValue | None = None
    transform: list["TileOp"] = field(default_factory=list)
    transform_extras: dict[str, IndexTransform] = field(default_factory=dict)
    transform_source_name: str | None = None  # chain source name for eval_expr_chain
    # Register-resident epilogue extras: maps a transform-chain leaf value name
    # to the name of a SECOND persistent-MMA accumulator (same geometry as the
    # store's primary acc). Unlike `transform_extras` (device-buffer reads),
    # these are read per-lane from `thread_elements()` of a sibling simdgroup
    # accumulator — lets a multi-accumulator elementwise epilogue (e.g.
    # dot_q4_k_silu's `silu(gate)*up`) run register-resident with no shmem spill.
    acc_extras: dict[str, str] = field(default_factory=dict)
    col_slice: ColumnSliceInfo | None = None  # column-slice epilogue bounds
    row_indices: TileValue | None = None
    col_indices: TileValue | None = None
    row_stride: int | None = None
    base_offset: TileValue | None = None
    # When True, the MMA epilogue rounds the f32 acc to the store's narrow
    # dtype before the transform chain, matching eager precision for tile-binop
    # epilogues (e.g. `gemm + residual_tile`, where eager rounds the gemm output
    # to bf16 before adding). Set by the fusion engine for tile_consumer only.
    round_acc_for_eager: bool = False

    # FA-2 forward post-loop normalization: when set, the persistent acc
    # `value` is multiplied by `acc_post_scale[row]` per-lane before the
    # device write. Avoids materializing the (BM, D) acc to shmem just for
    # the `o = o * (1/l)` epilogue. Per-row scalar (TileValue with shape
    # (M, 1)) — its per-thread expression is read by tid<M, written to a
    # small shmem broadcast buffer, then read per-lane during the store.
    acc_post_scale: TileValue | None = None
    # Atomic reduction store: "" = plain overwrite; "add" = atomic_fetch_add into
    # the (overlapping) destination — for scatter-accumulate epilogues like the MoE
    # grouped-down combine, where TOP_K expert tiles add into the same Y[token] row.
    # Marks the destination buffer `atomic<float>*` (see compiler buffer typing).
    reduce: str = ""

    # `transform` is nested IR, not operands of this Store.
    _operand_fields: ClassVar[tuple[str, ...]] = (
        "ptr",
        "offsets",
        "value",
        "mask",
        "row_indices",
        "col_indices",
        "base_offset",
        "acc_post_scale",
    )


@dataclass
class Load4Vec(TileOp):
    """Vectorized load: read 4 consecutive f16 values as half4.

    ptr: base pointer, offsets: scalar element offset (thread-specific).
    Emits: *(device const half4*)(ptr + offsets * 4)
    Result is an opaque 4-element vector used by Dot4.
    """

    ptr: TileValue | None = None
    offsets: TileValue | None = None  # element index (each thread loads 4 elements at offsets*4)
    _operand_fields: ClassVar[tuple[str, ...]] = ("ptr", "offsets")


@dataclass
class LoadWide(TileOp):
    """Read ONE scalar of a wider type `wide` at a BYTE offset from a byte
    buffer. Emits *(device const <wide>*)((device const char*)ptr + offsets).

    `offsets` is in the buffer's element units (bytes for a u8 buffer). Matches
    llama.cpp's `(half)xb->d` / `(uint16_t*)xb->scales` — one aligned wide load
    instead of assembling a value from 2 byte loads + shift/or."""

    ptr: TileValue | None = None
    offsets: TileValue | None = None  # byte offset
    wide: str = "u16"
    _operand_fields: ClassVar[tuple[str, ...]] = ("ptr", "offsets")


@dataclass
class Dot4(TileOp):
    """Vectorized dot product: dot(float4(a), float4(b)) → scalar f32.

    a, b are Load4Vec results (half4 vectors).
    Emits: dot(float4(a), float4(b))
    """

    a: TileValue | None = None
    b: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("a", "b")


@dataclass
class Unpack4(TileOp):
    """Extract one scalar component from a vec4 (Load4Vec result).

    `lane` selects the component: 0=.x, 1=.y, 2=.z, 3=.w. Result is a
    scalar matching the elementwise dtype of the source vec.

    Lets a kernel issue one vectorised load and use the components as scalar
    register values for downstream FMAs; the alternative is 4 separate scalar
    loads that emit 4× the LSU instructions.
    """

    a: TileValue | None = None
    lane: int = 0
    _operand_fields: ClassVar[tuple[str, ...]] = ("a",)


@dataclass
class AsChar4(TileOp):
    """Reinterpret one uint component of a uint4 (load4_vec on a u32-viewed
    buffer) as 4 signed int8 codes, promoted to float4.

    `lane` selects the component: 0=.x .. 3=.w. Emits
    `float4 r = float4(as_type<char4>(a.{x,y,z,w}));` — one hardware
    reinterpret + one convert. Lets a quantized-KV kernel fetch 16 int8 codes
    with ONE 16-byte load (uint4) and feed dot4 float4 operands: 4x fewer
    load-issue slots than four char4 loads on the load-issue-bound decode
    attention path."""

    a: TileValue | None = None
    lane: int = 0
    _operand_fields: ClassVar[tuple[str, ...]] = ("a",)


@dataclass
class InterleaveVec4(TileOp):
    """Interleave two uchar4/half4 vectors → uchar4/half4.

    Combines components of `lo` and `hi` so that the result holds two adjacent
    pairs from each, controlled by `half`:
      half=0: result = vec4(lo.x, hi.x, lo.y, hi.y)
      half=1: result = vec4(lo.z, hi.z, lo.w, hi.w)

    Use case: Q4_K nibble unpacking. `lo = raw4 & 0x0F` and
    `hi = (raw4 >> 4) & 0x0F` give "every-other-K" weight nibbles; this op
    re-aligns them with consecutive-K activations for dot4 consumption.
    """

    lo: TileValue | None = None
    hi: TileValue | None = None
    half: int = 0
    _operand_fields: ClassVar[tuple[str, ...]] = ("lo", "hi")


@dataclass
class Dot(TileOp):
    """Tile matrix multiply: (M, K) x (K, N) -> (M, N).

    When acc is set, this dot accumulates into persistent simdgroup registers
    owned by the acc value (a Zeros op). The emitter keeps accumulators across
    ForLoop iterations instead of materializing to shared memory each step.
    """

    lhs: TileValue | None = None  # (M, K) tile, or (K, M) if transpose_lhs
    rhs: TileValue | None = None  # (K, N) tile, or (N, K) if transpose_rhs
    transpose_lhs: bool = False  # if True, lhs is (K, M) and transposed
    transpose_rhs: bool = False  # if True, rhs is (N, K) and transposed
    acc: TileValue | None = None  # if set, accumulate into this value's registers
    acc_pre_scale: TileValue | None = None  # FA-2 forward per-row alpha
    _operand_fields: ClassVar[tuple[str, ...]] = ("lhs", "rhs", "acc", "acc_pre_scale")


@dataclass
class Barrier(TileOp):
    """Threadgroup synchronization barrier."""

    pass
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class Reduce(TileOp):
    """Reduce a tile along an axis."""

    input: TileValue | None = None
    axis: int = 0
    op: str = ""  # "sum", "max", "min"
    _operand_fields: ClassVar[tuple[str, ...]] = ("input",)


@dataclass
class SimdReduce(TileOp):
    """SIMD-level reduction across the simdgroup (32 lanes).

    Emits metal::simd_sum (or simd_max etc.) for hardware-accelerated
    cross-lane reduction. Input must be a scalar per thread.
    """

    input: TileValue | None = None
    op: str = "sum"  # "sum", "max", "min"
    _operand_fields: ClassVar[tuple[str, ...]] = ("input",)


@dataclass
class ForLoop(TileOp):
    """A loop with loop-carried values (e.g., K-loop in GEMM).

    carried: list of (init_value, body_final_value) pairs.
    The emitter declares init_value before the loop, updates in-place
    inside the body, and reads body_final_value after the loop.

    start/end/step can be int (compile-time) or TileValue (runtime).
    """

    var: str = ""
    start: int | TileValue = 0
    end: int | TileValue = 0
    step: int | TileValue = 1
    body: list[TileOp] = field(default_factory=list)
    carried: list[tuple[TileValue, TileValue]] = field(default_factory=list)

    # `body` is nested IR (not operands of THIS ForLoop). start/end/step may be
    # TileValues — the fast walker handles both int and TileValue via runtime
    # isinstance, same as the generic walker.
    _operand_fields: ClassVar[tuple[str, ...]] = ("start", "end", "step", "carried")


@dataclass
class WhileLoop(TileOp):
    """A while loop with a runtime condition and loop-carried values.

    cond_body: ops that compute the condition value (re-evaluated each iteration).
    cond: the TileValue that is the boolean condition.
    body: ops executed when condition is true.
    carried: list of (init_value, body_final_value) pairs, same as ForLoop.
    """

    cond_body: list[TileOp] = field(default_factory=list)
    cond: TileValue | None = None
    body: list[TileOp] = field(default_factory=list)
    carried: list[tuple[TileValue, TileValue]] = field(default_factory=list)

    _operand_fields: ClassVar[tuple[str, ...]] = ("cond", "carried")


@dataclass
class ProgramId(TileOp):
    """Get the program (threadgroup) index along an axis."""

    axis: int = 0
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class Constant(TileOp):
    """A compile-time constant value (constexpr or literal)."""

    value: Any = None
    _operand_fields: ClassVar[tuple[str, ...]] = ()


# --- Per-thread ops (shared memory, atomics, SIMD, control flow) ---


@dataclass
class ThreadId(TileOp):
    """Get the thread index within the threadgroup."""

    pass
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class NumPrograms(TileOp):
    """Get the number of threadgroups along an axis."""

    axis: int = 0
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class SharedAlloc(TileOp):
    """Allocate threadgroup shared memory."""

    size: int = 0
    dtype: str = "f32"
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class LocalAlloc(TileOp):
    """Allocate thread-local array."""

    size: int = 0
    dtype: str = "f32"
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class IndexLoad(TileOp):
    """Scalar load from array by index: base[index]."""

    base: TileValue | None = None
    index: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("base", "index")


@dataclass
class IndexStore(TileOp):
    """Scalar store to array by index: base[index] = value."""

    base: TileValue | None = None
    index: TileValue | None = None
    value: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("base", "index", "value")


@dataclass
class Atomic(TileOp):
    """Atomic memory operation.

    op: "add", "max", "min", "cas", "xchg", "and", "or", "xor",
        "add_float", "max_float", "min_float"
    """

    op: str = ""
    ptr: TileValue | None = None
    index: TileValue | None = None
    value: TileValue | None = None
    expected: TileValue | None = None  # for CAS only
    _operand_fields: ClassVar[tuple[str, ...]] = ("ptr", "index", "value", "expected")


@dataclass
class SimdOp(TileOp):
    """SIMD group operation.

    op: "shuffle_xor", "shuffle", "shuffle_up", "shuffle_down",
        "prefix_exclusive_sum", "prefix_inclusive_sum",
        "all", "any", "id", "lane_id"
    """

    op: str = ""
    args: list[TileValue] = field(default_factory=list)
    _operand_fields: ClassVar[tuple[str, ...]] = ("args",)


@dataclass
class SimdMatrixOp(TileOp):
    """SIMD group matrix operation (manual MMA).

    op: "create", "load", "store", "mma"
    """

    op: str = ""
    args: list[TileValue] = field(default_factory=list)
    stride: int | None = None
    transpose: bool = False
    _operand_fields: ClassVar[tuple[str, ...]] = ("args",)


@dataclass
class DebugPrint(TileOp):
    """Debug printf for kernel debugging."""

    fmt: str = ""
    args: list[TileValue] = field(default_factory=list)
    _operand_fields: ClassVar[tuple[str, ...]] = ("args",)


@dataclass
class IfElse(TileOp):
    """Runtime conditional with body and optional else branch."""

    cond: TileValue | None = None
    body: list[TileOp] = field(default_factory=list)
    orelse: list[TileOp] = field(default_factory=list)
    # Variable merges: (result, true_val, false_val) — replaces Select for
    # IfElse scope safety. Result is pre-declared at outer scope, assigned
    # inside the if/else blocks.
    merges: list[tuple[TileValue, TileValue, TileValue]] = field(default_factory=list)

    # `body`/`orelse` are nested IR, not operands of THIS IfElse.
    _operand_fields: ClassVar[tuple[str, ...]] = ("cond", "merges")


@dataclass
class FlowControl(TileOp):
    """Break, continue, or return."""

    kind: str = ""  # "break", "continue", "return"
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class Copy(TileOp):
    """Identity copy — creates a distinct variable for the same value."""

    source: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("source",)


@dataclass
class Cast(TileOp):
    """Type cast."""

    input: TileValue | None = None
    target_dtype: str = "f32"
    _operand_fields: ClassVar[tuple[str, ...]] = ("input",)


@dataclass
class CoopLoad(TileOp):
    """Cooperative threadgroup load with auto-barrier."""

    dst: TileValue | None = None
    src: TileValue | None = None
    count: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("dst", "src", "count")


@dataclass
class Copy4(TileOp):
    """Vec4 memory copy: dst[dst_offset] ← src_ptr[src_offset] as float4."""

    dst: TileValue | None = None
    dst_offset: TileValue | None = None
    src_ptr: TileValue | None = None
    src_offset: TileValue | None = None
    _operand_fields: ClassVar[tuple[str, ...]] = ("dst", "dst_offset", "src_ptr", "src_offset")


@dataclass
class FusedElementwise(TileOp):
    """A sequence of elementwise 2D ops fused into a single row loop.

    Created by the `_opt_fuse_row_loops` pass in tile_opt.py.
    The emitter emits one `for (_c)` loop with all ops inlined.

    ops:          constituent BinOp/UnaryOp/Select/Compare/TernaryOp
    writeback:    set of result names that must be written to shmem
    source_buf:   name of the shmem buffer the chain reads from
    source_stride: stride of that buffer
    """

    ops: list[TileOp] = field(default_factory=list)
    writeback: set[str] = field(default_factory=set)
    source_buf: str = ""
    source_stride: int = 0
    # Flat-threaded emit: one element per thread (strided when tile >
    # NUM_THREADS). Set by `_opt_fuse_row_loops` when `_chain_is_flat_safe`.
    flat_threaded: bool = False

    # `ops` holds nested TileOps — walked by walk_ops, not operands of THIS op.
    _operand_fields: ClassVar[tuple[str, ...]] = ()


@dataclass
class RowPass(TileOp):
    """Multi-phase row-local computation: elementwise chains + row reductions.

    The emitter scans `ops` in order and groups them into phases. Each phase
    is one `for (_c)` column loop. Within a phase:

        * 2D elementwise ops (BinOp/UnaryOp/Select/Compare/TernaryOp with a
          2D result) are evaluated per column as part of the loop body.
        * Reduce(axis=1) ops accumulate into simd-resident scalars during the
          same column loop, in parallel with the elementwise ops.
        * Store ops in `ops` emit writes inside the column loop.

    A phase ends when a subsequent op depends (directly or transitively) on a
    Reduce result produced in the current phase. At that boundary the emitter
    issues the cross-lane butterfly for every pending reduction, then emits
    any 1D / scalar ops (e.g. `s / N`) outside the column loop. The next phase
    opens a new column loop that can read the 2D operands again and reference
    the now-resolved scalar reductions as broadcast values.

    Contract:
        * All 2D ops produce tiles of the same `(M, N)` shape.
        * 2D operands read by later phases must be live across phases — either
          shared-memory tiles or cooperative Loads re-emitted each phase.
        * `writeback` names the 2D results that downstream IR consumers need
          in shared memory; everything else stays as per-thread expressions.

    Created by the row-pass fusion planner; lowered by `_emit_row_pass`.
    """

    ops: list[TileOp] = field(default_factory=list)
    writeback: set[str] = field(default_factory=set)

    _operand_fields: ClassVar[tuple[str, ...]] = ()


# --- IR traversal utilities ---


def walk_ops(ops: list[TileOp]) -> Iterator[TileOp]:
    """Yield all ops recursively, descending into ForLoop/WhileLoop/IfElse bodies."""
    for op in ops:
        yield op
        if isinstance(op, ForLoop):
            yield from walk_ops(op.body)
        elif isinstance(op, WhileLoop):
            yield from walk_ops(op.cond_body)
            yield from walk_ops(op.body)
        elif isinstance(op, IfElse):
            yield from walk_ops(op.body)
            yield from walk_ops(op.orelse)
        elif isinstance(op, FusedElementwise):
            yield from walk_ops(op.ops)
        elif isinstance(op, RowPass):
            yield from walk_ops(op.ops)


def get_op_reads(op: TileOp) -> frozenset[str]:
    """Names of all TileValues read by an op (excluding result)."""
    d = op.__dict__
    cached = d.get("_reads_cache")
    if cached is None:
        cached = frozenset(v.name for v in op.operand_values())
        d["_reads_cache"] = cached
    return cached


def shallow_clone_for_fusion(func: "TileFunction") -> "TileFunction":
    """Clone a TileFunction for fusion-time mutation without deep-copying
    every op.

    Fusion mutates only the ops lists (tee-branch Stores get inserted) and
    Load/Store fields (transform, transform_extras, transform_source_name,
    col_slice). Everything else is shared. Load/Store ops are a minority of
    the op count, so this is substantially cheaper than copy.deepcopy(func).
    """
    def _clone_list(ops: list[TileOp]) -> list[TileOp]:
        return [_clone_op(op) for op in ops]

    def _clone_op(op: TileOp) -> TileOp:
        if isinstance(op, (Load, Store)):
            if isinstance(op, Store):
                return dataclasses.replace(
                    op,
                    transform=list(op.transform),
                    transform_extras=dict(op.transform_extras),
                    acc_extras=dict(op.acc_extras),
                )
            return dataclasses.replace(
                op,
                transform=list(op.transform),
                transform_extras=dict(op.transform_extras),
            )
        if isinstance(op, ForLoop):
            return dataclasses.replace(op, body=_clone_list(op.body))
        if isinstance(op, WhileLoop):
            return dataclasses.replace(
                op, cond_body=_clone_list(op.cond_body), body=_clone_list(op.body)
            )
        if isinstance(op, IfElse):
            return dataclasses.replace(
                op, body=_clone_list(op.body), orelse=_clone_list(op.orelse)
            )
        if isinstance(op, FusedElementwise):
            return dataclasses.replace(op, ops=_clone_list(op.ops))
        if isinstance(op, RowPass):
            return dataclasses.replace(op, ops=_clone_list(op.ops))
        return op

    return dataclasses.replace(
        func,
        params=list(func.params),
        ops=_clone_list(func.ops),
        constexpr_values=dict(func.constexpr_values),
        shape_vars=dict(func.shape_vars),
        buffer_shapes=dict(func.buffer_shapes),
        options=dict(func.options),
    )


# --- Tile function — the top-level IR unit ---


@dataclass
class TileFunction:
    """A complete kernel in tile IR form."""

    name: str
    params: list[TileParam] = field(default_factory=list)
    ops: list[TileOp] = field(default_factory=list)
    constexpr_values: dict[str, Any] = field(default_factory=dict)
    shape_vars: dict[str, tuple[int, ...]] = field(default_factory=dict)
    buffer_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dispatch_spec: DispatchContract | None = None
    # Compiler options — control IR passes and codegen, never emitted as MSL
    # constants. Populated from @al.tunable(options=...) and explicit kwargs.
    options: dict[str, Any] = field(default_factory=dict)

    def add_op(self, op: TileOp) -> TileValue | None:
        self.ops.append(op)
        return op.result

    @cached_property
    def fingerprint(self) -> str:
        """Stable hash of a lowered TileFunction for cache identity."""
        ir_dump = dump_tile_ir(self)
        fp = hashlib.sha256(ir_dump.encode("utf-8")).hexdigest()
        return fp


@dataclass
class TileParam:
    """A kernel parameter."""

    name: str
    is_constexpr: bool = False
    dtype: str = "f32"


# --- Kernel plan — all optimization decisions for a tile kernel ---


@dataclass
class ShmemBuffer:
    """A shared memory buffer in the plan."""

    name: str  # e.g., "_s0"
    rows: int
    cols: int
    stride: int  # cols + pad
    total_elems: int  # rows * stride


@dataclass
class TileKernelPlan:
    """All optimization decisions for compiling a tile kernel to MSL.

    Computed by plan_tile_kernel() from analysis of the tile IR.
    Consumed by the emitter — the emitter makes no optimization decisions.
    """

    # Thread model
    threads: int = 256  # total threads per threadgroup
    tpr: int = 1  # threads per row (1 = serial, 32 = simdgroup)

    # Data types
    dtype: str = "float"  # compute dtype (MSL type name)
    acc_dtype: str = "float"  # accumulator dtype
    shmem_dtype: str = "float"  # shared memory dtype

    # Register blocking (for GEMM / MMA)
    reg_m: int = 2  # register tile rows (1, 2, or 4)
    reg_n: int = 2  # register tile cols
    # Per-dot reg pick target: when set, `pick_dot_reg` will downsize
    # individual dots whose default (M, N)-only reg pick would leave many
    # simdgroups idle relative to other dots in the same kernel. Computed
    # by `_pass_thread_model` as max n_sg across all dots, plumbed through
    # to the emitter so per-dot guards/accumulators match planner's pick.
    n_sg_target: int | None = None

    # Shared memory
    shmem_plan: dict = field(default_factory=dict)  # val_name → (buf, rows, cols, stride)
    # 2D Load results that feed a Dot as a NON-shared operand (reuse scope = 1
    # simdgroup: nothing else in the threadgroup needs the bytes) — streamed
    # straight from device into the MMA via simdgroup_load, skipping the
    # cooperative-load → shmem → barrier round-trip (Flash-Attention K/V). Maps
    # val_name → ("lhs"|"rhs") slot in its consuming Dot.
    device_direct_loads: dict = field(default_factory=dict)
    shmem_buffers: list[ShmemBuffer] = field(default_factory=list)
    # Per-buffer shmem dtype override. Defaults to `shmem_dtype` (the kernel-
    # global dtype) when a buffer name isn't present here. Used when the
    # planner can keep bf16 input loads in bf16 shmem (halving their footprint)
    # while keeping precision-sensitive intermediate buffers (Dot results,
    # BinOps that consume Dot results) in f32 — relevant for SDPA bwd at
    # HIGH_PRECISION=1 where K-bias amplification needs f32 in `s - lse` but
    # the Q/K/V loads themselves can stay bf16 in shmem.
    shmem_buf_dtype: dict = field(default_factory=dict)  # buf_name → dtype

    # Column tiling
    col_tiled: bool = False
    block_n: int | None = None  # None = no column tiling

    # Register-resident mode (skip shared memory for small N)
    register_resident: bool = False

    # Row bounds
    row_bound: str | None = None  # "M" when M is in constexprs

    # Buffer classification
    buffer_params: list[str] = field(default_factory=list)
    outputs: set[str] = field(default_factory=set)

    # Per-buffer dtypes (for mixed precision)
    buffer_dtypes: dict[str, str] = field(default_factory=dict)  # param → MSL type

    # Vectorization
    vec_width: int = 4
    pad: int = 4

    # Double-buffering (K-loop latency hiding)
    double_buffer: bool = False


# --- Builder — convenient API for constructing tile IR ---


class TileBuilder:
    """Fluent API for building tile IR programs."""

    def __init__(self, name: str):
        self.func = TileFunction(name=name)
        self._counter = 0

    def _fresh(self, prefix: str = "v") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def add_param(
        self, name: str, is_constexpr: bool = False, dtype: str = "f32"
    ) -> TileValue | None:
        self.func.params.append(TileParam(name=name, is_constexpr=is_constexpr, dtype=dtype))
        if is_constexpr:
            return None
        return TileValue(name=name, shape=(), layout=Layout.REPLICATED, dtype=dtype)

    def set_constexprs(self, values: dict[str, Any]):
        self.func.constexpr_values = values

    def program_id(self, axis: int = 0) -> TileValue:
        v = TileValue(name=self._fresh("pid"), shape=(), layout=Layout.REPLICATED, dtype="i32")
        self.func.add_op(ProgramId(result=v, axis=axis))
        return v

    def make_range(self, start: int, end: int) -> TileValue:
        n = end - start
        v = TileValue(name=self._fresh("rng"), shape=(n,), layout=Layout.BLOCKED, dtype="i32")
        self.func.add_op(MakeRange(result=v, start=start, end=end))
        return v

    def constant(self, value: Any, dtype: str = "i32") -> TileValue:
        v = TileValue(name=self._fresh("c"), shape=(), layout=Layout.REPLICATED, dtype=dtype)
        self.func.add_op(Constant(result=v, value=value))
        return v

    def splat(self, value: TileValue, shape: tuple[int, ...]) -> TileValue:
        layout = Layout.BLOCKED if len(shape) > 0 else Layout.REPLICATED
        v = TileValue(name=self._fresh("spl"), shape=shape, layout=layout, dtype=value.dtype)
        self.func.add_op(Splat(result=v, value=value, shape=shape))
        return v

    def binop(self, op: str, lhs: TileValue, rhs: TileValue) -> TileValue:
        # Broadcast shapes
        shape = _broadcast_shape(lhs.shape, rhs.shape)
        layout = _pick_layout(lhs.layout, rhs.layout)
        dtype = _result_dtype(lhs.dtype, rhs.dtype, op)
        v = TileValue(name=self._fresh("t"), shape=shape, layout=layout, dtype=dtype)
        self.func.add_op(BinOp(result=v, op=op, lhs=lhs, rhs=rhs))
        return v

    def ternary(self, op: str, a: TileValue, b: TileValue, c: TileValue) -> TileValue:
        shape = _broadcast_shape(_broadcast_shape(a.shape, b.shape), c.shape)
        layout = _pick_layout(_pick_layout(a.layout, b.layout), c.layout)
        dtype = _result_dtype(a.dtype, b.dtype, op)
        v = TileValue(name=self._fresh("t"), shape=shape, layout=layout, dtype=dtype)
        self.func.add_op(TernaryOp(result=v, op=op, a=a, b=b, c=c))
        return v

    def compare(self, op: str, lhs: TileValue, rhs: TileValue) -> TileValue:
        shape = _broadcast_shape(lhs.shape, rhs.shape)
        layout = _pick_layout(lhs.layout, rhs.layout)
        v = TileValue(name=self._fresh("cmp"), shape=shape, layout=layout, dtype="bool")
        self.func.add_op(Compare(result=v, op=op, lhs=lhs, rhs=rhs))
        return v

    def bool_op(self, op: str, lhs: TileValue, rhs: TileValue) -> TileValue:
        shape = _broadcast_shape(lhs.shape, rhs.shape)
        layout = _pick_layout(lhs.layout, rhs.layout)
        v = TileValue(name=self._fresh("bop"), shape=shape, layout=layout, dtype="bool")
        self.func.add_op(BoolOp(result=v, op=op, lhs=lhs, rhs=rhs))
        return v

    def select(self, cond: TileValue, true_val: TileValue, false_val: TileValue) -> TileValue:
        shape = _broadcast_shape(true_val.shape, false_val.shape)
        layout = _pick_layout(true_val.layout, false_val.layout)
        v = TileValue(name=self._fresh("sel"), shape=shape, layout=layout, dtype=true_val.dtype)
        self.func.add_op(Select(result=v, cond=cond, true_val=true_val, false_val=false_val))
        return v

    def unary(self, op: str, input: TileValue) -> TileValue:
        v = TileValue(
            name=self._fresh("t"),
            shape=input.shape,
            layout=input.layout,
            dtype=input.dtype,
        )
        self.func.add_op(UnaryOp(result=v, op=op, input=input))
        return v

    def load(
        self,
        ptr: TileValue,
        offsets: TileValue,
        mask: TileValue | None = None,
        other: float = 0.0,
        dtype: str = "f32",
        row_indices: TileValue | None = None,
        col_indices: TileValue | None = None,
        row_stride: int | None = None,
        base_offset: TileValue | None = None,
        addr_transposed: bool = False,
        pack_factor: int = 0,
        pack_bits: int = 0,
        dequant_scale_ptr: TileValue | None = None,
        dequant_bias_ptr: TileValue | None = None,
        dequant_zero_point: float = 0.0,
        dequant_n_groups: int = 0,
        dequant_format: str = "",
        dequant_high_ptr: TileValue | None = None,
    ) -> TileValue:
        v = TileValue(
            name=self._fresh("ld"),
            shape=offsets.shape,
            layout=Layout.BLOCKED,
            dtype=dtype,
        )
        self.func.add_op(
            Load(
                result=v,
                ptr=ptr,
                offsets=offsets,
                mask=mask,
                other=other,
                row_indices=row_indices,
                col_indices=col_indices,
                row_stride=row_stride,
                base_offset=base_offset,
                addr_transposed=addr_transposed,
                pack_factor=pack_factor,
                pack_bits=pack_bits,
                dequant_scale_ptr=dequant_scale_ptr,
                dequant_bias_ptr=dequant_bias_ptr,
                dequant_zero_point=dequant_zero_point,
                dequant_n_groups=dequant_n_groups,
                dequant_format=dequant_format,
                dequant_high_ptr=dequant_high_ptr,
            )
        )
        return v

    def store(
        self,
        ptr: TileValue,
        offsets: TileValue,
        value: TileValue,
        mask: TileValue | None = None,
        row_indices: TileValue | None = None,
        col_indices: TileValue | None = None,
        row_stride: int | None = None,
        base_offset: TileValue | None = None,
        reduce: str = "",
    ):
        self.func.add_op(
            Store(
                ptr=ptr,
                offsets=offsets,
                value=value,
                mask=mask,
                row_indices=row_indices,
                col_indices=col_indices,
                row_stride=row_stride,
                base_offset=base_offset,
                reduce=reduce,
            )
        )

    def expand_dims(self, input: TileValue, axis: int) -> TileValue:
        shape = list(input.shape)
        shape.insert(axis, 1)
        v = TileValue(
            name=self._fresh("exp"),
            shape=tuple(shape),
            layout=input.layout,
            dtype=input.dtype,
        )
        self.func.add_op(ExpandDims(result=v, input=input, axis=axis))
        return v

    def dot(
        self,
        lhs: TileValue,
        rhs: TileValue,
        transpose_rhs: bool = False,
        transpose_lhs: bool = False,
    ) -> TileValue:
        assert lhs.rank == 2 and rhs.rank == 2
        assert not (transpose_lhs and transpose_rhs), "transpose both sides not supported"
        if transpose_rhs:
            M, K = lhs.shape
            N, K2 = rhs.shape
            assert K == K2, f"dot transpose_rhs: K mismatch {K} vs {K2}"
        elif transpose_lhs:
            K, M = lhs.shape
            K2, N = rhs.shape
            assert K == K2, f"dot transpose_lhs: K mismatch {K} vs {K2}"
        else:
            assert lhs.shape[1] == rhs.shape[0]
            M, K = lhs.shape
            _, N = rhs.shape
        v = TileValue(name=self._fresh("dot"), shape=(M, N), layout=Layout.MMA, dtype=lhs.dtype)
        self.func.add_op(
            Dot(
                result=v,
                lhs=lhs,
                rhs=rhs,
                transpose_rhs=transpose_rhs,
                transpose_lhs=transpose_lhs,
            )
        )
        return v

    def reduce(self, input: TileValue, axis: int, op: str) -> TileValue:
        shape = list(input.shape)
        if len(shape) >= 2:
            # 2D+ input: keepdim (replace axis with 1) for broadcastability
            shape[axis] = 1
        else:
            # 1D input: reduce to scalar
            shape.pop(axis)
        v = TileValue(
            name=self._fresh("red"),
            shape=tuple(shape),
            layout=input.layout,
            dtype=input.dtype,
        )
        self.func.add_op(Reduce(result=v, input=input, axis=axis, op=op))
        return v

    def load4_vec(self, ptr: TileValue, offsets: TileValue) -> TileValue:
        """Vectorized load: 4 consecutive values as a native vec4."""
        v = TileValue(
            name=self._fresh("ld4"),
            shape=(),  # opaque vector, scalar per thread
            layout=offsets.layout,
            dtype=ptr.dtype,
        )
        self.func.add_op(Load4Vec(result=v, ptr=ptr, offsets=offsets))
        return v

    def load_wide(self, ptr: TileValue, offsets: TileValue, wide: str) -> TileValue:
        """Read one scalar of type `wide` at a byte offset (reinterpret load)."""
        v = TileValue(
            name=self._fresh("ldw"),
            shape=(),
            layout=offsets.layout,
            dtype=wide,
        )
        self.func.add_op(LoadWide(result=v, ptr=ptr, offsets=offsets, wide=wide))
        return v

    def dot4(self, a: TileValue, b: TileValue) -> TileValue:
        """Vectorized dot: dot(float4(a), float4(b)) → scalar f32."""
        v = TileValue(
            name=self._fresh("dot4"),
            shape=(),
            layout=a.layout,
            dtype="f32",
        )
        self.func.add_op(Dot4(result=v, a=a, b=b))
        return v

    def unpack4(self, a: TileValue, lane: int) -> TileValue:
        """Extract one component (lane in 0..3) of a vec4 as a scalar."""
        if lane not in (0, 1, 2, 3):
            raise ValueError(f"unpack4 lane must be in 0..3, got {lane}")
        # Element dtype: load4_vec results carry the source-pointer dtype as
        # their dtype (see Load4Vec). Promote to f32 in the result so
        # downstream scalar arithmetic stays in float and matches dot4.
        v = TileValue(
            name=self._fresh("unp4"),
            shape=(),
            layout=a.layout,
            dtype="f32",
        )
        self.func.add_op(Unpack4(result=v, a=a, lane=int(lane)))
        return v

    def as_char4(self, a: TileValue, lane: int) -> TileValue:
        """Reinterpret uint component `lane` of a uint4 as char4 -> float4."""
        if lane not in (0, 1, 2, 3):
            raise ValueError(f"as_char4 lane must be in 0..3, got {lane}")
        v = TileValue(
            name=self._fresh("c4"),
            shape=(),
            layout=a.layout,
            dtype="f32",
        )
        self.func.add_op(AsChar4(result=v, a=a, lane=int(lane)))
        return v

    def interleave_vec4(self, lo: TileValue, hi: TileValue, half: int) -> TileValue:
        """Interleave two vec4s: half=0 → (lo.x, hi.x, lo.y, hi.y),
        half=1 → (lo.z, hi.z, lo.w, hi.w)."""
        v = TileValue(
            name=self._fresh("ilv4"),
            shape=(),
            layout=lo.layout,
            dtype=lo.dtype,
        )
        self.func.add_op(InterleaveVec4(result=v, lo=lo, hi=hi, half=int(half)))
        return v

    def simd_reduce(self, input: TileValue, op: str = "sum") -> TileValue:
        """SIMD cross-lane reduction. Input is scalar-per-thread, output is broadcast scalar."""
        v = TileValue(
            name=self._fresh("simd"),
            shape=(),
            layout=input.layout,
            dtype=input.dtype,
        )
        self.func.add_op(SimdReduce(result=v, input=input, op=op))
        return v

    def build(self) -> TileFunction:
        return self.func


# --- Shape / layout / dtype helpers ---


def _broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Numpy-style broadcast shape computation."""
    if a == ():
        return b
    if b == ():
        return a
    rank = max(len(a), len(b))
    a = (1,) * (rank - len(a)) + a
    b = (1,) * (rank - len(b)) + b
    result = []
    for da, db in zip(a, b):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            raise ValueError(f"Cannot broadcast shapes {a} and {b}")
    return tuple(result)


def _pick_layout(a: Layout, b: Layout) -> Layout:
    """Pick the dominant layout from two operands."""
    if a == b:
        return a
    # MMA dominates everything
    if a == Layout.MMA or b == Layout.MMA:
        return Layout.MMA
    # Blocked dominates Replicated
    if a == Layout.BLOCKED or b == Layout.BLOCKED:
        return Layout.BLOCKED
    return Layout.REPLICATED


def _result_dtype(a: str, b: str, op: str) -> str:
    """Pick result dtype from two operands."""
    # Float wins over int
    float_types = {"f32", "f16", "bf16"}
    if a in float_types or b in float_types:
        if a == "f32" or b == "f32":
            return "f32"
        if a == "f16" or b == "f16":
            return "f16"
        return "bf16"
    return a  # both int


# --- Pretty printer ---


def dump_tile_ir(func: TileFunction) -> str:
    """Dump tile IR as readable text."""
    lines = []
    params = []
    for p in func.params:
        s = p.name
        if p.is_constexpr:
            s += ": constexpr"
            val = func.constexpr_values.get(p.name)
            if val is not None:
                s += f" = {val}"
        else:
            s += f": {p.dtype}"
        params.append(s)
    lines.append(f"tile_func {func.name}({', '.join(params)}) {{")

    def _dump_transform(xf_ops, indent):
        pad = "  " * indent
        for xop in xf_ops:
            r = xop.result
            xp = f"{pad}{r.name}: {r.dtype}{list(r.shape)}" if r else pad
            if isinstance(xop, Constant):
                lines.append(f"{xp} = constant({xop.value})")
            elif isinstance(xop, BinOp):
                lines.append(f"{xp} = {xop.op}({xop.lhs.name}, {xop.rhs.name})")
            elif isinstance(xop, UnaryOp):
                lines.append(f"{xp} = {xop.op}({xop.input.name})")
            else:
                lines.append(f"{pad}<{type(xop).__name__}>")

    def _dump_ops(ops, indent):
        pad = "  " * indent
        for op in ops:
            r = op.result
            prefix = f"{pad}{r.name}: {r.dtype}{list(r.shape)} [{r.layout.name}]" if r else pad

            if isinstance(op, ProgramId):
                lines.append(f"{prefix} = program_id({op.axis})")
            elif isinstance(op, MakeRange):
                lines.append(f"{prefix} = make_range({op.start}, {op.end})")
            elif isinstance(op, Constant):
                lines.append(f"{prefix} = constant({op.value})")
            elif isinstance(op, Splat):
                lines.append(f"{prefix} = splat({op.value.name}, {list(op.shape)})")
            elif isinstance(op, ExpandDims):
                lines.append(f"{prefix} = expand_dims({op.input.name}, axis={op.axis})")
            elif isinstance(op, BinOp):
                lines.append(f"{prefix} = {op.op}({op.lhs.name}, {op.rhs.name})")
            elif isinstance(op, UnaryOp):
                lines.append(f"{prefix} = {op.op}({op.input.name})")
            elif isinstance(op, TernaryOp):
                lines.append(f"{prefix} = {op.op}({op.a.name}, {op.b.name}, {op.c.name})")
            elif isinstance(op, Compare):
                lines.append(f"{prefix} = {op.op}({op.lhs.name}, {op.rhs.name})")
            elif isinstance(op, BoolOp):
                lines.append(f"{prefix} = {op.op}({op.lhs.name}, {op.rhs.name})")
            elif isinstance(op, Load):
                mask = f", mask={op.mask.name}" if op.mask else ""
                other = f", other={op.other}" if op.mask else ""
                lines.append(f"{prefix} = load({op.ptr.name} + {op.offsets.name}{mask}{other})")
                if op.transform:
                    lines.append(f"{pad}  transform {{")
                    _dump_transform(op.transform, indent + 2)
                    lines.append(f"{pad}  }}")
            elif isinstance(op, Store):
                mask = f", mask={op.mask.name}" if op.mask else ""
                lines.append(
                    f"{pad}store({op.ptr.name} + {op.offsets.name}, {op.value.name}{mask})"
                )
                if op.transform:
                    lines.append(f"{pad}  transform {{")
                    _dump_transform(op.transform, indent + 2)
                    lines.append(f"{pad}  }}")
            elif isinstance(op, Dot):
                lines.append(f"{prefix} = dot({op.lhs.name}, {op.rhs.name})")
            elif isinstance(op, Reduce):
                lines.append(f"{prefix} = reduce_{op.op}({op.input.name}, axis={op.axis})")
            elif isinstance(op, Zeros):
                lines.append(f"{prefix} = zeros({list(op.shape)}, {op.dtype})")
            elif isinstance(op, Barrier):
                lines.append(f"{pad}barrier()")
            elif isinstance(op, Select):
                lines.append(
                    f"{prefix} = select({op.cond.name}, {op.true_val.name}, {op.false_val.name})"
                )
            elif isinstance(op, ForLoop):
                carried_str = ", ".join(f"{i.name}->{f.name}" for i, f in op.carried)
                lines.append(
                    f"{pad}for {op.var} in range({op.start}, {op.end}, {op.step}) carried=[{carried_str}] {{"
                )
                _dump_ops(op.body, indent + 1)
                lines.append(f"{pad}}}")
            elif isinstance(op, WhileLoop):
                carried_str = ", ".join(f"{i.name}->{f.name}" for i, f in op.carried)
                lines.append(f"{pad}while carried=[{carried_str}] {{")
                _dump_ops(op.cond_body, indent + 1)
                lines.append(f"{pad}  cond: {op.cond.name}")
                _dump_ops(op.body, indent + 1)
                lines.append(f"{pad}}}")
            elif isinstance(op, ThreadId):
                lines.append(f"{prefix} = thread_id()")
            elif isinstance(op, NumPrograms):
                lines.append(f"{prefix} = num_programs({op.axis})")
            elif isinstance(op, SharedAlloc):
                lines.append(f"{prefix} = shared_alloc({op.size}, {op.dtype})")
            elif isinstance(op, LocalAlloc):
                lines.append(f"{prefix} = local_alloc({op.size}, {op.dtype})")
            elif isinstance(op, IndexLoad):
                lines.append(f"{prefix} = index_load({op.base.name}[{op.index.name}])")
            elif isinstance(op, IndexStore):
                lines.append(f"{pad}index_store({op.base.name}[{op.index.name}], {op.value.name})")
            elif isinstance(op, Atomic):
                expected_str = f", expected={op.expected.name}" if op.expected else ""
                lines.append(
                    f"{prefix} = atomic_{op.op}({op.ptr.name}[{op.index.name}], {op.value.name}{expected_str})"
                )
            elif isinstance(op, SimdOp):
                args_str = ", ".join(a.name for a in op.args)
                lines.append(f"{prefix} = simd_{op.op}({args_str})")
            elif isinstance(op, SimdMatrixOp):
                args_str = ", ".join(a.name for a in op.args)
                extra = ""
                if op.stride is not None:
                    extra += f", stride={op.stride}"
                if op.transpose:
                    extra += ", transpose"
                lines.append(f"{prefix} = simd_matrix_{op.op}({args_str}{extra})")
            elif isinstance(op, DebugPrint):
                args_str = ", ".join(a.name for a in op.args)
                lines.append(f"{pad}debug_print({op.fmt!r}, {args_str})")
            elif isinstance(op, IfElse):
                lines.append(f"{pad}if ({op.cond.name}) {{")
                _dump_ops(op.body, indent + 1)
                if op.orelse:
                    lines.append(f"{pad}}} else {{")
                    _dump_ops(op.orelse, indent + 1)
                lines.append(f"{pad}}}")
            elif isinstance(op, FlowControl):
                lines.append(f"{pad}{op.kind}")
            elif isinstance(op, Cast):
                lines.append(f"{prefix} = cast({op.input.name}, {op.target_dtype})")
            elif isinstance(op, CoopLoad):
                lines.append(f"{pad}coop_load({op.dst.name}, {op.src.name}, {op.count.name})")
            else:
                lines.append(f"{pad}<unknown op: {type(op).__name__}>")

    _dump_ops(func.ops, 1)
    lines.append("}")
    return "\n".join(lines)
