"""Cast-op duplication for prologue fusion.

A widening cast (bf16 -> f32, f16 -> f32, etc.) consumed by N kernels
materializes a wider buffer that all N kernels then re-load. The
fusion engine's prologue path already absorbs a cast into ITS single
consumer's cooperative load, but `_find_best_prologue` requires the
producer to have exactly one consumer.

This pass duplicates a multi-consumer widening cast into N copies, one
per consumer. Each (cast_dup, consumer) pair is then a clean
single-producer/single-consumer chain the existing prologue fusion can
absorb naturally — the cast disappears at codegen time and the
consumer reads narrow source bytes from DRAM and casts in registers.

Bandwidth comparison (X = source bytes, Y = dest bytes, N consumers):
  Original:  X (read) + Y (write) + N*Y (consumer reads) = X + (N+1)*Y
  Dup-fused: N*X (each consumer re-reads narrow source)
  Savings:   X + (N+1)*Y - N*X = (1-N)*X + (N+1)*Y

For widening casts (Y > X) savings are positive for any N. Narrowing
casts (Y < X) break even around N=2-3 and lose for larger fanouts, so
we restrict the pass to widening casts. The canonical case is LoRA
adapters: a bf16 base activation cast to f32 is consumed by every
adapter branch on the same layer (e.g. lora_A_q and lora_A_v).
"""

from __future__ import annotations

from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._dispatch.lazy import LazyOp


_DTYPE_BYTES = {
    "f32": 4, "i32": 4, "u32": 4,
    "f16": 2, "bf16": 2, "i16": 2, "u16": 2,
    "i8": 1, "u8": 1,
    "i64": 8, "u64": 8,
}


def _is_widening_cast(op: LazyOp) -> bool:
    name = op.kernel.name
    if not name.startswith("cast_") or len(op.buffer_args) != 2:
        return False
    in_buf = out_buf = None
    for pname, buf in op.buffer_args:
        if pname in op.output_params:
            out_buf = buf
        else:
            in_buf = buf
    if in_buf is None or out_buf is None:
        return False
    return _DTYPE_BYTES.get(out_buf._dtype.ir, 4) > _DTYPE_BYTES.get(in_buf._dtype.ir, 4)


def dup_fuse_casts(
    ops: list[LazyOp], root_op_ids: set[int] | None = None
) -> list[LazyOp]:
    """Walk `ops` once. For each widening cast where prologue absorption is
    blocked by either:
      (a) multiple consumers (the planner's `_find_best_prologue` rejects
          producers with >1 consumer), or
      (b) the cast's output is a save-for-backward root (the planner's
          root-protect drops the prologue because inlining the cast would
          stop the materialized output from existing for the bwd graph),
    emit one duplicate per consumer (each writing to a fresh, NON-root
    output buffer) and rewire each consumer's buffer_args to point at its
    dup. The original cast stays in `ops` only when it is a root (so the
    save-for-bwd buffer keeps getting written); otherwise it is dropped.
    """
    if not ops:
        return ops

    root_ids = root_op_ids or set()
    op_idx: dict[int, int] = {id(op): i for i, op in enumerate(ops)}
    consumed_by: dict[int, list[int]] = {}
    for ci, cop in enumerate(ops):
        for pname, _ in cop.buffer_args:
            if pname in cop.output_params:
                continue
            prod = cop.input_producers.get(pname)
            if prod is None:
                continue
            pi = op_idx.get(id(prod))
            if pi is not None:
                consumed_by.setdefault(pi, []).append(ci)

    result: list[LazyOp] = []
    for i, op in enumerate(ops):
        if not _is_widening_cast(op):
            result.append(op)
            continue
        consumers = consumed_by.get(i, [])
        is_root = id(op) in root_ids
        # Trigger when prologue absorption would otherwise be blocked.
        if not consumers:
            result.append(op)
            continue
        # Single-consumer is_root casts: don't dup. The save-for-bwd buffer
        # already exists; duplicating would add a second wide-write of a
        # potentially large output (e.g. f32 logits). The original cast
        # dispatch writes once and both the in-FW consumer and the
        # saved-for-bwd reader share that buffer.
        if len(consumers) < 2:
            result.append(op)
            continue

        out_pname = next(p for p, _ in op.buffer_args if p in op.output_params)
        in_pname = next(p for p, _ in op.buffer_args if p not in op.output_params)
        orig_out = next(b for p, b in op.buffer_args if p == out_pname)
        in_buf = next(b for p, b in op.buffer_args if p == in_pname)

        # Keep the original when it is a root — the save-for-backward
        # output handle has to keep getting written.
        if is_root:
            result.append(op)

        for ci in consumers:
            cop = ops[ci]
            new_out = _alloc_aligned(orig_out.shape, orig_out._dtype)
            new_op = LazyOp(
                kernel=op.kernel,
                grid=op.grid,
                func=op.func,
                constexpr_values=dict(op.constexpr_values),
                buffer_args=[(in_pname, in_buf), (out_pname, new_out)],
                buffer_dtypes=dict(op.buffer_dtypes),
                buffer_shapes=dict(op.buffer_shapes),
                output_params=set(op.output_params),
                input_producers=dict(op.input_producers),
            )
            for pi, (pn, pb) in enumerate(cop.buffer_args):
                if pn in cop.output_params:
                    continue
                if pb is orig_out:
                    cop.buffer_args[pi] = (pn, new_out)
                    cop.input_producers[pn] = new_op
                    break
            result.append(new_op)
    return result
