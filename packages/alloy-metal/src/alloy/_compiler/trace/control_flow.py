"""Trace-time control-flow hooks emitted by AST rewriting."""

from __future__ import annotations

from dataclasses import dataclass

from alloy._compiler.tile_ir import (
    Copy,
    FlowControl,
    ForLoop,
    IfElse,
    Layout,
    SimdMatrixOp,
    TileOp,
    TileValue,
    WhileLoop,
    Zeros,
)
from alloy._compiler.trace.value import TracedValue, _add_op, _ctx, _ensure_traced


class IfScope:
    """Context manager for traced if-then blocks. Generates IfElse tile IR."""

    def __init__(self, cond: "TracedValue"):
        self._cond = cond._tv

    def __enter__(self):
        func = _ctx().builder.func
        self._saved_ops = func.ops
        self._body: list = []
        func.ops = self._body  # redirect new ops to body
        return self

    def __exit__(self, *exc):
        func = _ctx().builder.func
        func.ops = self._saved_ops  # restore

        func.add_op(IfElse(cond=self._cond, body=list(self._body)))


def trace_if(cond) -> IfScope:
    """Create a traced if-block: `with al.if_(condition): ...`"""
    cond = _ensure_traced(cond)
    return IfScope(cond)


# ---------------------------------------------------------------------------
# Traced for-loops — enter/exit hooks injected by AST rewrite
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Traced flow control — break, continue, return
# ---------------------------------------------------------------------------


def _trace_flow(kind):
    """Emit a FlowControl op (break, continue, return)."""
    _add_op(FlowControl(kind=kind))


def _trace_if_enter(cond, merge_names=None, *pre_vals):
    """Begin a traced if block. Returns opaque context."""
    ctx = _ctx()
    saved_ops = ctx.builder.func.ops
    pre_tvs = [v._tv if isinstance(v, TracedValue) else _ensure_traced(v)._tv for v in pre_vals]

    if isinstance(cond, (bool, int)):
        if cond:
            return ("const_true", saved_ops, merge_names or [], pre_tvs)
        else:
            ctx.builder.func.ops = []
            return ("const_false", saved_ops, merge_names or [], pre_tvs)

    cond_tv = _ensure_traced(cond)._tv
    body_ops: list[TileOp] = []
    ctx.builder.func.ops = body_ops
    return ("runtime", cond_tv, saved_ops, body_ops, merge_names or [], pre_tvs)


def _trace_if_else(if_ctx, *body_final_vals):
    """Transition from if-body to else-body."""
    ctx = _ctx()
    tag = if_ctx[0]

    if tag == "const_true":
        ctx.builder.func.ops = []
        return ("const_true_else", if_ctx[1], if_ctx[2], if_ctx[3])
    if tag == "const_false":
        ctx.builder.func.ops = if_ctx[1]
        return ("const_false_else", if_ctx[1], if_ctx[2], if_ctx[3])

    _, cond_tv, saved_ops, body_ops, merge_names, pre_tvs = if_ctx
    body_tvs = [
        v._tv if isinstance(v, TracedValue) else _ensure_traced(v)._tv for v in body_final_vals
    ]
    orelse_ops: list[TileOp] = []
    ctx.builder.func.ops = orelse_ops
    return (
        "runtime_else",
        cond_tv,
        saved_ops,
        body_ops,
        orelse_ops,
        merge_names,
        pre_tvs,
        body_tvs,
    )


def _trace_if_exit(if_ctx, *post_vals):
    """End a traced if block. Emits IfElse IR node with merges."""
    ctx = _ctx()
    tag = if_ctx[0]

    if tag == "const_true":
        # Body ran, values are correct as-is
        if post_vals:
            return post_vals[0] if len(post_vals) == 1 else post_vals
        return
    if tag == "const_false":
        ctx.builder.func.ops = if_ctx[1]
        # Body was discarded — return pre-if values
        pre_tvs = if_ctx[3]
        if pre_tvs:
            results = tuple(
                (
                    TracedValue(_ensure_traced(v)._tv if not isinstance(v, TileValue) else v)
                    if isinstance(v, TileValue)
                    else v
                )
                for v in pre_tvs
            )
            # Actually just return the pre-values as TracedValues
            pre_traced = []
            for tv in pre_tvs:
                pre_traced.append(TracedValue(tv))
            return pre_traced[0] if len(pre_traced) == 1 else tuple(pre_traced)
        return
    if tag == "const_true_else":
        ctx.builder.func.ops = if_ctx[1]
        if post_vals:
            return post_vals[0] if len(post_vals) == 1 else post_vals
        return
    if tag == "const_false_else":
        if post_vals:
            return post_vals[0] if len(post_vals) == 1 else post_vals
        return

    # Runtime if — emit IfElse with merges
    if tag == "runtime":
        _, cond_tv, saved_ops, body_ops, merge_names, pre_tvs = if_ctx
        ctx.builder.func.ops = saved_ops
        # Body values are post_vals, else values are pre_tvs (no else branch)
        merges = []
        results = []
        for pre_tv, post_v in zip(pre_tvs, post_vals):
            post_tv = post_v._tv if isinstance(post_v, TracedValue) else _ensure_traced(post_v)._tv
            if pre_tv is not post_tv:
                result_tv = TileValue(
                    name=ctx.builder._fresh("sel"),
                    shape=post_tv.shape,
                    layout=post_tv.layout,
                    dtype=post_tv.dtype,
                )
                merges.append((result_tv, post_tv, pre_tv))
                results.append(TracedValue(result_tv))
            else:
                results.append(post_v if isinstance(post_v, TracedValue) else TracedValue(pre_tv))
        _add_op(IfElse(cond=cond_tv, body=body_ops, merges=merges))
        if results:
            return results[0] if len(results) == 1 else tuple(results)
        return

    elif tag == "runtime_else":
        _, cond_tv, saved_ops, body_ops, orelse_ops, merge_names, pre_tvs, body_tvs = if_ctx
        ctx.builder.func.ops = saved_ops
        merges = []
        results = []
        for pre_tv, body_tv, post_v in zip(pre_tvs, body_tvs, post_vals):
            post_tv = post_v._tv if isinstance(post_v, TracedValue) else _ensure_traced(post_v)._tv
            if body_tv is not pre_tv or post_tv is not pre_tv:
                result_tv = TileValue(
                    name=ctx.builder._fresh("sel"),
                    shape=body_tv.shape,
                    layout=body_tv.layout,
                    dtype=body_tv.dtype,
                )
                merges.append((result_tv, body_tv, post_tv))
                results.append(TracedValue(result_tv))
            else:
                results.append(post_v if isinstance(post_v, TracedValue) else TracedValue(pre_tv))
        _add_op(IfElse(cond=cond_tv, body=body_ops, orelse=orelse_ops, merges=merges))
        if results:
            return results[0] if len(results) == 1 else tuple(results)
        return


# ---------------------------------------------------------------------------
# Traced loops — enter/cond/exit hooks
# ---------------------------------------------------------------------------


@dataclass
class LoopCtx:
    """State captured by _trace_loop_enter, consumed by _trace_loop_exit."""

    carry_names: list[str]
    init_tvs: list[TileValue]
    saved_ops: list
    cond_ops: list  # ops for the condition (while loops)
    body_ops: list
    cond_tv: TileValue | None = None


def _flatten_carry(val):
    """Yield leaf carry values depth-first. A carry may be a scalar/TracedValue
    (one leaf) or a (nested) list/tuple of them — so a loop can carry an array
    of accumulators (`o = [0.0]*N; ... o[d] = o[d]*a + ...`) without the kernel
    author flattening to o0..oN by hand. Enter and exit flatten identically, so
    the per-leaf carry pairs line up as long as the structure is loop-invariant."""
    if isinstance(val, (list, tuple)):
        for x in val:
            yield from _flatten_carry(x)
    else:
        yield val


def _rebuild_carry(val, leaf_iter):
    """Rebuild a structure parallel to `val`, substituting each leaf with the
    next value from `leaf_iter` (the fresh loop-body copy)."""
    if isinstance(val, list):
        return [_rebuild_carry(x, leaf_iter) for x in val]
    if isinstance(val, tuple):
        return tuple(_rebuild_carry(x, leaf_iter) for x in val)
    return next(leaf_iter)


def _trace_loop_enter(carry_names, *carry_vals):
    """Begin a traced loop. Called by AST-rewritten kernel code.

    Returns (LoopCtx, body_copy_0, body_copy_1, ...) — fresh TracedValues
    for each carried variable. The body uses these; _trace_loop_exit receives
    the final values.
    """
    ctx = _ctx()

    # De-alias each carried LEAF: Copy it BEFORE redirecting ops so the Copy
    # lands in the outer scope (like the AST path); the Copy'd value becomes the
    # carried init. Skip values that can't be scalar-copied: Zeros (emitter uses
    # arrays), SimdMatrixOp (Metal simdgroup type). A carry may be a scalar (one
    # leaf) or a (nested) list of accumulators — flatten to leaves, then rebuild
    # the same structure with the fresh body copies so `o[d] = ...` carries work.
    def _process_leaf(leaf):
        if not isinstance(leaf, TracedValue):
            leaf = _ensure_traced(leaf)
        src_op = ctx.op_map.get(leaf._tv.name)
        # Skip the scalar de-alias Copy for values the emitter carries in place:
        #   - Zeros: emitted as a register/local array, mutated in place.
        #   - SimdMatrixOp: a Metal simdgroup_matrix register type.
        #   - 2D tiles: carried in a persistent threadgroup buffer (the body reads
        #     the tile as a GEMM operand — needs its "shared" location — and the
        #     loop-tail writes the update back into the same buffer). A scalar
        #     `float lc = <tile>` Copy would both mis-type it and strip the shared
        #     buffer, so the matmul operand resolves to '???'. The loop-tail
        #     cooperative copy in `_emit_for_loop` provides the cross-iteration
        #     de-alias instead.
        if isinstance(src_op, (Zeros, SimdMatrixOp)) or len(leaf._tv.shape) == 2:
            return leaf, leaf._tv
        fresh_tv = TileValue(
            name=ctx.builder._fresh("lc"),
            shape=leaf._tv.shape,
            layout=leaf._tv.layout,
            dtype=leaf._tv.dtype,
        )
        _add_op(Copy(result=fresh_tv, source=leaf._tv))
        bc = TracedValue(fresh_tv, ptr_base=leaf._ptr_base)
        # For pointer expressions, the Copy IS the new offsets reference
        if leaf._ptr_offsets is not None:
            bc._ptr_offsets = bc
        return bc, fresh_tv

    body_copies = []
    init_tvs = []
    for cv in carry_vals:
        fresh_leaves = []
        for leaf in _flatten_carry(cv):
            bc, init_tv = _process_leaf(leaf)
            fresh_leaves.append(bc)
            init_tvs.append(init_tv)
        body_copies.append(_rebuild_carry(cv, iter(fresh_leaves)))

    # NOW redirect ops for condition/body
    saved_ops = ctx.builder.func.ops
    cond_ops: list[TileOp] = []
    body_ops: list[TileOp] = []
    ctx.builder.func.ops = cond_ops

    lctx = LoopCtx(
        carry_names=carry_names,
        init_tvs=init_tvs,
        saved_ops=saved_ops,
        cond_ops=cond_ops,
        body_ops=body_ops,
    )

    return (lctx,) + tuple(body_copies)


def _trace_for_var(lctx: LoopCtx, var_name: str):
    """Create symbolic i32 loop variable for ForLoop and switch to body ops."""
    ctx = _ctx()
    loop_var_tv = TileValue(
        name=var_name,
        shape=(),
        layout=Layout.REPLICATED,
        dtype="i32",
    )
    # Switch from cond_ops to body_ops (for-loops have no traced condition)
    ctx.builder.func.ops = lctx.body_ops
    return TracedValue(loop_var_tv)


def _trace_loop_cond(lctx: LoopCtx, cond_val):
    """Transition from condition to body phase. Records the condition value."""
    ctx = _ctx()
    if isinstance(cond_val, TracedValue):
        lctx.cond_tv = cond_val._tv
    elif isinstance(cond_val, bool):
        # Constant condition (e.g., for-loop converted to while with constexpr bounds)
        lctx.cond_tv = None
    else:
        lctx.cond_tv = None
    # Switch from cond_ops to body_ops
    ctx.builder.func.ops = lctx.body_ops


def _trace_loop_exit(lctx: LoopCtx, var_name, start, end, step, *final_vals):
    """End a traced loop. Emits ForLoop or WhileLoop IR node.

    If start/end/step are all provided (not None), emits ForLoop.
    Otherwise emits WhileLoop using the captured condition.
    """
    ctx = _ctx()
    ctx.builder.func.ops = lctx.saved_ops

    # Flatten final carry values to leaves the same way enter flattened the
    # inits, so per-leaf carry pairs line up (handles list/array accumulators).
    carried = []
    final_leaves = []
    for fv in final_vals:
        for leaf in _flatten_carry(fv):
            if not isinstance(leaf, TracedValue):
                leaf = _ensure_traced(leaf)
            final_leaves.append(leaf)
    for init_tv, fv in zip(lctx.init_tvs, final_leaves):
        final_tv = fv._tv
        if init_tv is not final_tv:
            carried.append((init_tv, final_tv))

    if start is not None:
        # Unwrap TracedValues to TileValues/ints for ForLoop bounds
        if isinstance(start, TracedValue):
            start = start._tv
        if isinstance(end, TracedValue):
            end = end._tv
        if isinstance(step, TracedValue):
            step = step._tv
        # ForLoop — merge cond_ops into body_ops (no separate condition)
        all_body_ops = lctx.cond_ops + lctx.body_ops
        _add_op(
            ForLoop(
                var=var_name,
                start=start,
                end=end,
                step=step,
                body=all_body_ops,
                carried=carried,
            )
        )
    else:
        # WhileLoop
        _add_op(
            WhileLoop(
                cond_body=lctx.cond_ops,
                cond=lctx.cond_tv,
                body=lctx.body_ops,
                carried=carried,
            )
        )
