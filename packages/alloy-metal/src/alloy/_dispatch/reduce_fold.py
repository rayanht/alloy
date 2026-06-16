"""Reduction-fold fusion: fold a row-reduction producer into a streaming matvec
anchor by riding the anchor's own K-loop and simd-reduce.

The new primitive. No existing mechanism folds a reduction-bearing producer
(rms_norm) into a consumer — the engine only attaches per-element transforms or
builds a fresh kernel. But a decode quant-GEMV (dot_q4_k_v2 / dot_q4_k_silu_v2)
already streams the full activation row through a ForLoop with simd-reduced
per-column accumulators. So this composer threads the producer's reduction
through that SAME loop as an added carry and reduces it with the SAME
simd_reduce — authoring no loop and no reduce, only the carry plumbing.

All compute is LIFTED from rms_norm's TileFunction (its square, its
mean/eps/rsqrt tail, its weight multiply) and re-emitted via copy+remap, exactly
as RowPass/MultiRoot lift their compute. The per-row scalar rrms factors out of
the matvec — out[c] = rrms · Σ_k (w[k]·x[k])·W[c,k] — so the whole fold is one
pass: x² accumulates in a carry, w·x folds into each load, rrms scales each
column result before the (silu/)store.

Scope (measured): single-consumer, small-N only. The folded reduce is recomputed
by every output-column threadgroup, so its redundant cost scales with N — a win
only at small N (<=4096: qwen3:0.6b's qkv/gate_up), sharply net-negative above it
(Llama-3.2-3B's qkv 5120 / gate_up 8192 fold to -43% decode), so N_OUT_FOLD_LIMIT
gates it hard. Fan-out>=2 (an attn norm feeding QK+V) was also measured
net-negative — the reduce runs in both — so the pass claims single-consumer
norms only. Gated to M==1 decode + the no-remainder superblock layout.

The reduce uses simdgroup-order summation vs rms_norm's tile reduce — equal up
to f32 ULPs; validated by greedy-token match on qwen3.5:4b + gemma4.
"""

from __future__ import annotations

import copy

from alloy._compiler.tile_ir import (
    BinOp,
    Constant,
    Copy,
    Dot4,
    ForLoop,
    Layout,
    Load,
    Load4Vec,
    Reduce,
    SimdReduce,
    Store,
    TileFunction,
    TileOp,
    TileParam,
    TileValue,
    UnaryOp,
    shallow_clone_for_fusion,
    walk_ops,
)

from alloy._dispatch.fusion_types import FusionUnsupported
from alloy._dispatch.lazy import LazyOp

ACT_PARAM = "A"
FUSIBLE_GEMVS = frozenset(
    {"dot_q4_k_v2", "dot_q4_k_silu_v2", "dot_mlx_q4_v2", "dot_mlx_q4_silu_v2"}
)
# Above this output-column count the redundant per-threadgroup reduce outweighs
# the saved norm dispatch/barrier. The crossover is sharp and low: measured
# net-positive for N<=4096 (qwen3:0.6b qkv 4096 / gate_up 3072), catastrophic at
# N>=5120 (Llama-3.2-3B qkv 5120 / gate_up 8192 -> -43% decode). 4096 keeps the
# small-model win and excludes every larger projection.
N_OUT_FOLD_LIMIT = 4096


class Namer:
    def __init__(self, taken: set[str]) -> None:
        self.taken = set(taken)
        self.n = 0

    def __call__(self, prefix: str = "rf") -> str:
        while True:
            self.n += 1
            nm = f"{prefix}{self.n}"
            if nm not in self.taken:
                self.taken.add(nm)
                return nm


def result_index(func: TileFunction) -> dict[str, TileOp]:
    return {o.result.name: o for o in walk_ops(func.ops) if o.result is not None}


def op_deps(name: str, by_res: dict[str, TileOp]) -> set[str]:
    out: set[str] = set()
    stack = [name]
    while stack:
        n = stack.pop()
        if n in out or n not in by_res:
            continue
        out.add(n)
        for v in by_res[n].operand_values():
            stack.append(v.name)
    return out


def ordered_ops(names: set[str], func: TileFunction) -> list[TileOp]:
    """Ops with result in `names`, in the source's topological (emission) order."""
    return [o for o in walk_ops(func.ops) if o.result is not None and o.result.name in names]


def apply_template(
    template: list[TileOp], inputs: dict[str, TileValue], namer: Namer
) -> tuple[list[TileOp], TileValue | None]:
    """Clone a lifted op template, remap its input placeholders to actual values,
    fresh-rename every result. Returns (new_ops, final_result)."""
    remap = dict(inputs)
    out: list[TileOp] = []
    for op in template:
        n = copy.copy(op)
        n.remap(remap)
        if n.result is not None:
            fresh = TileValue(namer(), n.result.shape, n.result.layout, n.result.dtype)
            remap[n.result.name] = fresh
            n.result = fresh
        out.append(n)
    return out, (out[-1].result if out and out[-1].result is not None else None)


class RmsPieces:
    """Lifted op templates + placeholder names extracted from rms_norm.func."""

    def __init__(self) -> None:
        self.square: list[TileOp] = []
        self.square_in = ""
        self.tail: list[TileOp] = []
        self.tail_in = ""
        self.wmul: list[TileOp] = []
        self.wmul_x = ""
        self.wmul_w = ""
        self.rrms_mul: list[TileOp] = []
        self.rrms_mul_acc = ""
        self.rrms_mul_rrms = ""


def extract_rms(rfunc: TileFunction) -> RmsPieces | None:
    by_res = result_index(rfunc)
    reduce_op = next((o for o in walk_ops(rfunc.ops) if isinstance(o, Reduce)), None)
    rsqrt_op = next(
        (o for o in walk_ops(rfunc.ops) if isinstance(o, UnaryOp) and o.op == "rsqrt"), None
    )
    if reduce_op is None or rsqrt_op is None or reduce_op.result is None or reduce_op.input is None:
        return None
    p = RmsPieces()

    # tail: ops strictly between the reduce result and rsqrt (div, add, consts, rsqrt)
    tail_names = op_deps(rsqrt_op.result.name, by_res) - op_deps(reduce_op.result.name, by_res)
    p.tail = ordered_ops(tail_names, rfunc)
    p.tail_in = reduce_op.result.name

    # square: the per-element contribution feeding the reduce carry
    carry_final = reduce_op.input
    add_op = by_res.get(carry_final.name)
    pass1 = next(
        (
            o
            for o in rfunc.ops
            if isinstance(o, ForLoop) and any(c[1].name == carry_final.name for c in o.carried)
        ),
        None,
    )
    if add_op is None or pass1 is None:
        return None
    init_name = next((c[0].name for c in pass1.carried if c[1].name == carry_final.name), None)
    sq_operands = [v for v in add_op.operand_values() if v.name != init_name]
    if len(sq_operands) != 1:
        return None
    sq_val = sq_operands[0]
    xload1 = next(
        (b for b in pass1.body if isinstance(b, Load) and b.ptr is not None and b.ptr.name == "x"),
        None,
    )
    if xload1 is None or xload1.result is None:
        return None
    sq_names = op_deps(sq_val.name, by_res) - op_deps(xload1.result.name, by_res)
    p.square = ordered_ops(sq_names, rfunc)
    p.square_in = xload1.result.name

    # weight fold (pass2): out_store.value = mul(weight_load, cast(...x..rrms))
    out_store = next(
        (
            o
            for o in walk_ops(rfunc.ops)
            if isinstance(o, Store) and o.ptr is not None and o.ptr.name == "out"
        ),
        None,
    )
    if out_store is None or out_store.value is None:
        return None
    mul36 = by_res.get(out_store.value.name)
    if not isinstance(mul36, BinOp) or mul36.op != "mul":
        return None
    pass2 = next(
        (
            o
            for o in rfunc.ops
            if isinstance(o, ForLoop)
            and o is not pass1
            and any(
                isinstance(b, Store) and b.ptr is not None and b.ptr.name == "out"
                for b in walk_ops(o.body)
            )
        ),
        None,
    )
    if pass2 is None:
        return None
    wload = next(
        (
            b
            for b in pass2.body
            if isinstance(b, Load) and b.ptr is not None and b.ptr.name == "weight"
        ),
        None,
    )
    xload2 = next(
        (b for b in pass2.body if isinstance(b, Load) and b.ptr is not None and b.ptr.name == "x"),
        None,
    )
    if wload is None or xload2 is None or wload.result is None or xload2.result is None:
        return None
    xside = next((v for v in mul36.operand_values() if v.name != wload.result.name), None)
    if xside is None:
        return None
    rrms_val = rsqrt_op.result
    # t31 = the rrms multiply on the x side
    t31 = next(
        (
            by_res[n]
            for n in op_deps(xside.name, by_res)
            if isinstance(by_res[n], BinOp)
            and by_res[n].op == "mul"
            and any(v.name == rrms_val.name for v in by_res[n].operand_values())
        ),
        None,
    )
    if t31 is None:
        return None
    cast29 = next((v for v in t31.operand_values() if v.name != rrms_val.name), None)
    if cast29 is None:
        return None
    # wmul: the x→weight chain MINUS the rrms multiply (rrms factors to the
    # output). Drop only the rrms-mul op (t31) and the rrms subtree — KEEP cast29
    # (= cast(x)); then rewire cast32's input t31 -> cast29.
    wmul_names = (
        op_deps(mul36.result.name, by_res)
        - op_deps(rrms_val.name, by_res)
        - {t31.result.name}
        - op_deps(wload.result.name, by_res)
        - op_deps(xload2.result.name, by_res)
    )
    wmul_names.add(mul36.result.name)
    wmul_ops = [copy.copy(o) for o in ordered_ops(wmul_names, rfunc)]
    for o in wmul_ops:
        o.remap({t31.result.name: cast29})
    p.wmul = wmul_ops
    p.wmul_x = xload2.result.name
    p.wmul_w = wload.result.name

    # rrms output multiply = the lifted t31 mul, applied to (acc, rrms)
    p.rrms_mul = [copy.copy(t31)]
    p.rrms_mul_acc = cast29.name
    p.rrms_mul_rrms = rrms_val.name
    return p


def scalar_val(name: str, dtype: str = "f32") -> TileValue:
    return TileValue(name=name, shape=(), layout=Layout.REPLICATED, dtype=dtype)


def vec_val(name: str, ref: TileValue) -> TileValue:
    return TileValue(name=name, shape=ref.shape, layout=ref.layout, dtype=ref.dtype)


def compose_reduce_fold(
    gemv_op: LazyOp, rms_op: LazyOp, weight_param: str = "rms_weight"
) -> tuple[TileFunction, list, tuple[int, int, int]]:
    """Compose rms_norm (rms_op) into the GEMV (gemv_op) by reduction-fold.
    Returns (fused_func, buf_arrs, grid). Raises FusionUnsupported if unshapeable.
    """
    a_shape = gemv_op.buffer_shapes.get(ACT_PARAM)
    if not a_shape or len(a_shape) != 2 or int(a_shape[0]) != 1:
        raise FusionUnsupported("reduce-fold: not an M==1 matvec")
    k_dim = int(a_shape[1])
    if k_dim % 256 != 0 or (k_dim // 256) % 4 != 0:
        raise FusionUnsupported("reduce-fold: remainder superblock layout")
    c_shape = gemv_op.buffer_shapes.get("C")
    n_out = 1
    for d in c_shape or ():
        n_out *= int(d)
    if n_out > N_OUT_FOLD_LIMIT:
        raise FusionUnsupported("reduce-fold: output N too large (vocab-scale anchor)")

    pieces = extract_rms(rms_op.func)
    if pieces is None:
        raise FusionUnsupported("reduce-fold: could not extract rms_norm pieces")

    func = shallow_clone_for_fusion(gemv_op.func)
    func.name = f"{rms_op.kernel.name}_{gemv_op.kernel.name}"  # e.g. rms_norm_dot_q4_k_v2
    namer = Namer(
        {p.name for p in func.params} | {o.result.name for o in walk_ops(func.ops) if o.result}
    )

    m_loop = next((o for o in func.ops if isinstance(o, ForLoop) and o.var == "m"), None)
    if m_loop is None:
        raise FusionUnsupported("reduce-fold: no m-loop")
    k_loop = next(
        (
            o
            for o in m_loop.body
            if isinstance(o, ForLoop)
            and any(
                isinstance(b, (Load, Load4Vec)) and b.ptr is not None and b.ptr.name == ACT_PARAM
                for b in o.body
            )
        ),
        None,
    )
    if k_loop is None:
        raise FusionUnsupported("reduce-fold: no activation K-loop")

    w_ptr = scalar_val(weight_param)
    body = k_loop.body
    a_loads = [
        b
        for b in body
        if isinstance(b, (Load, Load4Vec)) and b.ptr is not None and b.ptr.name == ACT_PARAM
    ]
    if not a_loads:
        raise FusionUnsupported("reduce-fold: no activation loads in K-loop")
    last_idx = max(i for i, b in enumerate(body) if b in a_loads)

    # Σx² carry init (scalar 0.0 -> Copy, mirroring the dot accumulators)
    czero = scalar_val(namer("rf_c"))
    sq_carry = scalar_val(namer("rf_acc"))
    before: list[TileOp] = [Constant(result=czero, value=0.0), Copy(result=sq_carry, source=czero)]

    sumsq_ops: list[TileOp] = []
    fold_ops: list[TileOp] = []
    remap_loads: dict[str, TileValue] = {}
    acc = sq_carry
    for ld in a_loads:
        if isinstance(ld, Load4Vec):
            # vec4 rms pieces: dot4(v,v) for Σx², weight vec4 ⊙ activation for w·x.
            sq = scalar_val(namer("rf_sq"))
            sumsq_ops.append(Dot4(result=sq, a=ld.result, b=ld.result))
            nxt = scalar_val(namer("rf_acc"))
            sumsq_ops.append(BinOp(result=nxt, op="add", lhs=acc, rhs=sq))
            acc = nxt
            w4 = vec_val(namer("rf_w4"), ld.result)
            fold_ops.append(Load4Vec(result=w4, ptr=w_ptr, offsets=ld.offsets))
            weighted = vec_val(namer("rf_wx"), ld.result)
            fold_ops.append(BinOp(result=weighted, op="mul", lhs=w4, rhs=ld.result))
            remap_loads[ld.result.name] = weighted
            continue
        # Σx²: lifted square on the RAW load, accumulated into the carry
        sq, sq_v = apply_template(pieces.square, {pieces.square_in: ld.result}, namer)
        sumsq_ops.extend(sq)
        nxt = vec_val(namer("rf_acc"), ld.result)
        sumsq_ops.append(BinOp(result=nxt, op="add", lhs=acc, rhs=sq_v))
        acc = nxt
        # w·x: load weight at the activation offset, lifted weight-mul (rrms factored out)
        wl = vec_val(namer("rf_w"), ld.result)
        fold_ops.append(Load(result=wl, ptr=w_ptr, offsets=ld.offsets))
        wm, wm_v = apply_template(pieces.wmul, {pieces.wmul_x: ld.result, pieces.wmul_w: wl}, namer)
        fold_ops.extend(wm)
        remap_loads[ld.result.name] = wm_v
    sumsq_final = acc

    tail_after = list(body[last_idx + 1 :])
    for o in tail_after:
        o.remap(remap_loads)
    k_loop.body = body[: last_idx + 1] + sumsq_ops + fold_ops + tail_after
    k_loop.carried = list(k_loop.carried) + [(sq_carry, sumsq_final)]

    # after the loop: simd-reduce the carry, then lifted rrms tail
    ss = vec_val(namer("rf_ss"), sumsq_final)
    rrms_pre: list[TileOp] = [SimdReduce(result=ss, input=sumsq_final, op="sum")]
    tail, rrms = apply_template(pieces.tail, {pieces.tail_in: ss}, namer)
    rrms_pre.extend(tail)
    if rrms is None:
        raise FusionUnsupported("reduce-fold: rrms tail produced no value")

    # scale each column's simd-reduced result by rrms (lifted rrms multiply)
    loop_pos = m_loop.body.index(k_loop)
    post = m_loop.body[loop_pos + 1 :]
    scaled_post: list[TileOp] = []
    out_remap: dict[str, TileValue] = {}
    for o in post:
        if out_remap:
            o.remap(out_remap)
        scaled_post.append(o)
        if isinstance(o, SimdReduce) and o.result is not None:
            sc, sc_v = apply_template(
                pieces.rrms_mul, {pieces.rrms_mul_acc: o.result, pieces.rrms_mul_rrms: rrms}, namer
            )
            scaled_post.extend(sc)
            if sc_v is not None:
                out_remap[o.result.name] = sc_v
    m_loop.body = before + m_loop.body[:loop_pos] + [k_loop] + rrms_pre + scaled_post

    # weight param after the output buffer, before constexprs
    insert_at = next((i for i, p in enumerate(func.params) if p.is_constexpr), len(func.params))
    func.params.insert(insert_at, TileParam(name=weight_param, is_constexpr=False, dtype="f32"))

    # buffer bindings: A -> the raw rms_norm input x; + weight; (rrms_out dropped)
    rms_buf = dict(rms_op.buffer_args)
    x_buf = rms_buf.get("x")
    w_buf = rms_buf.get("weight")
    if x_buf is None or w_buf is None:
        raise FusionUnsupported("reduce-fold: rms_norm missing x/weight buffers")
    gemv_buf = dict(gemv_op.buffer_args)
    buf_arrs: list = []
    for p in func.params:
        if p.is_constexpr:
            continue
        if p.name == ACT_PARAM:
            buf_arrs.append(x_buf)
        elif p.name == weight_param:
            buf_arrs.append(w_buf)
        else:
            buf_arrs.append(gemv_buf[p.name])
    return func, buf_arrs, gemv_op.grid or (1, 1, 1)
