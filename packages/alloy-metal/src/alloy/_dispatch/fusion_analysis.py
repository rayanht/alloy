"""Fusion analysis — DAG analysis, candidate proposal, and plan construction.

Pure analysis layer: examines LazyOp graphs and produces FusionPlan decisions.
No IR mutation, no compilation, no Metal interaction.

Pipeline: _plan_fusion(ops) → list[FusionPlan]
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from alloy._dispatch.buf_utils import _NON_FUSABLE_ELEM_KERNELS
from alloy._dispatch.fusion_transform import extract_ir_transform
from alloy._dispatch.fusion_types import FusionGroup, FusionKind, FusionPlan
from alloy._dispatch.multi_root import find_islands, Island
from alloy._dispatch.row_pass import _is_row_reduce, grow_group
from alloy._dispatch.lazy import LazyOp
from alloy._runtime.alloy_buffer import AlloyBuffer


# ===================================================================
# Section 2: DAG analysis helpers
# ===================================================================


def _output_shape(op: LazyOp) -> tuple[int, ...] | None:
    for pname, arr in op.buffer_args:
        if pname in op.output_params:
            return tuple(arr.shape)
    return None


def _fused_name(ops: list[LazyOp], indices: set[int] | list[int]) -> str:
    seen: list[str] = []
    for idx in indices:
        name = ops[idx].kernel.name
        if not seen or seen[-1] != name:
            seen.append(name)
    return "_".join(seen)


def _elem_chain_has_future_dependency(
    start_idx: int,
    chain: list[int],
    ops: list[LazyOp],
    op_idx: dict[int, int],
) -> bool:
    fused_indices = set(chain)
    for chain_idx in chain[1:]:
        for producer in ops[chain_idx].input_producers.values():
            prod_idx = op_idx.get(id(producer))
            if prod_idx is None or prod_idx in fused_indices:
                continue
            if prod_idx > start_idx:
                return True
    return False


def _primary_input_chains(consumer_op: LazyOp, producer_op: LazyOp) -> bool:
    """True if `producer_op`'s output can feed into `consumer_op` as the
    chain input during ELEM_CHAIN fusion.

    `_apply_chain` → `_identify_chain_input` picks the chain slot by
    matching the previous output against the consumer's primary Load
    (the single load whose result directly flows into the Store value)
    OR one of the named-alias extra Loads from the same store-dep graph.
    If the producer's output buffer doesn't alias ANY of those, the
    composer raises FusionUnsupported. Gather-like kernels are a
    concrete example: `k_gather(src, idx)` sets Store.value to the
    `src`-load, so `idx` isn't registered as an extra; a chain that
    tries to deliver data through `idx` has no route through the fused
    kernel.
    """
    producer_out: AlloyBuffer | None = None
    for pname, buf in producer_op.buffer_args:
        if pname in producer_op.output_params:
            producer_out = buf
            break
    if producer_out is None:
        return False

    # extract_ir_transform walks the TileFunction to find the primary and
    # extra load ptrs. Deepcopy to avoid mutating the cached op.func.
    func = copy.deepcopy(consumer_op.func)
    raw = extract_ir_transform(func)
    if raw is None:
        return False
    _xf_ops, _load_name, _store_value, load_ptr, _store_ptr, extras = raw
    buf_by_param = {pname: buf for pname, buf in consumer_op.buffer_args}

    primary_buf = buf_by_param.get(load_ptr) if load_ptr else None
    if primary_buf is not None and primary_buf.shares_allocation(producer_out):
        return True
    for ename in extras.keys():
        earr = buf_by_param.get(ename)
        if earr is not None and earr.shares_allocation(producer_out):
            return True
    return False


def _elem_chain_changes_shape(chain: list[int], ops: list[LazyOp]) -> bool:
    """True if any consecutive pair of ops in the chain has a differing
    *element count*. Rank/shape annotations may differ (e.g. `_binary_dispatch`'s
    broadcast path emits a flat `(N,)` output for an op whose semantic shape
    is `(B, S, H)`); what matters for flat-indexing elem chains is that each
    op iterates over the same number of elements."""

    def _numel(shape: tuple[int, ...] | None) -> int | None:
        if shape is None:
            return None
        n = 1
        for d in shape:
            n *= int(d)
        return n

    for prev_idx, cur_idx in zip(chain, chain[1:], strict=False):
        prev_n = _numel(_output_shape(ops[prev_idx]))
        cur_n = _numel(_output_shape(ops[cur_idx]))
        if prev_n is None or cur_n is None:
            continue
        if prev_n != cur_n:
            return True
    return False


def _find_reconvergence(
    ops: list[LazyOp],
    consumed_by: dict[int, list[int]],
    consumers: list[int],
    planned: set[int],
) -> tuple[list[int], int] | None:
    """Find a reconvergence point for a 2-consumer fork.

    Returns (branch_ops, reconvergence_idx) where branch_ops is the longer
    branch and reconvergence_idx is where both branches meet. Handles nested
    2-consumer forks within the branch by recursing.
    """
    c0, c1 = consumers
    if c0 in planned or c1 in planned:
        return None
    if not ops[c0].is_elem_op() or not ops[c1].is_elem_op():
        return None
    for long_start, short in [(c0, c1), (c1, c0)]:
        chain: list[int] = [long_start]
        cur = long_start
        while True:
            cur_consumers = set(consumed_by.get(cur, []))
            if short in cur_consumers and short not in planned and ops[short].is_elem_op():
                return chain, short
            nc = consumed_by.get(cur, [])
            if len(nc) == 1:
                nxt = nc[0]
                if nxt in planned or not ops[nxt].is_elem_op():
                    break
                chain.append(nxt)
                cur = nxt
            elif len(nc) == 2:
                # Nested diamond — recurse
                inner = _find_reconvergence(ops, consumed_by, nc, planned)
                if inner is None:
                    break
                inner_branch, inner_reconv = inner
                chain.extend(inner_branch)
                chain.append(inner_reconv)
                cur = inner_reconv
            else:
                break
        last_consumers = set(consumed_by.get(cur, []))
        if short in last_consumers and short not in planned and ops[short].is_elem_op():
            return chain, short
    return None


def _output_elems(op: LazyOp) -> int:
    """Total elements in the op's output buffer."""
    for pn, buf in op.buffer_args:
        if pn in op.kernel._output_params:
            return buf.size
    return 0


def _build_epi(
    ops: list[LazyOp],
    consumed_by: dict[int, list[int]],
    anchor_idx: int,
    planned: set[int],
    roots: set[int] | None = None,
) -> list[int]:
    anchor_elems = _output_elems(ops[anchor_idx])
    roots = roots if roots is not None else set()
    epi: list[int] = []
    cur_c: list[int] = consumed_by.get(anchor_idx, [])
    if len(cur_c) == 2:
        rc = _find_reconvergence(ops, consumed_by, cur_c, planned)
        if rc is not None:
            branch, reconv = rc
            epi.extend(branch)
            epi.append(reconv)
            nc: list[int] = consumed_by.get(reconv, [])
            cur_c = nc if len(nc) == 1 else []
        else:
            cur_c = []
    else:
        cur_c = cur_c if len(cur_c) == 1 else []
    while cur_c:
        c: int = cur_c[0]
        if c in planned or not ops[c].is_elem_op():
            break
        # Don't fuse if consumer output is larger than anchor output —
        # the anchor's store loop can't expand to more elements.
        if anchor_elems > 0 and _output_elems(ops[c]) > anchor_elems:
            break
        epi.append(c)
        nc = consumed_by.get(c, [])
        if len(nc) == 1:
            cur_c = nc
        elif len(nc) == 2:
            rc = _find_reconvergence(ops, consumed_by, nc, planned)
            if rc is not None:
                branch, reconv = rc
                epi.extend(branch)
                epi.append(reconv)
                nc2 = consumed_by.get(reconv, [])
                cur_c = nc2 if len(nc2) == 1 else []
            else:
                break
        else:
            # Fan-out: `c` has >2 consumers. Normally we stop, since
            # register-forwarding past `c` would hide its raw value
            # from the extra consumers. But if `c` is a saved-for-bwd
            # root, the root-split logic below will emit a tee Store
            # for it, so the raw value IS materialized. In that case
            # we can keep extending the chain through the smallest-
            # idx elem consumer (= the in-sort-order successor), as
            # long as every OTHER consumer sits past that successor
            # (sort-order guard — an external consumer at idx ≤
            # successor would be placed in an earlier dep-group and
            # read stale memory).
            if c in roots:
                elem_nc = [
                    cc
                    for cc in nc
                    if cc not in planned
                    and ops[cc].is_elem_op()
                    and (anchor_elems == 0 or _output_elems(ops[cc]) <= anchor_elems)
                ]
                if elem_nc:
                    successor = min(elem_nc)
                    others = [cc for cc in nc if cc != successor]
                    if all(cc > successor for cc in others):
                        cur_c = [successor]
                        continue
            break
    return epi


def _build_multi_epi(
    ops: list[LazyOp],
    consumed_by: dict[int, list[int]],
    anchor_idx: int,
    planned: set[int],
    roots: set[int] | None = None,
) -> tuple[list[int], list[list[int]]]:
    consumers: list[int] = consumed_by.get(anchor_idx, [])
    if len(consumers) == 1:
        return _build_epi(ops, consumed_by, anchor_idx, planned, roots), []
    if len(consumers) == 2:
        rc = _find_reconvergence(ops, consumed_by, consumers, planned)
        if rc is not None:
            branch, reconv = rc
            return list(branch) + [reconv], []
    if len(consumers) >= 2 and all(c not in planned and ops[c].is_elem_op() for c in consumers):
        branches = [[c] + _build_epi(ops, consumed_by, c, planned, roots) for c in consumers]
        if all(branches):
            # Multi-branch fusion is only safe when branches are disjoint.
            # If two branches share a tail op, they've reconverged into a
            # merge node that the composer would emit twice (once per
            # branch's clone_param store) — the second write races the
            # first and silently drops one contribution. Concrete failure:
            # bwd of `0.5*x*(1+tanh(x))` where both `mul_2 → mul_5` and
            # `mul_3 → mul_6` branches terminate at `add_1 = mul_5 + mul_6`;
            # the composer emits add_1 twice and only one branch survives.
            all_sets = [set(b) for b in branches]
            disjoint = all(
                not (all_sets[i] & all_sets[j])
                for i in range(len(all_sets))
                for j in range(i + 1, len(all_sets))
            )
            if disjoint and not _branches_feed_gemms(ops, consumed_by, branches):
                return branches[0], branches[1:]
    return [], []


def _branches_feed_gemms(
    ops: list[LazyOp],
    consumed_by: dict[int, list[int]],
    branches: list[list[int]],
) -> bool:
    """True if any branch's terminal output feeds a non-elementwise op (a GEMM).

    Multi-branch fusion materializes each branch output for downstream reuse, which
    is correct when the branches write FINAL outputs (AdamW's m/v/param writebacks).
    But when branches feed GEMMs — gemma4 vision's clipped-linear gate/up/qkv, all
    clamping a shared rms_norm input — those GEMMs reconverge downstream (at the
    gated `act(gate)*up` mul, or the attention op consuming q/k/v) and the
    multi-branch writeback buffers alias incorrectly. Keep such branches as
    separate dispatches (each clamp then prologue-fuses into its own GEMM). Text's
    plain gated MLP never reaches the multi-branch case — its rms_norm feeds the
    GEMMs directly, so the anchor's consumers aren't elementwise."""
    return any(
        not ops[c].is_elem_op() for branch in branches for c in consumed_by.get(branch[-1], [])
    )


def _find_best_prologue(
    ops: list[LazyOp],
    consumed_by: dict[int, list[int]],
    op_idx: dict[int, int],
    anchor_idx: int,
    planned: set[int],
    roots: set[int] | None = None,
) -> list[int]:
    """Find the longest absorbable prologue chain ending at anchor_idx.

    Skips chain starts whose op is a save-for-backward root — root-protect
    will drop any group that includes them anyway, and the dup-fuse pass
    has already inserted a non-root duplicate of those casts that we want
    the search to find instead.
    """
    roots = roots or set()
    best: list[int] = []
    for start in range(anchor_idx):
        if start in planned or not ops[start].is_elem_op():
            continue
        if start in roots:
            continue
        chain: list[int] = [start]
        cur_c = consumed_by.get(start, [])
        if len(cur_c) != 1:
            continue
        cur = cur_c[0]
        while cur != anchor_idx:
            if cur in planned or not ops[cur].is_elem_op():
                break
            chain.append(cur)
            nc = consumed_by.get(cur, [])
            if len(nc) == 1:
                cur = nc[0]
            elif len(nc) == 2:
                rc = _find_reconvergence(ops, consumed_by, nc, planned)
                if rc is not None:
                    branch, reconv = rc
                    chain.extend(branch)
                    chain.append(reconv)
                    nc2 = consumed_by.get(reconv, [])
                    cur = nc2[0] if len(nc2) == 1 else -1
                else:
                    break
            else:
                break
        if cur == anchor_idx and len(chain) > len(best):
            if not _elem_chain_has_future_dependency(start, chain, ops, op_idx):
                # Block prologues with extra side inputs — the cooperative
                # load addressing uses K (inner dim) but extras get N
                # (output dim), producing wrong index math.
                #
                # `extras` here means OFF-CHAIN producers in excess of the
                # chain's primary source. A length-N chain naturally has N
                # input buffers: the first op's input is the new anchor
                # source, and each subsequent op's chain-link input comes
                # from the previous op in the chain. Anything beyond that
                # — e.g. `mul(a, b)` where both a and b are external — is a
                # side input the cooperative load can't address.
                chain_set = set(chain)
                total_inputs = 0
                external_inputs = 0
                for ci in chain:
                    for pn, _ in ops[ci].buffer_args:
                        if pn in ops[ci].output_params:
                            continue
                        total_inputs += 1
                        prod = ops[ci].input_producers.get(pn)
                        pi = op_idx.get(id(prod)) if prod is not None else None
                        if pi is None or pi not in chain_set:
                            external_inputs += 1
                # Each chain op consumes one chain-link input (or the
                # primary source for chain[0]); extras = inputs beyond that.
                extras = total_inputs - len(chain)
                if extras <= 0 and external_inputs <= 1:
                    best = chain
    return best


# ===================================================================
# Section 3: Proposal pipeline
# ===================================================================


@dataclass(slots=True)
class FusionPassContext:
    ops: list[LazyOp]
    consumed_by: dict[int, list[int]]
    op_idx: dict[int, int]
    roots: set[int]


@dataclass(slots=True)
class FusionPassState:
    planned: set[int] = field(default_factory=set)


FUSIBLE_GEMVS = frozenset(
    {"dot_q4_k_v2", "dot_q4_k_silu_v2", "dot_mlx_q4_v2", "dot_mlx_q4_silu_v2"}
)


class RmsNormFoldPass:
    """Fold an rms_norm row-reduction into the decode GEMV that consumes it.

    Matches an rms_norm whose `out` is read as the activation (`A`) of exactly
    one fusible GEMV, and whose `rrms_out` is otherwise unused (the decode
    shape). Claims both ops; the compiler rides the GEMV's K-loop/simd_reduce.
    """

    kind: FusionKind = FusionKind.REDUCE_FOLD
    name: str = kind.value

    def propose(self, context: FusionPassContext, state: FusionPassState) -> list[FusionGroup]:
        ops = context.ops
        groups: list[FusionGroup] = []
        # buffer-key -> indices that READ it; and the writer's op index per key
        readers: dict[tuple[int, int], list[int]] = {}
        writer: dict[tuple[int, int], int] = {}
        for ci, cop in enumerate(ops):
            for pn, arr in cop.buffer_args:
                if pn in cop.output_params:
                    writer[arr.buffer_key] = ci
                else:
                    readers.setdefault(arr.buffer_key, []).append(ci)

        for i, op in enumerate(ops):
            if i in state.planned or op.kernel is None or op.kernel.name != "rms_norm":
                continue
            if op.constexpr_values.get("M", 1) != 1 and op.buffer_shapes.get("x", (1,))[0] != 1:
                continue
            out_key = rrms_key = None
            for pn, arr in op.buffer_args:
                if pn == "out":
                    out_key = arr.buffer_key
                elif pn == "rrms_out":
                    rrms_key = arr.buffer_key
            if out_key is None:
                continue
            # rrms_out must be unused (only the reduce writes it)
            if rrms_key is not None and readers.get(rrms_key):
                continue
            out_readers = [c for c in readers.get(out_key, []) if c != i]
            if len(out_readers) != 1:
                continue
            gemv = out_readers[0]
            if gemv in state.planned or ops[gemv].kernel.name not in FUSIBLE_GEMVS:
                continue
            # the GEMV must read `out` as its activation param A
            if not any(
                pn == "A" and arr.buffer_key == out_key for pn, arr in ops[gemv].buffer_args
            ):
                continue
            state.planned.update({i, gemv})
            groups.append(
                FusionGroup(
                    op_indices=[i, gemv],
                    kind=FusionKind.REDUCE_FOLD,
                    anchor_idx=gemv,
                    pro_chain=[i],
                )
            )
        return groups


class RowPass:
    kind: FusionKind = FusionKind.ROW_PASS
    name: str = kind.value

    def propose(self, context: FusionPassContext, state: FusionPassState) -> list[FusionGroup]:
        groups: list[FusionGroup] = []
        for i, op in enumerate(context.ops):
            if op.kernel is None or not _is_row_reduce(op):
                continue
            if i in state.planned:
                continue
            group = grow_group(context.ops, i, state.planned, roots=context.roots)
            if group is None:
                continue
            state.planned.update(group.op_indices)
            groups.append(FusionGroup(op_indices=group.op_indices, kind=FusionKind.ROW_PASS))
        return groups


class MultiRootPass:
    kind: FusionKind = FusionKind.MULTI_ROOT
    name: str = kind.value

    def propose(self, context: FusionPassContext, state: FusionPassState) -> list[FusionGroup]:
        groups: list[FusionGroup] = []
        islands = find_islands(
            context.ops,
            context.op_idx,
            context.consumed_by,
            planned=set(state.planned),
            roots=context.roots,
        )
        for island in islands:
            bad_root = False
            for idx in island.indices:
                if idx in context.roots and idx not in island.writebacks:
                    bad_root = True
                    break
            if bad_root:
                continue
            if self._anchor_absorbable(island, context, state.planned):
                continue
            state.planned.update(island.indices)
            groups.append(
                FusionGroup(
                    op_indices=sorted(island.indices),
                    kind=FusionKind.MULTI_ROOT,
                    extra_branches=[island.order],
                    epi_chain=sorted(island.writebacks),
                )
            )
        return groups

    def _anchor_absorbable(
        self, island: Island, context: FusionPassContext, planned: set[int]
    ) -> bool:
        """True if a GEMM-class anchor's epilogue fusion would absorb the WHOLE
        island. Then defer: AnchorFusionPass folds these elementwise ops into the
        anchor's store (zero extra dispatches) instead of emitting a standalone
        multi-root kernel (one extra dispatch + an extra read of the anchor's
        output). Siblings over a NON-anchor producer (e.g. cos+sin sharing the
        non-fusable `cat` emb) have no absorbing anchor, so they still fire."""
        members = set(island.indices)
        anchors: set[int] = set()
        for idx in members:
            for producer in context.ops[idx].input_producers.values():
                pi = context.op_idx.get(id(producer))
                if pi is None or pi in members:
                    continue
                ap = context.ops[pi]
                if ap.is_elem_op() or not ap.output_params:
                    continue
                if ap.kernel is None or ap.kernel.name in _NON_FUSABLE_ELEM_KERNELS:
                    continue
                anchors.add(pi)
        for anchor_idx in anchors:
            epi_chain, extra_branches = _build_multi_epi(
                context.ops, context.consumed_by, anchor_idx, planned, context.roots
            )
            absorbed = set(epi_chain)
            for branch in extra_branches:
                absorbed.update(branch)
            if members <= absorbed:
                return True
        return False


class AnchorFusionPass:
    kind: FusionKind = FusionKind.ANCHOR
    name: str = kind.value

    def propose(self, context: FusionPassContext, state: FusionPassState) -> list[FusionGroup]:
        groups: list[FusionGroup] = []
        candidates: list[tuple[int, int]] = []
        for i, op in enumerate(context.ops):
            if (
                op.is_elem_op()
                or not op.output_params
                or op.kernel.name in _NON_FUSABLE_ELEM_KERNELS
            ):
                continue
            if i in state.planned:
                continue
            pro = _find_best_prologue(
                context.ops,
                context.consumed_by,
                context.op_idx,
                i,
                set(),
                context.roots,
            )
            candidates.append((i, len(pro)))
        candidates.sort(key=lambda x: x[0])

        for anchor_idx, _ in candidates:
            if anchor_idx in state.planned:
                continue

            anchor_name = context.ops[anchor_idx].kernel.name
            if "attention" in anchor_name:
                pro_chain: list[int] = []
            elif anchor_name == "layernorm":
                pro_chain = []
            elif anchor_name in ("dot", "dot_add", "dot_transpose_rhs"):
                m_dim = context.ops[anchor_idx].buffer_shapes.get("A", (0,))[0]
                if 0 < m_dim <= 32:
                    pro_chain = []
                else:
                    pro_chain = _find_best_prologue(
                        context.ops,
                        context.consumed_by,
                        context.op_idx,
                        anchor_idx,
                        state.planned,
                        context.roots,
                    )
            else:
                pro_chain = _find_best_prologue(
                    context.ops,
                    context.consumed_by,
                    context.op_idx,
                    anchor_idx,
                    state.planned,
                    context.roots,
                )

            epi_chain, extra_branches = _build_multi_epi(
                context.ops,
                context.consumed_by,
                anchor_idx,
                state.planned,
                context.roots,
            )
            all_epi: set[int] = set(epi_chain)
            for branch in extra_branches:
                all_epi.update(branch)

            fusion_group = all_epi | set(pro_chain) | {anchor_idx}
            if epi_chain:
                anchor_consumers = context.consumed_by.get(anchor_idx, [])
                external = [
                    consumer for consumer in anchor_consumers if consumer not in fusion_group
                ]
                if external:
                    hi = max({anchor_idx} | all_epi)
                    unsafe = any(consumer <= hi for consumer in external)
                    if unsafe:
                        epi_chain = []
                        extra_branches = []
                        all_epi = set()
                    else:
                        extra_branches = [epi_chain] + extra_branches
                        epi_chain = []

            if pro_chain:
                for pro_idx in pro_chain:
                    for consumer in context.consumed_by.get(pro_idx, []):
                        if consumer not in fusion_group:
                            pro_chain = []
                            break
                    if not pro_chain:
                        break

            if any(idx in context.roots for idx in pro_chain):
                pro_chain = []
                epi_chain = []
                extra_branches = []
                all_epi = set()
            elif anchor_idx in context.roots and epi_chain:
                extra_branches = [epi_chain] + extra_branches
                epi_chain = []
            else:
                internal_root_positions = [
                    k for k in range(len(epi_chain) - 1) if epi_chain[k] in context.roots
                ]
                if internal_root_positions:
                    fused_max_current = max(
                        {anchor_idx}
                        | set(pro_chain)
                        | set(epi_chain)
                        | {idx for branch in extra_branches for idx in branch}
                    )
                    safe_splits: list[list[int]] = []
                    for k in internal_root_positions:
                        root_idx = epi_chain[k]
                        ext = [
                            consumer
                            for consumer in context.consumed_by.get(root_idx, [])
                            if consumer != epi_chain[k + 1]
                            and consumer != anchor_idx
                            and consumer not in set(pro_chain)
                            and consumer not in set(epi_chain)
                        ]
                        if all(consumer > fused_max_current for consumer in ext):
                            safe_splits.append(epi_chain[: k + 1])
                    if safe_splits:
                        extra_branches = safe_splits + extra_branches

            if epi_chain:
                fused_set = {anchor_idx} | set(epi_chain)
                truncate_at: int | None = None
                for epi_pos, epi_idx in enumerate(epi_chain):
                    for producer in context.ops[epi_idx].input_producers.values():
                        prod_idx = context.op_idx.get(id(producer))
                        if (
                            prod_idx is not None
                            and prod_idx not in fused_set
                            and prod_idx > anchor_idx
                        ):
                            truncate_at = epi_pos
                            break
                    if truncate_at is not None:
                        break
                if truncate_at is not None:
                    epi_chain = epi_chain[:truncate_at]
                    extra_branches = []
                    all_epi = set(epi_chain)

            all_idx: set[int] = set(pro_chain) | {anchor_idx} | all_epi
            if all_idx & state.planned:
                pro_chain = []
                all_idx = {anchor_idx} | all_epi
                if all_idx & state.planned:
                    all_idx = {anchor_idx}
                    epi_chain = []
                    extra_branches = []
                    all_epi = set()
                    if anchor_idx in state.planned:
                        continue

            state.planned.update(all_idx)
            groups.append(
                FusionGroup(
                    op_indices=sorted(all_idx),
                    kind=FusionKind.ANCHOR,
                    anchor_idx=anchor_idx,
                    pro_chain=pro_chain,
                    epi_chain=epi_chain,
                    extra_branches=extra_branches,
                )
            )
        return groups


class ElementChainPass:
    kind: FusionKind = FusionKind.ELEM_CHAIN
    name: str = kind.value

    def propose(self, context: FusionPassContext, state: FusionPassState) -> list[FusionGroup]:
        groups: list[FusionGroup] = []
        for i, op in enumerate(context.ops):
            if i in state.planned or not op.is_elem_op():
                continue
            consumers = context.consumed_by.get(i, [])
            if len(consumers) != 1:
                continue
            chain: list[int] = [i]
            cur = consumers[0]
            while cur not in state.planned:
                if not context.ops[cur].is_elem_op():
                    break
                if not _primary_input_chains(context.ops[cur], context.ops[chain[-1]]):
                    break
                chain.append(cur)
                next_consumers = context.consumed_by.get(cur, [])
                if len(next_consumers) == 1:
                    cur = next_consumers[0]
                elif len(next_consumers) == 2:
                    reconvergence = _find_reconvergence(
                        context.ops,
                        context.consumed_by,
                        next_consumers,
                        state.planned,
                    )
                    if reconvergence is None:
                        break
                    branch, reconv = reconvergence
                    prev_check = chain[-1]
                    branch_ok = True
                    for branch_idx in branch:
                        if not context.ops[branch_idx].is_elem_op() or not _primary_input_chains(
                            context.ops[branch_idx], context.ops[prev_check]
                        ):
                            branch_ok = False
                            break
                        prev_check = branch_idx
                    if branch_ok and _primary_input_chains(
                        context.ops[reconv], context.ops[prev_check]
                    ):
                        chain.extend(branch)
                        chain.append(reconv)
                        reconv_consumers = context.consumed_by.get(reconv, [])
                        if len(reconv_consumers) == 1:
                            cur = reconv_consumers[0]
                        else:
                            break
                    else:
                        break
                else:
                    break
            if (
                len(chain) > 1
                and not (set(chain) & state.planned)
                and not (set(chain[:-1]) & context.roots)
                and not _elem_chain_changes_shape(chain, context.ops)
            ):
                state.planned.update(chain)
                groups.append(FusionGroup(op_indices=chain, kind=FusionKind.ELEM_CHAIN))
        return groups


class IndividualPass:
    kind: FusionKind = FusionKind.INDIVIDUAL
    name: str = kind.value

    def propose(self, context: FusionPassContext, state: FusionPassState) -> list[FusionGroup]:
        groups: list[FusionGroup] = []
        for i in range(len(context.ops)):
            if i not in state.planned:
                state.planned.add(i)
                groups.append(FusionGroup(op_indices=[i], kind=FusionKind.INDIVIDUAL))
        return groups


FusionPass = (
    RmsNormFoldPass | RowPass | MultiRootPass | AnchorFusionPass | ElementChainPass | IndividualPass
)

FUSION_PASSES: tuple[FusionPass, ...] = (
    RmsNormFoldPass(),
    RowPass(),
    MultiRootPass(),
    AnchorFusionPass(),
    ElementChainPass(),
    IndividualPass(),
)


def propose_candidates(
    ops: list[LazyOp],
    consumed_by: dict[int, list[int]],
    root_indices: set[int] | None = None,
) -> list[FusionGroup]:
    """Generate fusion candidates with ALL correctness constraints."""
    if not ops:
        return []
    if len(ops) == 1:
        return [FusionGroup(op_indices=[0], kind=FusionKind.INDIVIDUAL)]

    context = FusionPassContext(
        ops=ops,
        consumed_by=consumed_by,
        op_idx={id(op): i for i, op in enumerate(ops)},
        roots=root_indices or set(),
    )
    state = FusionPassState()
    candidates: list[FusionGroup] = []
    for fusion_pass in FUSION_PASSES:
        candidates.extend(fusion_pass.propose(context, state))
    candidates.sort(key=lambda group: min(group.op_indices))
    return candidates


_FUSION_DISABLED = False  # set True to disable all fusion (debug)


def _plan_fusion(
    ops: list[LazyOp], root_op_ids: set[int] | None = None
) -> list[tuple[int, FusionPlan]]:
    """Pure analysis: plan which ops to fuse. No compilation."""
    if _FUSION_DISABLED:
        return [(i, FusionPlan(FusionKind.INDIVIDUAL, {i}, idx=i)) for i in range(len(ops))]
    if len(ops) <= 1:
        return [(0, FusionPlan(FusionKind.INDIVIDUAL, {0}, idx=0))] if ops else []
    root_indices: set[int] = {
        i for i, op in enumerate(ops) if root_op_ids is not None and id(op) in root_op_ids
    }

    _out_direct: dict[int, list[tuple[int, AlloyBuffer]]] = {}
    _out_root: dict[tuple[int, int], list[tuple[int, AlloyBuffer]]] = {}
    for _ci, _cop in enumerate(ops):
        for _pn, _ca in _cop.buffer_args:
            if _pn in _cop.output_params:
                _out_direct.setdefault(_ca.data_ptr, []).append((_ci, _ca))
                _out_root.setdefault(_ca.buffer_key, []).append((_ci, _ca))

    consumed_by: dict[int, list[int]] = {}
    for _ci, _cop in enumerate(ops):
        _matched: set[int] = set()
        for _pn, _ca in _cop.buffer_args:
            if _pn in _cop.output_params:
                continue
            _ptr = _ca.data_ptr
            for _j, _oa in _out_direct.get(_ptr, ()):
                if _j < _ci and _j not in _matched:
                    if _ca.shares_allocation(_oa):
                        consumed_by.setdefault(_j, []).append(_ci)
                        _matched.add(_j)
            for _j, _oa in _out_root.get(_ca.buffer_key, ()):
                if _j < _ci and _j not in _matched:
                    if _ca.shares_allocation(_oa):
                        consumed_by.setdefault(_j, []).append(_ci)
                        _matched.add(_j)

    groups: list[FusionGroup] = propose_candidates(ops, consumed_by, root_indices)
    plans: list[tuple[int, FusionPlan]] = []
    for g in groups:
        topo_idx = min(g.op_indices)
        match g.kind:
            case FusionKind.ANCHOR:
                plans.append(
                    (
                        topo_idx,
                        FusionPlan(
                            g.kind,
                            set(g.op_indices),
                            anchor_idx=g.anchor_idx,
                            pro_chain=g.pro_chain,
                            epi_chain=g.epi_chain,
                            extra_branches=g.extra_branches,
                        ),
                    )
                )
            case FusionKind.ELEM_CHAIN:
                plans.append(
                    (
                        topo_idx,
                        FusionPlan(g.kind, set(g.op_indices), chain=g.op_indices),
                    )
                )
            case FusionKind.ROW_PASS:
                plans.append(
                    (
                        topo_idx,
                        FusionPlan(g.kind, set(g.op_indices), chain=g.op_indices),
                    )
                )
            case FusionKind.MULTI_ROOT:
                plans.append(
                    (
                        topo_idx,
                        FusionPlan(
                            g.kind,
                            set(g.op_indices),
                            chain=g.extra_branches[0],  # topo order
                            epi_chain=g.epi_chain,  # writebacks
                        ),
                    )
                )
            case FusionKind.REDUCE_FOLD:
                plans.append(
                    (
                        topo_idx,
                        FusionPlan(
                            g.kind,
                            set(g.op_indices),
                            anchor_idx=g.anchor_idx,
                            pro_chain=g.pro_chain,
                        ),
                    )
                )
            case FusionKind.INDIVIDUAL:
                plans.append(
                    (
                        topo_idx,
                        FusionPlan(g.kind, set(g.op_indices), idx=g.op_indices[0]),
                    )
                )

    plans.sort(key=lambda x: x[0])
    return plans
