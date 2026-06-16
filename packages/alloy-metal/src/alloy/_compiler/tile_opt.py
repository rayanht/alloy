"""Tile IR optimization passes — rewrite tile IR for better codegen.

Each pass is a function(func: TileFunction) → None that rewrites the IR
in-place. Passes run before planning and emission.

Architecture:
    trace → tile IR → [tile_opt: IR rewrites] → [tile_plan: plan] → [tile_msl: emit]

Prologue/epilogue transforms are attached directly during fusion injection
(fusion.py:inject_prologue_ir sets Load.transform; epilogue transforms set
Store.transform). tile_opt only handles IR-structural rewrites like persistent MMA.
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields

from alloy._compiler.tile_ir import (
    Barrier,
    BinOp,
    Compare,
    Constant,
    Dot,
    ForLoop,
    FusedElementwise,
    IfElse,
    Load,
    Reduce,
    RowPass,
    Select,
    SimdMatrixOp,
    Store,
    TernaryOp,
    TileFunction,
    TileOp,
    TileValue,
    UnaryOp,
    WhileLoop,
    Zeros,
    walk_ops,
)

# Ops that must not be hoisted out of loops.
# Load: shmem lifetime depends on loop position.
# SimdMatrixOp/Dot: register declarations scoped to loop body.
# Barrier/Store: side effects.
_NO_HOIST_OPS = (Barrier, Store, SimdMatrixOp, Dot, Load)


def optimize_tile_ir(func: TileFunction):
    """Run all tile IR optimization passes."""
    for opt in _PASSES:
        opt(func)


def _opt_persistent_mma(func: TileFunction):
    """Fuse loop-carried dot accumulators into persistent simdgroup registers.

    Detects the pattern (at any nesting level):
        Zeros z (M, N)
        ForLoop carried: (z → t_add)
            Dot(A, B) → d
            BinOp(add, z, d) → t_add

    Rewrites to:
        Zeros z (M, N)
        ForLoop carried: (z → d)
            Dot(A, B, acc=z) → d

    Handles nested ForLoops (e.g., chained GEMM where the inner loop's
    intermediate accumulator feeds the outer loop's Dot).
    """
    # Dot is mutated in-place when we set acc (persistent rewrite). The Dot
    # object can be shared across cached pipeline runs (e.g. when the same
    # kernel is recompiled with different planner inputs), but the rest of
    # the IR (op.body, op.carried, Store.value) is rebuilt fresh each time.
    # Reset Dot.acc so the pass always starts from a known clean state and
    # rewrites match what's actually in the body.
    for op in walk_ops(func.ops):
        if isinstance(op, Dot):
            op.acc = None
            op.acc_pre_scale = None

    _apply_pmma(func.ops, func.ops, top_level=True)
    _absorb_post_loop_scale(func.ops)


def _absorb_post_loop_scale(scope_ops: list[TileOp]) -> None:
    """Absorb `ForLoop.carry_out → BinOp(mul, scalar) → Store` into Store.transform.

    Runs after persistent_mma has updated outer ForLoop carries to point at
    persistent Dot results. The trailing scalar Mul (e.g., the SDPA bwd
    `dk = dk * SCALE` post-loop) gets folded into the Store epilogue so the
    persistent acc is consumed directly by Store without an intermediate
    materialization to shmem (which would defeat the persistent acc).
    """
    forloop_carry_finals: set[str] = set()
    persistent_acc_finals: set[str] = set()
    for op in scope_ops:
        if isinstance(op, ForLoop):
            for _, final in op.carried:
                forloop_carry_finals.add(final.name)
                # Identify finals that are persistent-acc Dot results — only
                # those flow into _emit_persistent_mma_store and can use the
                # acc_post_scale path. Non-persistent carries can't.
                for body_op in op.body:
                    if (
                        isinstance(body_op, Dot)
                        and body_op.acc is not None
                        and body_op.result is not None
                        and body_op.result.name == final.name
                    ):
                        persistent_acc_finals.add(final.name)
                        break
    if not forloop_carry_finals:
        return

    # Find Mul(carry_final, scalar_or_per_row) → t_scaled, Store(t_scaled).
    # `per_row_kind` is "scalar" (0-dim, transform-fold) or "row" (FA-2: per-row
    # broadcast (M, 1), use Store.acc_post_scale).
    mul_by_name: dict[str, tuple[BinOp, str]] = {}
    for op in scope_ops:
        if not isinstance(op, BinOp) or op.op != "mul":
            continue
        if op.result is None:
            continue
        lhs_name = op.lhs.name if op.lhs else ""
        rhs_name = op.rhs.name if op.rhs else ""
        # Carry-out is one side; classify the other side's shape.
        if lhs_name in forloop_carry_finals and op.rhs is not None:
            other = op.rhs
            carry_name = lhs_name
        elif rhs_name in forloop_carry_finals and op.lhs is not None:
            other = op.lhs
            carry_name = rhs_name
        else:
            continue
        if len(other.shape) == 0:
            mul_by_name[op.result.name] = (op, "scalar")
        elif (
            len(other.shape) == 2
            and op.result is not None
            and len(op.result.shape) == 2
            and other.shape[0] == op.result.shape[0]
            and other.shape[1] == 1
            and carry_name in persistent_acc_finals
        ):
            # Per-row broadcast — FA-2 forward `o = o * (1/l)`.
            # Only apply when the carry final is a persistent-acc Dot result;
            # otherwise the Store hits the regular emit path which ignores
            # acc_post_scale, silently dropping the per-row mul.
            mul_by_name[op.result.name] = (op, "row")

    if not mul_by_name:
        return

    to_remove: list[TileOp] = []
    for op in scope_ops:
        if not isinstance(op, Store) or op.value is None:
            continue
        entry = mul_by_name.get(op.value.name)
        if entry is None:
            continue
        mul_op, kind = entry
        lhs_name = mul_op.lhs.name if mul_op.lhs else ""
        rhs_name = mul_op.rhs.name if mul_op.rhs else ""
        if lhs_name in forloop_carry_finals:
            new_val = mul_op.lhs
            other_val = mul_op.rhs
        else:
            new_val = mul_op.rhs
            other_val = mul_op.lhs
        op.value = new_val
        if kind == "scalar":
            op.transform = list(op.transform) + [mul_op]
            op.transform_source_name = new_val.name
        else:
            # Per-row: emit-time path applies `acc *= acc_post_scale[row]`
            # per-lane during the persistent-MMA device store.
            op.acc_post_scale = other_val
        to_remove.append(mul_op)

    if to_remove:
        scope_ops[:] = [s for s in scope_ops if s not in to_remove]


def _apply_pmma(
    scope_ops: list[TileOp],
    all_parent_ops: list[TileOp],
    top_level: bool = False,
    ancestor_op_maps: tuple[dict[str, TileOp], ...] = (),
    external_uses: frozenset[str] = frozenset(),
):
    """Apply persistent MMA rewrite to ForLoops in scope_ops.

    scope_ops: the ops list containing the ForLoops to process
    all_parent_ops: the top-level func.ops (for Store reference updates)
    top_level: if True, require directly-stored check (top-level ForLoops
               can only be persistent MMA if the result goes straight to Store)
    ancestor_op_maps: op_maps from outer scopes, searched (innermost first)
                     when the current scope doesn't define the carry init.
                     SDPA bwd nests `for g in range(KV_GROUP)` around `for _ib`,
                     and the dv/dk Zeros live at top scope — without this
                     lookup, the inner BinOp(add, z67, dot) can't be matched
                     back to its outer Zeros init.
    """
    # Build op map from current scope (for Zeros lookup)
    op_map: dict[str, TileOp] = {}
    for op in scope_ops:
        if op.result:
            op_map[op.result.name] = op

    def _lookup_init(name: str) -> TileOp | None:
        op = op_map.get(name)
        if op is not None:
            return op
        for anc in ancestor_op_maps:
            op = anc.get(name)
            if op is not None:
                return op
        return None

    for op in scope_ops:
        if not isinstance(op, ForLoop):
            continue

        # Recursively handle nested ForLoops first (innermost rewrites first).
        # Propagate this loop's carry-final names as external_uses so an inner
        # loop with a matching carry can see its final value as "used" by the
        # outer carry (the outer loop forwards the inner result to its own
        # iterations / next outer iter / post-loop scope).
        nested_externals = external_uses | frozenset(f.name for _, f in op.carried)
        _apply_pmma(
            op.body,
            all_parent_ops,
            top_level=False,
            ancestor_op_maps=(op_map,) + ancestor_op_maps,
            external_uses=nested_externals,
        )

        body_map: dict[str, TileOp] = {}
        for bop in op.body:
            if bop.result:
                body_map[bop.result.name] = bop

        new_carried = []
        rewrites = []  # (init_name, dot_name, add_name)

        for init_val, final_val in op.carried:
            init_op = _lookup_init(init_val.name)
            if not isinstance(init_op, Zeros) or len(init_op.shape) != 2:
                new_carried.append((init_val, final_val))
                continue

            final_op = body_map.get(final_val.name)
            if not isinstance(final_op, BinOp) or final_op.op != "add":
                new_carried.append((init_val, final_val))
                continue

            lhs_name = final_op.lhs.name if final_op.lhs else ""
            rhs_name = final_op.rhs.name if final_op.rhs else ""

            pre_scale_op: BinOp | None = None
            pre_scale_val: TileValue | None = None
            if lhs_name == init_val.name:
                dot_name = rhs_name
            elif rhs_name == init_val.name:
                dot_name = lhs_name
            else:
                # FA-2 forward rescale: `add(mul(z, alpha), dot)` shape.
                lhs_op_check = body_map.get(lhs_name)
                rhs_op_check = body_map.get(rhs_name)
                if (
                    isinstance(lhs_op_check, BinOp) and lhs_op_check.op == "mul"
                    and lhs_op_check.lhs is not None and lhs_op_check.rhs is not None
                    and (lhs_op_check.lhs.name == init_val.name or lhs_op_check.rhs.name == init_val.name)
                ):
                    pre_scale_op = lhs_op_check
                    dot_name = rhs_name
                elif (
                    isinstance(rhs_op_check, BinOp) and rhs_op_check.op == "mul"
                    and rhs_op_check.lhs is not None and rhs_op_check.rhs is not None
                    and (rhs_op_check.lhs.name == init_val.name or rhs_op_check.rhs.name == init_val.name)
                ):
                    pre_scale_op = rhs_op_check
                    dot_name = lhs_name
                else:
                    new_carried.append((init_val, final_val))
                    continue
                pre_scale_val = (
                    pre_scale_op.rhs
                    if pre_scale_op.lhs is not None and pre_scale_op.lhs.name == init_val.name
                    else pre_scale_op.lhs
                )
                if pre_scale_val is None or pre_scale_val.shape != (init_val.shape[0], 1):
                    new_carried.append((init_val, final_val))
                    continue
                scaled_z_name = pre_scale_op.result.name if pre_scale_op.result else None
                if scaled_z_name is None:
                    new_carried.append((init_val, final_val))
                    continue
                external_consumers = 0
                for other in op.body:
                    if other is pre_scale_op or other is final_op:
                        continue
                    if any(v.name == scaled_z_name for v in other.operand_values()):
                        external_consumers += 1
                if external_consumers > 0:
                    new_carried.append((init_val, final_val))
                    continue

            dot_op = body_map.get(dot_name)
            if not isinstance(dot_op, Dot):
                new_carried.append((init_val, final_val))
                continue

            # Check: the carried-out value is used after the loop.
            # Top-level: must be directly stored (the emitter can't handle
            # elementwise ops on MMA-layout values; only Store.transform works).
            # We also accept the pattern `Store(BinOp(mul, final_val, scalar))`
            # — the post-loop scalar Mul gets absorbed into Store.transform as
            # an epilogue, letting MMA accumulators with a final scale (SDPA
            # bwd dq/dkdv: `acc * 1/sqrt(D)`) stay register-resident across
            # the loop instead of materializing per-iter to shmem.
            #
            # Nested: any usage suffices (materialization handles MMA→shmem).
            # The pass must be idempotent: a cached pipeline run may have
            # already re-routed `final_val` to `dot_op.result` or folded
            # the post-loop Mul into Store.transform / Store.acc_post_scale,
            # so the is_used check looks up every equivalent name.
            equiv_names = {final_val.name, dot_op.result.name}

            scale_store_info: tuple[BinOp, Store] | None = None
            if top_level:
                # Direct-store fast path: Store(final_val) routes through
                # _emit_persistent_mma_store and writes straight from the
                # simdgroup_matrix registers to device memory — no shmem
                # round-trip and no need for the planner to allocate a slot.
                is_used = False
                for post_op in scope_ops:
                    if (
                        isinstance(post_op, Store)
                        and post_op.value
                        and post_op.value.name in equiv_names
                    ):
                        is_used = True
                        break
                if not is_used:
                    # `final_val → Mul(scalar) → Store` folds the scalar into
                    # Store.transform (acc_post_scale path).
                    for nm in equiv_names:
                        scale_store_info = _find_scale_then_store(nm, scope_ops)
                        if scale_store_info is not None:
                            is_used = True
                            break
                # Any other post-loop usage (e.g. SiLU epilogue in
                # dot_q4_k_silu: `silu(acc_gate) * acc_up` before the Store).
                # The planner sees `consumed_by_other` and allocates a shmem
                # slot for the persistent dot result; the consumer-side
                # auto-spill in _emit_op materializes the persistent acc once
                # at first-use, after which the elementwise chain reads from
                # shmem normally. Without this, the (Zeros, ForLoop carry,
                # add) pattern stays non-persistent and each K iter spills
                # the partial MMA via shmem — the worst case.
                if not is_used:
                    for nm in equiv_names:
                        if _value_used_after(nm, op, scope_ops):
                            is_used = True
                            break
            else:
                is_used = any(_value_used_after(nm, op, scope_ops) for nm in equiv_names)
                if not is_used and (final_val.name in external_uses or dot_op.result.name in external_uses):
                    # Final value is consumed by an outer loop's carry — counts
                    # as a usage for nested-loop persistent MMA propagation.
                    is_used = True
            if not is_used:
                new_carried.append((init_val, final_val))
                continue

            # Pattern matched — rewrite
            dot_op.acc = init_val
            if pre_scale_val is not None:
                dot_op.acc_pre_scale = pre_scale_val
            rewrites.append((init_val.name, dot_name, final_val.name, scale_store_info, pre_scale_op))
            new_carried.append((init_val, dot_op.result))

        if not rewrites:
            continue

        op.carried = new_carried

        # Remove BinOp(add) ops from body, plus FA-2 rescale Mul ops.
        add_names = {r[2] for r in rewrites}
        rescale_mul_ids = {id(r[4]) for r in rewrites if r[4] is not None}
        op.body = [
            bop for bop in op.body
            if not (bop.result and bop.result.name in add_names)
            and id(bop) not in rescale_mul_ids
        ]

        # Update all references to the old carried-out value (BinOp result)
        # with the Dot result. For top-level, this is Store references.
        # For nested, this includes Dot inputs in the parent scope.
        for _, dot_name, add_name, scale_store_info, _pre_scale_op in rewrites:
            dot_op = body_map[dot_name]
            for target_ops in (all_parent_ops, scope_ops):
                _replace_value_refs(target_ops, add_name, dot_op.result)
            # If we matched the scale-then-store pattern, fold the Mul into
            # Store.transform and drop the standalone Mul from scope_ops.
            if scale_store_info is not None:
                mul_op, store_op = scale_store_info
                store_op.value = dot_op.result
                store_op.transform = list(store_op.transform) + [mul_op]
                store_op.transform_source_name = dot_op.result.name
                scope_ops[:] = [
                    s for s in scope_ops
                    if not (isinstance(s, BinOp) and s is mul_op)
                ]


def _find_scale_then_store(
    val_name: str, scope_ops: list[TileOp]
) -> tuple[BinOp, Store] | None:
    """Look for `val_name → BinOp(mul, val_name, scalar) → Store` in scope_ops.

    Returns (mul_op, store_op) if matched, else None. The scalar must be a
    0D / replicated value (not a 2D tile) so the Mul can fold into Store as
    a per-element scaling.
    """
    def _is_scalar_like(value: TileValue) -> bool:
        return len(value.shape) == 0 or all(dim == 1 for dim in value.shape)

    mul_op: BinOp | None = None
    for op in scope_ops:
        if not isinstance(op, BinOp) or op.op != "mul":
            continue
        lhs_name = op.lhs.name if op.lhs else ""
        rhs_name = op.rhs.name if op.rhs else ""
        # One side must be the carried-out value; the other a 0D scalar.
        if lhs_name == val_name and op.rhs is not None and _is_scalar_like(op.rhs):
            mul_op = op
            break
        if rhs_name == val_name and op.lhs is not None and _is_scalar_like(op.lhs):
            mul_op = op
            break
    if mul_op is None or mul_op.result is None:
        return None
    mul_result_name = mul_op.result.name
    for op in scope_ops:
        if (
            isinstance(op, Store)
            and op.value
            and op.value.name == mul_result_name
        ):
            return mul_op, op
    return None


def _value_used_after(val_name: str, loop_op: ForLoop, scope_ops: list[TileOp]) -> bool:
    """Check if val_name is used by any op after loop_op in scope_ops."""
    found_loop = False
    for op in scope_ops:
        if op is loop_op:
            found_loop = True
            continue
        if not found_loop:
            continue
        if _op_references(op, val_name):
            return True
    return False


def _op_references(op: TileOp, val_name: str) -> bool:
    """Check if op references val_name as an input (including nested ForLoop bodies)."""
    for v in op.operand_values():
        if v.name == val_name:
            return True
    if isinstance(op, ForLoop):
        for body_op in op.body:
            if _op_references(body_op, val_name):
                return True
    return False


def _replace_value_refs(ops: list[TileOp], old_name: str, new_val: TileValue):
    """Replace all references to old_name with new_val in ops (recursively)."""
    mapping = {old_name: new_val}
    for op in walk_ops(ops):
        op.remap(mapping)


def _opt_fuse_row_loops(func: TileFunction):
    """Fuse consecutive elementwise 2D ops into FusedElementwise nodes.

    Gated by the `_fuse_loops` constexpr flag (default 0 = off).
    The tuner sweeps `_fuse_loops=[0, 1]` and picks the best per shape.

    Scans ForLoop bodies for chains of BinOp/UnaryOp/Select/Compare/TernaryOp
    that operate on the same shared-memory tile rows. Replaces them with a
    single FusedElementwise op so the emitter can emit one column loop.

    Determines which intermediate results need shmem writeback vs can stay
    in registers by checking whether they're consumed after the chain.
    """
    if not func.options.get("fuse_loops", 0):
        return

    _FUSABLE = (BinOp, UnaryOp, Select, Compare, TernaryOp)

    def _find_chains(body: list[TileOp]) -> list[TileOp]:
        """Return a new body list with fusable chains replaced by FusedElementwise."""
        # First pass: identify which 2D ops produce results consumed by a Dot
        # (these are "anchor" values that must stay in shmem). Used to determine
        # the source buffer for the chain.
        dot_inputs: set[str] = set()
        for op in body:
            if isinstance(op, Dot):
                if op.lhs:
                    dot_inputs.add(op.lhs.name)
                if op.rhs:
                    dot_inputs.add(op.rhs.name)

        # Build consumed-after map for each possible chain position
        # consumed_after[name] = True if name is referenced by any op after
        # the op that produces it (beyond the immediate next fusable op)
        all_result_names: dict[str, int] = {}  # name → index in body
        for idx, op in enumerate(body):
            if op.result:
                all_result_names[op.result.name] = idx

        # Values known to be in shared memory: 2D Load results and Dot results
        in_shmem: set[str] = set()
        for op in body:
            if isinstance(op, Load) and op.result and len(op.result.shape) == 2:
                in_shmem.add(op.result.name)
            if isinstance(op, Dot) and op.result:
                in_shmem.add(op.result.name)

        new_body: list[TileOp] = []
        i = 0
        while i < len(body):
            op = body[i]
            # Only start chains at 2D fusable ops whose input is in shmem
            if not isinstance(op, _FUSABLE) or not op.result or len(op.result.shape) != 2:
                new_body.append(op)
                i += 1
                continue

            # At least one input must be in shared memory (Dot/Load result)
            if not any(v.name in in_shmem for v in op.operand_values()):
                new_body.append(op)
                i += 1
                continue

            rows = op.result.shape[0]
            chain: list[TileOp] = [op]
            chain_names: set[str] = {op.result.name}
            scan_end = i + 1

            for j in range(i + 1, len(body)):
                nxt = body[j]
                # Hard breaks
                if isinstance(nxt, (Barrier, Dot, Load, Store, ForLoop, WhileLoop, IfElse)):
                    scan_end = j
                    break
                if isinstance(nxt, Reduce):
                    scan_end = j
                    break
                # Skip non-2D ops (address math, constants)
                if not isinstance(nxt, _FUSABLE):
                    continue
                if not nxt.result or len(nxt.result.shape) != 2 or nxt.result.shape[0] != rows:
                    continue
                # Must consume a chain value
                if not any(v.name in chain_names for v in nxt.operand_values()):
                    continue
                # Reject Select with (M, N) condition (causal mask) — the
                # condition mixes row and column indices that can't be
                # separated by a simple tid → _c substitution.
                if isinstance(nxt, Select) and nxt.cond and len(nxt.cond.shape) == 2:
                    if nxt.cond.shape[0] > 1 and nxt.cond.shape[1] > 1:
                        scan_end = j
                        break
                chain.append(nxt)
                chain_names.add(nxt.result.name)
            else:
                scan_end = len(body)

            if len(chain) < 2:
                new_body.append(op)
                i += 1
                continue

            # Backward prepend: pull in fusable ops before chain[0] that
            # are separated from it only by Barrier/Load. Lets a pre-load
            # elementwise (e.g. SDPA bwd `s *= scale` between dot79 and the
            # Mask Load) absorb into the chain — its writeback to shmem is
            # elided, and the chain reads its source operand directly,
            # killing one shmem round-trip + two barriers per inner iter.
            prepend_ops: list[TileOp] = []
            scan_back = i - 1
            while scan_back >= 0:
                cand = body[scan_back]
                # Hard stops: anything that breaks the data-flow continuity
                # to the chain.
                if isinstance(cand, (Dot, Store, ForLoop, WhileLoop, IfElse,
                                     Reduce, FusedElementwise, RowPass)):
                    break
                # Skippable: Barrier, Load (cooperative, stays out of chain),
                # and any op whose result isn't consumed by the chain head
                # (e.g. index math `t127 = add(...)`, bound-check bools
                # `t132 = bitand(...)` — these emit per-thread or as
                # constants, don't touch the chain's shmem buffers, and
                # are independent of the prepend candidate's data flow).
                if isinstance(cand, (Barrier, Load)):
                    scan_back -= 1
                    continue
                head_ops = prepend_ops + [chain[0]]
                head_operands = {v.name for hop in head_ops for v in hop.operand_values()}
                if cand.result is None or cand.result.name not in head_operands:
                    scan_back -= 1
                    continue
                # Consumed by the chain head: must be a fusable 2D op with
                # matching row count, and its operands must all be
                # shmem-resident, broadcast, or scalar.
                if (not isinstance(cand, _FUSABLE)
                        or len(cand.result.shape) != 2
                        or cand.result.shape[0] != rows):
                    break
                # All cand operands valid (shmem / chain-internal / broadcast).
                ok = True
                for v in cand.operand_values():
                    if not v.shape:
                        continue
                    if v.name in in_shmem:
                        continue
                    if len(v.shape) == 1:
                        continue
                    if len(v.shape) == 2 and (v.shape[0] == 1 or v.shape[1] == 1):
                        continue
                    ok = False
                    break
                if not ok:
                    break
                # cand.result must have NO consumer outside head_ops — its
                # shmem writeback goes away, so any external reader (e.g.
                # later kernel ops, downstream chains) would see unscaled
                # values from the source buffer.
                external = 0
                for op_other in walk_ops(body):
                    if op_other is cand or any(op_other is h for h in head_ops):
                        continue
                    for v in op_other.operand_values():
                        if v.name == cand.result.name:
                            external += 1
                if external > 0:
                    break
                prepend_ops.insert(0, cand)
                chain_names.add(cand.result.name)
                scan_back -= 1

            if prepend_ops:
                chain = prepend_ops + chain
                prepend_ids = set(id(o) for o in prepend_ops)
                new_body[:] = [o for o in new_body if id(o) not in prepend_ids]

            # Determine writeback set: which chain results must go to shmem
            wb: set[str] = set()
            # (a) consumed by ops after the chain
            for aop in walk_ops(body[scan_end:]):
                for v in aop.operand_values():
                    if v.name in chain_names:
                        wb.add(v.name)
            # (b) consumed by a Dot (must be in shmem for MMA)
            wb |= chain_names & dot_inputs
            # (c) always write back the last result (safety)
            wb.add(chain[-1].result.name)
            # Propagate: written-back results are now "in shmem" for
            # subsequent chains in the same body
            in_shmem |= wb

            # Emit non-chain ops in [i, scan_end) before the fused node
            chain_ids = set(id(c) for c in chain)
            for k in range(i, scan_end):
                if id(body[k]) not in chain_ids:
                    new_body.append(body[k])

            flat_safe = _chain_is_flat_safe(chain, in_shmem)
            fused = FusedElementwise(
                result=chain[-1].result,
                ops=list(chain),
                writeback=wb,
                flat_threaded=flat_safe,
            )
            new_body.append(fused)
            i = scan_end

        return new_body

    # Apply to all ForLoop bodies (recursively)
    for op in walk_ops(func.ops):
        if isinstance(op, ForLoop):
            op.body = _find_chains(op.body)


def _chain_is_flat_safe(chain: list[TileOp], in_shmem: set[str]) -> bool:
    """Decide whether a fused elementwise chain can use flat threading
    (one thread per element, all 256 threads of the threadgroup) instead
    of the row-iter `if (_row < rows) for (_c)` form (16 threads × 16
    serial iters at BM=BN=16).

    Criteria:
      1. Tile produces a regular 2D result (M*N ≥ 16).
      2. Every operand of every chain op is one of:
           - in `in_shmem` (a 2D shmem tile read at [_row*S+_c]), OR
           - a chain-internal value, OR
           - 1D / row-or-col-broadcast / scalar operand. The fused emit's
             `_resolve` substitutes `tid → _row` for (M, 1) operands and
             `tid → _c` for (1, N) operands under flat threading.
    """
    if not chain or chain[-1].result is None:
        return False
    res_shape = chain[-1].result.shape
    if len(res_shape) != 2:
        return False
    M, N = res_shape
    if M * N < 16:
        return False

    chain_names = {op.result.name for op in chain if op.result}
    for op in chain:
        for v in op.operand_values():
            if v.name in chain_names:
                continue
            if v.name in in_shmem:
                continue
            if not v.shape:
                continue
            if len(v.shape) == 1:
                continue
            if len(v.shape) == 2 and (v.shape[0] == 1 or v.shape[1] == 1):
                continue
            return False
    return True


_ROW_PASS_OP_TYPES = (BinOp, UnaryOp, Select, Compare, TernaryOp, Reduce, Constant)


def _opt_fuse_row_pass(func: TileFunction):
    """Wrap `elem → reduce → elem` chains inside a kernel body into a single
    RowPass node, so the emitter can lower them as one multi-phase pass
    (straight-line per-thread code + cross-lane butterfly between phases).

    Triggered when the kernel's top-level ops contain at least one
    `Reduce(axis=0)` on a 1D tile plus elementwise ops both before (producing
    the reduce input) and after (consuming the reduce output). Handles
    chains with multiple sequential reductions — the boundary discovery
    inside `_emit_row_pass` partitions them into phases at emit time.

    Conservative scope: only rewrites when the kernel body is flat (no
    ForLoop / IfElse / WhileLoop between the participating ops). Ops
    inside those control-flow constructs are left untouched.
    """
    if any(isinstance(op, (ForLoop, IfElse, WhileLoop, RowPass, FusedElementwise))
           for op in func.ops):
        return

    # Locate every Reduce(axis=0) on a 1D tile in the body.
    reduce_idxs = [
        i for i, op in enumerate(func.ops)
        if isinstance(op, Reduce)
        and op.axis == 0
        and op.input is not None
        and len(op.input.shape) == 1
        and op.input.shape[0] > 1
    ]
    if not reduce_idxs:
        return

    # Find the Store and trace back from the value being stored through
    # the elementwise/reduce DAG. Only those ops we reach from the store's
    # value AND whose output is 1D (or a scalar consumed by a 1D op) are
    # candidates for the RowPass.
    store_op = next((o for o in func.ops if isinstance(o, Store)), None)
    if store_op is None or store_op.value is None:
        return

    name_to_op = {op.result.name: op for op in func.ops if op.result is not None}

    picked: set[str] = set()
    queue = [store_op.value.name]
    while queue:
        nm = queue.pop()
        if nm in picked or nm not in name_to_op:
            continue
        op = name_to_op[nm]
        if not isinstance(op, _ROW_PASS_OP_TYPES):
            continue
        picked.add(nm)
        for v in op.operand_values():
            queue.append(v.name)

    # Require at least one reduction in the picked set — otherwise there's
    # nothing for RowPass to gain over FusedElementwise.
    if not any(isinstance(name_to_op[n], Reduce) for n in picked):
        return

    compute = [op for op in func.ops if op.result is not None and op.result.name in picked]
    if len(compute) < 3:
        # Nothing to gain below a 3-op chain.
        return

    new_ops: list[TileOp] = []
    placed = False
    for op in func.ops:
        if op.result is not None and op.result.name in picked:
            if not placed:
                new_ops.append(RowPass(ops=compute, writeback=set()))
                placed = True
            continue
        new_ops.append(op)
    func.ops = new_ops


def _opt_absorb_load_scale(func: TileFunction):
    """Fold `Load → BinOp(mul, scalar) → consumer` into `Load.transform`.

    Detects the pattern (in any scope, including ForLoop bodies):
        ld = Load(...)               # 2D cooperative load → shmem
        scaled = ld * scalar         # elementwise multiply by scalar
        ... uses scaled ...          # ld is dead after the multiply

    Rewrites to:
        ld' = Load(... transform=[Mul]) → consumers see ld' instead of scaled

    The cooperative-load emitter already evaluates `Load.transform` per
    element during the load loop, so the multiply happens in the same
    register pass with no extra shmem buffer. Saves one BLOCK_M×D shmem
    slot per absorbed pattern (the scaled tile no longer needs its own
    buffer), which lets bigger BM configs fit in the 32 KB threadgroup
    budget — concretely, the SDPA bwd dq kernel goes from needing two
    K-sized slots (K, K_scaled) to one.

    Critical contract: this pass must NOT mutate operand fields on
    consumer ops. `shallow_clone_for_fusion` shares BinOp/Dot/Reduce
    objects across cloned funcs, so in-place operand reassignment would
    silently corrupt every other clone. Instead, when a
    consumer needs its operand redirected from the BinOp's result back to
    the Load's result, REPLACE the consumer in `scope_ops` with a fresh
    `dataclasses.replace` clone — leaving the original (shared) op
    untouched. Loads are cloned by `shallow_clone_for_fusion` so mutating
    `ld.transform` is safe.
    """
    # Build per-scope op maps and a global use-count for tile values
    use_count: dict[str, int] = {}
    for op in walk_ops(func.ops):
        for v in op.operand_values():
            use_count[v.name] = use_count.get(v.name, 0) + 1
        if isinstance(op, Load) and op.transform:
            for xf in op.transform:
                for v in xf.operand_values():
                    use_count[v.name] = use_count.get(v.name, 0) + 1

    def _process(scope_ops: list[TileOp]) -> None:
        # Recurse into nested loop bodies first
        for op in scope_ops:
            if isinstance(op, ForLoop):
                _process(op.body)
            elif isinstance(op, WhileLoop):
                _process(op.body)
            elif isinstance(op, IfElse):
                _process(op.body)
                if op.orelse:
                    _process(op.orelse)

        # Build a name → producing-op map for this scope
        prod: dict[str, TileOp] = {}
        for op in scope_ops:
            if op.result is not None:
                prod[op.result.name] = op

        to_remove: list[TileOp] = []
        replacements: dict[str, TileValue] = {}
        for op in scope_ops:
            if not isinstance(op, BinOp) or op.op != "mul":
                continue
            if op.lhs is None or op.rhs is None or op.result is None:
                continue
            # Identify which side is the load tile and which is the scalar.
            # Scalar = 0-D tile; tile = 2-D matching load output.
            if len(op.lhs.shape) == 2 and len(op.rhs.shape) == 0:
                tile_val = op.lhs
            elif len(op.rhs.shape) == 2 and len(op.lhs.shape) == 0:
                tile_val = op.rhs
            else:
                continue
            # Tile must come from a Load in this scope
            ld = prod.get(tile_val.name)
            if not isinstance(ld, Load):
                continue
            # Load result must be consumed ONLY by this Mul (so the un-scaled
            # tile isn't read elsewhere). Count global uses; the Load's own
            # transform list is excluded above.
            if use_count.get(tile_val.name, 0) != 1:
                continue
            # Absorb: append the Mul to Load.transform, redirect downstream
            # references from the Mul's result to the Load's result. Loads are
            # already cloned per-func by `shallow_clone_for_fusion`, so
            # mutating `ld.transform` is safe.
            ld.transform = list(ld.transform) + [op]
            replacements[op.result.name] = ld.result
            to_remove.append(op)

        if not to_remove:
            return
        scope_ops[:] = [s for s in scope_ops if s not in to_remove]
        if replacements:
            _replace_value_refs_clone(scope_ops, replacements)

    _process(func.ops)


def _replace_value_refs_clone(
    scope_ops: list[TileOp],
    mapping: "dict[str, TileValue]",
) -> None:
    """Replace tile-value references by cloning ops, not mutating shared ones.

    `shallow_clone_for_fusion` shares non-Load/Store TileOps (BinOp, Dot,
    Reduce, ...) across cloned `TileFunction`s. Mutating operand fields on
    those shared objects would silently corrupt every other clone that holds
    the same `op`. So instead of mutating, this walker REPLACES each op in
    `scope_ops` with a `dataclasses.replace` clone whose operand fields are
    rewritten to the mapped TileValues. Original op objects are left untouched.

    Recurses into ForLoop/WhileLoop/IfElse bodies (also cloned). Leaves
    Load/Store `transform` chains alone — the absorb pass owns those.
    """

    def _maybe_clone(op: TileOp) -> TileOp:
        # Recurse into nested bodies — clone them so the nested mutations
        # don't leak to other shared parents.
        if isinstance(op, ForLoop):
            new_body = list(op.body)
            _walk(new_body)
            new_carried = [
                (mapping.get(i.name, i), mapping.get(f.name, f))
                for i, f in op.carried
            ]
            if new_body != op.body or new_carried != op.carried:
                return dataclasses.replace(op, body=new_body, carried=new_carried)
            return op
        if isinstance(op, WhileLoop):
            new_body = list(op.body)
            _walk(new_body)
            new_cond = list(op.cond_body)
            if new_cond:
                _walk(new_cond)
            new_carried = [
                (mapping.get(i.name, i), mapping.get(f.name, f))
                for i, f in op.carried
            ]
            kw = {"body": new_body, "carried": new_carried}
            if new_cond and hasattr(op, "cond_body"):
                kw["cond_body"] = new_cond
            return dataclasses.replace(op, **kw)
        if isinstance(op, IfElse):
            new_body = list(op.body)
            _walk(new_body)
            new_orelse = list(op.orelse)
            if new_orelse:
                _walk(new_orelse)
            return dataclasses.replace(op, body=new_body, orelse=new_orelse)

        # Plain op: collect any operand-field rewrites needed
        rewrites: dict[str, TileValue] = {}
        for fld in fields(op):
            if fld.name == "transform":
                continue
            val = op.__dict__.get(fld.name)
            if isinstance(val, TileValue) and val.name in mapping:
                rewrites[fld.name] = mapping[val.name]
        if not rewrites:
            return op
        return dataclasses.replace(op, **rewrites)

    def _walk(ops: list[TileOp]) -> None:
        for i, op in enumerate(ops):
            new_op = _maybe_clone(op)
            if new_op is not op:
                ops[i] = new_op

    _walk(scope_ops)


def _opt_fuse_acc_epilogue(func: TileFunction):
    """Fold a multi-accumulator elementwise epilogue into the Store transform.

    dot_q4_k_silu computes two persistent MMA accumulators (gate, up), then a
    SiLU chain `silu(gate) * up`, then stores. Without this pass the chain reads
    the accumulators as ordinary 2D tiles, which forces each to spill to shared
    memory (the auto-spill in `compiler._emit_op`) — ~8KB of float staging that
    roughly halves threadgroup occupancy. This pass runs after
    `_opt_persistent_mma` (so the accumulators are register-resident), detects
    `≥2 persistent accs → pure elementwise chain → one Store`, and folds the
    chain into `Store.transform`. The non-source accumulators land in
    `Store.acc_extras`; `_emit_persistent_mma_store` then evaluates the chain
    per-lane on the simdgroup `thread_elements()` and writes straight to device,
    with no shmem spill.
    """
    acc_dot: dict[str, Dot] = {}
    for op in walk_ops(func.ops):
        if isinstance(op, Dot) and op.acc is not None and op.result is not None:
            acc_dot[op.result.name] = op
    if len(acc_dot) < 2:
        return

    top_ops = func.ops
    top_by_result = {op.result.name: op for op in top_ops if op.result is not None}
    rewrites = []  # (store, primary_dot, transform, source_name, acc_extras, chain_set)

    for store in top_ops:
        if not isinstance(store, Store) or store.value is None:
            continue
        if store.transform or store.acc_extras:
            continue

        chain_set: set[str] = set()
        leaf_set: set[str] = set()
        leaves: list[str] = []
        ok = True
        stack = [store.value.name]
        seen: set[str] = set()
        while stack:
            nm = stack.pop()
            if nm in seen:
                continue
            seen.add(nm)
            if nm in acc_dot:
                if nm not in leaf_set:
                    leaf_set.add(nm)
                    leaves.append(nm)
                continue
            producer = top_by_result.get(nm)
            if producer is None:
                ok = False  # leaf isn't a persistent acc (e.g. loop-internal) — unsafe
                break
            if isinstance(producer, Constant):
                chain_set.add(nm)
                continue
            if (
                isinstance(producer, (BinOp, UnaryOp, Select))
                and producer.result is not None
                and len(producer.result.shape) == 2
            ):
                chain_set.add(nm)
                for v in producer.operand_values():
                    stack.append(v.name)
            else:
                ok = False
                break
        if not ok or len(leaf_set) < 2 or not chain_set:
            continue

        # Every chain intermediate must be consumed ONLY within the chain (or by
        # this store) — otherwise dropping the chain ops would break a consumer.
        bail = False
        for other in walk_ops(top_ops):
            if other is store:
                continue
            if other.result is not None and other.result.name in chain_set:
                continue  # a chain op; its operand refs are internal
            for v in other.operand_values():
                if v.name in chain_set:
                    bail = True
                    break
            if bail:
                break
        if bail:
            continue

        transform = [
            op for op in top_ops if op.result is not None and op.result.name in chain_set
        ]
        primary = leaves[0]
        extras = leaves[1:]
        rewrites.append(
            (
                store,
                acc_dot[primary],
                transform,
                primary,
                {ex: acc_dot[ex].acc.name for ex in extras},
                chain_set,
            )
        )

    if not rewrites:
        return

    drop: set[str] = set()
    for store, primary_dot, transform, source, acc_extras, chain_set in rewrites:
        store.value = primary_dot.result
        store.transform = transform
        store.transform_source_name = source
        store.acc_extras = acc_extras
        drop |= chain_set

    func.ops = [
        op for op in func.ops if not (op.result is not None and op.result.name in drop)
    ]


_PASSES = [
    _opt_absorb_load_scale,
    _opt_persistent_mma,
    _opt_fuse_acc_epilogue,
    _opt_fuse_row_loops,
    _opt_fuse_row_pass,
]
