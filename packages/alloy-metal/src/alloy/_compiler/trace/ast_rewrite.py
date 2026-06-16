"""AST rewriting for trace-time control-flow capture."""

from __future__ import annotations

import ast

# AST rewrite — convert for/while loops into traced loop hooks
# ---------------------------------------------------------------------------


def _collect_assigned(stmts: list[ast.stmt]) -> set[str]:
    """Collect all variable names assigned in a list of statements."""
    names: set[str] = set()
    for s in stmts:
        for node in ast.walk(s):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
    return names


def _subscript_base(node: ast.expr) -> str | None:
    """Base name of a (possibly nested) subscript target: `o[g][d]` -> 'o'."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _scan_carried(
    stmts: list[ast.stmt], carried: set[str], written: set[str], rbw: set[str]
) -> None:
    """Scan a statement block IN ORDER, threading `written`/`read_before_write`
    so cross-statement read-before-write is detected across nested blocks too.

    Recurses into for/if/while bodies in statement order — `ast.walk` (BFS) would
    visit a later write before an earlier read (e.g. `m = mn` before the `m` in
    `mn = max(m, sc)` nested in an unrolled loop), missing the carry.
    """

    def _reads(value: ast.expr) -> set[str]:
        return {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}

    def _mark(target: ast.expr, rhs_names: set[str]) -> None:
        base = target.id if isinstance(target, ast.Name) else _subscript_base(target)
        if base is None:
            return
        if base in rhs_names and base not in written:
            carried.add(base)  # self-referential (x = f(x) / x[i] = f(x[i]))
        written.add(base)

    for s in stmts:
        if isinstance(s, ast.AugAssign):
            rhs = _reads(s.value)
            base = (
                s.target.id if isinstance(s.target, ast.Name) else _subscript_base(s.target)
            )
            for nm in rhs:
                if nm not in written:
                    rbw.add(nm)
            if base is not None:
                carried.add(base)  # += always reads the target
                written.add(base)
        elif isinstance(s, ast.Assign):
            rhs = _reads(s.value)
            for nm in rhs:
                if nm not in written:
                    rbw.add(nm)
            for t in s.targets:
                _mark(t, rhs)
        elif isinstance(s, (ast.For, ast.While)):
            if isinstance(s, ast.For) and isinstance(s.target, ast.Name):
                written.add(s.target.id)  # loop var isn't a carry
            _scan_carried(s.body, carried, written, rbw)
            _scan_carried(s.orelse, carried, written, rbw)
        elif isinstance(s, ast.If):
            _scan_carried(s.body, carried, written, rbw)
            _scan_carried(s.orelse, carried, written, rbw)
        else:
            for node in ast.walk(s):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in written:
                        rbw.add(node.id)


def _collect_carried(stmts: list[ast.stmt]) -> set[str]:
    """Collect variable names that are loop-carried (read before written).

    A variable is carried if it's WRITTEN in the body AND either:
    - Used via augmented assignment (+=, etc.)
    - Self-referential (x = x + ...), incl. accumulator arrays (x[i] = f(x[i]))
    - Read before its first write in the body (m = mn, where m was read earlier),
      across nested for/if blocks (e.g. an al.unroll loop body)

    Variables only READ (never written) in the body are NOT carried — they're
    references to outer-scope values.
    """
    carried: set[str] = set()
    written: set[str] = set()
    rbw: set[str] = set()
    _scan_carried(stmts, carried, written, rbw)
    carried |= rbw & written
    return carried


_loop_counter: int = 0


def _rewrite_loops(func_def: ast.FunctionDef) -> bool:
    """Rewrite for/while/if into traced hooks.

    Returns True if any rewrites were performed.
    """
    defined = {arg.arg for arg in func_def.args.args}
    # Collect constexpr param names for if-rewrite decisions
    constexpr_names: set[str] = set()
    for arg in func_def.args.args:
        if arg.annotation is not None:
            ann = arg.annotation
            if (isinstance(ann, ast.Attribute) and ann.attr == "constexpr") or (
                isinstance(ann, ast.Name) and ann.id == "constexpr"
            ):
                constexpr_names.add(arg.arg)
    return _rewrite_stmts(func_def, "body", defined, constexpr_names)


def _parse_range_args(call: ast.Call) -> tuple[ast.expr, ast.expr, ast.expr]:
    """Parse range() call args into (start_node, end_node, step_node)."""
    args = call.args
    if len(args) == 1:
        return ast.Constant(value=0), args[0], ast.Constant(value=1)
    elif len(args) == 2:
        return args[0], args[1], ast.Constant(value=1)
    else:
        return args[0], args[1], args[2]


_BODY_PARENTS = (ast.FunctionDef, ast.For, ast.While, ast.If)
_ORELSE_PARENTS = (ast.For, ast.While, ast.If)


def _read_stmt_list(parent: ast.AST, attr: str) -> list[ast.stmt]:
    if attr == "body" and isinstance(parent, _BODY_PARENTS):
        return parent.body
    if attr == "orelse" and isinstance(parent, _ORELSE_PARENTS):
        return parent.orelse
    raise TypeError(f"Unsupported statement list {type(parent).__name__}.{attr}")


def _write_stmt_list(parent: ast.AST, attr: str, stmts: list[ast.stmt]) -> None:
    if attr == "body" and isinstance(parent, _BODY_PARENTS):
        parent.body = stmts
        return
    if attr == "orelse" and isinstance(parent, _ORELSE_PARENTS):
        parent.orelse = stmts
        return
    raise TypeError(f"Unsupported statement list {type(parent).__name__}.{attr}")


def _rewrite_stmts(
    parent: ast.AST,
    attr: str,
    defined: set[str],
    constexpr_names: set[str] | None = None,
) -> bool:
    """Rewrite loops in a statement list. Modifies parent.attr in place.

    For-loops get direct ForLoop handling (symbolic loop var, not carried).
    While-loops get WhileLoop handling (condition traced, vars carried).

    Recurses into nested loops and if/else bodies.
    """
    global _loop_counter
    stmts = _read_stmt_list(parent, attr)
    new_stmts: list[ast.stmt] = []
    changed = False

    for stmt in stmts:
        # --- For loops: direct ForLoop IR (loop var is symbolic, not carried) ---
        if isinstance(stmt, ast.For) and _is_range_call(stmt.iter):
            var_name = stmt.target.id if isinstance(stmt.target, ast.Name) else "_k"
            start_node, end_node, step_node = _parse_range_args(stmt.iter)

            # Recursively rewrite nested constructs in body
            body_defined = set(defined)
            body_defined.add(var_name)
            _rewrite_stmts(stmt, "body", body_defined, constexpr_names)

            # Carried = defined before loop AND modified in body (aug-assign/self-ref), excluding loop var
            modified_in_body = _collect_carried(stmt.body)
            modified_in_body.discard(var_name)
            carry_names = sorted(defined & modified_in_body)

            _loop_counter += 1
            lctx_name = f"_lctx_{_loop_counter}"

            # _lctx, carry0, ... = _trace_loop_enter([names], carry0, ...)
            enter_targets = [ast.Name(id=lctx_name, ctx=ast.Store())]
            for cn in carry_names:
                enter_targets.append(ast.Name(id=cn, ctx=ast.Store()))

            enter_call = ast.Call(
                func=ast.Name(id="_trace_loop_enter", ctx=ast.Load()),
                args=[
                    ast.List(
                        elts=[ast.Constant(value=n) for n in carry_names],
                        ctx=ast.Load(),
                    ),
                ]
                + [ast.Name(id=n, ctx=ast.Load()) for n in carry_names],
                keywords=[],
            )
            enter_assign = ast.Assign(
                targets=[ast.Tuple(elts=enter_targets, ctx=ast.Store())],
                value=enter_call,
            )

            # VAR = _trace_for_var(_lctx, 'var_name') — creates symbolic i32 loop var
            var_assign = ast.Assign(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="_trace_for_var", ctx=ast.Load()),
                    args=[
                        ast.Name(id=lctx_name, ctx=ast.Load()),
                        ast.Constant(value=var_name),
                    ],
                    keywords=[],
                ),
            )

            # _trace_loop_exit(_lctx, 'var', start, end, step, carry0, ...)
            exit_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_trace_loop_exit", ctx=ast.Load()),
                    args=[
                        ast.Name(id=lctx_name, ctx=ast.Load()),
                        ast.Constant(value=var_name),
                        start_node,
                        end_node,
                        step_node,
                    ]
                    + [ast.Name(id=n, ctx=ast.Load()) for n in carry_names],
                    keywords=[],
                )
            )

            new_stmts.append(enter_assign)
            new_stmts.append(var_assign)
            new_stmts.extend(stmt.body)
            new_stmts.append(exit_call)
            changed = True
            defined.update(modified_in_body)
            continue

        # --- While loops: WhileLoop IR (condition traced, vars carried) ---
        if isinstance(stmt, ast.While):
            body_defined = set(defined)
            _rewrite_stmts(stmt, "body", body_defined, constexpr_names)

            modified_in_body = _collect_carried(stmt.body)
            carry_names = sorted(defined & modified_in_body)

            _loop_counter += 1
            lctx_name = f"_lctx_{_loop_counter}"

            enter_targets = [ast.Name(id=lctx_name, ctx=ast.Store())]
            for cn in carry_names:
                enter_targets.append(ast.Name(id=cn, ctx=ast.Store()))

            enter_call = ast.Call(
                func=ast.Name(id="_trace_loop_enter", ctx=ast.Load()),
                args=[
                    ast.List(
                        elts=[ast.Constant(value=n) for n in carry_names],
                        ctx=ast.Load(),
                    ),
                ]
                + [ast.Name(id=n, ctx=ast.Load()) for n in carry_names],
                keywords=[],
            )
            enter_assign = ast.Assign(
                targets=[ast.Tuple(elts=enter_targets, ctx=ast.Store())],
                value=enter_call,
            )

            cond_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_trace_loop_cond", ctx=ast.Load()),
                    args=[ast.Name(id=lctx_name, ctx=ast.Load()), stmt.test],
                    keywords=[],
                )
            )

            exit_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_trace_loop_exit", ctx=ast.Load()),
                    args=[
                        ast.Name(id=lctx_name, ctx=ast.Load()),
                        ast.Constant(value=None),
                        ast.Constant(value=None),
                        ast.Constant(value=None),
                        ast.Constant(value=None),
                    ]
                    + [ast.Name(id=n, ctx=ast.Load()) for n in carry_names],
                    keywords=[],
                )
            )

            new_stmts.append(enter_assign)
            new_stmts.append(cond_call)
            new_stmts.extend(stmt.body)
            new_stmts.append(exit_call)
            changed = True
            defined.update(modified_in_body)

        # Step 3: Rewrite bare break/continue/return
        elif isinstance(stmt, (ast.Break, ast.Continue, ast.Return)):
            kind = {ast.Break: "break", ast.Continue: "continue", ast.Return: "return"}[type(stmt)]
            new_stmts.append(_make_flow_call(kind))
            changed = True

        # Step 4: Rewrite if statements that contain flow control or
        # are inside a rewritten loop (where conditions may be symbolic)
        elif isinstance(stmt, ast.If) and _needs_if_rewrite(stmt, constexpr_names):
            _loop_counter += 1
            if_ctx_name = f"_ifctx_{_loop_counter}"

            # Find variables modified in if body that existed before
            assigned_in_if = _collect_assigned(stmt.body)
            if stmt.orelse:
                assigned_in_if |= _collect_assigned(stmt.orelse)
            merge_names = sorted(defined & assigned_in_if)

            # _trace_if_enter(cond, [merge_names], pre_val0, pre_val1, ...)
            enter = ast.Assign(
                targets=[ast.Name(id=if_ctx_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="_trace_if_enter", ctx=ast.Load()),
                    args=[
                        stmt.test,
                        ast.List(
                            elts=[ast.Constant(value=n) for n in merge_names],
                            ctx=ast.Load(),
                        ),
                    ]
                    + [ast.Name(id=n, ctx=ast.Load()) for n in merge_names],
                    keywords=[],
                ),
            )
            new_stmts.append(enter)

            _rewrite_stmts(stmt, "body", set(defined), constexpr_names)
            # Guard body execution: skip for const_false (concrete False condition)
            body_guard = ast.If(
                test=ast.Compare(
                    left=ast.Subscript(
                        value=ast.Name(id=if_ctx_name, ctx=ast.Load()),
                        slice=ast.Constant(value=0),
                        ctx=ast.Load(),
                    ),
                    ops=[ast.NotEq()],
                    comparators=[ast.Constant(value="const_false")],
                ),
                body=stmt.body,
                orelse=[],
            )
            new_stmts.append(body_guard)

            if stmt.orelse:
                # _trace_if_else(_ifctx, body_val0, body_val1, ...)
                else_call = ast.Assign(
                    targets=[ast.Name(id=if_ctx_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="_trace_if_else", ctx=ast.Load()),
                        args=[ast.Name(id=if_ctx_name, ctx=ast.Load())]
                        + [ast.Name(id=n, ctx=ast.Load()) for n in merge_names],
                        keywords=[],
                    ),
                )
                new_stmts.append(else_call)
                _rewrite_stmts(stmt, "orelse", set(defined), constexpr_names)
                # Guard else execution: skip for const_true (concrete True condition)
                else_guard = ast.If(
                    test=ast.Compare(
                        left=ast.Subscript(
                            value=ast.Name(id=if_ctx_name, ctx=ast.Load()),
                            slice=ast.Constant(value=0),
                            ctx=ast.Load(),
                        ),
                        ops=[ast.NotIn()],
                        comparators=[
                            ast.Tuple(
                                elts=[
                                    ast.Constant(value="const_true"),
                                    ast.Constant(value="const_true_else"),
                                ],
                                ctx=ast.Load(),
                            )
                        ],
                    ),
                    body=stmt.orelse,
                    orelse=[],
                )
                new_stmts.append(else_guard)

            # v0, v1, ... = _trace_if_exit(_ifctx, post_val0, post_val1, ...)
            exit_args = [ast.Name(id=if_ctx_name, ctx=ast.Load())] + [
                ast.Name(id=n, ctx=ast.Load()) for n in merge_names
            ]
            if merge_names:
                exit_targets = [ast.Name(id=n, ctx=ast.Store()) for n in merge_names]
                exit_stmt = ast.Assign(
                    targets=[
                        (
                            ast.Tuple(elts=exit_targets, ctx=ast.Store())
                            if len(exit_targets) > 1
                            else exit_targets[0]
                        )
                    ],
                    value=ast.Call(
                        func=ast.Name(id="_trace_if_exit", ctx=ast.Load()),
                        args=exit_args,
                        keywords=[],
                    ),
                )
            else:
                exit_stmt = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="_trace_if_exit", ctx=ast.Load()),
                        args=exit_args,
                        keywords=[],
                    )
                )
            new_stmts.append(exit_stmt)
            changed = True

        elif isinstance(stmt, ast.If):
            # Regular if — recurse but keep as Python if
            c1 = _rewrite_stmts(stmt, "body", set(defined), constexpr_names)
            c2 = _rewrite_stmts(stmt, "orelse", set(defined), constexpr_names)
            changed = changed or c1 or c2
            new_stmts.append(stmt)

        elif isinstance(stmt, ast.For):
            # Non-range for-loop (e.g. `for d in al.unroll(range(...))`): left as
            # native Python and unrolled at trace, but still recurse so any traced
            # constructs in its body get rewritten.
            body_defined = set(defined)
            if isinstance(stmt.target, ast.Name):
                body_defined.add(stmt.target.id)
            c = _rewrite_stmts(stmt, "body", body_defined, constexpr_names)
            changed = changed or c
            new_stmts.append(stmt)

        else:
            new_stmts.append(stmt)
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    defined.add(stmt.target.id)

    if changed:
        _write_stmt_list(parent, attr, new_stmts)

    return changed


def _is_range_call(node: ast.AST) -> bool:
    """Check if an AST node is a call to range()."""
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
    )


def _needs_if_rewrite(stmt: ast.If, constexpr_names: set[str] | None = None) -> bool:
    """Check if an if statement needs tracing rewrite.

    Rewrite if the condition could be symbolic (involves non-constexpr
    variables). Skip if the condition only involves constexpr params
    and literals — Python handles those correctly.
    """
    if constexpr_names is None:
        return True
    # Check if condition only references constexpr variables and literals
    for node in ast.walk(stmt.test):
        if isinstance(node, ast.Name) and node.id not in constexpr_names:
            return True  # references a non-constexpr variable → might be symbolic
    return False  # all references are constexpr or literals → Python handles it


def _make_flow_call(kind: str) -> ast.Expr:
    """Create AST for _trace_flow("kind") call."""
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id="_trace_flow", ctx=ast.Load()),
            args=[ast.Constant(value=kind)],
            keywords=[],
        )
    )


def _rewrite_kernel_source(func_ast: ast.Module) -> ast.Module | None:
    """Apply the loop/flow-control rewrite to a kernel AST in place; returns
    the rewritten AST, or None if no rewrite was needed.

    `func_ast` is mutated directly. Both callers hand us a throwaway
    `ast.parse(source)` they don't retain, so deep-copying it first (as this
    did) was pure overhead — and a costly one: the copy ran on every kernel
    trace and dominated eager-compile time (~7s of `copy.deepcopy` across a
    full model load, ~8M AST-node copies). `ast.parse` already returns a fresh
    tree, so re-parsing — not copying — is the way to get an isolated AST.
    """
    tree = func_ast
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    if func_def is None:
        return None

    # Strip decorators — the rewritten function is executed raw, not re-decorated
    func_def.decorator_list = []

    changed = _rewrite_loops(func_def)
    if not changed:
        return None

    ast.fix_missing_locations(tree)
    return tree


# ---------------------------------------------------------------------------
