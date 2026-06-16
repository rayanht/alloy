"""Snapshot tests for trace-time AST control-flow rewriting."""

from __future__ import annotations

import ast
import textwrap

import alloy._compiler.trace.ast_rewrite as ast_rewrite


def _rewrite_source(source: str) -> str:
    ast_rewrite._loop_counter = 0
    tree = ast_rewrite._rewrite_kernel_source(ast.parse(textwrap.dedent(source)))
    assert tree is not None
    return ast.unparse(tree)


def test_for_loop_rewrite_snapshot() -> None:
    rewritten = _rewrite_source(
        """
        def k(N: al.constexpr):
            acc = 0
            for i in range(0, N, 4):
                acc = acc + i
            return acc
        """
    )

    assert (
        rewritten
        == """def k(N: al.constexpr):
    acc = 0
    _lctx_1, acc = _trace_loop_enter(['acc'], acc)
    i = _trace_for_var(_lctx_1, 'i')
    acc = acc + i
    _trace_loop_exit(_lctx_1, 'i', 0, N, 4, acc)
    _trace_flow('return')"""
    )


def test_while_loop_rewrite_snapshot() -> None:
    rewritten = _rewrite_source(
        """
        def k(N: al.constexpr):
            i = 0
            while i < N:
                i = i + 1
            return i
        """
    )

    assert (
        rewritten
        == """def k(N: al.constexpr):
    i = 0
    _lctx_1, i = _trace_loop_enter(['i'], i)
    _trace_loop_cond(_lctx_1, i < N)
    i = i + 1
    _trace_loop_exit(_lctx_1, None, None, None, None, i)
    _trace_flow('return')"""
    )


def test_if_else_rewrite_snapshot() -> None:
    rewritten = _rewrite_source(
        """
        def k(x):
            y = 0
            if x > 0:
                y = x
            else:
                y = -x
            return y
        """
    )

    assert (
        rewritten
        == """def k(x):
    y = 0
    _ifctx_1 = _trace_if_enter(x > 0, ['y'], y)
    if _ifctx_1[0] != 'const_false':
        y = x
    _ifctx_1 = _trace_if_else(_ifctx_1, y)
    if _ifctx_1[0] not in ('const_true', 'const_true_else'):
        y = -x
    y = _trace_if_exit(_ifctx_1, y)
    _trace_flow('return')"""
    )
