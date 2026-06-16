"""Snapshot tests for trace-time address decomposition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import alloy as al
from alloy._compiler.tile_ir import (
    BinOp,
    Constant,
    ExpandDims,
    Load,
    Splat,
    TileFunction,
    TileOp,
    TileValue,
    walk_ops,
)
from alloy._compiler.trace import trace_kernel

ConstValue: TypeAlias = bool | int | float | str | tuple[int, ...]
Shape: TypeAlias = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AddressSnapshot:
    offset_shape: Shape | None
    row_shape: Shape | None
    col_shape: Shape | None
    row_stride: int | None
    base_offset_shape: Shape | None
    addr_transposed: bool
    offset_signature: tuple[str, ...]


def _trace(
    fn: Callable[..., None],
    constexprs: Mapping[str, ConstValue],
    param_names: tuple[str, ...],
    constexpr_params: set[str],
    buffer_shapes: Mapping[str, Shape] | None = None,
) -> TileFunction:
    shape_dict: dict[str, Shape] = {}
    if buffer_shapes is not None:
        shape_dict = dict(buffer_shapes)
    return trace_kernel(
        fn,
        fn.__name__,
        dict(constexprs),
        param_names=list(param_names),
        constexpr_params=constexpr_params,
        buffer_shapes=shape_dict,
        output_params={"out"},
    )


def _single_load(func: TileFunction) -> Load:
    loads = [op for op in walk_ops(func.ops) if isinstance(op, Load)]
    assert len(loads) == 1
    return loads[0]


def _shape(tv: TileValue | None) -> Shape | None:
    if tv is None:
        return None
    return tv.shape


def _offset_signature(func: TileFunction, tv: TileValue | None) -> tuple[str, ...]:
    if tv is None:
        return ()
    op_by_result: dict[str, TileOp] = {
        op.result.name: op for op in walk_ops(func.ops) if op.result is not None
    }
    signature: list[str] = []
    seen: set[str] = set()

    def visit(cur: TileValue) -> None:
        if cur.name in seen:
            return
        seen.add(cur.name)
        op = op_by_result.get(cur.name)
        if op is None:
            signature.append(f"leaf:{cur.shape}:{cur.dtype}")
            return
        result = op.result
        assert result is not None
        if isinstance(op, BinOp):
            signature.append(f"binop:{op.op}:{result.shape}:{result.dtype}")
            visit_required(op.lhs)
            visit_required(op.rhs)
        elif isinstance(op, Splat):
            signature.append(f"splat:{result.shape}:{result.dtype}")
            visit_required(op.value)
        elif isinstance(op, Constant):
            signature.append(f"const:{op.value}:{result.dtype}")
        elif isinstance(op, ExpandDims):
            signature.append(f"expand:{op.axis}:{result.shape}:{result.dtype}")
            visit_required(op.input)
        else:
            signature.append(f"{type(op).__name__}:{result.shape}:{result.dtype}")

    def visit_required(cur: TileValue | None) -> None:
        assert cur is not None
        visit(cur)

    visit(tv)
    return tuple(signature)


def _snapshot(func: TileFunction) -> AddressSnapshot:
    load = _single_load(func)
    return AddressSnapshot(
        offset_shape=_shape(load.offsets),
        row_shape=_shape(load.row_indices),
        col_shape=_shape(load.col_indices),
        row_stride=load.row_stride,
        base_offset_shape=_shape(load.base_offset),
        addr_transposed=load.addr_transposed,
        offset_signature=_offset_signature(func, load.offsets),
    )


def one_dimensional(x, out, N):
    pid = al.program_id(0)
    offs = pid * 8 + al.arange(0, 8)
    mask = offs < N
    al.store(out + offs, al.load(x + offs, mask=mask), mask=mask)


def two_dimensional(A, out, M, N, BM):
    pm = al.program_id(0)
    rm = pm * BM + al.arange(0, BM)
    rn = al.arange(0, N)
    a = al.load(A + rm[:, None] * N + rn[None, :], mask=rm[:, None] < M)
    al.store(out + rm[:, None] * N + rn[None, :], a, mask=rm[:, None] < M)


def strided(x, out, N):
    offs = al.arange(0, N)
    al.store(out + offs, al.load(x + offs))


def broadcast(x, out, N):
    offs = al.arange(0, N)
    al.store(out + offs, al.load(x + offs))


def transposed(A, out, M, N, BM, BN):
    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BM + al.arange(0, BM)
    rn = pn * BN + al.arange(0, BN)
    a = al.load(
        A + rn[None, :] * M + rm[:, None],
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )
    al.store(out + rm[:, None] * N + rn[None, :], a, mask=(rm[:, None] < M) & (rn[None, :] < N))


def test_one_dimensional_address_snapshot() -> None:
    func = _trace(one_dimensional, {"N": 64}, ("x", "out", "N"), {"N"})

    assert _snapshot(func) == AddressSnapshot(
        offset_shape=(8,),
        row_shape=None,
        col_shape=None,
        row_stride=None,
        base_offset_shape=None,
        addr_transposed=False,
        offset_signature=(
            "binop:add:(8,):i32",
            "binop:mul:():i32",
            "ProgramId:():i32",
            "const:8:i32",
            "MakeRange:(8,):i32",
        ),
    )


def test_two_dimensional_address_snapshot() -> None:
    func = _trace(
        two_dimensional, {"M": 16, "N": 8, "BM": 4}, ("A", "out", "M", "N", "BM"), {"M", "N", "BM"}
    )

    assert _snapshot(func) == AddressSnapshot(
        offset_shape=(4, 8),
        row_shape=(4,),
        col_shape=(8,),
        row_stride=8,
        base_offset_shape=None,
        addr_transposed=False,
        offset_signature=(
            "binop:add:(4, 8):i32",
            "binop:mul:(4, 1):i32",
            "expand:1:(4, 1):i32",
            "binop:add:(4,):i32",
            "binop:mul:():i32",
            "ProgramId:():i32",
            "const:4:i32",
            "MakeRange:(4,):i32",
            "const:8:i32",
            "expand:0:(1, 8):i32",
            "MakeRange:(8,):i32",
        ),
    )


def test_strided_address_snapshot() -> None:
    func = _trace(
        strided,
        {"N": 8, "_x_shape": (2, 4), "_x_strides": (8, 1)},
        ("x", "out", "N"),
        {"N"},
        {"x": (2, 4)},
    )

    assert _snapshot(func) == AddressSnapshot(
        offset_shape=(8,),
        row_shape=None,
        col_shape=None,
        row_stride=None,
        base_offset_shape=None,
        addr_transposed=False,
        offset_signature=(
            "binop:add:(8,):i32",
            "binop:mod:(8,):i32",
            "MakeRange:(8,):i32",
            "splat:(8,):i32",
            "const:4:i32",
            "binop:mul:(8,):i32",
            "binop:floordiv:(8,):i32",
            "splat:(8,):i32",
            "const:8:i32",
        ),
    )


def test_broadcast_address_snapshot() -> None:
    func = _trace(
        broadcast,
        {"N": 8, "_x_shape": (2, 4), "_x_strides": (0, 1)},
        ("x", "out", "N"),
        {"N"},
        {"x": (2, 4)},
    )

    assert _snapshot(func) == AddressSnapshot(
        offset_shape=(8,),
        row_shape=None,
        col_shape=None,
        row_stride=None,
        base_offset_shape=None,
        addr_transposed=False,
        offset_signature=(
            "binop:mod:(8,):i32",
            "MakeRange:(8,):i32",
            "splat:(8,):i32",
            "const:4:i32",
        ),
    )


def test_transposed_address_snapshot() -> None:
    func = _trace(
        transposed,
        {"M": 16, "N": 8, "BM": 4, "BN": 4},
        ("A", "out", "M", "N", "BM", "BN"),
        {"M", "N", "BM", "BN"},
    )

    assert _snapshot(func) == AddressSnapshot(
        offset_shape=(4, 4),
        row_shape=(4,),
        col_shape=(4,),
        row_stride=16,
        base_offset_shape=None,
        addr_transposed=True,
        offset_signature=(
            "binop:add:(4, 4):i32",
            "binop:mul:(1, 4):i32",
            "expand:0:(1, 4):i32",
            "binop:add:(4,):i32",
            "binop:mul:():i32",
            "ProgramId:():i32",
            "const:4:i32",
            "MakeRange:(4,):i32",
            "const:16:i32",
            "expand:1:(4, 1):i32",
            "binop:add:(4,):i32",
            "binop:mul:():i32",
            "ProgramId:():i32",
            "const:4:i32",
            "MakeRange:(4,):i32",
        ),
    )
