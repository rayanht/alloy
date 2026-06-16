"""Tests for tile IR types, builder, and pretty printer."""

import pytest
from alloy._compiler.tile_ir import (
    Layout,
    TileBuilder,
    TileValue,
    _broadcast_shape,
    _pick_layout,
    dump_tile_ir,
)


class TestTileValue:
    def test_scalar(self):
        v = TileValue("x", shape=(), layout=Layout.REPLICATED)
        assert v.rank == 0 and v.is_scalar and v.numel == 1

    def test_1d(self):
        v = TileValue("x", shape=(1024,), layout=Layout.BLOCKED)
        assert v.rank == 1 and not v.is_scalar and v.numel == 1024

    def test_2d(self):
        v = TileValue("x", shape=(64, 32), layout=Layout.MMA)
        assert v.rank == 2 and v.numel == 2048


class TestBroadcastShape:
    @pytest.mark.parametrize("a,b,expected", [
        ((), (1024,), (1024,)),
        ((), (64, 32), (64, 32)),
        ((64, 32), (64, 32), (64, 32)),
        ((64, 1), (1, 32), (64, 32)),
        ((32,), (64, 32), (64, 32)),
    ])
    def test_broadcast(self, a, b, expected):
        assert _broadcast_shape(a, b) == expected


class TestPickLayout:
    @pytest.mark.parametrize("a,b,expected", [
        (Layout.MMA, Layout.BLOCKED, Layout.MMA),
        (Layout.BLOCKED, Layout.REPLICATED, Layout.BLOCKED),
        (Layout.REPLICATED, Layout.REPLICATED, Layout.REPLICATED),
        (Layout.MMA, Layout.REPLICATED, Layout.MMA),
    ])
    def test_layout_dominance(self, a, b, expected):
        assert _pick_layout(a, b) == expected


class TestTileBuilder:
    def test_build_simple_program(self):
        b = TileBuilder("test")
        b.add_param("x", is_constexpr=False)
        b.add_param("N", is_constexpr=True)
        b.set_constexprs({"N": 1024})
        pid = b.program_id(0)
        assert pid.shape == ()
        func = b.build()
        assert func.name == "test"
        assert len(func.params) == 2

    def test_binop_shape_propagation(self):
        b = TileBuilder("test")
        r = b.make_range(0, 1024)
        c = b.constant(2, "i32")
        s = b.splat(c, (1024,))
        result = b.binop("mul", r, s)
        assert result.shape == (1024,)

    def test_dot_shape(self):
        b = TileBuilder("test")
        lhs = TileValue("a", (8, 16), Layout.BLOCKED, "f32")
        rhs = TileValue("b", (16, 8), Layout.BLOCKED, "f32")
        result = b.dot(lhs, rhs)
        assert result.shape == (8, 8)
        assert result.layout == Layout.MMA


class TestDumpTileIR:
    def test_roundtrip_readable(self):
        b = TileBuilder("vec_add")
        b.add_param("x")
        b.add_param("N", is_constexpr=True)
        b.set_constexprs({"N": 1024})
        b.program_id(0)
        text = dump_tile_ir(b.build())
        assert "tile_func vec_add" in text
        assert "program_id(0)" in text
