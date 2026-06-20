"""Tests for IndexTransform — MSL index expression generation.

Each transform maps a flat or (row, col) index to an MSL expression.
Tests verify the generated expression strings are correct.
"""

from alloy._compiler.fusion_transforms import (
    BroadcastTransform,
    ColumnBroadcastTransform,
    IdentityTransform,
    RowBroadcastTransform,
    ScalarBroadcastTransform,
    ScatterTransform,
    StridedTransform,
)


class TestIdentityTransform:
    def test_flat_passthrough(self):
        assert IdentityTransform().flat("idx") == "idx"

    def test_tile_2d(self):
        assert IdentityTransform().tile_2d("r", "c") == "r * _N + c"


class TestScalarBroadcastTransform:
    def test_flat_always_zero(self):
        assert ScalarBroadcastTransform().flat("idx") == "0"

    def test_tile_2d_always_zero(self):
        assert ScalarBroadcastTransform().tile_2d("r", "c") == "0"


class TestBroadcastTransform:
    def test_flat_modular(self):
        xf = BroadcastTransform(size=64, inner_repeat=1, row_stride=0)
        assert xf.flat("idx") == "(idx) % 64u"

    def test_flat_with_inner_repeat(self):
        xf = BroadcastTransform(size=64, inner_repeat=4, row_stride=0)
        assert xf.flat("idx") == "((idx) / 4u) % 64u"

    def test_tile_2d_with_row_stride(self):
        xf = BroadcastTransform(size=64, inner_repeat=1, row_stride=128)
        expr = xf.tile_2d("r", "c")
        assert "128u" in expr
        assert "% 64u" in expr


class TestRowBroadcastTransform:
    def test_tile_2d_uses_col_only(self):
        assert RowBroadcastTransform().tile_2d("r", "c") == "c"

    def test_flat_passthrough(self):
        # row broadcast falls through to identity in flat context
        assert RowBroadcastTransform().flat("idx") == "idx"


class TestColumnBroadcastTransform:
    def test_tile_2d_uses_row_only(self):
        assert ColumnBroadcastTransform().tile_2d("r", "c") == "r"

    def test_flat_passthrough(self):
        assert ColumnBroadcastTransform().flat("idx") == "idx"


class TestStridedTransform:
    def test_tile_2d(self):
        xf = StridedTransform(row_stride=256)
        assert xf.tile_2d("r", "c") == "r * 256u + c"

    def test_tile_2d_encodes_actual_stride(self):
        """Row stride must be the buffer's actual stride, not some ambient N."""
        xf = StridedTransform(row_stride=128)
        expr = xf.tile_2d("_gm", "_gn")
        assert "128u" in expr
        assert "_N" not in expr  # must NOT reference ambient constexpr

    def test_flat_passthrough(self):
        xf = StridedTransform(row_stride=64)
        assert xf.flat("idx") == "idx"


class TestScatterTransform:
    def test_3d_flat(self):
        xf = ScatterTransform(nd_shape=(2, 4, 8), nd_strides=(64, 1, 4))
        expr = xf.flat("idx")
        assert "64u" in expr
        assert "4u" in expr

    def test_2d_tile(self):
        xf = ScatterTransform(nd_shape=(4, 8), nd_strides=(16, 1))
        expr = xf.tile_2d("r", "c")
        assert "16u" in expr
