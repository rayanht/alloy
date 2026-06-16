"""Integration tests for AlloyBuffer — views, reshape, transpose, slice, identity."""

import numpy as np
from alloy._dispatch.buf_utils import _alloc_aligned
from alloy._compiler.dtypes import float32
from alloy._runtime.alloy_buffer import AlloyBuffer
import alloy as al


class TestSharesAllocation:
    def test_same_buffer(self):
        buf = _alloc_aligned((16,), float32)
        assert buf.shares_allocation(buf)

    def test_view_shares(self):
        buf = _alloc_aligned((4, 4), float32)
        view = buf.reshape((16,))
        assert buf.shares_allocation(view) and view.shares_allocation(buf)

    def test_slice_shares(self):
        buf = _alloc_aligned((16,), float32)
        sliced = buf.slice(0, 4, 8)
        assert buf.shares_allocation(sliced)

    def test_column_slice_shares(self):
        buf = _alloc_aligned((4, 8), float32)
        col0 = buf.slice(1, 0, 4)
        col1 = buf.slice(1, 4, 8)
        assert col0.shares_allocation(col1)

    def test_transpose_shares(self):
        buf = _alloc_aligned((4, 8), float32)
        assert buf.shares_allocation(buf.transpose())

    def test_separate_allocations_differ(self):
        a = _alloc_aligned((16,), float32)
        b = _alloc_aligned((16,), float32)
        assert not a.shares_allocation(b)

    def test_numpy_wrap_shares(self):
        arr = np.zeros((8,), dtype=np.float32)
        buf = AlloyBuffer(arr=arr)
        assert buf.shares_allocation(buf.reshape((2, 4)))

    def test_numpy_separate_differs(self):
        a = AlloyBuffer(arr=np.zeros(8, dtype=np.float32))
        b = AlloyBuffer(arr=np.zeros(8, dtype=np.float32))
        assert not a.shares_allocation(b)


class TestTranspose:
    def test_2d(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        lb = al.dot(x[:, :3], x[:3, :])
        np.testing.assert_allclose(np.array(lb.transpose()), np.array(lb).T)

    def test_shape_correct(self):
        buf = AlloyBuffer(arr=np.zeros((5, 7), dtype=np.float32))
        assert buf.transpose().shape == (7, 5)


class TestReshape:
    def test_contiguous(self):
        buf = _alloc_aligned((4, 8), float32)
        r = buf.reshape((32,))
        assert r.shape == (32,)
        assert r.is_contiguous()

    def test_preserves_data(self):
        arr = np.arange(12, dtype=np.float32)
        buf = AlloyBuffer(arr=arr)
        r = buf.reshape((3, 4))
        np.testing.assert_array_equal(np.asarray(r), arr.reshape(3, 4))
