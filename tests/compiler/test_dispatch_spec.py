"""Tests for DispatchContract — symbolic expressions and dispatch evaluation."""

import pytest
from alloy._compiler.dispatch_spec import (
    Add, AxisSpec, CeilDiv, Const, DispatchContract,
    FloorDiv, FromConstexpr, FromDerived, FromInputShape, Mul, OutputSpec, Sym,
)


class TestExprAlgebra:
    @pytest.mark.parametrize("expr,bindings,expected", [
        (Const(42), {}, 42),
        (Sym("N"), {"N": 128}, 128),
        (Add(Const(3), Const(4)), {}, 7),
        (Mul(Sym("M"), Const(2)), {"M": 64}, 128),
        (FloorDiv(Const(10), Const(3)), {}, 3),
        (CeilDiv(Const(10), Const(3)), {}, 4),
        (CeilDiv(Const(9), Const(3)), {}, 3),
        (CeilDiv(Const(1), Const(32)), {}, 1),
        (CeilDiv(Sym("M"), Sym("BM")), {"M": 100, "BM": 32}, 4),
    ])
    def test_evaluate(self, expr, bindings, expected):
        assert expr.evaluate(bindings) == expected

    def test_sym_missing_raises(self):
        with pytest.raises(KeyError, match="Unresolved symbol 'N'"):
            Sym("N").evaluate({})

    def test_operator_overloads(self):
        assert (Sym("N") + 1).evaluate({"N": 10}) == 11
        assert (Sym("N") * 2).evaluate({"N": 10}) == 20
        assert (Sym("N") // 4).evaluate({"N": 10}) == 2


class TestBindingResolution:
    def test_from_constexpr(self):
        spec = DispatchContract(bindings={"BH": FromConstexpr("BH")})
        assert spec.resolve_bindings({"BH": 4}, {})["BH"] == 4

    def test_from_input_shape(self):
        spec = DispatchContract(bindings={
            "M": FromInputShape("x", 0), "N": FromInputShape("x", 1),
        })
        b = spec.resolve_bindings({}, {"x": (512, 768)})
        assert b["M"] == 512 and b["N"] == 768

    def test_from_derived(self):
        spec = DispatchContract(bindings={
            "BH": FromConstexpr("BH"),
            "Q0": FromInputShape("Q", 0),
            "N": FromDerived(FloorDiv(Sym("Q0"), Sym("BH"))),
        })
        assert spec.resolve_bindings({"BH": 4}, {"Q": (512, 64)})["N"] == 128

    def test_derived_chain(self):
        spec = DispatchContract(bindings={
            "A": FromConstexpr("A"),
            "B": FromDerived(Sym("A") * 2),
            "C": FromDerived(Sym("B") + 1),
        })
        b = spec.resolve_bindings({"A": 10}, {})
        assert b["B"] == 20 and b["C"] == 21


class TestGridEvaluation:
    def test_1d_grid(self):
        spec = DispatchContract(
            grid_axes={0: AxisSpec(block=Const(1024), bound=Sym("N"))},
            bindings={"N": FromConstexpr("N")},
        )
        b = spec.resolve_bindings({"N": 4096}, {})
        assert spec.evaluate_grid(b) == (4, 1, 1)

    def test_2d_grid(self):
        spec = DispatchContract(
            grid_axes={
                0: AxisSpec(block=Sym("BM"), bound=Sym("M")),
                1: AxisSpec(block=Sym("BN"), bound=Sym("N")),
            },
            bindings={
                "M": FromConstexpr("M"), "N": FromConstexpr("N"),
                "BM": FromConstexpr("BM"), "BN": FromConstexpr("BN"),
            },
        )
        b = spec.resolve_bindings({"M": 64, "N": 128, "BM": 32, "BN": 64}, {})
        assert spec.evaluate_grid(b) == (2, 2, 1)

    def test_unresolved_raises(self):
        spec = DispatchContract(unresolved_axes=[0])
        with pytest.raises(RuntimeError, match="unresolved bounds"):
            spec.evaluate_grid({})


class TestOutputShape:
    def test_basic_output(self):
        spec = DispatchContract(
            outputs={"out": OutputSpec(shape=(Sym("M"), Sym("N")), dtype="f32")},
            bindings={"M": FromConstexpr("M"), "N": FromConstexpr("N")},
        )
        b = spec.resolve_bindings({"M": 64, "N": 128}, {})
        assert spec.evaluate_output_shape("out", b) == (64, 128)

    def test_missing_output(self):
        spec = DispatchContract()
        assert spec.evaluate_output_shape("missing", {}) is None
