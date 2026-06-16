"""DispatchContract — canonical dispatch specification for grid, output shape, and bindings.

Built during kernel tracing, evaluated at dispatch time with concrete bindings.
Single source of truth for grid dimensions, output shapes/dtypes, and symbolic bindings.
All consumers (runtime launch, lazy fusion, codegen) use this contract's evaluation APIs.

Schema version is embedded for cache invalidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Contract schema version — bump when evaluation semantics change.
DISPATCH_CONTRACT_VERSION = 2


# ---------------------------------------------------------------------------
# Symbolic expression algebra
# ---------------------------------------------------------------------------


class Expr:
    """Base class for symbolic expressions. Evaluate with concrete bindings."""

    def evaluate(self, bindings: dict[str, int]) -> int:
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, int):
            other = Const(other)
        return Add(self, other)

    def __radd__(self, other):
        if isinstance(other, int):
            other = Const(other)
        return Add(other, self)

    def __mul__(self, other):
        if isinstance(other, int):
            other = Const(other)
        return Mul(self, other)

    def __rmul__(self, other):
        if isinstance(other, int):
            other = Const(other)
        return Mul(other, self)

    def __floordiv__(self, other):
        if isinstance(other, int):
            other = Const(other)
        return FloorDiv(self, other)


@dataclass(frozen=True)
class Const(Expr):
    value: int

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.value

    def __repr__(self):
        return str(self.value)


@dataclass(frozen=True)
class Sym(Expr):
    """Symbolic variable — resolved from bindings at dispatch time."""

    name: str

    def evaluate(self, bindings: dict[str, int]) -> int:
        if self.name not in bindings:
            raise KeyError(
                f"Unresolved symbol '{self.name}' — available bindings: {list(bindings.keys())}"
            )
        return bindings[self.name]

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Add(Expr):
    lhs: Expr
    rhs: Expr

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.lhs.evaluate(bindings) + self.rhs.evaluate(bindings)

    def __repr__(self):
        return f"({self.lhs} + {self.rhs})"


@dataclass(frozen=True)
class Mul(Expr):
    lhs: Expr
    rhs: Expr

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.lhs.evaluate(bindings) * self.rhs.evaluate(bindings)

    def __repr__(self):
        return f"({self.lhs} * {self.rhs})"


@dataclass(frozen=True)
class FloorDiv(Expr):
    lhs: Expr
    rhs: Expr

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.lhs.evaluate(bindings) // self.rhs.evaluate(bindings)

    def __repr__(self):
        return f"({self.lhs} // {self.rhs})"


@dataclass(frozen=True)
class CeilDiv(Expr):
    lhs: Expr
    rhs: Expr

    def evaluate(self, bindings: dict[str, int]) -> int:
        a = self.lhs.evaluate(bindings)
        b = self.rhs.evaluate(bindings)
        return (a + b - 1) // b

    def __repr__(self):
        return f"ceildiv({self.lhs}, {self.rhs})"


# ---------------------------------------------------------------------------
# Binding sources — where symbols get their values at dispatch time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FromConstexpr:
    """Value comes from a constexpr parameter."""

    name: str


@dataclass(frozen=True)
class FromInputShape:
    """Value comes from an input buffer's shape dimension."""

    param: str
    dim: int


@dataclass(frozen=True)
class FromDerived:
    """Value is computed from other symbols."""

    expr: Expr


BindingSource = FromConstexpr | FromInputShape | FromDerived


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass
class AxisSpec:
    """Grid axis specification: block size and total bound."""

    block: Expr
    bound: Expr


@dataclass
class OutputWritePattern:
    """Write pattern from a single al.store to an output buffer.

    Captured at trace time, merged at assembly to derive OutputSpec.shape.
    Each dim is a tuple (kind, info):
      ("bound", [pid_axes]) → extent = product of bound(ax)
      ("grid",  [pid_axes]) → extent = product of grid_count(ax)
      ("const", int)        → extent is a concrete value
    """

    param_name: str
    dtype: str
    value_shape: tuple[int, ...]  # tile shape, for tiebreaking
    dims: list[tuple[str, Any]]  # outermost to innermost


@dataclass
class OutputSpec:
    """Output buffer shape specification."""

    shape: tuple[Expr, ...]
    dtype: str = "f32"


@dataclass
class DispatchContract:
    """Canonical dispatch specification for a kernel.

    Built during tracing, evaluated at dispatch time. Single source of truth
    for grid dimensions, output shapes/dtypes, and symbolic bindings.
    """

    version: int = DISPATCH_CONTRACT_VERSION
    grid_axes: dict[int, AxisSpec] = field(default_factory=dict)
    outputs: dict[str, OutputSpec] = field(default_factory=dict)
    bindings: dict[str, BindingSource] = field(default_factory=dict)
    unresolved_axes: list[int] = field(default_factory=list)
    unresolved_outputs: dict[str, list[int]] = field(default_factory=dict)  # pname → [dim indices]

    def resolve_bindings(
        self,
        constexpr_values: dict[str, Any],
        buffer_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, int]:
        """Resolve all symbolic bindings to concrete ints."""
        resolved: dict[str, int] = {}

        # First pass: resolve direct sources (constexpr, input_shape)
        for name, src in self.bindings.items():
            if isinstance(src, FromConstexpr):
                if src.name in constexpr_values:
                    resolved[name] = int(constexpr_values[src.name])
            elif isinstance(src, FromInputShape):
                shape = buffer_shapes.get(src.param)
                if shape and src.dim < len(shape):
                    resolved[name] = shape[src.dim]

        # Iterative resolution of derived bindings (may depend on each other)
        derived = [
            (name, src) for name, src in self.bindings.items() if isinstance(src, FromDerived)
        ]
        for _ in range(len(derived) + 1):  # at most N passes for N derived bindings
            progress = False
            for name, src in derived:
                if name not in resolved:
                    try:
                        resolved[name] = src.expr.evaluate(resolved)
                        progress = True
                    except KeyError:
                        pass
            if not progress:
                break

        return resolved

    def evaluate_grid(self, bindings: dict[str, int]) -> tuple[int, ...]:
        """Evaluate grid dimensions from resolved bindings.

        Raises RuntimeError if any axis has an unresolved bound.
        """
        if self.unresolved_axes:
            raise RuntimeError(
                f"Cannot evaluate grid: unresolved bounds for axes {self.unresolved_axes}. "
                f"No mask comparison (offs < BOUND) found and no input shape to infer from."
            )
        grid = [1, 1, 1]
        for axis, spec in sorted(self.grid_axes.items()):
            if axis < 3:
                bound = spec.bound.evaluate(bindings)
                block = spec.block.evaluate(bindings)
                grid[axis] = (bound + block - 1) // block
        return tuple(grid)

    def evaluate_output_shape(
        self,
        param_name: str,
        bindings: dict[str, int],
    ) -> tuple[int, ...] | None:
        """Evaluate output shape for a parameter. None if param not in outputs."""
        out = self.outputs.get(param_name)
        if out is None:
            return None
        return tuple(dim.evaluate(bindings) for dim in out.shape)

    def validate(self) -> list[str]:
        """Check contract for completeness. Returns list of issues (empty = valid)."""
        issues = []
        for axis in self.unresolved_axes:
            issues.append(f"Grid axis {axis}: bound is unresolved")
        for pname, dims in self.unresolved_outputs.items():
            for dim_idx in dims:
                issues.append(f"Output '{pname}' dim {dim_idx}: unresolved bound")
        return issues

    def evaluate_dispatch(
        self,
        constexpr_values: dict[str, Any],
        buffer_shapes: dict[str, tuple[int, ...]],
        grid_override: tuple | None = None,
        output_params: set[str] | None = None,
        kernel_name: str = "<unknown>",
    ) -> tuple[dict[str, int], tuple[int, int, int], dict[str, tuple[int, ...]]]:
        """One-shot dispatch evaluation: bindings + grid + output shapes.

        Args:
            constexpr_values: Resolved constexpr parameters.
            buffer_shapes: Input buffer shapes.
            grid_override: Explicit grid (skips auto-derivation if set).
            output_params: Output parameter names (for shape evaluation).
            kernel_name: For error messages.

        Returns:
            (bindings, grid_3d, output_shapes) where output_shapes maps
            param_name → shape tuple for each output param.

        Raises:
            RuntimeError: If grid cannot be derived (unresolved axes, no override).
        """
        bindings = self.resolve_bindings(constexpr_values, buffer_shapes)

        if grid_override is not None:
            grid = grid_override
        elif self.unresolved_axes:
            raise RuntimeError(
                f"Kernel '{kernel_name}': cannot derive grid — unresolved "
                f"bounds for program_id axes {self.unresolved_axes}. "
                f"Add a mask (offs < BOUND) or provide an explicit grid."
            )
        elif self.grid_axes:
            grid = self.evaluate_grid(bindings)
        else:
            grid = (1, 1, 1)

        # Normalize to 3D
        n = len(grid)
        if n == 0:
            grid_3d = (1, 1, 1)
        elif n == 1:
            grid_3d = (int(grid[0]), 1, 1)
        elif n == 2:
            grid_3d = (int(grid[0]), int(grid[1]), 1)
        else:
            grid_3d = (int(grid[0]), int(grid[1]), int(grid[2]))

        output_shapes: dict[str, tuple[int, ...]] = {}
        if output_params:
            for pname in output_params:
                shape = self.evaluate_output_shape(pname, bindings)
                if shape is not None:
                    output_shapes[pname] = shape

        return bindings, grid_3d, output_shapes
