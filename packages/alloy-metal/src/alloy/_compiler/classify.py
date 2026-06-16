"""AST classification for Alloy kernels — fusion eligibility detection."""

import ast

# --- Kernel classification ---

_THREADGROUP_OPS = {
    "shared",
    "barrier",
    "thread_id",
    "local",
    "coop_load",
    "simd_shuffle_xor",
    "simd_shuffle",
    "simd_shuffle_up",
    "simd_shuffle_down",
    "simd_prefix_exclusive_sum",
    "simd_prefix_inclusive_sum",
    "simd_all",
    "simd_any",
    "simd_id",
    "simd_lane_id",
    "simd_matrix",
    "simd_load",
    "simd_store",
    "simd_mma",
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_cas",
    "atomic_xchg",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_add_float",
    "atomic_max_float",
    "atomic_min_float",
}

_TILE_REDUCE_OPS = {"sum", "max", "min"}


def _ast_call_name(func):
    """Extract the function name from an AST Call node's func field."""
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def has_threadgroup_ops(tree: ast.Module) -> bool:
    """Check if a kernel AST uses threadgroup primitives (shared, barrier, etc.)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _ast_call_name(node.func)
            if name in _THREADGROUP_OPS:
                return True
    return False


def has_non_elem_constructs(tree: ast.Module) -> bool:
    """Check if a kernel uses constructs that prevent elementwise fusion.

    Non-elem indicators: break, continue, return, while/for loops,
    multi-axis program_id.
    """
    pid_axes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.While, ast.For)):
            return True
        if isinstance(node, ast.Call):
            name = _ast_call_name(node.func)
            if name == "program_id":
                if node.args and isinstance(node.args[0], ast.Constant):
                    pid_axes.add(node.args[0].value)
    return len(pid_axes) > 1


def is_tile_kernel(tree: ast.Module) -> bool:
    """Check if a kernel uses tile IR patterns (tile_dot, tile reductions, 2D expand_dims)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _ast_call_name(node.func)
            if name == "tile_dot":
                return True
            if name in _TILE_REDUCE_OPS:
                return True
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple):
            elts = node.slice.elts
            if len(elts) == 2:
                has_none = any(isinstance(e, ast.Constant) and e.value is None for e in elts)
                has_slice = any(
                    isinstance(e, ast.Slice) and e.lower is None and e.upper is None for e in elts
                )
                if has_none and has_slice:
                    return True
    return False
