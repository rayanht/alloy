"""Op registry — utility functions for reductions and dot ops.

No kernel generation or special-casing. Standard kernels are defined in the
alloy.std package as real @al.kernel functions.
"""


def _reduce_num_groups(N, block_size=256):
    """Compute number of threadgroups for multi-group reductions."""
    min_per_group = block_size * 32
    max_groups = min(256, N // min_per_group)
    max_groups = max(max_groups, 1)
    n_groups = max_groups
    n_per_group = ((N + n_groups - 1) // n_groups + block_size - 1) // block_size * block_size
    n_groups = (N + n_per_group - 1) // n_per_group
    return n_groups, n_per_group


def resolve_reduction_variant(op_name, M=None, axis=1):
    """Resolve a reduction op name to its variant key (flat/row/col)."""
    reduce_op = op_name.split("_", 1)[1] if op_name.startswith("reduce_") else op_name
    if M is not None:
        if axis == 0:
            return f"col_{op_name}" if op_name == "mean" else f"col_reduce_{reduce_op}"
        else:
            return f"row_{op_name}" if op_name == "mean" else f"row_reduce_{reduce_op}"
    return op_name
