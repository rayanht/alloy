"""Kernel inspection: view generated MSL, tile IR, and compilation details."""

from alloy._compiler.tile_ir import dump_tile_ir
from alloy._compiler.trace import trace_kernel
from alloy._dispatch.kernel import KernelFunction


def inspect(kernel, level="msl", **constexpr_values):
    """Print compilation details for a kernel.

    Args:
        kernel: A KernelFunction (from @al.kernel).
        level: "msl" (default) or "tile-ir".
        **constexpr_values: Compile-time constant values for the kernel.

    Returns:
        The generated source as a string (also printed).
    """
    if not isinstance(kernel, KernelFunction):
        raise TypeError(f"Expected a @al.kernel function, got {type(kernel)}")

    if level == "msl":
        msl = kernel.compile_to_msl(**constexpr_values)
        print(msl)
        return msl

    if level == "tile-ir":
        # Separate buffer shapes from constexpr values
        buffer_shapes = {}
        ce = {}
        buf_params = set(kernel._param_names) - kernel._constexpr_params - kernel._output_params
        for k, v in constexpr_values.items():
            if k in buf_params and isinstance(v, tuple):
                buffer_shapes[k] = v
            else:
                ce[k] = v
        func = trace_kernel(
            kernel.fn,
            kernel.name,
            ce,
            param_names=kernel._param_names,
            constexpr_params=kernel._constexpr_params,
            source=kernel._source,
            buffer_shapes=buffer_shapes or None,
            output_params=kernel._output_params,
        )
        text = dump_tile_ir(func)
        print(text)
        return text

    raise ValueError(f"Unknown inspection level: {level!r}. Supported: 'msl', 'tile-ir'")
