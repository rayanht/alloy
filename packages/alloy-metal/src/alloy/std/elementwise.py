"""Generated elementwise kernels."""

import alloy as al

# --- Elementwise kernel templates ---


def _make_elementwise_binary(name, op_fn):
    @al.kernel
    def k(x, y, out: al.output, N: al.constexpr, BLOCK_SIZE: al.constexpr = 1024):
        pid = al.program_id(0)
        offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        a = al.load(x + offs, mask=mask)
        b = al.load(y + offs, mask=mask)
        al.store(out + offs, op_fn(a, b), mask=mask)

    k.name = name
    k._source = f"# {name}\n" + k._source
    return k


def _make_elementwise_unary(name, op_fn):
    @al.kernel
    def k(x, out: al.output, N: al.constexpr, BLOCK_SIZE: al.constexpr = 1024):
        pid = al.program_id(0)
        offs = pid * BLOCK_SIZE + al.arange(0, BLOCK_SIZE)
        mask = offs < N
        v = al.load(x + offs, mask=mask)
        al.store(out + offs, op_fn(v), mask=mask)

    k.name = name
    k._source = f"# {name}\n" + k._source
    return k


add = _make_elementwise_binary("add", lambda a, b: a + b)
# silu(g) * u for the unfused gate/up GEMM pair — the expression tree matches
# the fused dot_*_silu epilogue exactly (same MSL ops, bit-identical output).
silu_mul = _make_elementwise_binary(
    "silu_mul", lambda g, u: g * (1.0 / (1.0 + al.exp(-g))) * u
)
# gelu_tanh(g) * u for the unfused gate/up GEMM pair — matches the fused
# dot_q4_k_gelu_v2 epilogue (same al.gelu_tanh primitive, bit-identical).
gelu_tanh_mul = _make_elementwise_binary(
    "gelu_tanh_mul", lambda g, u: al.gelu_tanh(g) * u
)
sub = _make_elementwise_binary("sub", lambda a, b: a - b)
mul = _make_elementwise_binary("mul", lambda a, b: a * b)
neg = _make_elementwise_unary("neg", lambda v: -v)
k_gelu = _make_elementwise_unary("gelu", lambda v: al.gelu(v))
k_gelu_tanh_approx = _make_elementwise_unary("gelu_tanh", lambda v: al.gelu_tanh(v))
k_relu = _make_elementwise_unary("relu", lambda v: al.relu(v))
k_sigmoid = _make_elementwise_unary("sigmoid", lambda v: al.sigmoid(v))
k_floor = _make_elementwise_unary("k_floor", lambda v: al.floor(v))
