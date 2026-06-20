# Alloy

Kernel authoring DSL, `torch.compile` backend and LLM serving for Apple Silicon.

Alloy is a compiler and runtime for GPU compute kernels on Apple Silicon. You write
kernels in Python. Alloy compiles them to Metal through a tile IR pipeline; 
covering everything from per-thread scalar kernels to cooperative
tiled GEMM with simdgroup MMA and automatic operator fusion for multi-kernel pipelines.

**Status**: technical preview. Requires Apple Silicon (M1+) and macOS 13+. The
Python packages need Python 3.10–3.12.

## Install

**Python (pip / uv)**

```bash
pip install 'alloy-kit[serve]'   # local LLM server + CLI + torch.compile backend
pip install alloy-kit            # lean: just the GPU kernel compiler (no torch)
pip install 'alloy-kit[all]'     # + training / vision / audio research extras

import alloy as al
```

The PyPI distribution is **`alloy-kit`**. The brackets are optional
dependency groups: the lean base provides `@al.kernel` with the tile IR, MSL emitter and Metal dispatch machinery,
and `[serve]` adds everything needed to run the server and the `alloy` CLI.

**Standalone (no Python required):**

```bash
curl -fsSL https://raw.githubusercontent.com/rayanht/alloy/main/installer/install.sh | sh
```

Installs a self-contained `alloy` CLI into `/usr/local`.

**From source (contributors):** see [Contributing](#contributing).

## Inference server - Quickstart

Alloy serves a loopback HTTP API that's drop-in compatible with the OpenAI, Anthropic and
Ollama clients.

> [!IMPORTANT]
> Run `alloy tune <model>` before serving for optimal performance

```bash
# Start the server in the foreground; loads the model
# from a local Ollama cache or Hugging Face if present.

alloy serve -m qwen3:0.6b                                   # Ollama tag
alloy serve -m bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M  # HF model
```

```bash
# OpenAI:
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"hi"}]}'

# Ollama:
curl http://127.0.0.1:11434/api/chat \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"hi"}]}'
```

```bash
# Claude Code
alloy launch claude
```

The default port is `11434`. Pass `--port 11435` to `alloy serve` (or set `ALLOY_PORT`) to override.

### Features

| Feature | Status |
|---|---|
| Warm-prefix KV reuse (bookmarks + branching) | Stable |
| On-GPU sampling (temp / top-p / top-k / min-p / seed) | Stable |
| Constrained decoding (xgrammar JSON + tool grammars) | Stable |
| Tool calling (OpenAI / Anthropic / Ollama, per-family parsers) | Stable |
| Reasoning / thinking split | Stable |
| MoE inference | Stable (Qwen3.5-MoE) |
| Vision input | Stable (gemma4) |
| Audio input | Stable (gemma4) |
| Embeddings | Stable (nomic-embed-text) |
| Speculative decoding — PLD (prompt lookup) | Opt-in (`--spec pld`) |
| Speculative decoding — MTP | Opt-in (`--spec mtp`, Qwen3.5) |
| Speculative decoding — DFlash (block diffusion) | Opt-in (`--spec dflash`) |
| Paged KV cache | Opt-in (`ALLOY_KV=paged`) |
| KV cache quantization (int8 + fp16 scales) | Opt-in (`--kv-quant q8_0`) |

<details>
<summary>Supported quantizations</summary>

**Model weights**

| source | format | supported |
|---|---|:---:|
| GGUF | Q4_K (Q4_K_M / Q4_K_S) | ✅ |
| GGUF | Q5_0 | ✅ |
| GGUF | Q6_K | ✅ |
| GGUF | Q8_0 | ✅ |
| GGUF | F16 / BF16 / F32 | ✅ |
| GGUF | Q2_K / Q3_K / Q5_K | ❌ |
| GGUF | Q4_0 / Q4_1 / Q5_1 | ❌ |
| GGUF | IQ1 / IQ2 / IQ3 / IQ4 (IQ4_XS, IQ4_NL) | ❌ |
| MLX | 4-bit affine (group size 64 / 128) | ✅ |
| MLX | 2-bit / 3-bit / 6-bit / 8-bit | ❌ |

**KV cache**

| format | supported |
|---|:---:|
| fp16 (default) | ✅ |
| q8_0 | ✅ |
| q4 / other | ❌ |

</details>


## torch.compile backend

Alloy includes a `torch.compile` backend that compiles covered PyTorch FX graphs to
fused Metal compute kernels.

```python
import torch
import transformers
import alloy_torch  # registers the "alloy" backend

model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
compiled = torch.compile(model, backend="alloy")

input_ids = torch.randint(0, model.config.vocab_size, (1, 16))
output = compiled(input_ids=input_ids)
```

The backend handles: FX graph decomposition, operator fusion (RMSNorm, RoPE, GELU,
batched QKV, GEMM+LayerNorm, scalar broadcast), GQA-native attention, compiled
dispatch plans, and autotuning.

Runnable model examples live in [`examples/torch/`](examples/torch/):

- [`mlp.py`](examples/torch/mlp.py) — multi-layer perceptron (Linear / LayerNorm / GELU)
- [`resnet.py`](examples/torch/resnet.py) — GroupNorm ResNet (Conv2d + residual blocks)
- [`transformer.py`](examples/torch/transformer.py) — pre-norm encoder block (SDPA + GELU MLP)

### Training preview

A full `torch.compile` training step (forward, backward, and the optimizer
update) runs end to end through Alloy and matches PyTorch eager within
floating-point tolerance for dense transformer-style models: embeddings, linear
layers, normalization, residual blocks, attention, cross-entropy, and the common
optimizers (SGD, Adam, AdamW, RMSprop). A small language model trains end to end,
and LoRA fine-tuning of a pretrained transformer works in `model.train()`. Enable
it before `torch.compile`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import alloy_torch  # registers the "alloy" backend
from alloy_torch.training import set_training_mode

set_training_mode(True)  # before torch.compile

model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 1))
step = torch.compile(model, backend="alloy")
opt = torch.optim.AdamW(model.parameters(), lr=0.05)

x, y = torch.randn(32, 64), torch.randn(32, 1)
for _ in range(20):
    opt.zero_grad()
    loss = F.mse_loss(step(x), y)
    loss.backward()
    opt.step()
```

Fine-tuning a pretrained transformer with [PEFT](https://github.com/huggingface/peft) LoRA is the same shape:

```python
import peft
import transformers

model = peft.get_peft_model(
    transformers.AutoModelForCausalLM.from_pretrained("gpt2"),
    peft.LoraConfig(target_modules=["c_attn"], task_type="CAUSAL_LM"),
)
step = torch.compile(model, backend="alloy")
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)

model.train()
for input_ids in batches:
    opt.zero_grad()
    loss = step(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    opt.step()
```

Runnable training examples live in [`examples/torch/`](examples/torch/):

- [`train_mlp.py`](examples/torch/train_mlp.py) — MLP regression (Linear / LayerNorm / GELU, AdamW)
- [`train_transformer.py`](examples/torch/train_transformer.py) — transformer block + cross-entropy (SGD)
- [`train_lm.py`](examples/torch/train_lm.py) — tiny language model (Embedding + attention + cross-entropy)
- [`finetune_lora.py`](examples/torch/finetune_lora.py) — LoRA fine-tuning of gpt2 (PEFT + transformers)

It is still a preview. The backward pass does not yet cover convolutions or
pooling, so CNN training is not supported. Inference is the primary, fully
validated path.

## Benchmarks

Reproduce with `alloy bench <HF_OR_OLLAMA_TAG>`

### Causal LM Inference

#### HF Models

| model | quant | pp512 | tg128 |
|---|---|---:|---:|
| LFM2.5-1.2B-Instruct-GGUF | Q4_K_M | 4222 | 508 |
| bartowski/Llama-3.2-3B-Instruct-GGUF | Q4_K_M | 2061 | 198 |
| Qwen_Qwen3-0.6B-GGUF | Q4_K_M | 8311 | 612 |

#### Ollama Models

| model | quant | pp512 | tg128 |
|---|---|---:|---:|
| qwen2.5:0.5b | Q4_K_M | 12102 | 505 |
| qwen3:0.6b | Q4_K_M | 10077 | 584 |
| llama3.2:1b | Q8_0 | 5653 | 324 |
| qwen3.5:0.8b | Q8_0 | 6141 | 349 |
| deepseek-r1:1.5b | Q4_K_M | 3295 | 274 |
| qwen2.5:1.5b | Q4_K_M | 3295 | 270 |
| qwen3.5:2b | Q8_0 | 3247 | 187 |
| gemma4:e2b | Q4_K_M | 2121 | 175 |
| qwen2.5:3b | Q4_K_M | 1617 | 185 |
| gemma4:e4b | Q4_K_M | 1079 | 115 |
| qwen3.5:4b | Q4_K_M | 1098 | 122 |
| qwen3.5:9b | Q4_K_M | 598 | 78.6 |
| qwen3.6:35b | Q4_K_M | 988 | 121 |

#### MLX Models

| model | quant | pp512 | tg128 |
|---|---|---:|---:|
| Qwen/Qwen3-0.6B-MLX-4bit | 4-bit g128 | 10063 | 710 |
| LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit | 4-bit g64 | 5688 | 589 |
| mlx-community/Llama-3.2-3B-Instruct-4bit | 4-bit g64 | 2173 | 220 |
| mlx-community/Qwen3-4B-4bit | 4-bit g64 | 1673 | 174 |
| mlx-community/Qwen3-8B-4bit | 4-bit g64 | 866 | 102 |

### Multimodal Inference

| model | vision ms | alloy TTFT | alloy dec | alloy wall |
|---|---:|---:|---:|---:|
| gemma4:e2b | 229 | 455 | 172 | 1193 |
| gemma4:e4b | 257 | 665 | 99.0 | 1949 |

### Embeddings Inference

Per-regime encoder tok/s from `alloy bench nomic-embed-text --dataset embeddings`.

| regime  | batch | seq | tok/s |
|---|---:|---:|---:|
| q_short  | 1 |  10 | 5094 |
| q_long   | 1 | 256 | 19161 |
| b8_short | 8 |  10 | 14142 |
| b8_long  | 8 | 128 | 11840 |

## Write a kernel

```python
import numpy as np
import alloy as al

@al.kernel
def blur(src, dst: al.output, W: al.constexpr, H: al.constexpr):
    x = al.program_id(0)
    y = al.program_id(1)
    acc = 0.0
    count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx = x + dx
            ny = y + dy
            if nx >= 0:
                if nx < W:
                    if ny >= 0:
                        if ny < H:
                            acc = acc + al.load(src + ny * W + nx)
                            count = count + 1
    al.store(dst + y * W + x, acc / count)

W, H = 1920, 1080
img = np.random.rand(H, W).astype(np.float32)

out = blur[W, H](img.ravel(), W=W, H=H)
print(np.asarray(out).reshape(H, W))
```

NumPy and PyTorch arrays can be bound directly as inputs for covered contiguous
host-memory paths. The kernel's `al.output` is allocated automatically and returned as an
`AlloyBuffer` — convert with `np.asarray(...)` or `.numpy()`. Some interop paths
allocate Alloy-owned shared buffers or require layout copies, so this is not a blanket
promise that every input type and view is no-copy.

More runnable examples live in [`examples/kernel/`](examples/kernel/):

- [`vector_add.py`](examples/kernel/vector_add.py) — masked elementwise add
- [`elementwise.py`](examples/kernel/elementwise.py) — fused GELU / sigmoid / SiLU
- [`softmax.py`](examples/kernel/softmax.py) — row-wise softmax, manual vs. builtin
- [`blur.py`](examples/kernel/blur.py) — 2D box blur (shown above)
- [`matmul.py`](examples/kernel/matmul.py) — naive vs. tiled GEMM with simdgroup MMA
- [`histogram.py`](examples/kernel/histogram.py) — atomics
- [`nbody.py`](examples/kernel/nbody.py) — N-body simulation
- [`mandelbrot.py`](examples/kernel/mandelbrot.py) — divergent per-thread iteration
- [`flash_attention.py`](examples/kernel/flash_attention.py) — online-softmax attention

## Tiled GEMM

```python
@al.kernel
def matmul(A, B_T, C: al.output,
           BLOCK_M: al.constexpr = 64, BLOCK_N: al.constexpr = 64, BLOCK_K: al.constexpr = 16):
    M, K = A.shape
    N = B_T.shape[0]
    pm = al.program_id(0)
    pn = al.program_id(1)
    rm = pm * BLOCK_M + al.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + al.arange(0, BLOCK_N)
    rk = al.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * K + rk[None, :]
    b_ptrs = B_T + rn[:, None] * K + rk[None, :]
    acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
    for k in range(0, K, BLOCK_K):
        a = al.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K))
        b = al.load(b_ptrs, mask=(rn[:, None] < N) & (rk[None, :] < K))
        acc += al.tile_dot(a, b, transpose_rhs=True)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    al.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
```

This compiles to Metal with simdgroup matrix multiply-accumulate (MMA), cooperative tile loads, threadgroup shared memory staging, and optional double buffering all generated automatically from the tile IR.

## Why Alloy

**The problem.** Metal compute is powerful but painful to program. You write MSL in a C++ dialect, manually manage buffer bindings, compile pipeline state objects, and set up command encoders. There's no equivalent of Triton, Numba, or CuPy for Metal.

**What Alloy does.** Python → tile IR → MSL, with a runtime that handles dispatch, caching, and optimization:

- **Shared-memory execution** — Apple Silicon CPU and GPU share physical memory. Alloy
  binds caller buffers directly where the storage layout supports it, and uses
  Alloy-owned shared buffers when plan safety or alignment requires it.
- **Tile IR compiler** — Python kernel source → AST → tile IR (loads, stores, reductions, MMA, barriers) → Metal Shading Language. Handles threadgroup sizing, shared memory allocation, simdgroup decomposition, and barrier placement automatically.
- **Automatic dispatch** — builtins return lazy buffers that queue GPU work automatically. Reading results triggers a single fused Metal command buffer commit. No manual batch management needed.
- **Operator fusion** — adjacent elementwise kernels fuse automatically, eliminating intermediate buffers. Elementwise ops fuse as prologues and epilogues into reductions, GEMM, softmax, and layernorm. Transposes fuse via stride absorption.
- **Autotuning** — exhaustive search over tile sizes, loop unrolling, double buffering, and matvec strategies.

## Built-in ops

High-performance implementations of common operations, written in the Alloy DSL and compiled through the tile IR pipeline:

```python
C = al.dot_transpose_rhs(A, B)               # tiled GEMM with autotuning
s = al.softmax(x)                            # fused row-wise softmax
y = al.layernorm(x, gamma, beta)             # fused layer normalization
y, _ = al.rms_norm(x, weight)                # fused RMS normalization (+ per-row 1/rms)
L = al.cross_entropy(logits, labels)         # fused cross-entropy loss kernel
```

Builtins infer output shapes and constexpr values from input arrays. They compose with fusion. e.g. `al.dot` followed by an elementwise kernel automatically fuses the elementwise op as an epilogue.

## Kernel primitives

```python
# Grid and thread indexing
pid = al.program_id(0)        # threadgroup position (block index)
tid = al.thread_id()          # thread position within threadgroup
offs = pid * 1024 + al.arange(0, 1024)  # block-level offsets

# Memory
x = al.load(ptr + offs, mask=mask)      # masked global load
al.store(ptr + offs, val, mask=mask)    # masked global store
buf = al.shared(256)                     # threadgroup shared memory
loc = al.local(8)                        # per-thread register array
al.barrier()                             # threadgroup memory barrier
al.coop_load(buf, src_ptr, size)         # cooperative threadgroup load + barrier
al.copy4(dst, offset, src_ptr)           # vectorized 4-element load

# Tile operations (2D blocks for GEMM, attention, etc.)
acc = al.zeros((BLOCK_M, BLOCK_N), dtype=al.float32)
acc += al.tile_dot(a, b, transpose_rhs=True)  # simdgroup MMA
reduced = al.simd_reduce(val)                  # cross-lane reduction

# Simdgroup (warp-level)
al.simd_shuffle_xor(val, offset)         # butterfly shuffle
al.simd_shuffle(val, lane)               # read from specific lane
acc = al.simd_matrix()                   # 8x8 matrix accumulator
al.simd_load(src, offset, stride)        # load into simd matrix
al.simd_mma(acc, a, b)                   # matrix multiply-accumulate

# Atomics
al.atomic_add(ptr, idx, val)             # atomic fetch-and-add (int32)
al.atomic_max(ptr, idx, val)
al.atomic_cas(ptr, idx, expected, desired)  # compare-and-swap

# Control flow — plain Python
if cond: ...
for i in range(N): ...
while cond: ...
```

## Automatic fusion

```python
# These three kernels fuse into one — no intermediate buffers allocated.
# Each call returns a lazy AlloyBuffer; feed it straight into the next:
t1 = scale[grid](x, N=N)          # t1 = x * 2.0
t2 = bias[grid](t1, N=N)          # t2 = t1 + 1.0
result = activate[grid](t2, N=N)  # result = relu(t2)

# Reading the result triggers one fused GPU submission:
print(result[0])
```

## Framework interop

Pass PyTorch tensors or MLX arrays directly when their storage layout is supported:

```python
import torch
x = torch.randn(32, 128)    # CPU tensor, lives in unified memory
result = my_kernel[grid](x, M=32, N=128)  # x bound directly; result returned as an AlloyBuffer
```

Alloy's compiled plans may convert PyTorch input storage to Alloy-owned shared
memory on first execution so subsequent dispatches can resolve Metal buffers by handle. That keeps subsequent dispatches free of per-call input copies for stable storage.

## Inspect generated code

```python
al.inspect(my_kernel, N=8192)                      # prints MSL source
al.inspect(my_kernel, level="tile-ir", N=8192)     # prints tile IR
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, test commands, and PR conventions.

## License

[Apache License 2.0](LICENSE).
