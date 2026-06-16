# Contributing to Alloy

Thanks for your interest. Alloy is a GPU compiler, `torch.compile` backend and LLM serving engine for
Apple Silicon. Most contributions will involve macOS, Metal, PyTorch, or some
combination of the three.

## Development setup

You need a Mac with an Apple Silicon GPU (M1 or later) and Xcode Command Line
Tools installed. Other platforms cannot run Alloy.

```bash
git clone https://github.com/rayanht/alloy.git
cd alloy
uv sync                        # installs all four packages in editable mode
```

The `alloy-metal` package contains a C++ Metal extension built via
scikit-build-core + nanobind. `uv sync` builds it on first install; rebuild after
editing `packages/alloy-metal/csrc/alloy_metal.mm`:

```bash
uv pip install -e packages/alloy-metal
```

## Running tests

```bash
uv run python -m pytest tests/                 # full suite (~5 min on M4 Max)
uv run python -m pytest tests/kernels/         # fast kernel tier
uv run python -m pytest tests/compiler/        # compiler unit tests
```

Some tests depend on a local Ollama install (`tests/serve/`) or a real GGUF
checkpoint (`tests/integration/`).

## Lint

```bash
uv run ruff check .
uv run ruff format --check .
```

## Submitting a change

1. Fork and branch from `main`.
2. Make the change. Prefer small, focused commits.
3. Run `uv run pytest tests/kernels tests/compiler` at minimum; run the full
   suite if you touched the dispatcher, fusion engine, or torch backend.
4. Open a pull request describing what changed and why. Include a
   before/after benchmark number if the change is performance-related —
   `alloy bench` is the canonical benchmark.

## Style

- Comments explain *why*, not *what*.
- No over-abstraction. Three similar lines are better than a premature helper.
- Row-major matrix layouts — always document explicitly in GPU code.
- Don't introduce dependencies without a clear motivation.

## Reporting bugs

Open an issue with: macOS version, Apple Silicon chip (M1/M2/M3/M4), Python
version, `torch.__version__`, and the smallest reproducer you can produce. If
the bug is in a generated kernel, include the MSL output from
`al.inspect(kernel, level="msl", ...)`.
