# alloy-kit

Local LLM inference and GPU kernels for Apple Silicon.

```bash
pip install alloy-kit            # kernel compiler/runtime (no torch)
pip install 'alloy-kit[serve]'   # + OpenAI/Ollama server, torch.compile backend, CLI
pip install 'alloy-kit[all]'     # + training / vision / audio
```

The distribution is `alloy-kit`; it provides `import alloy` and the `alloy`
command. Requires Apple Silicon (M1+) and macOS 13+. See the
[project README](https://github.com/rayanht/alloy#readme) for usage.
