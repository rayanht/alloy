# alloy-cli

Command-line interface for the Alloy local LLM server.

```bash
alloy serve -m qwen3:0.6b            # run the server in the foreground
alloy doctor                         # diagnose server health
```

Chat with it from any OpenAI/Ollama/Anthropic client:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"hello"}]}'
```

The server binds to loopback by default and exposes both Ollama-compatible
(`/api/*`) and OpenAI-compatible (`/v1/*`) endpoints. See the
[monorepo README](https://github.com/rayanht/alloy) for the full surface.

Requires macOS on Apple Silicon. Pulls in `alloy-torch` and `alloy-metal` as
dependencies.
