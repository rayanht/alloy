"""Tool-call parsing and emission, dispatched by model family.

Each family emits a different tool-call syntax (Hermes JSON, qwen3.5 XML,
deepseek special-token blocks, gemma4 `call:name{...}`); `extract_tool_calls`
detects the family and parses to structured `ToolCall`s. The opener markers and
`find_tool_opener` are also used by the streaming path to detect a call anywhere
in the content. No HTTP — the dialects render the parsed calls per wire shape.
"""

from __future__ import annotations

import ast
import json
import re
import uuid

from alloy_server.schema import ToolCall

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# qwen3.5 XML tool-call body (inside <tool_call> tags): <function=name>...
# <parameter=key>\nvalue\n</parameter>...</function>. NOT Hermes JSON.
QWEN_XML_FN_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
QWEN_XML_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
# deepseek-r1: <｜tool▁call▁begin｜>type<｜tool▁sep｜>name\n```json\n{args}\n```<｜tool▁call▁end｜>
# (the bars/underscores are the fullwidth U+FF5C / U+2581 deepseek control glyphs).
DEEPSEEK_BEGIN = "<｜tool▁calls▁begin｜>"
DEEPSEEK_CALL_RE = re.compile(
    "<｜tool▁call▁begin｜>[^\n]*?<｜tool▁sep｜>([^\n]+)\n```(?:json)?\n(.*?)\n```",
    re.DOTALL,
)
# gemma4: <|tool_call>call:name{gemma-args}<tool_call|> where strings are <|"|>..<|"|>.
GEMMA_CALL_MARKER = "<|tool_call>call:"
GEMMA_CALL_END = "<tool_call|>"
GEMMA_STR_DELIM = '<|"|>'
# LFM2: <|tool_call_start|>[fn(arg=val, ...), ...]<|tool_call_end|> — a pythonic
# list of function calls (kwargs syntax), parsed with `ast`.
LFM2_CALL_START = "<|tool_call_start|>"
LFM2_CALL_END = "<|tool_call_end|>"


def coerce_tool_json(blob: str) -> dict | None:
    """Parse one tool-call object, tolerating the formats small Qwen GGUFs emit:
    clean JSON, or the doubled-brace `{{...}}` that some copy from the template's
    literal example. Must have a 'name'; 'arguments' is normalized to a dict."""
    blob = blob.strip()
    candidates = [blob]
    if blob.startswith("{{") and blob.endswith("}}"):
        candidates.append(blob[1:-1])  # strip one doubled brace layer
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}
    return None


def coerce_scalar(raw: str) -> object:
    """A tool-arg value emitted as text: JSON-decode it so numbers/bools/objects/
    arrays parse to their real types, else keep the raw string (the common case for
    free-text args, which aren't valid JSON on their own)."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def parse_hermes_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """Hermes `<tool_call>{json}</tool_call>` blocks (qwen2.5/qwen3) or a bare
    tool-call JSON object as the whole output (llama3.2 `{"name","parameters"}`)."""
    blocks = TOOL_CALL_RE.findall(text)
    raw_objs: list[str] = []
    content = text
    if blocks:
        raw_objs = blocks
        content = TOOL_CALL_RE.sub("", text).strip()
    else:
        stripped = text.strip()
        if stripped.startswith("{") and coerce_tool_json(stripped) is not None:
            raw_objs = [stripped]
            content = ""
    pairs = [(obj["name"], obj["arguments"])
             for raw in raw_objs if (obj := coerce_tool_json(raw)) is not None]
    return content, pairs


def parse_qwen_xml_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """qwen3.5 emits XML tool calls wrapped in <tool_call> tags:
    `<tool_call><function=name><parameter=k>\\nv\\n</parameter>...</function></tool_call>`."""
    pairs: list[tuple[str, dict]] = []
    for block in TOOL_CALL_RE.findall(text):
        for fn in QWEN_XML_FN_RE.finditer(block):
            args = {p.group(1).strip(): coerce_scalar(p.group(2))
                    for p in QWEN_XML_PARAM_RE.finditer(fn.group(2))}
            pairs.append((fn.group(1).strip(), args))
    return TOOL_CALL_RE.sub("", text).strip(), pairs


def parse_deepseek_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """deepseek-r1: `<｜tool▁call▁begin｜>type<｜tool▁sep｜>name\\n```json\\n{...}\\n``` `."""
    pairs: list[tuple[str, dict]] = []
    for m in DEEPSEEK_CALL_RE.finditer(text):
        try:
            args = json.loads(m.group(2))
        except (json.JSONDecodeError, ValueError):
            args = {}
        pairs.append((m.group(1).strip(), args if isinstance(args, dict) else {}))
    return text.split(DEEPSEEK_BEGIN)[0].strip(), pairs


def coerce_gemma_scalar(raw: str) -> object:
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_gemma4_value(s: str, i: int) -> tuple[object, int]:
    """Parse one gemma4 arg value at index `i`; return (value, next_index). gemma4's
    format: strings are `<|"|>..<|"|>`, objects `{k:v,..}`, arrays `[v,..]`, and bare
    numbers/bools/null. Keys are unquoted at every level (template escape_keys=False)."""
    n = len(s)
    while i < n and s[i] in " \n\t":
        i += 1
    if s.startswith(GEMMA_STR_DELIM, i):
        i += len(GEMMA_STR_DELIM)
        end = s.find(GEMMA_STR_DELIM, i)
        if end < 0:
            end = n
        return s[i:end], end + len(GEMMA_STR_DELIM)
    if i < n and s[i] == "{":
        obj: dict = {}
        i += 1
        while i < n and s[i] != "}":
            while i < n and s[i] in " \n\t,":
                i += 1
            if i >= n or s[i] == "}":
                break
            if s.startswith(GEMMA_STR_DELIM, i):
                key, i = parse_gemma4_value(s, i)
            else:
                colon = s.find(":", i)
                if colon < 0:
                    break
                key, i = s[i:colon].strip(), colon
            while i < n and s[i] in " \n\t":
                i += 1
            if i < n and s[i] == ":":
                i += 1
            value, i = parse_gemma4_value(s, i)
            obj[str(key)] = value
        return obj, i + 1
    if i < n and s[i] == "[":
        arr: list = []
        i += 1
        while i < n and s[i] != "]":
            while i < n and s[i] in " \n\t,":
                i += 1
            if i >= n or s[i] == "]":
                break
            value, i = parse_gemma4_value(s, i)
            arr.append(value)
        return arr, i + 1
    j = i
    while j < n and s[j] not in ",}]":
        j += 1
    return coerce_gemma_scalar(s[i:j].strip()), j


def parse_gemma4_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """gemma4: `<|tool_call>call:name{gemma-args}<tool_call|>` (custom, nestable)."""
    pairs: list[tuple[str, dict]] = []
    spans: list[tuple[int, int]] = []
    idx = 0
    while (start := text.find(GEMMA_CALL_MARKER, idx)) >= 0:
        body = start + len(GEMMA_CALL_MARKER)
        brace = text.find("{", body)
        if brace < 0:
            break
        name = text[body:brace].strip()
        value, end = parse_gemma4_value(text, brace)
        close = text.find(GEMMA_CALL_END, end)
        span_end = close + len(GEMMA_CALL_END) if close >= 0 else end
        pairs.append((name, value if isinstance(value, dict) else {}))
        spans.append((start, span_end))
        idx = span_end
    content = text
    for s0, s1 in reversed(spans):
        content = content[:s0] + content[s1:]
    return content.strip(), pairs


def _lfm2_call_name(func: ast.expr) -> str | None:
    """Function name from an ast call target — `fn` or dotted `mod.fn`."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


_LFM2_NAME_CONSTS = {
    "true": True, "false": False, "null": None,  # JSON-style bare literals
    "True": True, "False": False, "None": None,  # Python-style
}


def _lfm2_literal(node: ast.expr) -> object:
    """Evaluate an arg-value node, accepting both Python (`True`) and JSON-style
    (`true`/`null`) bare literals — LFM2's pythonic calls mix the two."""
    if isinstance(node, ast.Name):
        return _LFM2_NAME_CONSTS.get(node.id, node.id)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_lfm2_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_lfm2_literal(k): _lfm2_literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _lfm2_literal(node.operand)
        return -v if isinstance(v, (int, float)) else v
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def parse_lfm2_inner(inner: str) -> list[tuple[str, dict]]:
    """Parse the pythonic call list inside the LFM2 markers: `[fn(k=v), ...]` (or a
    bare `fn(k=v)`). Values are Python/JSON literals; non-literal args become None."""
    inner = inner.strip()
    if not inner:
        return []
    try:
        body = ast.parse(inner, mode="eval").body
    except SyntaxError:
        return []
    nodes = body.elts if isinstance(body, ast.List) else [body]
    pairs: list[tuple[str, dict]] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _lfm2_call_name(node.func)
        if name is None:
            continue
        args = {kw.arg: _lfm2_literal(kw.value) for kw in node.keywords if kw.arg is not None}
        pairs.append((name, args))
    return pairs


def parse_lfm2_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """LFM2: `<|tool_call_start|>[fn(arg=val, ...)]<|tool_call_end|>` — pythonic
    function calls. The call block is stripped from the returned content."""
    pairs: list[tuple[str, dict]] = []
    spans: list[tuple[int, int]] = []
    idx = 0
    while (start := text.find(LFM2_CALL_START, idx)) >= 0:
        body = start + len(LFM2_CALL_START)
        close = text.find(LFM2_CALL_END, body)
        inner = text[body:close] if close >= 0 else text[body:]
        span_end = close + len(LFM2_CALL_END) if close >= 0 else len(text)
        pairs.extend(parse_lfm2_inner(inner))
        spans.append((start, span_end))
        idx = span_end
    content = text
    for s0, s1 in reversed(spans):
        content = content[:s0] + content[s1:]
    return content.strip(), pairs


# Markers that open a tool-call block, used by the STREAMING path to detect a
# call anywhere in the content (models routinely emit prose before the call —
# qwen3.5 especially). Bare-JSON calls (llama3.2) have no marker and are only
# detectable at content start.
TOOL_CALL_OPENERS: tuple[str, ...] = (
    "<tool_call>",            # Hermes JSON + qwen3.5 XML wrapper
    DEEPSEEK_BEGIN,           # deepseek <｜tool▁calls▁begin｜>
    "<｜tool▁call▁begin｜>",  # deepseek single call without the outer wrapper
    "<|tool_call>",           # gemma4
    LFM2_CALL_START,          # lfm2 <|tool_call_start|>
)


def find_tool_opener(text: str) -> int:
    """Index of the earliest tool-call opener in `text`, or -1."""
    best = -1
    for marker in TOOL_CALL_OPENERS:
        i = text.find(marker)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def extract_tool_calls(text: str, active: bool) -> tuple[str, tuple[ToolCall, ...]]:
    """Split generated text into (assistant_content, tool_calls), only when `active`
    (tools were supplied). Each model family emits a different tool-call syntax, so
    dispatch on a distinctive marker: deepseek special-token blocks, gemma4
    `call:name{...}`, qwen3.5 XML `<function=..>`, else Hermes/bare JSON (qwen2.5/
    qwen3/llama3.2)."""
    if not active or not text:
        return text, ()
    if DEEPSEEK_BEGIN in text:
        content, pairs = parse_deepseek_calls(text)
    elif GEMMA_CALL_MARKER in text:
        content, pairs = parse_gemma4_calls(text)
    elif LFM2_CALL_START in text:
        content, pairs = parse_lfm2_calls(text)
    elif "<function=" in text and "<tool_call>" in text:
        content, pairs = parse_qwen_xml_calls(text)
    else:
        content, pairs = parse_hermes_calls(text)
    calls = tuple(
        ToolCall(id=f"call_{uuid.uuid4().hex[:24]}", name=name, arguments=args)
        for name, args in pairs
    )
    return content, calls


def openai_tool_calls(calls: tuple[ToolCall, ...]) -> list[dict]:
    """OpenAI shape: arguments serialized to a JSON string, with an id + type."""
    return [
        {"id": c.id, "type": "function",
         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
        for c in calls
    ]


def ollama_tool_calls(calls: tuple[ToolCall, ...]) -> list[dict]:
    """Ollama shape: arguments as an object, no id/type."""
    return [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]
