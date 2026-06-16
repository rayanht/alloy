"""OpenAI audio API rendering: `/v1/audio/transcriptions` + `/v1/audio/translations`.

Parses the multipart form into a transcribe() call and renders the result per
`response_format` (json / verbose_json → JSON; text / srt / vtt → a `Rendered`
text body). Transcription is OpenAI-only; the other dialects don't expose it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus

from alloy_server.dialects.wire import sse_data
from alloy_server.schema import JsonObject, Rendered, RequestError
from alloy_server.transcription import Segment, TranscriptionResult


def parse_float(value: object, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "temperature must be a number") from exc


def run_transcription(model: object, data: dict, *, task: str) -> JsonObject | Rendered:
    audio = data.get("file")
    if not isinstance(audio, (bytes, bytearray)):
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "missing 'file' upload")
    language = data.get("language") or None
    prompt = data.get("prompt") or None
    response_format = data.get("response_format") or "json"
    temperature = parse_float(data.get("temperature"), 0.0)
    result: TranscriptionResult = model.transcribe(  # type: ignore[attr-defined]
        bytes(audio), task=task, language=language, prompt=prompt, temperature=temperature,
    )
    return render_result(result, response_format)


def transcription_payload(model: object, data: dict) -> JsonObject | Rendered:
    return run_transcription(model, data, task="transcribe")


def translation_payload(model: object, data: dict) -> JsonObject | Rendered:
    return run_transcription(model, data, task="translate")


def wants_stream(data: dict) -> bool:
    return str(data.get("stream", "")).strip().lower() in ("true", "1", "yes")


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    model: object
    audio: bytes
    task: str
    language: str | None
    prompt: str | None
    temperature: float


def parse_request(model: object, data: dict, *, task: str) -> TranscriptionRequest:
    audio = data.get("file")
    if not isinstance(audio, (bytes, bytearray)):
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid_request", "missing 'file' upload")
    return TranscriptionRequest(
        model=model, audio=bytes(audio), task=task,
        language=data.get("language") or None,
        prompt=data.get("prompt") or None,
        temperature=parse_float(data.get("temperature"), 0.0),
    )


def parse_transcription(model: object, data: dict) -> TranscriptionRequest:
    return parse_request(model, data, task="transcribe")


def parse_translation(model: object, data: dict) -> TranscriptionRequest:
    return parse_request(model, data, task="translate")


def stream_transcription(request: TranscriptionRequest) -> Iterator[bytes]:
    """OpenAI transcription stream: transcript.text.delta frames, then a
    transcript.text.done frame with the full text."""
    stream_fn = request.model.stream_transcribe  # type: ignore[attr-defined]
    parts: list[str] = []
    for delta in stream_fn(
        request.audio, task=request.task, language=request.language,
        prompt=request.prompt, temperature=request.temperature,
    ):
        parts.append(delta)
        yield sse_data({"type": "transcript.text.delta", "delta": delta})
    yield sse_data({"type": "transcript.text.done", "text": "".join(parts).strip()})


def render_result(result: TranscriptionResult, response_format: str) -> JsonObject | Rendered:
    if response_format == "json":
        return {"text": result.text}
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": result.language,
            "duration": result.duration,
            "text": result.text,
            "segments": [segment_dict(s) for s in result.segments],
        }
    if response_format == "text":
        return Rendered("text/plain; charset=utf-8", result.text + "\n")
    if response_format == "srt":
        return Rendered("text/plain; charset=utf-8", to_srt(result.segments))
    if response_format == "vtt":
        return Rendered("text/vtt; charset=utf-8", to_vtt(result.segments))
    raise RequestError(
        HTTPStatus.BAD_REQUEST, "invalid_request",
        f"unsupported response_format {response_format!r} "
        "(expected one of json, verbose_json, text, srt, vtt)",
    )


def segment_dict(s: Segment) -> JsonObject:
    return {
        "id": s.id, "seek": s.seek, "start": s.start, "end": s.end, "text": s.text,
        "tokens": list(s.tokens), "temperature": s.temperature,
        "avg_logprob": s.avg_logprob, "compression_ratio": s.compression_ratio,
        "no_speech_prob": s.no_speech_prob,
    }


def clock(seconds: float, millis_sep: str) -> str:
    millis = int(round(seconds * 1000.0))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def to_srt(segments: tuple[Segment, ...]) -> str:
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{clock(s.start, ',')} --> {clock(s.end, ',')}")
        lines.append(s.text)
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: tuple[Segment, ...]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{clock(s.start, '.')} --> {clock(s.end, '.')}")
        lines.append(s.text)
        lines.append("")
    return "\n".join(lines)
