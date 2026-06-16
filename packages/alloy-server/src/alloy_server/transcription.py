"""Dialect-agnostic speech-to-text served types. `TranscriptionModel` is the
abstraction the server's transcription modality dispatches to (mirrors
embedding.py:EmbeddingModel). The wire/dialect layer renders a TranscriptionResult
into json/text/srt/vtt/verbose_json."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Word:
    word: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: tuple[int, ...]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: tuple[Segment, ...] = ()
    words: tuple[Word, ...] = ()  # populated only when word timestamps are requested


# transcribe(audio_bytes, *, task, language, prompt, temperature, granularities)
TranscribeFn = Callable[..., TranscriptionResult]
# stream_transcribe(audio_bytes, *, task, language, prompt, temperature) -> text deltas
StreamTranscribeFn = Callable[..., "Iterator[str]"]


@dataclass(frozen=True, slots=True)
class TranscriptionModel:
    name: str
    transcribe: TranscribeFn
    stream_transcribe: StreamTranscribeFn | None = None
    languages: tuple[str, ...] = field(default_factory=tuple)
