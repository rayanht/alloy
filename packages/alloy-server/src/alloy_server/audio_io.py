"""Decode arbitrary audio bytes/paths to a mono float32 waveform at a target sample
rate. ffmpeg (broad formats: mp3/m4a/webm) when present, else soundfile (wav/flac/ogg);
resampling via librosa. Mirrors the gemma4 audio decode policy."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_with_ffmpeg(data: bytes, target_sr: int) -> np.ndarray:
    """Decode to mono float32 PCM at target_sr via ffmpeg (reads stdin, writes stdout)."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
        "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "pipe:1",
    ]
    proc = subprocess.run(cmd, input=data, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def decode_with_soundfile(data: bytes, target_sr: int) -> np.ndarray:
    """Decode wav/flac/ogg via soundfile; downmix to mono; resample if needed."""
    wav, sr = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa  # scoped: heavy (numba) dep, only when resampling

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def load_audio(source: bytes | str | Path, target_sr: int = 16000) -> np.ndarray:
    """Bytes or a path → mono float32 waveform at target_sr. Tries soundfile first
    (wav/flac/ogg — an in-process libsndfile read, ~1 ms), falling back to ffmpeg only
    for what it can't decode (mp3/m4a/webm). ffmpeg is a subprocess spawn (~20 ms), so
    preferring it unconditionally taxed every wav request — the common serving case."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    try:
        return decode_with_soundfile(data, target_sr)
    except Exception:  # noqa: BLE001 — soundfile can't decode it (mp3/m4a/webm) → ffmpeg
        if have_ffmpeg():
            return decode_with_ffmpeg(data, target_sr)
        raise
