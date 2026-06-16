"""Structured logging for alloy.

    log = alloy.get_logger(__name__)
    log.info("compiled", model=name, n_plans=7, took_ms=68000)
    log.debug("warm_splice", hit=True, prefix_len=2048)

Two output channels render the same events:
  * **Console** — pretty (`structlog.dev.ConsoleRenderer`) on a TTY,
    JSON when piped. Goes to stderr. Tuned for the human watching the
    CLI / tail -f.
  * **File** — JSON, rotating (10 MB × 5 by default). Always JSON
    regardless of TTY, so the file is machine-greppable.

Config via env (read once at first import):

    ALLOY_LOG=<level>             global level (default: info)
    ALLOY_LOG_<SUBSYSTEM>=<level> per-logger override; e.g.
                                  ALLOY_LOG_FUSION=debug raises only
                                  the alloy.fusion logger to DEBUG
    ALLOY_LOG_FORMAT=pretty|json  console format
                                  (default: pretty on TTY, json otherwise)
    ALLOY_LOG_FILE=<path>         where the JSON log goes; default
                                  ~/Library/Logs/Alloy/alloy.log.
                                  Set ALLOY_LOG_FILE='' to disable.

`configure()` is idempotent — re-call to change level / format / paths
at runtime.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import TextIO

import structlog
from tqdm import tqdm


_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_SUBSYSTEM_LOGGERS = {
    # env-var suffix → fully-qualified logger name
    "DISPATCH": "alloy.dispatch",
    "FUSION": "alloy.fusion",
    "TUNE": "alloy.tune",
    "RUNTIME": "alloy.runtime",
    "COMPILER": "alloy.compiler",
    "BACKEND": "alloy_torch.backend",
    "GENERATION": "alloy_server.generation",
    "SERVER": "alloy_server",
    "GGUF": "alloy_server.gguf",
    "REWRITES": "alloy_torch.rewrites",
    "BENCH": "alloy_cli.bench",
}

_DEFAULT_LOG_FILE = Path("~/Library/Logs/Alloy/alloy.log").expanduser()
_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_FILE_BACKUP_COUNT = 5

_configured = False
_console_handler: logging.Handler | None = None
_file_handler: logging.Handler | None = None


def _level_from_env(default: int = logging.INFO) -> int:
    raw = os.environ.get("ALLOY_LOG", "").lower().strip()
    return _LEVELS.get(raw, default)


def _console_format_from_env(stream: TextIO) -> str:
    raw = os.environ.get("ALLOY_LOG_FORMAT", "auto").lower().strip()
    if raw in ("pretty", "json"):
        return raw
    # auto: pretty when stderr is a TTY, JSON otherwise (pipe, daemon log file)
    return "pretty" if stream.isatty() else "json"


def _file_path_from_env() -> Path | None:
    raw = os.environ.get("ALLOY_LOG_FILE")
    if raw is None:
        return _DEFAULT_LOG_FILE
    raw = raw.strip()
    if not raw:
        return None  # explicit opt-out
    return Path(raw).expanduser()


def _apply_subsystem_overrides() -> None:
    """Raise individual subsystem loggers above the global level when
    ALLOY_LOG_<SUBSYSTEM> is set (e.g. ALLOY_LOG_FUSION=debug)."""
    for env_suffix, logger_name in _SUBSYSTEM_LOGGERS.items():
        raw = os.environ.get(f"ALLOY_LOG_{env_suffix}", "").lower().strip()
        if raw in _LEVELS:
            logging.getLogger(logger_name).setLevel(_LEVELS[raw])


class AlloyOnlyFilter(logging.Filter):
    """Drop records that didn't originate from an alloy* logger.

    We attach to our handlers so foreign loggers (httpx, transformers,
    torch._dynamo, …) don't end up in the alloy log file even when the
    user has set root to INFO. Those foreign records still go to any
    handler the user installed via logging.basicConfig — we just don't
    capture them in ours.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("alloy")


class TqdmAwareStreamHandler(logging.StreamHandler):
    """Routes log output through `tqdm.write()`.

    Without this, structlog lines and tqdm progress bars interleave on
    stderr — tqdm's \\r-based redraw and structlog's \\n-terminated
    lines clash, leaving the bar smeared across multiple lines. tqdm.write
    clears active bars, writes the message, then redraws the bars below.
    """

    def emit(self, record: logging.LogRecord) -> None:
        stream = self.stream
        if stream is None or stream.closed:
            return
        try:
            msg = self.format(record)
            tqdm.write(msg, file=stream, end=self.terminator)
            self.flush()
        except (ValueError, OSError):
            return
        except Exception:
            self.handleError(record)


def _make_console_handler(stream: TextIO, fmt: str, pre_chain: list) -> logging.Handler:
    handler = TqdmAwareStreamHandler(stream)
    if fmt == "json":
        renderer: object = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=stream.isatty())
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=pre_chain,
    ))
    handler.addFilter(AlloyOnlyFilter())
    return handler


def _make_file_handler(path: Path, pre_chain: list) -> logging.Handler | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=_FILE_MAX_BYTES, backupCount=_FILE_BACKUP_COUNT,
        delay=True,  # don't open the file until the first record is emitted
    )
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=pre_chain,
    ))
    handler.addFilter(AlloyOnlyFilter())
    return handler


def configure(
    *,
    level: int | str | None = None,
    console_format: str | None = None,
    console_stream: TextIO | None = None,
    json_path: Path | str | None | bool = True,
) -> None:
    """Configure the structlog + stdlib logging pipeline. Idempotent.

    All args win over env vars when explicitly passed.

    `json_path`:
        True (default)  → resolve via ALLOY_LOG_FILE or _DEFAULT_LOG_FILE
        Path/str        → write JSON here
        None / False    → no file output
    """
    global _configured, _console_handler, _file_handler
    out = console_stream if console_stream is not None else sys.stderr

    if isinstance(level, str):
        resolved_level = _LEVELS.get(level.lower(), logging.INFO)
    elif isinstance(level, int):
        resolved_level = level
    else:
        resolved_level = _level_from_env()

    resolved_console_fmt = (
        console_format if console_format in ("pretty", "json") else _console_format_from_env(out)
    )

    if json_path is True:
        resolved_file_path: Path | None = _file_path_from_env()
    elif json_path in (None, False):
        resolved_file_path = None
    else:
        resolved_file_path = Path(json_path).expanduser()  # type: ignore[arg-type]

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # BoundLogger (not filtering_bound_logger) so per-logger stdlib levels
    # gate output; cache off so re-configure affects already-bound loggers.
    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    # Swap our handlers in place; leave foreign handlers untouched.
    root = logging.getLogger()
    root.setLevel(resolved_level)
    if _console_handler is not None:
        root.removeHandler(_console_handler)
    if _file_handler is not None:
        root.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None

    _console_handler = _make_console_handler(out, resolved_console_fmt, shared_processors)
    root.addHandler(_console_handler)

    if resolved_file_path is not None:
        _file_handler = _make_file_handler(resolved_file_path, shared_processors)
        if _file_handler is not None:
            root.addHandler(_file_handler)

    _apply_subsystem_overrides()
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger for `name`. Triggers one-time configure() on first call."""
    if not _configured:
        configure()
    return structlog.get_logger(name)


configure_logging = configure
