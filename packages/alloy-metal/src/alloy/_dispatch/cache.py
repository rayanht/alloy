"""Unified cache for all Alloy compilation and dispatch artifacts.

Consolidates:
  L1: MSL source (disk)          — keyed by kernel source + config + emitter digest
  L2: Compiled pipelines (memory) — keyed by MSL hash + device
  L3: Trace results (memory)      — keyed by kernel source + constexprs + shapes
  L4: Dispatch results (memory)   — opaque (single-op) and fused (multi-op)

Lifetime management:
  clear()          — reset everything (disk + memory), used between test runs
  clear_dispatch() — reset only fused_cache (L4), used between forward passes
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import dataclass

from alloy.log import get_logger

from alloy._runtime.metal import CompiledKernel

if TYPE_CHECKING:
    from alloy._compiler.tile_ir import TileFunction

# Alloy version — included in all cache keys
ALLOY_VERSION = "0.65.0"
CACHE_FORMAT_VERSION = "4"

Grid3D = tuple[int, int, int]

logger = get_logger("alloy.compiler")


def _emitter_source_digest() -> str:
    """SHA-256 over the codegen source tree (``alloy/_compiler/``).

    Folded into every :class:`KernelKey` so that ANY change to the compiler —
    the MSL emitter (``_compiler/msl/``), the planner (``tile_plan``), the IR
    opt passes (``tile_opt``), the tracer (``trace/``) — automatically
    invalidates the on-disk L1 MSL cache. Without it the key is derived from the
    *kernel's* Python source + config only, so editing the emitter is silently
    shadowed by stale cached ``.metal`` files: every dispatch keeps running the
    old codegen until the cache dir is hand-cleared. Hashing only ``msl/`` would
    miss planner/opt-pass edits that change the emitted MSL just as much, so we
    cover the whole ``_compiler/`` tree — over-invalidation is a grid-shrink
    recompile, under-invalidation is a silent-staleness bug. Computed once at
    import; the walk is a few dozen small files, sorted for a deterministic
    digest.
    """
    compiler_dir = Path(__file__).resolve().parent.parent / "_compiler"
    h = hashlib.sha256()
    for path in sorted(compiler_dir.rglob("*.py")):
        h.update(path.relative_to(compiler_dir).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


# Digest of the codegen source. Changes to the emitter/planner/IR passes bust
# the disk MSL cache automatically (computed once at import).
_EMITTER_DIGEST = _emitter_source_digest()

@dataclass(frozen=True, slots=True)
class CachedFusionEntry:
    """Cached launch metadata plus the op buffer slots to rebind."""

    pso_handle: int
    grid: Grid3D
    threadgroup: Grid3D
    buffer_slots: tuple[tuple[int, int], ...]
    op_indices: frozenset[int]
    write_indices: frozenset[int]
    debug_name: str


def _compute_hash(*parts: str) -> str:
    """SHA-256 hash of concatenated parts."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
    return h.hexdigest()


def msl_hash(msl_source: str) -> str:
    """Compute the cache key hash for an MSL source string."""
    return _compute_hash(msl_source)


class KernelKey:
    """Unique identifier for a kernel compilation.

    Composed of:
    - kernel_source: The Python source code of the kernel function
    - config: Tune configuration dict (or empty dict)
    - alloy_version: Version string for cache invalidation
    - emitter digest: SHA-256 of the codegen tree, so emitter/planner edits
      invalidate the cache automatically (see `_emitter_source_digest`)
    """

    __slots__ = ("kernel_source", "config", "alloy_version", "cache_format_version")

    def __init__(self, kernel_source: str, config: dict[str, int | float | bool | str | tuple[int, ...] | dict[str, str]] | None = None) -> None:
        self.kernel_source = kernel_source
        self.config: dict[str, int | float | bool | str | tuple[int, ...] | dict[str, str]] = config or {}
        self.alloy_version = ALLOY_VERSION
        self.cache_format_version = CACHE_FORMAT_VERSION

    @property
    def hash(self) -> str:
        """Compute the cache key hash.

        Deterministic: config is serialized with sorted keys so that
        ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` produce the same hash.
        """
        config_json = json.dumps(self.config, sort_keys=True, separators=(",", ":"))
        return _compute_hash(
            self.kernel_source,
            config_json,
            self.alloy_version,
            self.cache_format_version,
            _EMITTER_DIGEST,
        )


class CacheManager:
    """Unified cache for all Alloy compilation and dispatch artifacts.

    Singleton with test override: CacheManager.default() for production,
    CacheManager.set_default(mock) for testing.
    """

    __slots__ = (
        "_cache_dir",
        "_msl_dir",
        "_pipeline_cache",
        "trace_cache",
        "opaque_cache",
        "fused_cache",
        "_lock",
        "_l1_hits",
        "_l1_misses",
        "_l2_hits",
        "_l2_misses",
    )

    _default: CacheManager | None = None
    _default_lock: threading.Lock = threading.Lock()

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "alloy"
        self._cache_dir = Path(cache_dir)
        version_dir = self._cache_dir / f"v{ALLOY_VERSION}-cf{CACHE_FORMAT_VERSION}"
        self._msl_dir = version_dir / "msl"
        _emit_version_mismatch_warning(self._cache_dir, version_dir)

        # L2: Compiled pipelines (memory)
        self._pipeline_cache: dict[tuple[str, str], CompiledKernel] = {}

        # L3: Trace results (memory)
        self.trace_cache: dict[
            tuple[object, ...], tuple[TileFunction, Grid3D, tuple[int, ...] | None]
        ] = {}

        # L4: Dispatch results (memory)
        self.opaque_cache: dict[
            tuple[object, ...], tuple[CompiledKernel, Grid3D, Grid3D]
        ] = {}
        self.fused_cache: dict[tuple[object, ...], list[CachedFusionEntry]] = {}

        # Thread safety (for L1/L2 — L3/L4 are single-threaded dispatch path)
        self._lock = threading.Lock()

        # Statistics
        self._l1_hits: int = 0
        self._l1_misses: int = 0
        self._l2_hits: int = 0
        self._l2_misses: int = 0

    # --- L1: MSL source (disk) ---

    def _msl_path(self, kernel_key: KernelKey) -> Path:
        return self._msl_dir / f"{kernel_key.hash}.metal"

    def get_msl(self, kernel_key: KernelKey) -> str | None:
        """Look up cached MSL source code. Returns None on miss."""
        path = self._msl_path(kernel_key)
        try:
            data = path.read_text(encoding="utf-8")
            if not data:
                path.unlink(missing_ok=True)
                with self._lock:
                    self._l1_misses += 1
                return None
            with self._lock:
                self._l1_hits += 1
            return data
        except (OSError, UnicodeDecodeError):
            with self._lock:
                self._l1_misses += 1
            return None

    def put_msl(self, kernel_key: KernelKey, msl_source: str) -> None:
        """Store MSL source code in the disk cache (atomic write)."""
        self._msl_dir.mkdir(parents=True, exist_ok=True)
        dest = self._msl_path(kernel_key)
        fd, tmp_path = tempfile.mkstemp(dir=str(self._msl_dir), suffix=".tmp", prefix=".cache_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(msl_source)
            os.replace(tmp_path, str(dest))
        except OSError as exc:
            logger.error(
                "l1_cache_write_failed", path=str(dest), errno=exc.errno, error=str(exc),
            )
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # --- L2: Compiled pipelines (memory) ---

    def get_pipeline(self, msl_hash_key: str, device_key: str) -> CompiledKernel | None:
        """Look up a compiled pipeline. Returns None on miss."""
        with self._lock:
            result = self._pipeline_cache.get((msl_hash_key, device_key))
            if result is not None:
                self._l2_hits += 1
            else:
                self._l2_misses += 1
            return result

    def put_pipeline(self, msl_hash_key: str, device_key: str, pipeline: CompiledKernel) -> None:
        """Store a compiled pipeline in the in-memory L2 cache."""
        with self._lock:
            self._pipeline_cache[(msl_hash_key, device_key)] = pipeline

    # --- Lifetime management ---

    def clear(self) -> None:
        """Reset all caches (disk + memory). Called between test runs."""
        with self._lock:
            self._pipeline_cache.clear()
            self._l1_hits = 0
            self._l1_misses = 0
            self._l2_hits = 0
            self._l2_misses = 0
        self.trace_cache.clear()
        self.opaque_cache.clear()
        self.fused_cache.clear()
        if self._msl_dir.exists():
            shutil.rmtree(self._msl_dir, ignore_errors=True)

    def clear_dispatch(self) -> None:
        """Clear the per-batch dispatch caches (L4 opaque + fused) between
        dispatch batches.

        The L3 `trace_cache` is intentionally NOT cleared here. Unlike the
        opaque/fused caches — which are keyed by op identity / buffer id, both
        recycled across batches and therefore stale-prone — the trace cache is
        keyed purely by value: (contract version, kernel source, constexprs,
        dtypes, shapes, grid). It holds the traced IR, a pure function of that
        key with no buffer references, so persisting it across batches is both
        safe and necessary: clearing it forced every kernel to re-trace once
        per layer during compile (~1.2k traces for ~94 unique kernels)."""
        self.opaque_cache.clear()
        self.fused_cache.clear()

    # --- Statistics ---

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "l1_hits": self._l1_hits,
                "l1_misses": self._l1_misses,
                "l2_hits": self._l2_hits,
                "l2_misses": self._l2_misses,
                "compilations_avoided": self._l1_hits + self._l2_hits,
                "trace_cache_size": len(self.trace_cache),
                "opaque_cache_size": len(self.opaque_cache),
                "fused_cache_size": len(self.fused_cache),
                "pipeline_cache_size": len(self._pipeline_cache),
            }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"CacheManager("
            f"l1={s['l1_hits']}/{s['l1_hits'] + s['l1_misses']}, "
            f"l2={s['l2_hits']}/{s['l2_hits'] + s['l2_misses']}, "
            f"dir={self._msl_dir})"
        )

    # --- Singleton ---

    @classmethod
    def default(cls) -> CacheManager:
        """Get or create the default CacheManager instance. Thread-safe."""
        if cls._default is None:
            with cls._default_lock:
                if cls._default is None:
                    cls._default = CacheManager()
                    # Operator/debug feature only — off by default so importing
                    # alloy as a library doesn't print a stats line at exit.
                    if os.environ.get("ALLOY_CACHE_STATS"):
                        atexit.register(_emit_cache_stats_snapshot, cls._default)
        return cls._default

    @classmethod
    def set_default(cls, instance: CacheManager) -> None:
        """Override the default instance (for testing)."""
        cls._default = instance


def _emit_version_mismatch_warning(cache_dir: Path, current_version_dir: Path) -> None:
    """Warn once when we find stale v*-cf* subdirs from previous alloy
    releases. Those are now orphan caches taking disk space."""
    if not cache_dir.is_dir():
        return
    stale = [
        p for p in cache_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and "-cf" in p.name
        and p != current_version_dir
    ]
    if stale:
        logger.warning(
            "cache_format_version_mismatch",
            current=current_version_dir.name,
            stale=[p.name for p in stale],
            cache_dir=str(cache_dir),
        )


def _emit_cache_stats_snapshot(manager: CacheManager) -> None:
    """atexit hook: dump final L1-L4 hit/miss stats so an operator can
    see the cache effectiveness for this process at a glance."""
    try:
        snapshot = manager.stats()
    except Exception:
        return
    logger.info("cache_stats_snapshot", **snapshot)
