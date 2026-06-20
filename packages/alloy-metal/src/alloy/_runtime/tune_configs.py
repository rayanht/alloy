"""Static kernel config resolution — replaces runtime autotuning.

Loads pre-tuned configs from JSON files in alloy/configs/ and resolves
them at kernel dispatch time via O(1) dict lookup.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from itertools import product
from pathlib import Path
from typing import Any

from alloy.log import get_logger
from alloy._runtime.device import detect_device

logger = get_logger("alloy.tune")

# Pipeline version — bump when tile IR, codegen, or MSL emission changes.
# Configs generated with an older version are ignored.
_PIPELINE_VERSION = 6

# grid-shrunk chunk prefill compiles its plan at SEQ_LEN = M_MAX (the model's native
# context), but kernel tile configs are M-saturated: past a few thousand rows the
# optimal (BLOCK_M, BLOCK_N, BLOCK_K) for a GEMM/attention stops changing (the grid
# just gains more identical tiles). Measured on qwen3.5:0.8b, the M=16384 and
# M=262144 grid-shrink configs are identical for 3/4 GEMM shapes (the 4th differs only
# in BLOCK_N). So tuning at native M_MAX is wasted work — and intractable: the cold
# single-pass attention is O(M^2), ~1hr+/shape at 262144.
#
# Instead tune ONCE at this representative M — the production large-chunk prefill
# size — and, when a plan compiled at a larger M_MAX resolves configs, its M-scaled
# key fields cap down to it (`set_grid_shrink_resolve_cap` + `_apply_grid_shrink_cap`).
# A key field that is k*M_MAX (a position/sequence-scaled dim or stride, so
# `v % m_max == 0 and v >= m_max`) maps to k*REP_M; weight / reduction / output-width
# dims (all < M_MAX) are left untouched.
GRID_SHRINK_REP_M = 4096
_grid_shrink_cap = {"m_max": 0, "rep_m": 0}


def set_grid_shrink_resolve_cap(m_max: int, rep_m: int = GRID_SHRINK_REP_M) -> None:
    """Activate (m_max>0) / clear (m_max=0) the grid-shrink M-scaled key cap.

    Set around the shrink-capable plan compile (generation.eager_compile_all + grid-recipe
    discovery) with `m_max` = the SEQ_LEN that plan is compiled at. A no-op when
    `rep_m >= m_max` (native context already <= the representative M)."""
    _grid_shrink_cap["m_max"] = int(m_max)
    _grid_shrink_cap["rep_m"] = int(rep_m)


def _is_kv_cache_field(name: str) -> bool:
    """A KV-cache CONTEXT field: `KV_LEN`, or a K/V buffer dim/stride (`_K_dim*`,
    `_V_dim*`, `K_*`, `V_*`).

    All of these are sized by the NATIVE CONTEXT (the KV cache allocation), not the
    prompt length: `KV_LEN` is the K-scan extent constexpr = the full cache (the causal
    mask, not KV_LEN, bounds the actually-attended span per query); `_K_dim0` etc. are
    the K/V buffer dims. They are IDENTICAL in the representative-M tune (which uses the
    native cache) and the native deploy, so they must NOT be capped. `KV_GROUP` is not
    matched (it is < M_MAX, never capped regardless)."""
    return name == "KV_LEN" or name.startswith(("_K_", "_V_", "K_", "V_"))


def _apply_grid_shrink_cap(key_values: dict[str, int]) -> dict[str, int]:
    """Map each prompt-scaled k*M_MAX key field to k*REP_M (see GRID_SHRINK_REP_M).

    Caps SEQ_LEN, KV_LEN, and the M-scaled activation dims/strides (`_Q_*`, `_O_*`,
    `_A_*`, `_C_*`, `_x_*`, `Q_*`, ...). Leaves K/V cache-buffer fields (context-sized,
    see `_is_kv_cache_field`) and all sub-M_MAX dims (weights, reductions, widths,
    head_dim, ...) untouched. No-op when inactive."""
    m_max = _grid_shrink_cap["m_max"]
    rep = _grid_shrink_cap["rep_m"]
    if not m_max or not rep or rep >= m_max:
        return key_values
    return {
        k: (v // m_max) * rep
        if (v >= m_max and v % m_max == 0 and not _is_kv_cache_field(k))
        else v
        for k, v in key_values.items()
    }


# ---------------------------------------------------------------------------
# TuneConfig — used by @al.tunable decorator and al.tune()
# ---------------------------------------------------------------------------


@dataclass
class TuneConfig:
    """One candidate configuration to benchmark."""

    constexprs: dict[str, Any]
    threadgroup_size: tuple[int, int, int] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        parts = [f"{k}={v}" for k, v in self.constexprs.items()]
        for k, v in self.options.items():
            parts.append(f"{k}={v}")
        if self.threadgroup_size:
            parts.append(f"tg={self.threadgroup_size}")
        return f"Config({', '.join(parts)})"


def generate_configs(
    param_ranges: dict[str, list],
    threadgroup_sizes: list[tuple[int, int, int]] | None = None,
    option_ranges: dict[str, list] | None = None,
) -> list[TuneConfig]:
    """Generate cartesian product of param ranges x option ranges x threadgroup sizes."""
    param_names = sorted(param_ranges.keys())
    param_values = [param_ranges[k] for k in param_names]

    opt_names = sorted(option_ranges.keys()) if option_ranges else []
    opt_values = [option_ranges[k] for k in opt_names] if option_ranges else []

    all_names = param_names + opt_names
    all_values = param_values + opt_values

    configs = []
    for combo in product(*all_values):
        vals = dict(zip(all_names, combo))
        constexprs = {k: vals[k] for k in param_names}
        options = {k: vals[k] for k in opt_names}
        if threadgroup_sizes:
            for tg in threadgroup_sizes:
                configs.append(
                    TuneConfig(
                        constexprs=dict(constexprs), threadgroup_size=tg, options=dict(options)
                    )
                )
        else:
            configs.append(TuneConfig(constexprs=constexprs, options=options))
    return configs


# ---------------------------------------------------------------------------
# StaticConfig — resolved config at dispatch time
# ---------------------------------------------------------------------------

class StaticConfig:
    """Resolved kernel config: constexprs + compiler options."""

    __slots__ = ("constexprs", "options", "is_default")

    def __init__(self, constexprs: dict[str, Any], options: dict[str, Any] | None = None):
        self.constexprs = constexprs
        self.options = options or {}
        # True for _CONSERVATIVE_DEFAULTS entries (set below). Lets shape-aware
        # resolvers (attention's _legal_fallback_blocks) distinguish "tuned for
        # this shape" from "kernel-wide default": the attention default
        # (32, 64) busts the 32KB shmem budget at head_dim >= 256 — it must
        # never be trusted over the head_dim-aware fallback.
        self.is_default = False

    def __repr__(self) -> str:
        return f"StaticConfig({self.constexprs}, options={self.options})"


# ---------------------------------------------------------------------------
# Conservative defaults — always work, never optimal
# ---------------------------------------------------------------------------

_CONSERVATIVE_DEFAULTS: dict[str, StaticConfig] = {
    "dot": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "_reg": 1, "_matvec": 0},
        {"double_buffer": 0, "async_copy": 0},
    ),
    "dot_transpose_rhs": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "_reg": 1, "_matvec": 0},
        {"double_buffer": 0, "async_copy": 0},
    ),
    "dot_transpose_rhs_silu": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "_matvec": 0},
    ),
    "dot_dequant": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "_matvec": 0},
    ),
    "dot_dequant_silu": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "_matvec": 0},
    ),
    "softmax": StaticConfig({"BLOCK_SIZE": 256}),
    "layernorm": StaticConfig({"BLOCK_SIZE": 256}),
    "rms_norm": StaticConfig({"BLOCK_SIZE": 256}),
    "cross_entropy": StaticConfig({"BLOCK_SIZE": 256}),
    "attention": StaticConfig({"BLOCK_M": 32, "BLOCK_N": 64}, {"fuse_loops": 0}),
    "attention_masked_by_batch": StaticConfig({"BLOCK_M": 32, "BLOCK_N": 64}, {"fuse_loops": 0}),
    "attention_strided": StaticConfig({"BLOCK_M": 32, "BLOCK_N": 64}, {"fuse_loops": 0}),
    "attention_strided_masked_by_batch": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 64}, {"fuse_loops": 0}
    ),
    "attention_strided_logsumexp": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32}, {"fuse_loops": 0}
    ),
    "attention_strided_logsumexp_masked_by_batch": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32}, {"fuse_loops": 0}
    ),
    # Prefill attention used by the alloy serve path. Without a conservative
    # entry here the tuner's reference config falls back to a generic
    # BLOCK_K-bearing config that doesn't match this kernel's signature,
    # `_compute_reference` either fails to compile or runs the wrong code,
    # and the candidate sweep ends up validating against a poisoned reference.
    "attention_strided_masked_by_batch_with_lse": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32}, {"fuse_loops": 0}
    ),
    "attention_strided_with_lse": StaticConfig(
        {"BLOCK_M": 32, "BLOCK_N": 32}, {"fuse_loops": 0}
    ),
    # SDPA bwd: 8×8 is the only block size that fits f32 shmem at HEAD_DIM=128
    # (HIGH_PRECISION=1, Qwen-class K-bias models): 6 tiles × 16 × 128 × 4 = 49KB
    # overflows the 32KB threadgroup-memory budget at BM=16 BN=16. The tuner
    # uses this conservative config to compute its reference output for
    # correctness checking — if the reference fails to compile, the tuner
    # silently skips the correctness check and can pick a fast-but-wrong config
    # (e.g., the dq kernel produces incorrect output at BM=16 BN=8).
    "attention_strided_backward_dq": StaticConfig({"BLOCK_M": 8, "BLOCK_N": 8}, {"fuse_loops": 0}),
    "attention_strided_backward_dq_masked_by_batch": StaticConfig(
        {"BLOCK_M": 8, "BLOCK_N": 8}, {"fuse_loops": 0}
    ),
    "attention_strided_backward_dkdv": StaticConfig(
        {"BLOCK_M": 8, "BLOCK_N": 8}, {"fuse_loops": 0}
    ),
    "attention_strided_backward_dkdv_masked_by_batch": StaticConfig(
        {"BLOCK_M": 8, "BLOCK_N": 8}, {"fuse_loops": 0}
    ),
    "attention_kv_update": StaticConfig({"BLOCK_M": 8, "BLOCK_N": 8}, {"fuse_loops": 0}),
}
for _default_cfg in _CONSERVATIVE_DEFAULTS.values():
    _default_cfg.is_default = True

# ---------------------------------------------------------------------------
# Loaded config store
# ---------------------------------------------------------------------------

# kernel_name → {shape_key_tuple → StaticConfig}
_STATIC_CONFIGS: dict[str, dict[tuple, StaticConfig]] = {}
_loaded_device: str | None = None
_warned_keys: set[tuple[str, tuple]] = set()


# ---------------------------------------------------------------------------
# Config locations: shipped baseline (read-only, in the package) + user overlay
# ---------------------------------------------------------------------------

def shipped_configs_dir():
    """Read-only baseline configs that ship inside the alloy package."""
    return resources.files("alloy.configs")


def user_config_dir() -> Path:
    """User-writable dir where `alloy tune` writes, independent of the install.

    The package dir is read-only on a system install and wiped on every
    reinstall/upgrade, so tuned configs can't live there. Overridable via
    ALLOY_CONFIG_DIR; defaults to the XDG data dir — a *data* dir, not a cache:
    a tuned map costs minutes to regenerate and must outlive a cache clear.
    """
    override = os.environ.get("ALLOY_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "alloy" / "configs"


def user_config_file(device: str) -> Path:
    return user_config_dir() / f"{device}.json"


def _read_config_data(base_dir, device: str) -> dict | None:
    """Version-matched config data for `device` under `base_dir`, trying the
    exact device then its family (apple9_max → apple9). None if missing or
    stale — a stale shipped file must not block a current user overlay."""
    candidates = [device]
    if "_" in device:
        candidates.append(device.split("_")[0])
    for name in candidates:
        try:
            text = base_dir.joinpath(f"{name}.json").read_text(encoding="utf-8")
            data = json.loads(text)
        except (FileNotFoundError, TypeError, json.JSONDecodeError):
            continue
        if data.get("pipeline_version", 0) != _PIPELINE_VERSION:
            return None  # stale configs — ignore silently
        return data
    return None


def _apply_entries(data: dict) -> None:
    for kernel_name, entries in data.get("entries", {}).items():
        kernel_map = _STATIC_CONFIGS.setdefault(kernel_name, {})
        for entry in entries:
            key = tuple(sorted(entry["key"].items()))
            kernel_map[key] = StaticConfig(
                constexprs=entry["config"],
                options=entry.get("options", {}),
            )


def _load_configs(device: str | None = None) -> None:
    """Load configs for `device`: the package-shipped baseline first, then the
    user-tuned overlay on top, so a user's `alloy tune` wins over the baseline
    and survives reinstalls (the baseline ships in the package; the overlay
    lives in `user_config_dir()`)."""
    global _loaded_device
    if device is None:
        device = detect_device()
    if device == _loaded_device:
        return

    _STATIC_CONFIGS.clear()

    shipped = _read_config_data(shipped_configs_dir(), device)
    if shipped is not None:
        _apply_entries(shipped)

    user = _read_config_data(user_config_dir(), device)
    if user is not None:
        _apply_entries(user)

    _loaded_device = device


# ---------------------------------------------------------------------------
# Resolution (called from _kernel.py at dispatch time)
# ---------------------------------------------------------------------------

def _round_up(v: int, multiple: int) -> int:
    return ((v + multiple - 1) // multiple) * multiple


def resolve_config(
    kernel_name: str,
    key_values: dict[str, int],
) -> StaticConfig:
    """Look up a static config for (kernel, shape). O(1) dict lookup.

    Falls back to rounding dimensions to common tile boundaries, then to
    a conservative default with a warning.
    """
    _load_configs()

    key_values = _apply_grid_shrink_cap(key_values)
    exact_key = tuple(sorted(key_values.items()))
    entries = _STATIC_CONFIGS.get(kernel_name, {})

    # 1. Exact match
    if exact_key in entries:
        return entries[exact_key]

    # 2. Fallback: round dimensions to common tile boundaries
    for round_to in (64, 32, 16, 8):
        rounded = tuple((k, _round_up(v, round_to)) for k, v in sorted(key_values.items()))
        if rounded in entries:
            return entries[rounded]

    # 3. No match — conservative default + one-time warning per kernel
    default = _CONSERVATIVE_DEFAULTS.get(kernel_name)

    if default is None:
        return StaticConfig({})

    warn_key = (kernel_name, exact_key)
    if warn_key not in _warned_keys:
        _warned_keys.add(warn_key)
        dims = {k: v for k, v in key_values.items() if k.startswith("_") and "dim" in k}
        shape_str = ", ".join(f"{k}={v}" for k, v in sorted(dims.items())) if dims else str(key_values)
        logger.warning("runtime_config_miss", kernel=kernel_name, shape=shape_str)
    return default
