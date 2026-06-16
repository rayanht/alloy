"""Shared Alloy-Torch execution mode flags."""

from alloy._runtime.tune import register_training_mode_hooks

_training_mode_enabled: bool = False


def is_training_mode_enabled() -> bool:
    return _training_mode_enabled


def set_training_mode_enabled(mode: bool) -> None:
    global _training_mode_enabled
    _training_mode_enabled = mode


# The tuner toggles training mode around training-kernel sweeps; alloy can't
# import this module to reach the flag (the base package must stay torch-free),
# so we hand it the accessors at import time instead.
register_training_mode_hooks(is_training_mode_enabled, set_training_mode_enabled)
