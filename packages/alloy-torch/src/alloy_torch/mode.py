"""Shared Alloy-Torch execution mode flags."""

from alloy._runtime.tune import register_training_mode_hooks

_training_mode_enabled: bool = False


def is_training_mode_enabled() -> bool:
    return _training_mode_enabled


def set_training_mode_enabled(mode: bool) -> None:
    global _training_mode_enabled
    _training_mode_enabled = mode


# The base package stays torch-free and can't import this module, so hand it the
# accessors at import time for the tuner's training-kernel sweeps.
register_training_mode_hooks(is_training_mode_enabled, set_training_mode_enabled)
