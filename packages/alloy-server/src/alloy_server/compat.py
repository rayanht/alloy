"""Import-time transformers suppression, run before any model class loads.

The package `__init__` imports this module ahead of `alloy_server.models`, so the
env + probe patches below take effect before transformers configures its logger
or a model class triggers the optional sklearn/scipy imports.
"""

from __future__ import annotations

import os


def _apply() -> None:
    # Set the verbosity BEFORE transformers is imported: it reads
    # TRANSFORMERS_VERBOSITY when it lazily configures its root logger on first
    # use, so the env must win that race. `setdefault` keeps a user override.
    # Silences the benign load noise alloy never acts on (torchvision-less
    # image-processor fallbacks — vision routes through our kernels — and the
    # fla "fast path not available" notice — alloy compiles its own DeltaNet).
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    # transformers gates several optional imports on `is_sklearn_available` /
    # `is_scipy_available`: candidate_generator eagerly does `from sklearn.metrics
    # import roc_curve`, and the object-detection loss modules (pulled by
    # modeling_utils) do `from scipy.optimize import linear_sum_assignment`. Both
    # fire the moment a model class loads, dragging sklearn + scipy (~0.4s) into
    # every import. alloy only runs causal-LM / embedding / vision / audio paths,
    # never HF assisted generation or detection models — force both probes False.
    # Patch the source and the transformers.utils re-export (callers bind by name).
    import transformers.utils.import_utils  # scoped: must follow the env set above so verbosity wins transformers' lazy logger config

    transformers.utils.import_utils.is_sklearn_available = lambda: False
    transformers.utils.import_utils.is_scipy_available = lambda: False
    transformers.utils.is_sklearn_available = lambda: False
    transformers.utils.is_scipy_available = lambda: False


_apply()
