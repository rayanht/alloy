"""Apple Silicon chip detection for static config lookup."""

from __future__ import annotations

from alloy._runtime.metal import default_device

_CHIP_TABLE: dict[str, str] = {
    "Apple M1": "apple6",
    "Apple M1 Pro": "apple6_pro",
    "Apple M1 Max": "apple6_max",
    "Apple M1 Ultra": "apple6_ultra",
    "Apple M2": "apple7",
    "Apple M2 Pro": "apple7_pro",
    "Apple M2 Max": "apple7_max",
    "Apple M2 Ultra": "apple7_ultra",
    "Apple M3": "apple8",
    "Apple M3 Pro": "apple8_pro",
    "Apple M3 Max": "apple8_max",
    "Apple M3 Ultra": "apple8_ultra",
    "Apple M4": "apple9",
    "Apple M4 Pro": "apple9_pro",
    "Apple M4 Max": "apple9_max",
}

_cached_device: str | None = None


def detect_device() -> str:
    """Return canonical device name like 'apple9_max'.

    Uses Metal device name (e.g. 'Apple M4 Max') and maps through _CHIP_TABLE.
    Falls back to gpu_family (e.g. 'apple9') if the exact chip isn't in the table.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    dev = default_device()
    chip_name = dev.name  # e.g. "Apple M4 Max"

    canonical = _CHIP_TABLE.get(chip_name)
    if canonical is None:
        # Unknown chip — fall back to gpu_family
        canonical = dev.gpu_family  # e.g. "apple9"

    _cached_device = canonical
    return canonical
