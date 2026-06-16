"""Shared pytest fixtures for Alloy tests."""

import pytest
from alloy._dispatch.dispatch import _engine


@pytest.fixture(autouse=True)
def _reset_alloy_caches():
    """Clear caches between tests."""
    _engine.clear()
    yield
