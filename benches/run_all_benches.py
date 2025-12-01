"""
Smoke tests for the one-button clinical suite runner.

Goal: verify that
- benches.run_all_benches imports cleanly
- (optionally) exposes a callable entrypoint

We keep this light so CI stays fast and stable.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def test_run_all_benches_module_imports() -> None:
    """Module must import without raising (wires + dependencies OK)."""
    module = importlib.import_module("benches.run_all_benches")
    assert isinstance(module, ModuleType)


def test_run_all_benches_has_entrypoint() -> None:
    """
    If the module defines a `run_all` function, it must be callable.

    This is a soft contract: if you haven't added `run_all` yet,
    the test will simply skip that assertion.
    """
    module = importlib.import_module("benches.run_all_benches")
    run_all = getattr(module, "run_all", None)
    if run_all is None:
        # No strong contract yet; just ensure we didn't crash above.
        return
    assert callable(run_all)
