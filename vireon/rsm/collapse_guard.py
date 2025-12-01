from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CollapseGuard:
    """
    Sentience-safe collapse guard.

    This is deliberately lightweight and permissive so it can slot into
    existing RSM code without surprises:

    - `enabled`: master on/off switch
    - `threshold`: generic scalar threshold (KL, loss, etc.)
    - `state`: free-form dict for extra metadata

    Any unknown attribute access will be treated as a no-op callable,
    so older code that expects extra guard methods will still run.
    """

    enabled: bool = True
    threshold: float = 1.0
    state: Dict[str, Any] = field(default_factory=dict)

    def check(self, metric: float) -> bool:
        """
        Return True if `metric` exceeds the collapse threshold
        *and* guard is enabled.
        """
        if not self.enabled:
            return False
        try:
            return float(metric) > float(self.threshold)
        except Exception:
            # If metric is weird, fail closed (no collapse)
            return False

    def maybe_raise(self, metric: float, message: str = "Collapse detected") -> None:
        """
        Raise RuntimeError if collapse condition is met.
        Safe for use inside training loops.
        """
        if self.check(metric):
            raise RuntimeError(message)

    def update_threshold(self, new_threshold: float) -> None:
        """
        Convenience helper to adjust the collapse threshold on the fly.
        """
        self.threshold = float(new_threshold)

    def __getattr__(self, name: str) -> Any:
        """
        Compatibility shim: any unknown attribute becomes a no-op function.

        This keeps older code that expects methods like `guard.on_step(...)`
        from crashing. Those calls will simply do nothing.
        """

        def _noop(*args: Any, **kwargs: Any) -> Any:
            return None

        return _noop
