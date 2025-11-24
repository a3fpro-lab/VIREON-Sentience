"""
VIREON metrics / instrumentation layer.

Standardizes evidence logging across all benches:
- TRP signals: dt_eff, KL, divergence, alpha
- RSM signals: KL_mirror, G_self, pressure
- Run metadata: bench name, seed, variant, timestamps

All logs are JSONL (one dict per line) for easy audit + plotting.
"""

from .logger import JSONLLogger
from .trp_metrics import TRPMetrics
from .rsm_metrics import RSMMetrics
from .run_context import RunContext

__all__ = ["JSONLLogger", "TRPMetrics", "RSMMetrics", "RunContext"]
