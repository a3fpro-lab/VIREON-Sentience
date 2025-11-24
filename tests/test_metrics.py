import os
import json
import tempfile

from vireon.metrics import JSONLLogger, TRPMetrics, RSMMetrics, RunContext


def test_jsonl_logger_writes_lines():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.jsonl")
        logger = JSONLLogger(path)
        logger.log({"a": 1})
        logger.log({"b": 2})
        logger.close()

        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        assert "a" in rec0 and "_ts_unix" in rec0


def test_trp_metrics_summary():
    m = TRPMetrics()
    m.update_step(dt_eff=0.5, kl_policy=0.1, divergence=2.0, alpha_t=1.0)
    s = m.summary()
    assert s["dt_eff"]["mean"] == 0.5


def test_rsm_metrics_summary():
    m = RSMMetrics()
    m.update_step(kl_mirror=0.2, g_self=0.03, pressure=1.2)
    s = m.summary()
    assert s["pressure"]["mean"] == 1.2


def test_run_context_dict():
    rc = RunContext(bench="x", variant="trp", seed=1, steps_or_episodes=10)
    d = rc.to_dict()
    assert d["bench"] == "x" and d["variant"] == "trp"
