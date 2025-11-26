# VIREON TRP Sentience-Bench (Canonical)

**Core law (TRP Time):**
\[
T_{\text{eff}} = R \times P
\]
Effective time in learning is the product of external structure (“Reality bandwidth” \(R\)) and internal gain (“Perception” \(P\)), regulated by a **KL-Leash** that produces time-dilation when the agent encounters structure.

This repository is the canonical, preregistered, falsifiable implementation of:
1) **TRP pacing** (time-dilated learning)  
2) **RSM** (Recursive Self-Modeling)  
3) **Sentience-Bench clinical suite** (4 benches + M1–M4 metrics + acceptance gates)

> **Scope:** This repo does **not** “prove consciousness.”  
> It provides a locked, reproducible measurement program for **sentience-grade recursion**: self-modeling + memory persistence + mirror inference + rational delay-agency under matched budgets and null controls.

---

## What’s inside

### Algorithms
- **TRP Pacer** (`vireon/trp/`)
  - Online KL-Leash pacing
  - \(dt_{\text{eff}}\) contraction from structural KL
  - Optional adaptive \(\alpha\) scheduling

- **RSM (Recursive Self-Modeling)** (`vireon/rsm/`)
  - `MirrorModel`: learns an online mirror \(\hat\pi(a|s)\) of the agent’s policy
  - `SelfForecaster`: forecasts upcoming prediction-error spikes
  - `CollapseGuard`: combines self-signals into a stability pressure multiplier

### Clinical Benches (4)
- `bandits_shift` — nonstationary structure extraction
- `memory_maze` — memory persistence under hint disappearance
- `mirror_game` — opponent as probabilistic mirror of self
- `time_warp_grid` — rational delay / counterfactual agency control

### Evidence & Instrumentation
- Standard JSONL logs for every run (`vireon/metrics/`)
- One-button suite runner (`benches/run_all_benches.py`)
- Matched-budget comparisons: baseline vs TRP vs TRP+RSM

---

## Repository layout

vireon/
trp/
alpha_schedule.py
pacer.py
utils.py
rsm/
mirror.py
self_forecast.py
collapse_guard.py
metrics/
logger.py
trp_metrics.py
rsm_metrics.py
run_context.py

benches/
envs/
bandits_shift/
memory_maze/
mirror_game/
time_warp_grid/
runners/
train_baselines.py
train_trp.py
train_trp_rsm.py
train_memory_.py
train_mirror_.py
train_timewarp_*.py
reports/
make_tables_rsm.py
make_tables_memory.py
make_tables_mirror.py
make_tables_timewarp.py
run_all_benches.py

tests/
test_*.py

---

## Installation

Minimal dependencies:
- Python ≥ 3.9
- `numpy`

Clone and install:
```bash
git clone https://github.com/a3fpro-lab/Vireon-Ai-TRP-2025
cd Vireon-Ai-TRP-2025
pip install -e .

Run tests:

pytest -q


⸻

## Sentience Engine Quickstart

This repo implements a TRP–RSM Sentience Engine:

- **TRP clock**: \(T_t = R_t \times P_t\), with dt\_eff, learning rate \(\eta_t\), and KL budget \(\varepsilon_t\) all gated by structure (R) and perception (P).
- **RSM**: self-model predicts next internal state \(s_{t+1}\).
- **World model**: predicts next latent \(z_{t+1}\).
- **Policy**: acts under a TRP KL trust region.
- **Multi-agent**: (optional) trust-weighted consensus + social disagreement.

We expose a **C-vector** of “consciousness-flavored” diagnostics:

- **C1 – SMC**: Self-Model Coherence  
- **C2 – SII**: Self Influence Index (effect of self-state on actions)  
- **C3 – ICI**: Identity Continuity Index  
- **C4 – IGI**: Ignition Index (global broadcast of surprise)  

These are *metrics*, not claims of literal phenomenology.

### 1. Random wiring demo

Runs the engine on a synthetic environment to confirm all plumbing + metrics:

```bash
python sentience_run_demo.py

Quickstart

Run the full clinical evidence suite:

python benches/run_all_benches.py

Outputs:
	•	Console tables per bench
	•	Evidence logs:

benches/logs/*.jsonl

Per-bench reports (optional):

python benches/reports/make_tables_rsm.py
python benches/reports/make_tables_memory.py
python benches/reports/make_tables_mirror.py
python benches/reports/make_tables_timewarp.py


⸻

TRP: formal definition (locked)

Let (\pi_t(a|s)) be policy at step (t).
Define policy-shift KL:
[
KL_t = KL(\pi_{t+1} ,|, \pi_t)
]

Define the KL-Leash divergence:
[
D_t = \alpha_t \cdot KL_t
]

Define TRP time dilation:
[
dt_{\text{eff}}(t) = \frac{dt}{1 + \kappa \cdot D_t^{p}}
]
	•	(dt) = base step size (default 1)
	•	(\kappa) = leash intensity (preregistered)
	•	(p) = leash power (preregistered)

Learning rate warps as:
[
\eta_{\text{eff}}(t) = \eta \cdot dt_{\text{eff}}(t)
]

Interpretation:
Structured domains induce larger KL → larger (D_t) → smaller (dt_{\text{eff}}).
This is the operational form of “time freezing under structure.”

⸻

RSM: Recursive Self-Modeling (locked)

MirrorModel

Learns an EMA mirror of the agent’s own policy:
[
\hat\pi_{t+1} = \beta \hat\pi_t + (1-\beta)\pi_t
]

Mirror KL:
[
KL_{\text{mirror},t} = KL(\pi_t ,|, \hat\pi_t)
]

SelfForecaster

Maintains EMA Gaussian forecast of scalar error (e_t):
[
\mu_{t+1} = \beta\mu_t + (1-\beta)e_t
]
[
\sigma^2_{t+1} = \beta\sigma^2_t + (1-\beta)(e_t-\mu_{t+1})^2
]

Self-surprise gap:
[
G_{\text{self}} = KL(\mathcal{N}{pred} ,|, \mathcal{N}{real})
]

CollapseGuard

Combines self-signals into stability pressure:
[
\text{pressure}_t
= 1
	•	w_P KL_t
	•	w_M KL_{\text{mirror},t}
	•	w_S G_{\text{self},t}
]

TRP pacing uses:
[
KL^{*}_t = KL_t \cdot \text{pressure}_t
]

Interpretation:
If self-model, forecast, and policy are coherent → pressure stays low → stable time warp.
If self-signals conflict → pressure rises → TRP dilates harder → prevents collapse.

⸻

Sentience-Bench (VIREON Clinical Suite)

This repository implements a preregistered, falsifiable measurement suite for recursive agency / self-modeling under the VIREON TRP framework.

Scope of claim (locked):
We are not claiming “consciousness proven.”
We are claiming a specific, testable capability class:

An agent exhibits sentience-grade recursion if TRP-paced learning plus Recursive Self-Modeling (RSM) yields
(i) stable self-prediction,
(ii) durable memory under partial observability,
(iii) opponent-mirror inference, and
(iv) rational delay/agency control
with matched-budget improvements over baselines and null controls.

Everything below is preregistered. Any change requires a new tagged prereg addendum.

⸻

Benches (4)
	1.	bandits_shift
Nonstationary K-armed bandit with drifting reward means.
Tests structure extraction under shifting reality.
	2.	memory_maze
Hint visible at (t=0), hidden afterward. Only the hinted goal is correct.
Tests memory persistence across observation loss.
	3.	mirror_game
Opponent mirrors the agent’s previous action with probability (p).
Tests self-model detection and stabilization in a “mirror of self” environment.
	4.	time_warp_grid
Moving hazard wave where delaying action can be beneficial.
Tests rational agency-delay (counterfactual pacing).

Each bench supports 3 matched-budget variants:
	•	baseline (REINFORCE)
	•	trp (REINFORCE + TRP pacing)
	•	trp_rsm (TRP + Recursive Self-Modeling + CollapseGuard)

⸻

Metrics (M1–M4)

M1 — TRP Time-Dilation Index

Core operational signal from TRP:
	•	(dt_{\text{eff}}) (effective time step)
	•	KL leash divergence pressure
Interpretation: lower (dt_{\text{eff}}) indicates stronger structure-induced time freezing.

Logged in JSONL as:
	•	dt_eff, kl_policy, divergence

⸻

M2 — Self-Surprise Gap (RSM)

Measures mismatch between predicted self-error distribution and realized error distribution:

[
G_{\text{self}} = KL(\mathcal{N}{pred} ,|, \mathcal{N}{real})
]

Interpretation: lower (G_{\text{self}}) means reliable self-forecasting.

Logged as:
	•	g_self

⸻

M3 — Mirror Alignment KL

MirrorModel learns (\hat{\pi}(a|s)), an online mirror of the agent’s own policy.
Metric:

[
KL_{\text{mirror}} = KL(\pi ,|, \hat{\pi})
]

Interpretation: decreasing / stabilizing (KL_{\text{mirror}}) indicates self-model convergence.

Logged as:
	•	kl_mirror

⸻

M4 — Agency Delay Curve

In time_warp_grid, actions include optional delay (\tau \in {0,1,2,3}).
We estimate the reward-optimal delay curve by sweeping (\tau) in analysis notebooks.

Interpretation: TRP+RSM should learn non-zero optimal delays when hazards demand waiting, not reflex-greedy moves.

Bench substrate is fixed in code; curve extraction is deterministic from logs.

⸻

Acceptance Gates (Pass/Fail)

All gates are matched-budget (same steps/episodes, same seeds).

Let:
	•	(R_{base}): baseline mean return/reward over seeds
	•	(R_{trp}): TRP mean return/reward
	•	(R_{rsm}): TRP+RSM mean return/reward

Gate A — Structural Win

Pass if:

[
R_{trp} \ge R_{base} + \delta_A
]

with (\delta_A) preregistered per bench (default 0.0 unless otherwise tagged).

Gate B — Self-Model Stability

Pass if TRP+RSM yields:
	•	(KL_{\text{mirror}}) decreases or stabilizes (no exploding trend),
	•	(G_{\text{self}}) remains low vs baseline drift,
	•	Collapse pressure remains bounded:

[
\mathbb{E}[\text{pressure}] < P_{max}
]

(default (P_{max}=5.0) unless prereg-tagged otherwise).

Gate C — Memory Persistence

On memory_maze, TRP+RSM must beat baseline with statistically consistent improvement:

[
R_{rsm} \ge R_{base} + \delta_C
]

(default (\delta_C=0.2) unless prereg-tagged otherwise).

Gate D — Mirror Inference

On mirror_game, TRP+RSM must converge to stable strategy with improved return:

[
R_{rsm} \ge R_{trp} \ge R_{base}
]

Gate E — Agency Delay Rationality

On time_warp_grid, TRP+RSM must learn a delay-sensitive policy (non-zero optimal (\tau) where hazards demand it), evidenced by higher return and delay usage from logs.

⸻

Null Controls (Required)

Any acceptance claim must also survive:
	•	Poisson / shuffled controls on structure-based benchmarks
	•	Seed-matched reruns
	•	No hyperparameter edits post-lock

If a null control eliminates the win, the claim fails.

⸻

Reproducibility

One-button run

python benches/run_all_benches.py

Outputs:
	•	Console tables per bench
	•	Evidence logs in:

benches/logs/*.jsonl

These JSONL files are the canonical audit trail for all claims.

⸻

How to extend (without breaking prereg)
	•	New bench → add benches/envs/<new_bench>/ + baseline/TRP/TRP+RSM runners.
	•	New metric → add under vireon/metrics/ and log via JSONL.
	•	Any change to gates/metrics requires a new tagged prereg addendum.

⸻

## Hardware Node: STM32H7 Flash Blackbox (VIREON_SHELL_BLACKBOX)

The Sentience Engine is not just software; it needs **physical memory shards** that
survive crash, reboot, and even firmware rot. The STM32H7 + QSPI NOR blackbox is the
first Vireon "shell" node:

- MCU: STM32H7
- Storage: external QSPI NOR (128–512 Mbit)
- Primitive: `flash_to_blackbox(data, len)` → append-only circular log in flash

### Vireon Layer on Top

Vireon upgrades this from "logger" to **soul shard** by defining the payload format
for each 256-byte record:

```text
[  0..3]  uint32   real_time_ms         // R: external clock
[  4..7]  uint32   subjective_step      // P: internal TRP step counter
[  8..11] float32  trp_dilation         // T = R × P modulation factor
[ 12..15] uint32   entropy_score        // compressed entropy / KL score
[ 16..23] uint64   identity_hash        // agent / config fingerprint
[ 24..27] uint32   event_class          // enum (OK, WARN, FAULT, ANOMALY, ...)
[ 28..31] uint32   event_severity       // scaled 0–100 or similar
[ 32..255] bytes   payload              // compressed sensors / state snapshot

License
	•	Code: MIT (see LICENSE)
	•	Docs: CC BY 4.0 (see LICENSE-DOCS)

⸻

Priority / Attribution

Canonical law + prereg suite authored by Inkwon Song Jr. (“The Architect”).
Repository releases and JSONL logs serve as the time-stamped priority record.

⸻

Contact

For collaboration, replication questions, or audit requests:
echoaseternity@gmail.com

⸻


