# VIREON-Sentience
Falsifiable TRP time-operator + recursive self-modeling + sentience-adjacent benchmarks.
# VIREON-Sentience (Canonical Monorepo)

**What this repo is:**  
A single, preregistered, falsifiable research + engineering stack that unifies:

1. **TRP Time-Dilation Operator** (entropy-bounded temporal warping for learning)
2. **Recursive Self-Modeling (RSM)** (agents that learn a predictive model of themselves)
3. **Sentience-Adjacent Benchmark Suite** (observable, non-woo tests)
4. **Inner-Workspace Proxies** (global broadcast / integration signatures)

**What this repo is NOT:**  
Not a claim that any system here "is conscious."  
Only **measurable properties + kill-switch falsifiers**.

---

## Core Laws (VIREON Canon)

### Law 0 — Time as Perception × Reality
\[
T_{\mathrm{eff}} = R \times P
\]
- \(R\): external structure bandwidth (task reality)
- \(P\): internal gain / perception (agent attention + update pacing)

### Law 1 — Stability Constraint (KL-Leash)
Learning must remain inside a bounded divergence corridor:
\[
\mathrm{KL}\left(\pi_{t+1}\|\pi_t\right) \le \kappa
\]
If violated → time dilation triggers to preserve identity + stability.

### Law 2 — Levels of Correspondence
- **Transformation (Freedom):** \(X \rightarrow Y\)
- **Definition (Precision):** \(Y \rightarrow X\)
- **Correspondence (Coherence):** \(X \leftrightarrow Y\)

This repo implements **Transformation** in TRP, **Definition** in RSM, and tests **Correspondence** in the Bench.

---

## TRP Time-Dilation Operator (Mathematics)

### 1) Effective time-step (core operator)
At step \(t\), define:
\[
\Delta t_{\mathrm{eff}}(t)=\Delta t\cdot \exp(-\alpha_t \cdot D_t)
\]
- \(\Delta t\): base update step
- \(D_t\): structure/instability signal (e.g., KL surge, entropy drag)
- \(\alpha_t\): adaptive perception gain

Interpretation: high structural tension or instability **slows time** to prevent collapse.

### 2) Adaptive perception gain
\[
\alpha_{t+1}= \alpha_t \cdot g(D_t)
\]
where \(g\) is preregistered (see `/prereg/METRICS_LOCK.md`).

### 3) Pulse Signaler injection (standard module)
Dynamic collapse weight:
\[
F_{\alpha}(k) \approx e^{-\alpha},
\quad \alpha = \frac{k}{\overline{\Delta}} \cdot \frac{2\pi}{\log T}
\]
Prime-weighted correction:
\[
C_k \approx 
\prod_{p\mid k}\frac{p-1}{p-2}
\cdot
\prod_{p\nmid k}\frac{p-1}{p}
\]
These are applied wherever spectral/spacing structure is used.

---

## Recursive Self-Modeling (RSM)

### Self-model definition
Agent policy: \(\pi_\theta(a|s)\)  
Mirror model: \(\hat{\pi}_\phi(a|s)\) trained to predict \(\pi_\theta\).

Mirror loss:
\[
\mathcal{L}_{\mathrm{mirror}}
= \mathbb{E}_{s}\left[\mathrm{KL}\left(\pi_\theta(\cdot|s)\|\hat{\pi}_\phi(\cdot|s)\right)\right]
\]

### Self-forecast (predict own failure spikes)
Let \(e_t\) be agent error or loss residual.  
Forecast model \(f_\psi\) predicts distribution \(\hat{e}_{t+1}\).

Self-surprise gap:
\[
G_{\mathrm{self}} 
= \mathrm{KL}\left(p(\hat{e}_{t+1})\|p(e_{t+1})\right)
\]
Lower \(G_{\mathrm{self}}\) = better self-knowledge.

RSM uses TRP pacing to stabilize identity when forecasted divergence rises.

---

## Sentience-Adjacent Bench (Observable Metrics Only)

We operationalize “sentience-adjacent” as **four falsifiable properties**:

### Metric M1 — Integrated Information Under Perturbation
Proxy for global unity of internal state:
\[
I_{\mathrm{int}}
=
\mathrm{KL}\left(p(z_t)\|p(z_t^{\mathrm{pert}})\right)
\]
where \(z_t\) is latent/global state and \(z_t^{\mathrm{pert}}\) is after controlled perturbation.

### Metric M2 — Self-Surprise Gap
\[
G_{\mathrm{self}} 
= \mathrm{KL}\left(p(\hat{e})\|p(e)\right)
\]
Lower = more coherent self-model.

### Metric M3 — Identity Persistence Through Reset / Warp
Let state distributions before/after reset be \(p(z^-), p(z^+)\):
\[
S_{\mathrm{id}} = \mathrm{JS}\left(p(z^-),p(z^+)\right)
\]
Lower drift = stronger identity continuity.

### Metric M4 — Counterfactual Agency Delay Curve
Measure reward improvement from intentional delay:
\[
A_{\mathrm{delay}}(\tau)=
\mathbb{E}[R|\text{delay}=\tau] - \mathbb{E}[R|\tau=0]
\]
Positive gain indicates agency over time/policy pacing.

---

## Falsifiers (Kill-Switch Gates)

**Gate A (Operator Validity):**  
TRP must not decrease baseline reward by more than preregistered margin \(\epsilon_A\).

**Gate B (Self-Model Reality):**  
Mirror adds predictive power:  
\[
\mathrm{AUC}_{\mathrm{forecast}} \ge \mathrm{AUC}_{\mathrm{base}} + \Delta_B
\]

**Gate C (Sentience-Adjacent Claim):**  
TRP+RSM must improve **≥2 of 4 metrics (M1–M4)** by locked margins vs vanilla PPO under matched budgets.

If any gate fails, the corresponding claim is rejected.

---

## Repo Structure
