# TRP–RSM Engine: Mathematical Specification

This document formalizes the TRP–RSM engine implemented in `engine_trp_rsm.py`
and the metrics in `sentience_metrics.py`.

---

## 1. Core Objects

Discrete time steps: \( t = 0, 1, 2, \dots \)

For a single agent:

- Observation: \( x_t \in \mathcal{X} \)
- Action: \( a_t \in \mathcal{A} \)
- Reward: \( r_t \in \mathbb{R} \)
- Hidden self-state: \( s_t \in \mathbb{R}^{d_s} \)
- World latent: \( z_t \in \mathbb{R}^{d_z} \)

Learned maps (with parameters collectively denoted \( \theta \)):

1. **Encoder**
   \[
   h_t = E_\theta(x_t, s_t) \in \mathbb{R}^{d_h}
   \]

2. **Self-model (RSM)**
   \[
   \hat{s}_{t+1} = M_\theta(s_t, h_t)
   \]

3. **World model**
   \[
   \hat{z}_{t+1} = F_\theta(s_t, h_t)
   \]

4. **Policy**
   \[
   \pi_\theta(a_t \mid s_t, h_t)
   \]

5. **Value function**
   \[
   V_\theta(s_t, h_t) \approx \mathbb{E}\Big[ \sum_{k \ge 0} \gamma^k r_{t+k} \,\Big|\, s_t, h_t \Big]
   \]

Additionally, a **shadow “no-self” policy** ignores the self-state:

\[
\pi_\theta^{\text{no-self}}(a_t \mid x_t, h_t)
\]

used to measure the behavioral influence of \( s_t \).

---

## 2. TRP Clock: Time, Reality, Perception

At each step we compute:

- **Reality signal** \( R_t \): structure / surprise in the external world  
- **Perception gain** \( P_t \): confidence / internal gain

In the reference implementation:

- \( R_t \) is derived from the L2 change in observation:
  \[
  R_t \approx \mathrm{mean}\big( (x_t - x_{t-1})^2 \big)
  \]
- \( P_t \) is a monotone function of **negative entropy** of the policy:
  \[
  H_t = - \sum_a \pi_\theta(a_t \mid s_t, h_t) \log \pi_\theta(a_t \mid s_t, h_t)
  \]
  \[
  P_t = \frac{1}{1 + H_t}
  \]

The **TRP time** is:

\[
T_t = R_t \cdot P_t
\]

We define an **effective time step** (time dilation):

\[
\mathrm{dt}_{\text{eff}, t}
  = \frac{1}{1 + \gamma \, T_t}
\]

with \(\gamma > 0\) a hyperparameter.

This yields:

- Per-step learning rate
  \[
  \eta_t = \eta_0 \, \mathrm{dt}_{\text{eff}, t}
  \]
- Per-step KL budget
  \[
  \varepsilon_t = \varepsilon_0 \, \mathrm{dt}_{\text{eff}, t}^{\beta}, \quad \beta \in (0,1]
  \]

High structure + high gain (large \(T_t\)) ⇒ small \(\eta_t\), small \(\varepsilon_t\):
the policy moves slowly and is tightly KL-leashed.

---

## 3. Losses and TRP-Gated Update

### 3.1 Task Loss (RL)

For an advantage signal \(A_t\),

\[
L_{\text{task}}(\theta)
= - \mathbb{E}_t\Big[ \log \pi_\theta(a_t \mid s_t, h_t) \, A_t \Big].
\]

A value loss regularizes the critic:

\[
L_{\text{value}}(\theta)
= \frac{1}{2} \, \mathbb{E}_t \Big[ (V_\theta(s_t, h_t) - G_t)^2 \Big],
\]

where \(G_t\) is a TD(λ) or Monte-Carlo return.

### 3.2 Self-Model Loss

Self-model prediction error:

\[
L_{\text{self}}(\theta)
= \mathbb{E}_t \Big[ D_{\text{self}}(s_{t+1} \Vert \hat{s}_{t+1}) \Big],
\]

with the implementation using squared error:

\[
D_{\text{self}}(a \Vert b) = \lVert a - b \rVert_2^2.
\]

### 3.3 World-Model Loss

World latent prediction error:

\[
L_{\text{world}}(\theta)
= \mathbb{E}_t \Big[ D_{\text{world}}(z_{t+1} \Vert \hat{z}_{t+1}) \Big],
\quad
D_{\text{world}}(a \Vert b) = \lVert a - b \rVert_2^2.
\]

### 3.4 TRP KL Constraint

Let

- \( p_t(a) = \pi_{\theta_t}(a \mid s_t, h_t) \) be the previous policy
- \( q_\theta(a) = \pi_\theta(a \mid s_t, h_t) \) be the candidate policy

We define a **policy KL divergence**:

\[
D_{\text{KL}}(q_\theta \Vert p_t)
= \mathbb{E}_{s_t, h_t} \Big[ \sum_a
   q_\theta(a \mid s_t, h_t)
   \log \frac{q_\theta(a \mid s_t, h_t)}{p_t(a \mid s_t, h_t)}
\Big].
\]

The TRP KL penalty is:

\[
L_{\text{KL}}(\theta)
= \lambda_{\text{KL}} \, 
  \max\Big(0, D_{\text{KL}}(q_\theta \Vert p_t) - \varepsilon_t \Big).
\]

### 3.5 Total Loss

The total per-step loss:

\[
L_{\text{total}}(\theta)
= L_{\text{task}}(\theta)
+ \alpha_{\text{value}} L_{\text{value}}(\theta)
+ \alpha_{\text{self}} L_{\text{self}}(\theta)
+ \alpha_{\text{world}} L_{\text{world}}(\theta)
+ L_{\text{KL}}(\theta).
\]

Hyperparameters \( \alpha_{\text{value}}, \alpha_{\text{self}}, \alpha_{\text{world}} \ge 0 \).

### 3.6 TRP-Gated Gradient Step

We approximate a **proximal** update:

\[
\theta_{t+1}
\approx \theta_t - \eta_t \, \nabla_\theta L_{\text{total}}(\theta_t),
\]

with a practical projection / rescaling to respect the KL budget:

1. Compute raw gradient \( g_t = \nabla_\theta L_{\text{total}}(\theta_t) \).
2. Candidate update \( \theta' = \theta_t - \eta_t g_t \).
3. If \( D_{\text{KL}}(\pi_{\theta'} \Vert \pi_{\theta_t}) > \varepsilon_t \),
   scale back the step, e.g. find \( \lambda \in (0,1] \) such that

   \[
   D_{\text{KL}}(\pi_{\theta_t - \lambda \eta_t g_t} \Vert \pi_{\theta_t})
   \le \varepsilon_t,
   \]

   and set

   \[
   \theta_{t+1} = \theta_t - \lambda \eta_t g_t.
   \]

In code, this is approximated via the KL penalty plus the TRP-scaled learning rate.

---

## 4. Satisfiability and Gates

We define three per-step error quantities:

- Self-model error:
  \[
  d_{\text{self}, t} = D_{\text{self}}(s_{t+1} \Vert \hat{s}_{t+1}).
  \]

- World-model error:
  \[
  d_{\text{world}, t} = D_{\text{world}}(z_{t+1} \Vert \hat{z}_{t+1}).
  \]

- Policy KL change:
  \[
  d_{\text{KL}, t} = D_{\text{KL}}(\pi_{\theta_{t+1}} \Vert \pi_{\theta_t}).
  \]

Given thresholds \( \tau_{\text{self}}, \tau_{\text{world}} \) and KL budget \( \varepsilon_t \),
we define a **satisfiability event**:

\[
\mathrm{sat}_t
= \mathbb{1}\Big\{
   d_{\text{self}, t} \le \tau_{\text{self}},
   d_{\text{world}, t} \le \tau_{\text{world}},
   d_{\text{KL}, t} \le \varepsilon_t
\Big\}.
\]

Over a window of length \(T\):

\[
\mathrm{sat\%}
= \frac{1}{T} \sum_{t=1}^T \mathrm{sat}_t.
\]

---

## 5. Multi-Agent Consensus and Trust

For \(N\) agents indexed by \( i = 1, \dots, N \):

Each agent has its own:

- self-state \( s_t^i \)
- world latent prediction \( \hat{z}_{t+1}^i \)
- parameters \( \theta^i \)
- TRP clock \( T_t^i \), learning rate \( \eta_t^i \), KL budget \( \varepsilon_t^i \).

### 5.1 Trust-Weighted Consensus

Let \( w_t^i \ge 0 \) be the **trust weights** such that
\( \sum_i w_t^i = 1 \).

The **consensus world latent** is:

\[
\bar{z}_{t+1}
= \frac{\sum_{i=1}^N w_t^i \, \hat{z}_{t+1}^i}{\sum_{i=1}^N w_t^i}
= \sum_{i=1}^N w_t^i \, \hat{z}_{t+1}^i.
\]

### 5.2 Social Disagreement and Trust Decay

For each agent, define a **social disagreement** measure (MSE):

\[
d_{\text{social}, t}^i
= \mathbb{E}\big[ \lVert \bar{z}_{t+1} - \hat{z}_{t+1}^i \rVert_2^2 \big].
\]

Given a social threshold \( \tau_{\text{social}} \) and dynamic
trust decay rate \( \lambda_t \), trust is updated by:

\[
w_{t+1}^i
\propto
  w_t^i \exp\Big( - \lambda_t \max(0, d_{\text{social}, t}^i - \tau_{\text{social}}) \Big),
\]
followed by normalization:
\[
w_{t+1}^i
= \frac{w_{t+1}^i}{\sum_j w_{t+1}^j}.
\]

The rate \( \lambda_t \) may itself depend on a global satisfiability
signal over a window, e.g.:

\[
\lambda_t =
\begin{cases}
0.8 \, \lambda_t & \text{if global sat\%}_\text{window} > 0.9, \\
\lambda_0 \big(1 + 0.5 (1 - \text{sat\%}_\text{window})\big) & \text{otherwise}.
\end{cases}
\]

### 5.3 Per-Agent Loss with Social Term

Each agent can also be penalized for deviation from consensus:

\[
L_{\text{social}}^i(\theta^i)
= \mathbb{E}_t\Big[
   D_{\text{social}}(\bar{z}_{t+1} \Vert \hat{z}_{t+1}^i)
 \Big],
\]

with e.g. \(D_{\text{social}}(a\Vert b) = \lVert a - b \rVert_2^2\).

The total per-agent loss becomes:

\[
L_{\text{total}}^i(\theta^i)
= L_{\text{task}}^i
+ \alpha_{\text{value}} L_{\text{value}}^i
+ \alpha_{\text{self}} L_{\text{self}}^i
+ \alpha_{\text{world}} L_{\text{world}}^i
+ \alpha_{\text{social}} L_{\text{social}}^i
+ L_{\text{KL}}^i.
\]

Each agent then performs its own TRP-gated update:

\[
\theta_{t+1}^i
\approx \theta_t^i - \eta_t^i \nabla_{\theta^i} L_{\text{total}}^i(\theta_t^i),
\]

with KL control via \(\varepsilon_t^i\).

---

## 6. Sentience Metrics (C-Vector)

The metrics in `sentience_metrics.py` are **diagnostics** that
quantify aspects of self-model quality, identity continuity, and
global ignition. They are **not** proofs of consciousness.

### C1. Self-Model Coherence (SMC)

Given step logs \( \{ \text{StepLog}_t \}_{t=1}^T \) and self threshold \( \tau_{\text{self}} \),

\[
\text{SMC}
= 1 - \frac{1}{T} \sum_{t=1}^T
  \min\Big(1, \frac{d_{\text{self}, t}}{\max(\tau_{\text{self}}, 10^{-8})}\Big).
\]

Values in \([0,1]\). Higher SMC = more coherent self-prediction.

### C2. Self Influence Index (SII)

Let:

- \( \ell_t^{\text{self}} \) = logits from the policy using \(s_t\)
- \( \ell_t^{\text{no-self}} \) = logits from the shadow policy ignoring \(s_t\)

We form distributions:

\[
p_t(a) = \text{softmax}(\ell_t^{\text{self}}),
\quad
q_t(a) = \text{softmax}(\ell_t^{\text{no-self}}).
\]

The SII is:

\[
\text{SII}
= \mathbb{E}_t \big[ D_{\text{KL}}(p_t \Vert q_t) \big].
\]

Higher SII ⇒ the internal self-state materially influences actions.

### C3. Identity Continuity Index (ICI)

Given a sequence of self-states \( \{ s_t \}_{t=1}^T \in \mathbb{R}^{d_s} \)
and a temporal offset \( \Delta \),

\[
\Delta_s(\Delta)
= \frac{1}{T - \Delta} \sum_{t=1}^{T-\Delta}
  \lVert s_{t+\Delta} - s_t \rVert_2^2.
\]

Define:

\[
\text{ICI}
= \frac{1}{1 + \Delta_s(\Delta)} \in (0,1].
\]

Higher ICI ⇒ stronger identity continuity over \(\Delta\) steps.

### C4. Ignition Index (IGI)

Let \(e^i_t\) be the per-step world-model error for agent \(i\),
and form a matrix \(E \in \mathbb{R}^{N \times T}\).

Let \(m\) be the median of all entries in \(E\).
Define a **surprise threshold**:

\[
\text{thresh} = \kappa \, m, \quad \kappa > 1.
\]

A **spike event** is a pair \((i_0, t_0)\) such that \(e^{i_0}_{t_0} > \text{thresh}\).

For each spike, within a window \([t_0, t_0 + w]\), we measure how many
agents respond with error above median:

\[
\text{resp\_frac}(i_0, t_0)
= \frac{1}{N}
  \big|\{ i : \max_{t_0 \le t \le t_0 + w} e^i_t > m \} \big|.
\]

The **Ignition Index** is:

\[
\text{IGI}
= \mathbb{E}_{(i_0, t_0) \in \text{spikes}} \big[ \text{resp\_frac}(i_0, t_0) \big].
\]

High IGI ⇒ surprising events broadcast through many agents.

---

## 7. C-Vector Summary

For a given run, the engine can report a **C-vector**:

\[
C = \big(
  \text{SMC},\,
  \text{SII},\,
  \text{ICI},\,
  \text{IGI}
\big),
\]

possibly extended with:

- RSS (role-structure stability),
- additional temporal / social statistics.

This vector characterizes aspects of:

- self-model coherence,
- behavioral role of the self,
- temporal identity continuity,
- global ignition / broadcast dynamics,

for the TRP–RSM multi-agent system.

These are structural and empirical properties, not claims of
phenomenal consciousness.

## 8. Basic Theorems

This section collects elementary but useful properties of the TRP–RSM
engine and C-metrics. They are intentionally modest and fully
formalizable from the definitions above.

### Theorem 1 (TRP Time Dilation Bounds and Monotonicity)

For fixed \(\gamma > 0\), reality \(R_t \ge 0\), perception \(P_t \ge 0\),
and
\[
T_t = R_t P_t, \qquad
\mathrm{dt}_{\text{eff}, t}
  = \frac{1}{1 + \gamma T_t},
\]
we have:

1. \(0 < \mathrm{dt}_{\text{eff}, t} \le 1\) for all \(t\).
2. \(\mathrm{dt}_{\text{eff}, t}\) is strictly decreasing in \(T_t\) whenever \(T_t > 0\).
3. The TRP learning rate \(\eta_t = \eta_0 \mathrm{dt}_{\text{eff}, t}\) satisfies
   \(0 < \eta_t \le \eta_0\) and is strictly decreasing in \(T_t\) for \(T_t > 0\).

**Proof.**

1. Since \(R_t, P_t \ge 0\), we have \(T_t \ge 0\). Thus
   \(1 + \gamma T_t \ge 1\), so
   \[
   0 < \frac{1}{1 + \gamma T_t} \le 1.
   \]
   Hence \(0 < \mathrm{dt}_{\text{eff}, t} \le 1\).

2. For \(T_t > 0\),
   \[
   \frac{\partial}{\partial T_t}
   \mathrm{dt}_{\text{eff}, t}
   = \frac{\partial}{\partial T_t}
     \frac{1}{1 + \gamma T_t}
   = - \frac{\gamma}{(1 + \gamma T_t)^2} < 0,
   \]
   so \(\mathrm{dt}_{\text{eff}, t}\) is strictly decreasing in \(T_t\).

3. Since \(\eta_t = \eta_0 \mathrm{dt}_{\text{eff}, t}\) with \(\eta_0 > 0\),
   the bounds follow from (1) and monotonicity from (2). \(\square\)

---

### Theorem 2 (KL Budget Bounds and Monotonicity)

Let \(\varepsilon_0 > 0\), \(\beta > 0\), and define
\[
\varepsilon_t
= \varepsilon_0 \, \mathrm{dt}_{\text{eff}, t}^{\beta}
= \varepsilon_0 \bigg(\frac{1}{1 + \gamma T_t}\bigg)^{\beta}.
\]
Then:

1. \(\varepsilon_t > 0\) for all \(t\).
2. \(\varepsilon_t \le \varepsilon_0\) for all \(t\).
3. \(\varepsilon_t\) is strictly decreasing in \(T_t\) whenever \(T_t > 0\).

**Proof.**

1. Since \(\mathrm{dt}_{\text{eff}, t} > 0\) and \(\beta > 0\), we have
   \(\mathrm{dt}_{\text{eff}, t}^{\beta} > 0\). With \(\varepsilon_0 > 0\), the product
   is positive.

2. From Theorem 1, \(0 < \mathrm{dt}_{\text{eff}, t} \le 1\). Raising to \(\beta > 0\)
   preserves the inequality, giving
   \(0 < \mathrm{dt}_{\text{eff}, t}^{\beta} \le 1\),
   so \(\varepsilon_t \le \varepsilon_0\).

3. Since \(\mathrm{dt}_{\text{eff}, t}\) is strictly decreasing in \(T_t > 0\) and
   \(x \mapsto x^{\beta}\) is strictly increasing on \((0,1]\) for \(\beta > 0\),
   their composition is strictly decreasing. Multiplication by \(\varepsilon_0 > 0\)
   preserves strict monotonicity. \(\square\)

---

### Proposition 3 (C-Metrics Bounds)

Let:

- SMC be defined as
  \[
  \mathrm{SMC}
  = 1 - \frac{1}{T} \sum_{t=1}^T
      \min\Big(1, \frac{d_{\text{self}, t}}{\max(\tau_{\text{self}}, 10^{-8})}\Big),
  \]
- ICI by
  \[
  \mathrm{ICI}
  = \frac{1}{1 + \Delta_s(\Delta)},
  \quad
  \Delta_s(\Delta) =
  \frac{1}{T - \Delta} \sum_{t=1}^{T-\Delta}
    \lVert s_{t+\Delta} - s_t \rVert_2^2,
  \]
- IGI as an average of fractions in \([0,1]\),
- SII as
  \[
  \mathrm{SII}
  = \mathbb{E}_t\big[D_{\text{KL}}(p_t \Vert q_t)\big]
  \]
  with \(p_t, q_t\) probability distributions over a finite action set.

Then:

1. \(0 \le \mathrm{SMC} \le 1\).
2. \(0 < \mathrm{ICI} \le 1\).
3. \(0 \le \mathrm{IGI} \le 1\).
4. \(\mathrm{SII} \ge 0\).

**Proof.**

1. For each \(t\),
   \[
   0 \le \min\Big(1, \frac{d_{\text{self}, t}}{\max(\tau_{\text{self}}, 10^{-8})}\Big) \le 1.
   \]
   The average lies in \([0,1]\); subtracting from 1 keeps us in \([0,1]\).

2. By definition, \(\Delta_s(\Delta) \ge 0\), so \(1 + \Delta_s(\Delta) \ge 1\),
   and thus
   \[
   0 < \frac{1}{1 + \Delta_s(\Delta)} \le 1.
   \]

3. IGI is defined as an average of response fractions where each fraction
   is in \([0,1]\), so the result lies in \([0,1]\).

4. KL divergence between two discrete distributions is always
   non-negative, so \(\mathrm{SII}\), as an expectation of KL terms,
   is also non-negative. \(\square\)

---

### Proposition 4 (Self Influence Index Detects Policy Equality)

Let \(p_t(a)\) and \(q_t(a)\) be two categorical policies over a finite
action set \(\mathcal{A}\), and define
\[
\mathrm{SII}
= \mathbb{E}_t \big[ D_{\text{KL}}(p_t \Vert q_t) \big].
\]

Assume that the expectation is taken over a distribution on \(t\) with
support on all encountered states. Then:

1. \(\mathrm{SII} = 0\) if and only if \(p_t(a) = q_t(a)\) for all \(a \in \mathcal{A}\) and for all \(t\) in the support.
2. If \(p_t \neq q_t\) on a set of non-zero measure in \(t\), then
   \(\mathrm{SII} > 0\).

**Proof.**

For each fixed \(t\), \(D_{\text{KL}}(p_t \Vert q_t) \ge 0\), with equality
if and only if \(p_t(a) = q_t(a)\) for all \(a\). If \(p_t = q_t\) for all
\(t\) in the support, then every KL term is zero and thus
\(\mathrm{SII} = 0\).

Conversely, if there exists a set of \(t\) of non-zero measure where
\(p_t \neq q_t\), then the KL divergence is strictly positive on that set,
and its expectation over \(t\) is strictly positive, yielding
\(\mathrm{SII} > 0\). \(\square\)

---
