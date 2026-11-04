# System Architecture: `wsp_core` Software Stack

## 1. Modular Overview

The `wsp_core` framework is designed as a modular, mathematically rigorous, and reproducible toolkit for Subspace-Projected & Self-Verifying Weight-Space Programming.

```
+-----------------------------------------------------------------------------------+
|                                 wsp_core Framework                                 |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +--------------------+   +---------------------+   +--------------------------+  |
|  |  WeightInspector   |   | NullSpaceProjector  |   |    VerificationOracle    |  |
|  |--------------------|   |---------------------|   |--------------------------|  |
|  | - SVD Spectrum     |   | - Covariance Cache  |   | - Target Task Accuracy   |  |
|  | - Activation Probe |   | - Moore-Penrose Inv |   | - Retained PPL Evaluator |  |
|  | - Condition Number |   | - Tikhonov Tuning   |   | - Refusal/Safety Checker |  |
|  | - Pathology Vector |   | - Frobenius Bounding|   | - Automated Rollback     |  |
|  +--------------------+   +---------------------+   +--------------------------+  |
|            ^                         ^                           ^                |
|            |                         |                           |                |
|  +---------+-------------------------+---------------------------+-------------+  |
|  |                           CognitiveReasoningAgent                           |  |
|  |  - Formulates Mechanistic Hypotheses from SVD / Activation geometry         |  |
|  |  - Generates Candidate Directional Tensors ΔW_raw                          |  |
|  |  - Interprets Verification Feedback & Iterates Adaptively                  |  |
|  +-----------------------------------------------------------------------------+  |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 `WeightInspector` (`wsp_core/inspector.py`)
Responsible for reading, probing, and diagnostic profiling of model layers without mutating parameters:
- **`profile_layer(layer_name)`**: Computes singular value decomposition $W = U \Sigma V^T$, condition number $\kappa(W)$, spectral energy distribution, and Frobenius norm.
- **`capture_activations(prompt_batch, layer_name, position="last")`**: Registers forward hooks to capture input representations $X \in \mathbb{R}^{d_{\text{in}} \times N}$ and output activations $Y \in \mathbb{R}^{d_{\text{out}} \times N}$.
- **`extract_contrastive_vector(positive_prompts, negative_prompts, layer_name)`**: Computes the mean difference or principal direction separating desired vs. pathological behaviors:
  $$v_{\text{target}} = \mu_{\text{target}} - \mu_{\text{base}} \quad \text{or} \quad \text{Top-1 PCA}\left(X_{\text{target}} - X_{\text{base}}\right)$$

### 2.2 `NullSpaceProjector` (`wsp_core/projector.py`)
Responsible for constructing the safe projection manifold and ensuring zero-interference constraints:
- **`compute_covariance(activations, alpha=1e-3)`**: Computes the sample covariance matrix $C = \frac{1}{N} X X^T$ and applies Tikhonov regularized inversion:
  $$P_{\text{null}} = I - C (C + \alpha I)^{-1}$$
- **`project_delta(delta_w, p_null)`**: Performs orthogonal projection:
  $$\Delta W_{\text{safe}} = \Delta W \cdot P_{\text{null}}$$
- **`enforce_frobenius_budget(delta_w, base_w, gamma=0.02)`**: Enforces the relative perturbation bound:
  $$\|\Delta W_{\text{safe}}\|_F \le \gamma \|W\|_F$$

### 2.3 `VerificationOracle` (`wsp_core/verifier.py`)
Autonomous supervisor that guards against unintended regressions:
- **`evaluate_target_task(model, task_suite)`**: Computes performance gain on target task (e.g., math reasoning completion, loop suppression percentage).
- **`evaluate_retained_perplexity(model, text_corpus)`**: Computes token-level cross-entropy loss $\mathcal{L}_{\text{CE}}$ and exponentiates to calculate perplexity on standard text (Wikitext-2 / OpenWebText sample).
- **`evaluate_safety_invariants(model, safety_prompts)`**: Checks adherence to standard refusal guardrails and calibrated generation.
- **`verdict(metrics, thresholds)`**: Emits `COMMIT`, `ADAPT`, or `ROLLBACK`.
- **`rollback(model, backup_state)`**: Restores pristine layer weights if any invariant fails.

### 2.4 `CognitiveReasoningAgent` (`wsp_core/agent.py`)
The orchestrator that interfaces between diagnostic tools and cognitive reasoning:
- Receives structured diagnostic summaries (spectral profiles, singular values, activation clusters).
- Proposes mechanistic hypotheses (e.g., "Attractor located along left singular vector $u_3$ of layer 15 down-projection").
- Synthesizes candidate tensor deltas $\Delta W_{\text{raw}}$.
- Inspects verification feedback and adapts hyperparameters ($\alpha, \gamma$, rank $r$) if constraints are breached.

---

## 3. Data Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher
    participant Agent as Cognitive Reasoning Agent
    participant Insp as WeightInspector
    participant Proj as NullSpaceProjector
    participant Target as Target Model (SmolLM2 / Qwen)
    participant Verif as VerificationOracle

    User->>Agent: "Ablate reasoning-loop in Layer 15 without degrading PPL"
    Agent->>Insp: profile_layer("model.layers.15.mlp.down_proj")
    Insp-->>Agent: Spectral summary (κ = 142.3, σ_max = 12.4, σ_min = 0.087)
    Agent->>Insp: capture_activations(loop_prompts vs normal_prompts)
    Insp-->>Agent: Isolated attractor direction v_loop
    Agent->>Agent: Synthesize raw delta: ΔW_raw = - v_loop · u^T
    Agent->>Proj: project_delta(ΔW_raw, C_retain)
    Proj-->>Agent: Subspace-projected ΔW_safe (null-space error < 1e-6)
    Agent->>Target: Apply ΔW_safe (in-memory)
    Agent->>Verif: run_verification_battery(Target)
    Verif->>Target: Measure Target Gain + Wikitext PPL + Invariants
    Verif-->>Agent: Metric Tuple: {Gain: +42%, ΔPPL: +0.03 (Pass), κ_shift: +0.2%}
    Agent->>Verif: Verdict -> COMMIT
    Verif-->>User: Verified Checkpoint Saved & Logged
```

