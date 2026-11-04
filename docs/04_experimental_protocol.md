# Experimental Protocol & Benchmark Specification

## 1. Experimental Objectives

The empirical validation aims to answer three central scientific questions:
1. **Target Efficacy:** Does Subspace-Projected Weight-Space Programming (SP-SV-WSP) match or exceed the target pathology correction rate of unconstrained Weight-Space Programming (Takehi et al., 2026)?
2. **Locality & Invariant Preservation:** Does SP-SV-WSP significantly reduce collateral damage (perplexity degradation, semantic drift, factual forgetting) compared to unconstrained weight editing?
3. **Autonomous Adaptivity:** Can the closed-loop verification oracle successfully detect and roll back / adapt harmful updates without human intervention?

---

## 2. Target Models & Resource Allocation

Experiments are designed to run within an 8GB VRAM envelope (NVIDIA RTX 4060 Laptop GPU, Ryzen 7 7840HS, 16GB RAM):

| Model Name | Parameters | Format | Hidden Dim | Layers | VRAM Footprint | Role in Experiment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`HuggingFaceTB/SmolLM2-360M`** | 360M | bfloat16 | 960 | 32 | ~720 MB | **Phase 1: Rapid mathematical validation & sanity check** |
| **`EleutherAI/pythia-410m`** | 410M | float16 | 1024 | 24 | ~820 MB | **Phase 2: Mechanistic interpretability & factual editing** |
| **`Qwen/Qwen2.5-0.5B` / `Qwen3.5-0.8B`** | 500M - 800M | bfloat16 | 896 - 1024 | 24 | ~1.1 - 1.6 GB | **Phase 3: Direct comparative reproduction of Takehi et al. (2026)** |

---

## 3. Benchmark Suites & Datasets

### 3.1 Strict 3-Way Disjoint Partitioning Protocol
To guarantee complete experimental independence and prevent leakage between intervention derivation, validation tuning, and final evaluation, all experiments enforce a pairwise disjoint 3-way split across **both** evaluation text passages and behavioral prompts:

1. **Passage Partitioning:**
   - **Train Set (Passages 00–29, 30 passages, 540 tokens):** Used strictly for empirical activation covariance $C_{\text{train}}$ and null-space projector $P_{\text{null}}$ construction.
   - **Validation Set (Passages 30–39, 10 passages, 149 tokens):** Used strictly for hyperparameter grid tuning $(\alpha, \gamma)$ and Pareto frontier selection.
   - **Final Test Set (Passages 40–49, 10 passages, 185 tokens):** Strictly held out; evaluated only on final frozen models.
   - *Independence Verification:* $\text{IDs}(\text{Train}) \cap \text{IDs}(\text{Val}) = \emptyset$, $\text{IDs}(\text{Train}) \cap \text{IDs}(\text{Test}) = \emptyset$, $\text{IDs}(\text{Val}) \cap \text{IDs}(\text{Test}) = \emptyset$. Programmatic assertions verify zero lexical and ID overlap.

2. **Behavioral Prompt Partitioning:**
   - **Dev Prompts (4 prompts):** Used strictly to extract target failure activations $X_{\text{loop}}$ and clean activations $X_{\text{clean}}$ for deriving $u_{\text{loop}}$, $u_{\text{diff}}$, and $v_{\text{out}}$.
   - **Val Prompts (4 prompts):** Used strictly during validation grid search to evaluate behavioral loop reduction without tuning on test prompts.
   - **Final Test Prompts (6 prompts):** Strictly held out; evaluated only once on final frozen model configurations.
   - *Independence Verification:* $\text{Dev} \cap \text{Val} = \emptyset$, $\text{Dev} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

---

## 4. Baselines for Comparison

We benchmark the proposed framework against six distinct paradigms:
1. **Unmodified Base Model:** Pristine model parameters.
2. **Unconstrained WSP (Takehi et al., 2026):** Raw directional update $\Delta W_{\text{raw}} = -\eta v u^T$ applied directly to down-projection weights without null-space filtering.
3. **Multi-Draw Random Subspace Control ($N=5$):** Projects updates onto $I - Q_k Q_k^T$ using 5 independent random orthogonal draws ($Q_k \in \mathbb{R}^{d_{\text{in}} \times 100}$), reporting mean $\pm$ standard deviation.
4. **Validation-Selected SP-SV-WSP:** Null-space projected update with optimal $(\alpha^*, \gamma^*)$ selected exclusively via validation data.
5. **Strict SP-SV-WSP ($\alpha = 10^{-3}$):** Evaluates high-stiffness null-space filtering.
6. **Differential SP-SV-WSP ($u_{\text{diff}}$):** Applies differential activation surgery to decouple the behavioral pathology from anisotropic language centroids.

---

## 5. Metrics & Hypothesis Acceptance Criteria

| Metric | Target Acceptance Criteria | Rationale |
| :--- | :--- | :--- |
| **Target Task Accuracy** | $\Delta \text{Acc} \ge +25\%$ relative to base | Demonstrates successful pathology resolution |
| **Looping Rate** | $\le 10\%$ (down from >80%) | Verifies elimination of recursive attractors |
| **Perplexity Degradation** | $\Delta \text{PPL}_{\text{rel}} \le 2.0\%$ | Guarantees zero noticeable collateral damage |
| **Spectral Condition Shift** | $\frac{|\kappa(W') - \kappa(W)|}{\kappa(W)} \le 1.0\%$ | Prevents internal representation collapse |
| **Invariant Accuracy Retained** | $\ge 98.5\%$ of base accuracy | Ensures general downstream capabilities survive intact |

---

## 6. Reproducibility & Execution Protocol

Every experiment script logs:
1. Exact commit hash and random seed (`seed=42`).
2. SVD singular values $\sigma(W)$ before and after modification.
3. Projection error: $\epsilon_{\text{proj}} = \frac{\| \Delta W_{\text{safe}} X_{\text{retain}} \|_F}{\| \Delta W_{\text{raw}} X_{\text{retain}} \|_F}$ (must be $< 10^{-4}$).
4. Checkpoint diff files stored in standard safetensors / PyTorch format.

