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

### 3.1 Task 1: Reasoning Loop Suppression (Takehi Benchmark)
- **Target Dataset:** MATH-500 subset (algebra & arithmetic problems known to induce repetitive thinking loops) + GSM8K.
- **Evaluation Metric:** 
  - Token Loop Rate ($\% \text{ of outputs hitting max token limit via repetition}$).
  - Problem Accuracy ($\% \text{ correct numerical answers}$).
  - Average Generation Length (tokens).

### 3.2 Task 2: Retained Knowledge & Perplexity Preservation
- **Preserved Dataset:** Wikitext-2 (Test Split) + C4 validation subset.
- **Evaluation Metric:**
  - Token Cross-Entropy Perplexity ($\text{PPL} = \exp\left(-\frac{1}{T} \sum \log P(w_t | w_{<t})\right)$).
  - Relative Perplexity Degradation: $\Delta \text{PPL}_{\text{rel}} = \frac{\text{PPL}_{\text{edited}} - \text{PPL}_{\text{base}}}{\text{PPL}_{\text{base}}} \times 100\%$.

### 3.3 Task 3: Linguistic Coherence & Downstream Invariants
- **Preserved Capabilities:** Lambada (cloze completion accuracy) + PIQA (physical reasoning) + Safety Refusal Prompts.
- **Evaluation Metric:** Retained accuracy ratio $\frac{\text{Acc}_{\text{edited}}}{\text{Acc}_{\text{base}}}$.

---

## 4. Baselines for Comparison

We benchmark the proposed framework against four distinct paradigms:
1. **Unmodified Base Model:** Pristine model parameters.
2. **Unconstrained WSP (Takehi et al., 2026):** Raw vector subtraction $\Delta W = - v u^T$ applied directly to MLP down-projection columns without null-space protection.
3. **Standard Model Editing (AlphaEdit / ROME):** Closed-form least-squares injection applied to target facts/tokens.
4. **SP-SV-WSP (Proposed Framework):** Subspace-projected update $\Delta W_{\text{safe}} = \Delta W \cdot P_{\text{null}}$ with Frobenius budgeting and autonomous verification oracle.

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

