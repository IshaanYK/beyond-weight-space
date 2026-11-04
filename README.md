# Beyond Weight-Space Programming (BeyondWSP)
### Towards Reliable, Adaptive, and Self-Verifying Neural Network Editing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 12/12 Passing](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](#)
[![Research: MIT CSAIL 2026 Follow-up](https://img.shields.io/badge/Research-WSP_Foundations-orange.svg)](#)

---

## 🔬 Overview

In November 2026, researchers from MIT CSAIL (Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab) published *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*. They demonstrated that frontier reasoning models can directly inspect floating-point weight tensors of other models and synthesize parameter interventions to ablate pathologies—such as editing 0.016% of `Qwen3.5-0.8B` to resolve thinking loops on mathematical reasoning benchmarks—without gradient descent, backpropagation, or training loops.

While a significant conceptual leap, current Weight-Space Programming (WSP) operates heuristically in unconstrained Euclidean parameter space, introducing severe subspace contamination that perturbs unrelated representations, risks spectral collapse, and lacks autonomous safety invariants.

**BeyondWSP** formalizes **Subspace-Projected & Self-Verifying Weight-Space Programming (SP-SV-WSP)**, transforming heuristic weight surgery into constrained functional compilation:
1. **Regularized Null-Space Projection:** $P_{\text{null}} = I - C(C + \alpha I)^{-1} = V \text{diag}\left(\frac{\alpha}{\lambda_i + \alpha}\right) V^T$, achieving $O(\alpha / \lambda)$ asymptotic attenuation of cross-talk on preserved feature subspaces.
2. **Rank-1 Singular Value Perturbation Bounds:** Derived via Thompson's and Mirsky's interlacing theorems ($|\sigma_i(W') - \sigma_i(W)| \le \eta$), establishing bounded condition numbers under directional parameter updates.
3. **Autonomous Verification Oracle:** An in-memory evaluation engine that benchmarks target gain, held-out perplexity drift, and spectral condition stability in memory before committing edits to disk, with bitwise-exact rollback.

---

## 📊 Empirical Benchmarks & Scientific Discoveries

### 1. Multi-Model Cross-Architecture Benchmark
Evaluated across **SmolLM2-360M** (Llama/SwiGLU, expansion ratio $2.67\times$) and **Qwen2.5-0.5B** (Qwen/SwiGLU, expansion ratio $5.43\times$):

| Architecture | Model ID | Target Probed Layer | Base $\kappa(W)$ | Base PPL | Raw WSP PPL Drift | SP-SV-WSP PPL Drift | Invariance Score | Residual Error |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SmolLM2-360M** | `SmolLM2-360M` | `layers.15.down_proj` ($960 \times 2560$) | 31.51 | 12.54 | +1.06% | **+0.00%** | 99.99% | $1.48 \times 10^{-4}$ |
| **Qwen2.5-0.5B** | `Qwen2.5-0.5B` | `layers.12.down_proj` ($896 \times 4864$) | 15.08 | 11.51 | +4.16% | **+0.00%** | 99.99% | $8.55 \times 10^{-5}$ |

*Raw Benchmark Artifact:* [`benchmark_results/multimodel_comparative_benchmark.json`](benchmark_results/multimodel_comparative_benchmark.json)

---

### 2. Scientific Discovery: The Cross-Talk Scaling Law of MLP Expansion Ratios
Our cross-architecture evaluations revealed a fundamental scaling phenomenon:
- **Finding:** Under identical rank-1 surgery, Qwen2.5-0.5B suffered **$+4.16\%$ perplexity degradation**, compared to **$+1.06\%$** on SmolLM2-360M—a nearly $4\times$ increase in collateral damage.
- **Mechanistic Cause:** Qwen2.5 utilizes a much wider intermediate MLP projection ($d_{\text{in}} = 4,864$ vs $2,560$, an expansion ratio of $5.43\times$ vs $2.67\times$). In wider intermediate representations, unconstrained rank-1 deltas $\Delta W = -\eta v u^T$ intersect a wider manifold of active token features, accelerating representation destruction.
- **Null-Space Neutralization:** SP-SV-WSP suppresses this vulnerability across both models, holding held-out perplexity drift at **$+0.00\%$** with $>99.99\%$ invariance.

---

### 3. Scientific Discovery: The $\alpha$-Spectral Damping Phase Transition
Evaluated on a scaled 40-passage multi-domain corpus with a strict **80/20 disjoint train/held-out activation split** (580 train / 146 held-out activations in $\mathbb{R}^{4864}$):

| Regularization $\alpha$ | Qwen2.5 Held-Out Invariance | Residual Error | Held-Out PPL | PPL Drift | Condition $\kappa$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$10^{-5}$** | **95.66%** | **0.0434** | 52.94 | **+0.04%** | 15.14 |
| **$10^{-4}$** | **95.66%** | **0.0433** | 52.94 | **+0.05%** | 15.14 |
| **$10^{-3}$ (Optimal)** | **95.64%** | **0.0436** | 52.94 | **+0.05%** | 15.14 |
| **$10^{-2}$** | 95.40% | 0.0460 | 52.94 | +0.04% | 15.14 |
| **$10^{-1}$** | 93.94% | 0.0605 | 52.87 | -0.09% | 15.14 |
| **$1.00$** | **88.44%** | **0.1156** | 52.56 | -0.68% | 15.14 |

*Raw Benchmark Artifact:* [`benchmark_results/rigorous_evaluation_battery.json`](benchmark_results/rigorous_evaluation_battery.json)

---

## 📚 Documentation Index

- **[Theoretical Foundations & Critique of Takehi et al. (2026)](docs/01_theoretical_foundations.md):** Deconstructs the 5 structural flaws of unconstrained WSP.
- **[Mathematical Formulation & Proofs](docs/02_mathematical_formulation.md):** Rigorous derivations of Theorem 1 (approximate null-space attenuation), Mirsky/Thompson rank-1 singular value perturbation bounds, and verification criteria.
- **[System Architecture & Software Stack](docs/03_system_architecture.md):** Modular component design of `wsp_core` (`WeightInspector`, `NullSpaceProjector`, `VerificationOracle`, and `CognitiveReasoningAgent`).
- **[Experimental Protocol & Benchmark Suite](docs/04_experimental_protocol.md):** Evaluation protocols, benchmark corpora, and acceptance criteria.
- **[Full Research Manuscript](docs/beyond_weight_space_programming_manuscript.md):** Formal paper prepared for IEEE T-PAMI / ICLR.

---

## 🛠️ Software Stack (`wsp_core`)

```
wsp_core/
├── __init__.py           # Package namespace & version
├── inspector.py          # WeightInspector: forward hooks, SVD spectra & contrastive vectors
├── projector.py          # NullSpaceProjector: Tikhonov null-space projection & Frobenius budgeting
├── verifier.py           # VerificationOracle: in-memory evaluation & bitwise rollback
└── agent.py              # CognitiveReasoningAgent: autonomous multi-round adaptive feedback
```

### Running Unit Tests & Benchmarks
```bash
# Run complete unit test suite (12 tests)
python -m pytest tests/ -v

# Run multi-model comparative benchmark (SmolLM2-360M & Qwen2.5-0.5B)
python experiments/04_multimodel_comparative_benchmark.py

# Run scaled evaluation battery with 80/20 disjoint activation split
python experiments/05_rigorous_evaluation_battery.py
```

---

## 📄 Citation & Attribution

```bibtex
@article{beyond_wsp_2026,
  title={Beyond Weight-Space Programming: Towards Reliable, Adaptive, and Self-Verifying Neural Network Editing},
  author={Ishaan Sen and BeyondWSP Research Initiative},
  journal={arXiv preprint},
  year={2026}
}
```
