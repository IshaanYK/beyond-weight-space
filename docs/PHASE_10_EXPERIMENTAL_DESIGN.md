# Phase 10 Experimental Design & Strict Protocol Specification

**Project:** Beyond Weight-Space Programming (SP-SV-WSP)  
**Phase:** Phase 10 — Scientifically Validated Research Contribution  
**Objective:** Establish reliable behavioral improvement, cross-architecture generalization, and defensible scientific evidence under a strict test-set protection protocol.

---

## 1. Experimental Hypotheses

1. **Hypothesis 1 (Argmax Threshold & Multi-Layer Coordination):**  
   Single-layer rank-1 weight editing with strict Frobenius budgeting ($\gamma = 0.01$) produces logit perturbations below the discrete argmax decision threshold ($\Delta z < z_{(1)} - z_{(2)}$) under greedy decoding. Coordinating rank-1 updates across dual mid-to-late feedforward layers (e.g. Layers 15 + 19 as in Takehi et al., 2026, or Layers 14 + 15) increases cumulative output perturbation past the decision boundary while distributing spectral perturbation, maintaining $>65\%$ out-of-sample forward invariance.

2. **Hypothesis 2 (Metric Stabilization via Absolute Reduction):**  
   Reporting absolute repetition score reduction $\Delta S = S_{\text{base}} - S_{\text{edit}}$, loop incidence ($S_{\text{rep}} \ge 0.15$), and distinct $n$-gram ratios eliminates the pathological ratio instability (e.g. $-600\%$) caused by dividing by near-zero base rates.

3. **Hypothesis 3 (Architectural Cross-Talk Vulnerability & Generalization):**  
   Qwen2.5-0.5B exhibits higher collateral vulnerability than SmolLM2-360M due to its wider intermediate MLP expansion ratio ($5.43\times$ vs $2.67\times$). Regularized spectral attenuation will consistently hold perplexity drift to $<0.1\%$ across both architectures while achieving comparable behavioral modification.

4. **Hypothesis 4 (Sample Covariance Rank & Feature Coverage):**  
   Increasing the retained passage corpus from 30 to 40 passages ($\approx 750$ tokens) expands the span of empirical covariance $C_{\text{train}}$, reducing out-of-sample forward perturbation residual error on held-out test data.

---

## 2. Strict Test-Set Protection Protocol

To eliminate the test-set reuse issue identified in Phase 9 (where 51 parameter combinations were evaluated directly on the test set):

```
+-----------------------------------------------------------------------------------------------+
|                                STRICT THREE-WAY DISJOINT PARTITIONS                           |
+-----------------------------------------------------------------------------------------------+
| 1. TRAINING PARTITION (Strictly Isolated for Covariance & Direction Extraction)               |
|    * Passages: 40 passages (Passages 00-39, ~750 tokens) across 8 academic domains            |
|    * Prompts: 8 Development Prompts (Repetitive & Clean contrastive pairs)                    |
|    * Usage: C_train, P_null, u_loop, u_diff, v_out extraction ONLY                            |
+-----------------------------------------------------------------------------------------------+
                                                |
                                        (ONE-WAY TRANSITION)
                                                v
+-----------------------------------------------------------------------------------------------+
| 2. VALIDATION PARTITION (Strictly Isolated for Hyperparameter Selection & Tuning)             |
|    * Passages: 15 passages (Passages 40-54, ~280 tokens) across 5 independent domains        |
|    * Prompts: 8 Validation Prompts (Repetitive sequences disjoint from Dev)                   |
|    * Usage: Tuning grid over (alpha, gamma, layer configuration, single vs dual layer)        |
|    * Output: Exactly ONE selected operating point per framework variant                      |
+-----------------------------------------------------------------------------------------------+
                                                |
                                        (FROZEN SELECTION)
                                                v
+-----------------------------------------------------------------------------------------------+
| 3. FINAL TEST PARTITION (Virgin Held-Out Benchmark Partition)                                 |
|    * Passages: 15 held-out passages (Passages 55-69, ~300 tokens) across 5 independent domains|
|    * Prompts: 12 Final Test Prompts (Virgin held-out prompts with verified baseline loops)    |
|    * Usage: Single, one-time frozen benchmark evaluation across 20 stochastic seeds           |
|    * Enforcement: Zero parameter tuning or configuration selection on Test Data               |
+-----------------------------------------------------------------------------------------------+
```

### Disjointness Programmatic Assertions:
```python
assert train_ids.isdisjoint(val_ids), "Train/Val passage ID collision!"
assert train_ids.isdisjoint(test_ids), "Train/Test passage ID collision!"
assert val_ids.isdisjoint(test_ids), "Val/Test passage ID collision!"
assert dev_prompts.isdisjoint(val_prompts), "Dev/Val prompt collision!"
assert dev_prompts.isdisjoint(test_prompts), "Dev/Test prompt collision!"
assert val_prompts.isdisjoint(test_prompts), "Val/Test prompt collision!"
```

---

## 3. Evaluated Model Topologies & Layer Mapping

| Model ID | Parameter Count | Layers | Hidden Dim $d$ | MLP Dim $d_{\text{mlp}}$ | Expansion Ratio | Probed Mid Layer | Probed Late Layer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HuggingFaceTB/SmolLM2-360M** | 360M | 32 | 960 | 2560 | $2.67\times$ | `layers.15.mlp.down_proj` | `layers.19.mlp.down_proj` |
| **Qwen/Qwen2.5-0.5B** | 490M | 24 | 896 | 4864 | $5.43\times$ | `layers.12.mlp.down_proj` | `layers.15.mlp.down_proj` |

---

## 4. Evaluated Baselines & Experimental Variants

1. **Pristine Baseline:** The unedited model checkpoint ($W' = W$).
2. **Raw WSP (Takehi et al., 2026):** Unconstrained rank-1 directional subtraction ($\Delta W_{\text{raw}} = -\eta v u^T$).
3. **Random-Subspace Control ($N=20$ Draws):** 20 independent orthogonal projections of codimension 100 ($P_{\text{rand}} = I - Q Q^T$).
4. **Strict SP-SV-WSP ($\gamma=0.01$):** Conservative single-layer regularized attenuation ($\alpha^*, \gamma^*=0.01$).
5. **Balanced SP-SV-WSP ($\gamma=0.05$):** Balanced single-layer Pareto operating point ($\alpha^*, \gamma^*=0.05$).
6. **Dual-Layer Coordinated SP-SV-WSP:** Coordinated rank-1 updates across mid + late layers (Layers 15+19 on SmolLM2, Layers 12+15 on Qwen2.5) with distributed Frobenius budgeting ($\gamma=0.025$ per layer).
7. **Differential SP-SV-WSP ($u_{\text{diff}}$):** Centroid-decoupled contrastive vector extraction ($u_{\text{diff}} = u_{\text{loop}} - u_{\text{clean}}$).

---

## 5. Statistical Rigor Standards

- **Seed Count:** $N = 20$ independent random seeds (`seeds = [10, 20, ..., 200]`).
- **Autoregressive Parameters:** `do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=25`.
- **Hypothesis Testing:**
  - Two-sided paired $t$-test (`scipy.stats.ttest_rel`) comparing baseline vs edited paired scores.
  - Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) with Pratt or Wilcox zero-difference handling.
  - Cohen's $d$ effect size: $d = (\bar{x}_1 - \bar{x}_2) / s_{\text{pooled}}$.
  - 95% Confidence Intervals: $\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{N}}$.
- **Negative Result Transparency:** All negative percentage swings, seed-level variances, and failures to cross decision boundaries will be reported explicitly without omission or cherry-picking.

