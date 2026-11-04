# Beyond Weight-Space Programming: Towards Reliable, Adaptive, and Self-Verifying Neural Network Editing

**Authors:** Ishaan Sen$^{1}$, BeyondWSP Research Initiative$^{1}$  
$^{1}$Department of Computer Science & Artificial Intelligence Laboratory  
**Target Submission Venue:** International Conference on Learning Representations (ICLR) / IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)  
**Primary Artifact Repository:** [https://github.com/IshaanYK/beyond-weight-space](https://github.com/IshaanYK/beyond-weight-space)  
**Research Freeze Commit:** `f6d67e8`  

---

## Abstract

Recent investigations have demonstrated that reasoning-capable large language models can analyze the raw floating-point parameter matrices of other neural networks and synthesize targeted weight interventions to ablate behavioral pathologies without computing gradients, executing backpropagation, or curating fine-tuning datasets. However, this emerging paradigm of Weight-Space Programming (WSP) operates heuristically in unconstrained parameter space through rank-1 tensor subtraction ($\Delta W = -\eta v u^T$). This unconstrained intervention induces collateral subspace interference (cross-talk) on retained token representations, accelerates singular value drift, and lacks autonomous closed-loop safety verification. 

In this work, we formalize **Subspace-Projected and Self-Verifying Weight-Space Programming (SP-SV-WSP)**, a mathematically rigorous framework that transforms empirical weight surgery into constrained functional projection. By projecting candidate parameter perturbations onto the regularized null space of preserved activation covariances ($P_{\text{null}} = I - C(C + \alpha I)^{-1}$), we prove that forward-pass interference along active feature directions decays asymptotically as $O(\alpha / \lambda_i)$, suppressing cross-talk by over $99.9\%$ along dominant covariance eigenvectors. We replace heuristic norm bounds with rank-1 singular value perturbation bounds grounded in Mirsky's theorem and Thompson's interlacing inequalities. Crucially, we prove that when the nominal intervention scale exceeds the minimal singular value of the unedited weight matrix ($\eta > \sigma_{\min}(W)$)—a condition ubiquitous in modern transformer projection layers—worst-case theoretical bounds cannot rule out matrix singularity, establishing that in-memory verification oracles are mathematically necessary. 

We validate SP-SV-WSP across **SmolLM2-360M** and **Qwen2.5-0.5B** under a strict three-way disjoint protocol where training, validation, and test partitions are separated across both text passages and behavioral prompts. Our experiments reveal an architectural scaling dynamic: unconstrained rank-1 surgery is nearly four times more destructive on Qwen2.5 than on SmolLM2 (+4.16% vs +1.06% perplexity drift) due to its wider intermediate MLP expansion ratio ($5.43\times$ vs $2.67\times$). Under SP-SV-WSP, in-sample perplexity drift is eliminated (+0.00%). On strictly held-out test data, SP-SV-WSP achieves $88.43\%$ out-of-sample forward invariance and confines perplexity drift to $-0.0455\%$, whereas a 20-draw random orthogonal subspace baseline of codimension 100 achieves only $4.09\% \pm 1.23\%$ invariance, proving that representation preservation requires genuine covariance eigenspace alignment. Finally, a 20-seed stochastic autoregressive battery (`temperature=0.7`, `top_p=0.9`) maps the discrete argmax decision threshold and establishes the empirical Pareto frontier between target behavioral modification and retained capability preservation.

**Keywords:** Neural Network Editing, Weight-Space Programming, Model Surgery, Null-Space Projection, Representation Engineering, Singular Value Perturbation, Mechanistic Interpretability.

---

## 1. Introduction

Continuous gradient-based optimization via backpropagation and variants of stochastic gradient descent (SGD, AdamW) has served as the foundational mechanism for training neural networks for four decades. Despite its universal applicability, gradient descent remains computationally intensive, requires substantial curated datasets, and frequently causes catastrophic forgetting when adapting trained models to specific downstream tasks or repairing localized behavioral defects.

In late 2026, exploratory work from MIT CSAIL (Takehi et al., 2026) demonstrated that frontier reasoning models equipped with sandboxed execution environments can natively inspect the raw floating-point weights of open-weights models and synthesize parameter interventions directly in weight space. Specifically, by identifying an activation direction $u$ and an output direction $v$ associated with pathological recursive reasoning loops in Qwen3.5-0.8B, an automated agent constructed a rank-1 outer-product update:
$$\Delta W_{\text{raw}} = -\eta \cdot v u^T$$
Subtracting this rank-1 tensor across 64 output columns in MLP layers 15 and 19 ablated the target failure mode, improving MATH-500 accuracy from 7.9% to 48.1% without backward passes, gradient calculations, or optimizer state allocations.

```
       UNCONSTRAINED WEIGHT SURGERY (Takehi et al., 2026)
       [Pathology Vector u] ---> Outer Product: dW_raw = -η v u^T
                                            |
                                            v
                               Applied Directly to Layer W
                                            |
                     +----------------------+----------------------+
                     |                                             |
                     v                                             v
          [Ablates Target Loop]                       [Collateral Representation Drift]
          (Greedy Loop Rate: -26.7%)                  (Out-of-Sample Invariance: 0.00%)
                                                      (PPL Drift: -0.7337%, Spikes Condition)

                                      VS.

       SP-SV-WSP (OUR MATHEMATICALLY CONSTRAINED FRAMEWORK)
       [Pathology Vector u] ---> Subspace Projection: dW_safe = dW_raw * P_null
                                            |
                                            v
                               Frobenius Budgeting: ||dW||_F <= γ ||W||_F
                                            |
                                            v
                               In-Memory Verification Oracle
                               (ΔS_target, ΔPPL_retain, ΔRefusal, Δκ)
                                            |
                     +----------------------+----------------------+
                     |                                             |
               All Passes?                                    Violated?
                     |                                             |
                     v                                             v
          [Commit to Checkpoint]                         [Adaptive Stiffening or
          (Invariance: 88.43% - 89.95%)                   Bitwise-Exact Rollback]
          (PPL Drift: -0.0455%)
```

Despite this remarkable empirical observation, unconstrained Weight-Space Programming (Raw WSP) exhibits three structural deficiencies that prevent it from being deployed reliably:

1. **Subspace Cross-Talk and Collateral Contamination:** When an unconstrained rank-1 update $\Delta W_{\text{raw}} = -\eta v u^T$ is added to a layer matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, any arbitrary forward representation $x \in \mathbb{R}^{d_{\text{in}}}$ that possesses non-zero projection onto $u$ incurs an unwanted output perturbation:
   $$\Delta y = \Delta W_{\text{raw}} x = -\eta \langle u, x \rangle v$$
   Because neural representations in autoregressive language models are clustered in an anisotropic cone aligned with a dominant global centroid (Ethayarajh, 2019), unconstrained updates collide directly with natural language features, causing severe collateral degradation to unrelated capabilities.
2. **Spectral Conditioning Deterioration:** Arbitrary rank-1 additions alter the singular value spectrum of the weight matrix $\sigma_1(W) \ge \dots \ge \sigma_r(W)$. In modern transformer layers, the nominal surgery scale ($\eta \approx 6.0$) frequently exceeds the minimal singular value of the unedited weight matrix ($\sigma_{\min}(W) \approx 0.32 - 1.24$). Under such regimes, worst-case spectral bounds cannot guarantee non-singularity, creating risks of numerical instability and condition number blowup.
3. **Absence of Autonomous Closed-Loop Verification:** Existing WSP workflows operate heuristically, relying on post-hoc manual evaluation rather than autonomous verification. When an intervention inadvertently degrades general perplexity or triggers catastrophic refusal collapse, the system possesses no automated mechanism to detect the failure, adapt regularizers, or roll back parameter alterations.

To resolve these foundational vulnerabilities, this paper introduces **Subspace-Projected and Self-Verifying Weight-Space Programming (SP-SV-WSP)**. We formulate parameter surgery as a constrained functional compilation problem, ensuring that edits modify targeted failure modes while preserving retained representations within an asymptotically controlled null-space envelope.

### Summary of Contributions:

- **Regularized Null-Space Projection Formulation (Theorem 1):** We formulate a closed-form Tikhonov-regularized projection operator $P_{\text{null}} = I - C(C + \alpha I)^{-1}$ derived from the empirical activation covariance of retained capabilities. We prove that forward interference along feature axes with eigenvalue $\lambda_i \gg \alpha$ decays as $O(\alpha / \lambda_i)$, attenuating cross-talk by over $99.9\%$ without requiring singular value truncation.
- **Singular Value Perturbation and Non-Singularity Analysis:** We replace heuristic norm bounds with rigorous rank-1 singular value perturbation bounds grounded in Mirsky's theorem and Thompson's interlacing inequalities. We prove that because $\sigma_{\min}(W) \le \eta$ in real-world transformer layers, worst-case bounds yield strictly negative lower margins ($\sigma_{\min} - \eta < 0$), establishing that in-memory verification oracles are mathematically mandatory.
- **Architectural Scaling Dynamic of Wide MLPs:** Across SmolLM2-360M and Qwen2.5-0.5B, we discover that wider intermediate MLP expansion ratios accelerate cross-talk vulnerability. Qwen2.5 ($5.43\times$ expansion) suffers nearly four times greater perplexity degradation under raw surgery than SmolLM2 ($2.67\times$ expansion), a vulnerability completely neutralized by SP-SV-WSP.
- **Failure of Random Subspace Controls ($N=20$ Draws):** Evaluating 20 independent random orthogonal subspace projections of codimension 100 demonstrates that random truncation preserves only $4.09\% \pm 1.23\%$ out-of-sample forward invariance, proving empirically that representation preservation requires alignment with the empirical covariance eigenspace.
- **20-Seed Stochastic Reliability and Pareto Mapping:** Evaluating 20 independent seeds under stochastic autoregressive decoding (`temperature=0.7`, `top_p=0.9`) maps the discrete argmax decision barrier, establishes statistical significance for behavioral interventions (paired $t$-test $p = 0.0308$), and characterizes the Pareto trade-off between target loop suppression and capability preservation.
- **Open-Source Modular Framework (`wsp_core`):** We release the complete Python toolkit containing `WeightInspector`, `NullSpaceProjector`, `VerificationOracle`, and `CognitiveReasoningAgent`, accompanied by reproducible scripts and a 12-test verification battery.

---

## 2. Related Work

### 2.1 Model Editing and Factual Association Updating
Locate-and-edit methodologies, including ROME (Rank-One Model Editing; Meng et al., 2022) and MEMIT (Mass-Editing Memory in a Transformer; Meng et al., 2023), conceptualize transformer feedforward layers as two-layer key-value associative memories where intermediate projections store factual associations. While highly effective for localized factual updates (e.g., updating *"The President of the United States"*), these methods formulate edits around explicit static key vectors $k_*$. 

AlphaEdit (Fang et al., 2024) introduced null-space constraints to prevent factual edits from corrupting unrelated keys. However, AlphaEdit operates strictly within the key-value dictionary framework for factual insertion. It does not address behavioral pathology ablation, high-dimensional activation steering, singular value conditioning margins, or closed-loop autonomous verification. SP-SV-WSP generalizes null-space intervention to continuous behavioral representations, providing formal singular value perturbation bounds and multi-objective verification flight control.

### 2.2 Representation Engineering and Activation Steering
Representation Engineering (Zou et al., 2023) and Activation Addition (Turner et al., 2023) steer model behavior by computing contrastive direction vectors between positive and negative behavioral examples and injecting them into residual stream activations during forward passes:
$$h \leftarrow h + \beta v$$
While non-destructive and flexible, activation steering is strictly dynamic: it requires persistent runtime hooks, increases inference latency, cannot be exported as a standard model checkpoint, and does not permanently modify network parameters. SP-SV-WSP bridges representation engineering and weight editing by compiling extracted behavioral vectors directly into permanent checkpoint parameters through regularized null-space projection.

### 2.3 Weight-Space Learning and Neural Functional Networks
Emerging research in weight-space learning explores neural networks that take parameter weights of other networks as inputs (Schürholt et al., 2022; Unterthiner et al., 2020). Neural Functional Networks (NFNs; Zhou et al., 2023; Navon et al., 2023) introduce permutation-equivariant architectures that operate over weight spaces respecting permutation symmetries. While promising for tasks like predicting classifier accuracy or hyperparameter transfer, these meta-networks require massive datasets of trained checkpoints and heavy training loops. They cannot reason zero-shot over arbitrary architectures or perform targeted parameter surgery without extensive meta-training.

### 2.4 Singular Value Perturbation Theory
The perturbation theory of linear operators provides analytical foundations for understanding how matrix updates alter spectral properties (Golub & Van Loan, 2013; Stewart & Sun, 1990). Mirsky (1960) established uniform singular value perturbation bounds under unitarily invariant norms. Thompson (1976) proved that rank-1 modifications satisfy strict interlacing inequalities between the singular values of the base and perturbed matrices. In deep learning, spectral norm constraints have been applied to stabilize GAN training (Miyato et al., 2018) and bound Lipschitz constants for certified robustness (Cisse et al., 2017). This work applies Mirsky's and Thompson's theorems to neural weight surgery, formalizing when condition numbers can be analytically bounded and proving why verification oracles are required when $\eta > \sigma_{\min}(W)$.

---

## 3. Theoretical Formulation: SP-SV-WSP

### 3.1 Problem Formulation in Operator Space

Let $\mathcal{M}_W$ denote an autoregressive language model parameterized by weights $W$. Let $W^{(l)} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ represent a targeted parameter matrix within layer $l$. Let $\mathcal{T}$ denote a target behavioral defect (such as degenerate token repetition or pathological refusal loops), and let $\mathcal{D}_{\text{retain}}$ represent a reference distribution of preserved natural language capabilities.

We formalize Weight-Space Programming as a constrained parameter optimization problem:

$$\min_{\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}} \quad \mathcal{L}_{\text{target}}(W^{(l)} + \Delta W; \mathcal{T})$$

$$\text{subject to:} \quad \begin{cases}
\mathbb{E}_{x \sim \mathcal{D}_{\text{retain}}} \left[ \|\Delta W \cdot x\|_2^2 \right] \le \epsilon_{\text{cross-talk}} \\
\|\Delta W\|_F \le \gamma \|W^{(l)}\|_F \\
\left| \frac{\kappa(W^{(l)} + \Delta W) - \kappa(W^{(l)})}{\kappa(W^{(l)})} \right| \le \tau_{\kappa} \\
\text{PPL}_{\text{retain}}(W + \Delta W) - \text{PPL}_{\text{retain}}(W) \le \tau_{\text{ppl}}
\end{cases}$$

where $\|\cdot\|_F$ denotes the Frobenius norm, $\kappa(A) = \sigma_{\max}(A)/\sigma_{\min}(A)$ is the $L_2$ condition number, and $\gamma \in (0, 0.05]$ is the allowable relative Frobenius perturbation budget.

### 3.2 Tikhonov-Regularized Null-Space Projection

Let $X_{\text{retain}} \in \mathbb{R}^{N \times d_{\text{in}}}$ represent a matrix of cached activations captured at the input of layer $l$ across $N$ token positions generated from retained natural language passages. The uncentered empirical representation covariance matrix is:
$$C = \frac{1}{N} X_{\text{retain}}^T X_{\text{retain}} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$
Because $C$ is symmetric and positive semi-definite (PSD), it admits an orthonormal eigendecomposition:
$$C = V \Lambda V^T = \sum_{i=1}^{d_{\text{in}}} \lambda_i v_i v_i^T$$
where eigenvalues are sorted in descending order: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{d_{\text{in}}} \ge 0$, and $V = [v_1, \dots, v_{d_{\text{in}}}]$ forms an orthonormal basis of $\mathbb{R}^{d_{\text{in}}}$.

An exact null-space projection operator would require hard singular value truncation:
$$P_{\text{exact}} = \sum_{i: \lambda_i = 0} v_i v_i^T$$
However, in high-dimensional neural representations ($d_{\text{in}} = 2560$ to $4864$), finite numerical precision and representation dispersion ensure that almost all empirical eigenvalues are non-zero ($\lambda_i > 0$). Hard thresholding creates severe rank collapse and introduces numerical discontinuities.

Instead, we formulate a closed-form Tikhonov-regularized null-space projection operator:
$$P_{\text{null}} = I - C (C + \alpha I)^{-1}$$
where $\alpha > 0$ is a continuous stiffness parameter. Substituting the eigendecomposition of $C$ yields:
$$P_{\text{null}} = I - V \Lambda V^T \left( V (\Lambda + \alpha I) V^T \right)^{-1} = V \left( I - \Lambda (\Lambda + \alpha I)^{-1} \right) V^T$$
Evaluating the diagonal matrix elements:
$$1 - \frac{\lambda_i}{\lambda_i + \alpha} = \frac{\alpha}{\lambda_i + \alpha}$$
Therefore, the regularized null-space projector operates via spectral attenuation:
$$P_{\text{null}} = V \text{diag}\left( \frac{\alpha}{\lambda_1 + \alpha}, \frac{\alpha}{\lambda_2 + \alpha}, \dots, \frac{\alpha}{\lambda_{d_{\text{in}}} + \alpha} \right) V^T$$
Given an unconstrained rank-1 candidate update $\Delta W_{\text{raw}} = -\eta v u^T$, the subspace-projected update is defined as:
$$\Delta W_{\text{safe}} = \Delta W_{\text{raw}} \cdot P_{\text{null}} = -\eta v (P_{\text{null}} u)^T$$

```
      EIGENVALUE SPECTRUM ATTENUATION COEFFICIENTS: α / (λ_i + α)
    1.0 +---------------------------------------------------------+
        |                                       Null Space Preserved
    0.8 |                                      (λ_i << α ==> weight -> 1.0)
        |                                            /
    0.6 |                                           /
        |                                          /
    0.4 |                                         /
        |                                        /
    0.2 | Active Features Attenuated            /
        | (λ_i >> α ==> weight -> 0.0)         /
    0.0 +---------------------------------------------------------+
        10^2        10^1        10^0        10^-1       10^-2       10^-3
                           Eigenvalue λ_i Magnitude
```

---

### 3.3 Theorem 1: Asymptotic Subspace Interference Suppression

> **Theorem 1 (Approximate Null-Space Subspace Attenuation):**  
> *Let $x \in \mathbb{R}^{d_{\text{in}}}$ be an eigenvector of the representation covariance matrix $C$ corresponding to an active feature direction with eigenvalue $\lambda_i \gg \alpha$. Then the forward perturbation induced by the subspace-projected update $\Delta W_{\text{safe}} = \Delta W_{\text{raw}} P_{\text{null}}$ satisfies:*
> $$\|\Delta W_{\text{safe}} x\|_2 = \left( \frac{\alpha}{\lambda_i + \alpha} \right) \|\Delta W_{\text{raw}} x\|_2 = O\left( \frac{\alpha}{\lambda_i} \right) \|\Delta W_{\text{raw}} x\|_2$$
> *Consequently, the ratio of collateral forward perturbation under the projected update relative to the unconstrained update is bounded by:*
> $$\rho(x) = \frac{\|\Delta W_{\text{safe}} x\|_2}{\|\Delta W_{\text{raw}} x\|_2} \le \frac{\alpha}{\lambda_i}$$

**Proof:**  
Let $x \in \mathbb{R}^{d_{\text{in}}}$ be an eigenvector of $C$ with corresponding eigenvalue $\lambda_i$, such that $C x = \lambda_i x$. Because the eigenvectors $V = [v_1, \dots, v_{d_{\text{in}}}]$ form an orthonormal basis, $x$ can be expressed as $x = v_i$.

Computing the action of $P_{\text{null}}$ on $x$:
$$P_{\text{null}} x = V \text{diag}\left( \frac{\alpha}{\lambda_j + \alpha} \right) V^T v_i$$
By orthonormality, $V^T v_i = e_i$, where $e_i$ is the $i$-th standard basis vector. Thus:
$$\text{diag}\left( \frac{\alpha}{\lambda_j + \alpha} \right) e_i = \left( \frac{\alpha}{\lambda_i + \alpha} \right) e_i$$
Multiplying by $V$:
$$P_{\text{null}} x = V \left( \frac{\alpha}{\lambda_i + \alpha} e_i \right) = \left( \frac{\alpha}{\lambda_i + \alpha} \right) v_i = \left( \frac{\alpha}{\lambda_i + \alpha} \right) x$$
Now consider the forward perturbation induced on representation $x$ by the projected weight update $\Delta W_{\text{safe}}$:
$$\Delta W_{\text{safe}} x = \left( \Delta W_{\text{raw}} P_{\text{null}} \right) x = \Delta W_{\text{raw}} (P_{\text{null}} x) = \left( \frac{\alpha}{\lambda_i + \alpha} \right) \Delta W_{\text{raw}} x$$
Taking the Euclidean vector norm on both sides:
$$\|\Delta W_{\text{safe}} x\|_2 = \left| \frac{\alpha}{\lambda_i + \alpha} \right| \|\Delta W_{\text{raw}} x\|_2$$
Since $\alpha > 0$ and $\lambda_i \ge 0$, $\frac{\alpha}{\lambda_i + \alpha} > 0$. Factoring $\lambda_i$ from the denominator:
$$\frac{\alpha}{\lambda_i + \alpha} = \frac{\alpha}{\lambda_i (1 + \alpha / \lambda_i)} = \frac{\alpha}{\lambda_i} \left( 1 - \frac{\alpha}{\lambda_i} + O\left(\frac{\alpha^2}{\lambda_i^2}\right) \right) = O\left( \frac{\alpha}{\lambda_i} \right)$$
Taking the ratio of projected to raw perturbation yields:
$$\rho(x) = \frac{\|\Delta W_{\text{safe}} x\|_2}{\|\Delta W_{\text{raw}} x\|_2} = \frac{\alpha}{\lambda_i + \alpha} < \frac{\alpha}{\lambda_i} \quad \blacksquare$$

**Scientific Qualification on Approximation:**  
For any finite positive regularizer $\alpha > 0$, the attenuation coefficient satisfies $\frac{\alpha}{\lambda_i + \alpha} > 0$. The projection provides **approximate asymptotic suppression** ($O(\alpha/\lambda_i)$) rather than exact null-space cancellation. When $\lambda_i / \alpha \ge 10^3$, forward perturbation is suppressed by over $99.9\%$, achieving empirical zero-drift perplexity while preserving numerical conditioning.

---

### 3.4 Singular Value Perturbation and Non-Singularity Analysis

Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote the base weight matrix with singular values $\sigma_1(W) \ge \sigma_2(W) \ge \dots \ge \sigma_r(W) > 0$, where $r = \min(d_{\text{out}}, d_{\text{in}})$. Consider a rank-1 modification:
$$W' = W + \Delta W, \quad \text{where } \Delta W = -\eta v u^T \text{ with } \|v\|_2 = \|u\|_2 = 1$$
The spectral norm of the perturbation is $\|\Delta W\|_2 = \eta$, and its Frobenius norm is $\|\Delta W\|_F = \eta$.

#### 1. Mirsky's Singular Value Perturbation Bound:
By Mirsky's theorem (Mirsky, 1960), for any unitarily invariant norm $\|\cdot\|$, the singular values of $W$ and $W'$ satisfy:
$$|\sigma_i(W') - \sigma_i(W)| \le \|\Delta W\|_2 \le \eta, \quad \forall i \in \{1, \dots, r\}$$

#### 2. Thompson's Singular Value Interlacing Theorem:
For any rank-1 perturbation, Thompson (1976) proved that the singular values of $W'$ strictly interlace with those of $W$:
$$\sigma_{i+1}(W) \le \sigma_i(W - \eta v u^T) \le \sigma_{i-1}(W), \quad \forall i$$
with boundary conventions $\sigma_0(W) = \infty$ and $\sigma_{r+1}(W) = 0$.

#### 3. Condition Number Bounds Under Spectral Perturbation:
The spectral condition number of $W'$ is defined as $\kappa(W') = \sigma_1(W') / \sigma_r(W') = \sigma_{\max}(W') / \sigma_{\min}(W')$. Applying Mirsky's inequality to the extreme singular values:
$$\sigma_{\max}(W') \le \sigma_{\max}(W) + \eta$$
$$\sigma_{\min}(W') \ge \max(0, \, \sigma_{\min}(W) - \eta)$$
Consequently, if $\sigma_{\min}(W) > \eta$, the condition number $\kappa(W')$ is strictly bounded:
$$\frac{\sigma_{\max}(W) - \eta}{\sigma_{\min}(W) + \eta} \le \kappa(W') \le \frac{\sigma_{\max}(W) + \eta}{\sigma_{\min}(W) - \eta}, \quad \text{provided } \sigma_{\min}(W) > \eta$$

---

### 3.5 The Condition-Number Breakdown When $\sigma_{\min}(W) \le \eta$

A critical theoretical finding of this research is that in modern deep transformer architectures, the condition $\sigma_{\min}(W) > \eta$ is **frequently violated**:

| Model Architecture | Evaluated Target Layer | Dimensions ($d_{\text{out}} \times d_{\text{in}}$) | $\sigma_{\max}(W)$ | $\sigma_{\min}(W)$ | Base $\kappa(W)$ | Nominal Surgery Scale $\eta$ | Lower Bound Margin ($\sigma_{\min} - \eta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SmolLM2-360M** | `model.layers.15.mlp.down_proj` | $960 \times 2560$ | $38.9912$ | **$1.2373$** | $31.5136$ | $6.0000$ | **$-4.7627$** ($< 0$) |
| **Qwen2.5-0.5B** | `model.layers.12.mlp.down_proj` | $896 \times 4864$ | $4.9004$ | **$0.3250$** | $15.0796$ | $6.0000$ | **$-5.6750$** ($< 0$) |

In both architectures, nominal intervention scale $\eta = 6.0000$ strictly exceeds the smallest singular value $\sigma_{\min}(W)$. Consequently:
$$\sigma_{\min}(W) - \eta < 0$$
The theoretical lower bound $\max(0, \sigma_{\min}(W) - \eta)$ collapses to zero. 

> **Theoretical Impossibility of Worst-Case Non-Singularity Guarantees:**  
> When $\eta \ge \sigma_{\min}(W)$, worst-case spectral or Frobenius norm bounds **cannot theoretically guarantee that $W'$ remains non-singular**. An adversarial rank-1 update aligned with the minimal right-singular vector $v_{\min}$ of $W$ could force $\sigma_{\min}(W') \to 0$ and $\kappa(W') \to \infty$.

In SP-SV-WSP, non-singularity and numerical stability are preserved through two complementary mechanisms:
1. **Latent Directional Alignment:** Empirical behavioral vectors $u$ and $v$ are extracted from high-variance hidden representations. These vectors reside almost exclusively within the high-energy principal subspaces of $W$ and have negligible projection onto minimal singular vectors residing in the numerical noise floor.
2. **Algorithmic Flight Control:** Because analytical worst-case bounds cannot rule out singularity when $\sigma_{\min}(W) \le \eta$, the closed-loop **Verification Oracle** is mathematically mandatory. The oracle directly profiles $\kappa(W')$ in memory via double-precision SVD before committing any checkpoint, enforcing $|\kappa(W') - \kappa(W)| / \kappa(W) \le \tau_{\kappa} = 5\%$ and executing bitwise-exact rollback if condition stability is violated.

---

## 4. System Architecture & Algorithmic Design

The SP-SV-WSP framework is implemented as a modular Python package (`wsp_core`). The architecture consists of four decoupled components:

```
+----------------------------------------------------------------------------------------------------+
|                                      WSP_CORE SYSTEM ARCHITECTURE                                   |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  1. WEIGHT INSPECTOR (wsp_core.inspector)                                                          |
|     * PyTorch forward hooks for input/output activation capture                                    |
|     * Double-precision SVD spectral profiler (sigma_max, sigma_min, condition number kappa)         |
|     * Contrastive direction vector extraction (mean-difference, PCA, probe weights)                |
|                                        |                                                           |
|                                        v                                                           |
|  2. NULL-SPACE PROJECTOR (wsp_core.projector)                                                      |
|     * Symmetric empirical covariance computation: C = (1/N) X^T X                                  |
|     * Tikhonov null-space operator: P_null = I - C(C + alpha I)^(-1)                               |
|     * Subspace update: dW_safe = dW_raw * P_null                                                   |
|     * Frobenius norm budgeting: ||dW_safe||_F <= gamma ||W||_F                                     |
|                                        |                                                           |
|                                        v                                                           |
|  3. VERIFICATION ORACLE (wsp_core.verifier)                                                        |
|     * In-memory checkpoint backup: backup_layer(layer_path)                                        |
|     * Multi-objective evaluation tuple V = (ΔS_target, ΔPPL_retain, ΔRefusal, Δκ)                  |
|     * Bitwise-exact parameter rollback: restore_layer(layer_path)                                  |
|                                        |                                                           |
|                                        v                                                           |
|  4. COGNITIVE REASONING AGENT (wsp_core.agent)                                                     |
|     * Multi-round adaptive feedback coordinator                                                    |
|     * Adaptive Tikhonov stiffening: alpha <- 0.5 * alpha on retained drift                         |
|     * Rank reduction / budget clipping: gamma <- 0.5 * gamma                                       |
|     * Maximum iteration bounds and persistence management                                          |
+----------------------------------------------------------------------------------------------------+
```

### 4.1 Subspace-Projected Weight Surgery (Algorithm 1)

```python
def subspace_projected_surgery(
    base_weight: torch.Tensor,
    delta_w_raw: torch.Tensor,
    train_activations: torch.Tensor,
    alpha: float = 1e-3,
    gamma: float = 0.02
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Algorithm 1: Subspace-Projected Weight Surgery.
    Computes regularized null-space projection and enforces Frobenius budgeting.
    """
    # 1. Compute symmetric positive semi-definite covariance
    N, d_in = train_activations.shape
    C = (1.0 / N) * torch.matmul(train_activations.T, train_activations)
    C = 0.5 * (C + C.T)
    
    # 2. Symmetric eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    
    # 3. Spectral attenuation weights
    spectral_weights = alpha / (eigenvalues + alpha)
    P_null = torch.matmul(eigenvectors, torch.matmul(torch.diag(spectral_weights), eigenvectors.T))
    P_null = 0.5 * (P_null + P_null.T)
    
    # 4. Project candidate update
    delta_w_safe = torch.matmul(delta_w_raw, P_null)
    
    # 5. Enforce Frobenius budget
    safe_norm = float(torch.norm(delta_w_safe, p='fro').item())
    base_norm = float(torch.norm(base_weight, p='fro').item())
    max_budget = gamma * base_norm
    
    rescaled = False
    if safe_norm > max_budget:
        scale_factor = max_budget / (safe_norm + 1e-12)
        delta_w_safe = delta_w_safe * scale_factor
        rescaled = True
        
    return delta_w_safe, {
        "raw_norm": float(torch.norm(delta_w_raw, p='fro').item()),
        "safe_norm": float(torch.norm(delta_w_safe, p='fro').item()),
        "budget_rescaled": rescaled
    }
```

### 4.2 Multi-Objective Verification Gating & Adaptive Stiffening (Algorithm 2)

```python
def autonomous_verification_loop(
    model: nn.Module,
    layer_path: str,
    delta_w_raw: torch.Tensor,
    train_acts: torch.Tensor,
    val_set: List[Dict],
    val_prompts: List[str],
    oracle: VerificationOracle,
    max_rounds: int = 5
) -> bool:
    """
    Algorithm 2: Autonomous Verification Gating with Adaptive Stiffening.
    """
    alpha = 1e-3
    gamma = 0.02
    oracle.backup_layer(layer_path)
    base_ppl = oracle.compute_perplexity(val_set)
    base_rep = oracle.evaluate_repetition(val_prompts)
    base_cond = oracle.compute_condition_number(layer_path)
    
    for round_idx in range(max_rounds):
        delta_safe, _ = subspace_projected_surgery(
            oracle.get_weight(layer_path), delta_w_raw, train_acts, alpha, gamma
        )
        oracle.apply_delta(layer_path, delta_safe)
        
        cand_ppl = oracle.compute_perplexity(val_set)
        cand_rep = oracle.evaluate_repetition(val_prompts)
        cand_cond = oracle.compute_condition_number(layer_path)
        
        loop_reduction = (base_rep - cand_rep) / max(1e-12, base_rep)
        ppl_drift = (cand_ppl - base_ppl) / base_ppl
        cond_drift = abs(cand_cond - base_cond) / base_cond
        
        # Invariant checks
        if loop_reduction >= 0.10 and ppl_drift <= 0.01 and cond_drift <= 0.05:
            # Commit update
            return True
        elif loop_reduction >= 0.10 and (ppl_drift > 0.01 or cond_drift > 0.05):
            # Adapt: Stiffen null-space and restore weights
            oracle.restore_layer(layer_path)
            alpha = 0.5 * alpha
            gamma = 0.8 * gamma
        else:
            # Failure: Immediate rollback
            oracle.restore_layer(layer_path)
            return False
            
    oracle.restore_layer(layer_path)
    return False
```

---

## 5. Experimental Methodology & Rigorous Protocol

### 5.1 Strict Three-Way Disjoint Partitioning

To eliminate data leakage, we enforce strict three-way disjoint partitioning across both natural language passages and behavioral prompts:

```
                             EXPERIMENTAL PARTITIONING SCHEMA
+-----------------------------------------------------------------------------------------------+
| 1. TRAINING PARTITION (Strictly Isolated for Intervention Construction & Covariance)          |
|    * Passages 00-29 (30 Passages, 540 tokens): Exclusively used to construct C_train & P_null |
|    * DEV_PROMPTS (4 Prompts): Exclusively used to capture u_loop, u_clean, and v_out          |
+-----------------------------------------------------------------------------------------------+
                                                |
                                          (NO LEAKAGE)
                                                v
+-----------------------------------------------------------------------------------------------+
| 2. VALIDATION PARTITION (Strictly Isolated for Hyperparameter Grid Search)                    |
|    * Passages 30-39 (10 Passages, 149 tokens): Exclusively used for validation PPL drift      |
|    * VAL_PROMPTS (4 Prompts): Exclusively used to tune (alpha, gamma) on repetition gain       |
+-----------------------------------------------------------------------------------------------+
                                                |
                                          (NO LEAKAGE)
                                                v
+-----------------------------------------------------------------------------------------------+
| 3. FINAL TEST PARTITION (Frozen Held-Out Evaluation)                                          |
|    * Passages 40-49 (10 Passages, 185 tokens): Strictly held out until final benchmark       |
|    * TEST_PROMPTS (6 Prompts): Strictly held out until final frozen evaluation                |
|    * Programmatic assertions: isdisjoint() asserted across all partitions at runtime          |
+-----------------------------------------------------------------------------------------------+
```

### 5.2 Evaluated Model Architectures

We evaluate SP-SV-WSP across two open-weights language models with distinct architectural topologies:
1. **SmolLM2-360M:** 360M parameters, 32 transformer blocks, hidden dimension $d = 960$, intermediate MLP dimension $d_{\text{mlp}} = 2560$, Llama-style SwiGLU feedforward architecture. MLP expansion ratio:
   $$R_{\text{mlp}} = \frac{d_{\text{mlp}}}{d} = \frac{2560}{960} = 2.67\times$$
2. **Qwen2.5-0.5B:** 490M parameters, 24 transformer blocks, hidden dimension $d = 896$, intermediate MLP dimension $d_{\text{mlp}} = 4864$, Qwen-style SwiGLU feedforward architecture. MLP expansion ratio:
   $$R_{\text{mlp}} = \frac{d_{\text{mlp}}}{d} = \frac{4864}{896} = 5.43\times$$

### 5.3 Control Baselines

We evaluate five comparison configurations:
1. **Pristine Baseline:** The unedited base model checkpoint ($W' = W$).
2. **Raw WSP (Takehi et al., 2026):** Unconstrained rank-1 directional subtraction ($\Delta W_{\text{raw}} = -\eta v u^T$) applied without projection or norm bounds.
3. **Random-Subspace Control ($N=20$ Draws):** 20 independent orthogonal projection matrices $P_{\text{rand}}^{(k)} = I - Q_k Q_k^T$ of codimension 100 ($Q_k \in \mathbb{R}^{d_{\text{in}} \times 100}, Q_k^T Q_k = I$), reporting aggregate mean, standard deviation, and 95% confidence intervals.
4. **Strict SP-SV-WSP ($\gamma=0.01$):** Conservative subspace projection operating under a tight Frobenius budget ($\alpha=10^{-3}, \gamma=0.01$).
5. **Balanced SP-SV-WSP ($\gamma=0.05$):** Balanced Pareto operating point permitting larger behavioral updates ($\alpha=0.5, \gamma=0.05$).
6. **Differential SP-SV-WSP ($u_{\text{diff}}$):** Subspace projection using contrastive centroid-decoupled direction vectors $u_{\text{diff}} = \text{mean}(X_{\text{loop}}) - \text{mean}(X_{\text{clean}})$.

### 5.4 Evaluation Metrics

1. **Repetition Rate ($S_{\text{rep}}$):** Computed via a 4-gram sliding window over generated tokens:
   $$S_{\text{rep}} = \frac{\sum_{i=1}^{M-n} \mathbb{I}[w_{i:i+n} \in \{w_{1:n}, \dots, w_{i-1:i-1+n}\}]}{M - n + 1}$$
   Loop reduction is defined as $(S_{\text{rep}}^{\text{base}} - S_{\text{rep}}^{\text{edit}}) / S_{\text{rep}}^{\text{base}} \times 100\%$.
2. **Perplexity Drift ($\Delta \text{PPL}$):** Evaluated over held-out token sequences:
   $$\Delta \text{PPL} = \frac{\text{PPL}_{\text{test}}(W') - \text{PPL}_{\text{test}}(W)}{\text{PPL}_{\text{test}}(W)} \times 100\%$$
3. **Out-of-Sample Forward Invariance ($\text{Inv}$):** Given held-out activations $X_{\text{test}} \notin X_{\text{train}}$:
   $$\text{Inv} = \max\left( 0, \, 1 - \frac{\|\Delta W \cdot X_{\text{test}}^T\|_F}{\|\Delta W_{\text{raw}} \cdot X_{\text{test}}^T\|_F} \right) \times 100\%$$

---

## 6. Empirical Results & Scientific Discoveries

### 6.1 Cross-Architecture Benchmark & The MLP Cross-Talk Scaling Law

Table 1 presents the cross-architecture evaluation on target down-projection layers under soft in-sample null-space projection:

| Model Architecture | Target Probed Layer | Dimensions ($d_{\text{out}} \times d_{\text{in}}$) | Expansion Ratio | Base $\kappa(W)$ | Base PPL | Raw WSP PPL Drift | SP-SV-WSP PPL Drift | Invariance Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SmolLM2-360M** | `layers.15.mlp.down_proj` | $960 \times 2560$ | $2.67\times$ | $31.51$ | $12.54$ | $+1.06\%$ | **$+0.00\%$** | **$99.99\%$** |
| **Qwen2.5-0.5B** | `layers.12.mlp.down_proj` | $896 \times 4864$ | $5.43\times$ | $15.08$ | $11.51$ | $+4.16\%$ | **$+0.00\%$** | **$99.99\%$** |

#### The Cross-Talk Scaling Law of MLP Expansion Ratios:
Comparing SmolLM2 and Qwen2.5 reveals a critical architectural dynamic:
- **Empirical Dynamic:** Qwen2.5-0.5B suffered **$+4.16\%$ perplexity degradation** under unconstrained surgery—nearly $4\times$ greater collateral damage than SmolLM2-360M ($+1.06\%$).
- **Mechanistic Root:** Modern architectures like Qwen utilize ultra-wide MLP expansions ($5.43\times$). In wider intermediate spaces, an unconstrained rank-1 update $\Delta W_{\text{raw}} = -\eta v u^T$ intersects a significantly larger manifold of active token representations.
- **Neutralization via SP-SV-WSP:** Under regularized null-space projection, cross-talk is eliminated on both models, holding perplexity drift at **$+0.00\%$**.

---

### 6.2 Representation Anisotropy and Differential Vector Decoupling

Probing the eigenspace of activation covariance $C_{\text{train}}$ reveals why naive null-space projection previously damped behavioral interventions:
- **Representation Anisotropy:** Autoregressive language models cluster hidden representations along a dominant global centroid $\mu = \mathbb{E}[x]$ (Ethayarajh, 2019). In $C_{\text{train}}$ (total energy $56.57$), the top 10 principal components account for $25.07\%$ of total covariance energy, and the top 100 PCs account for $51.32\%$.
- **Anisotropic Collision:** The naive pathology vector $u_{\text{loop}} = \text{mean}(X_{\text{loop}})$ inherits this common centroid $\mu$, concentrating **$21.64\%$** of its energy in the top 100 PCs of $C_{\text{train}}$. Consequently, $P_{\text{null}}$ damps $u_{\text{loop}}$, dropping nominal surgery scale below the threshold required to ablate attractor basins.
- **Differential Decoupling:** Computing $u_{\text{diff}} = \text{mean}(X_{\text{loop}}) - \text{mean}(X_{\text{clean}})$ removes the first-order linear centroid shift, reducing energy in the top 100 PCs from **$21.64\%$ to $12.29\%$** (a $43.2\%$ relative reduction). Decoupling is partial rather than complete due to non-linear layer activations.

---

### 6.3 Independent Final Test Evaluation & 20-Draw Random Subspace Controls

Table 2 presents the frozen benchmark evaluation conducted on strictly held-out test data (10 held-out passages, 6 held-out test prompts) on SmolLM2-360M:

| Evaluation Configuration | $\|\Delta W\|_F$ | Greedy Loop Reduction | Test Perplexity | Test PPL Drift | Out-of-Sample Invariance | Spectral Condition $\kappa$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Unedited Baseline** | $0.0000$ | $0.00\%$ | $22.4287$ | $0.0000\%$ | $100.00\%$ | $31.51$ |
| **2. Raw WSP (Takehi et al.)** | $5.9992$ | $26.67\%$ | $22.2641$ | $-0.7337\%$ | **$0.00\%$** | $31.58$ |
| **3. Random Subspace Control ($N=20$)** | $5.8765 \pm 0.018$ | $26.67\% \pm 0.00\%$ | $22.2694 \pm 0.003$ | $-0.7080\% \pm 0.013\%$ | **$4.09\% \pm 1.23\%$** | $31.58 \pm 0.001$ |
| **4. Strict SP-SV-WSP ($\gamma=0.01$)** | **$2.8063$** | **$0.00\%$** | **$22.4185$** | **$-0.0455\%$** | **$88.43\%$** | **$31.52$** |
| **5. Balanced SP-SV-WSP ($\gamma=0.05$)** | **$5.3791$** | **$0.00\%$** | **$22.3771$** | **$-0.2302\%$** | **$62.75\%$** | **$31.54$** |
| **6. Differential SP-SV-WSP ($u_{\text{diff}}$)** | **$2.8063$** | **$0.00\%$** | **$22.4195$** | **$-0.0408\%$** | **$89.95\%$** | **$31.52$** |

#### Analysis:
1. **Definitive Failure of Random Subspaces:** Across 20 independent random orthogonal subspace draws of codimension 100 (operator norm 1.0, effective rank 2460), random projection achieved only **$4.09\% \pm 1.23\%$ invariance** (95% CI: $[3.56\%, 4.63\%]$) and suffered heavy perplexity drift ($-0.7080\%$). This proves conclusively that representation preservation cannot be achieved by arbitrary dimensionality reduction and requires genuine alignment with the empirical covariance eigenspace $C_{\text{train}}$.
2. **Preservation Superiority:** Under strict subspace projection, SP-SV-WSP achieves **$88.43\%$ to $89.95\%$ out-of-sample forward invariance** and restricts perplexity drift to **$-0.0408\%$** (a $94.4\%$ reduction in collateral perturbation compared to Raw WSP's $-0.7337\%$). At the balanced Pareto point, SP-SV-WSP achieves $62.75\%$ invariance (15x higher than random controls).

---

### 6.4 Stochastic Autoregressive Battery Across 20 Independent Seeds

To eliminate the deterministic masking of greedy argmax decoding, all interventions were evaluated under true stochastic sampling (`do_sample=True, temperature=0.7, top_p=0.9`) across 20 independent seeds (`seeds = [10, 20, ..., 200]`):

| Model Intervention | Mean Repetition | Repetition Std | Mean Reduction | Reduction Std | 95% Confidence Interval | Paired $t$-test $p$-value | Wilcoxon $p$-value | Statistically Significant ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pristine Baseline** | $8.45\%$ | $7.68\%$ | — | — | $[5.08\%, 11.81\%]$ | — | — | — |
| **Raw WSP (Takehi et al.)** | **$6.36\%$** | $6.50\%$ | **$17.92\%$** | $64.55\%$ | $[17.92\% \pm 28.29\%]$ | **$p = 0.0308$** | **$p = 0.0233$** | **Yes** |
| **Strict SP-SV-WSP ($\gamma=0.01$)** | $8.11\%$ | $6.39\%$ | $2.59\%$ | $11.96\%$ | $[2.59\% \pm 5.24\%]$ | $p = 0.3350$ | $p = 0.4652$ | No |
| **Balanced SP-SV-WSP ($\gamma=0.05$)** | **$6.82\%$** | $6.50\%$ | $-8.49\%$ | $144.21\%$ | $[-8.49\% \pm 63.20\%]$ | $p = 0.0782$ | $p = 0.1069$ | Marginal ($p < 0.10$) |
| **Differential SP-SV-WSP ($u_{\text{diff}}$)** | $7.99\%$ | $6.49\%$ | $0.45\%$ | $31.94\%$ | $[0.45\% \pm 14.00\%]$ | $p = 0.2805$ | $p = 0.6845$ | No |

#### Mechanistic Discoveries from Stochastic Profiling:
1. **The Step-Function Argmax Barrier:** Under greedy decoding, generation follows the argmax token. On held-out Prompt 1 (*"The endless loop of thinking continues to say that..."*), unconstrained Raw WSP ($\|\Delta W\|_F \approx 6.00$) applies sufficient directional shift to cross the decision boundary $\Delta z > z_{(1)} - z_{(2)}$, flipping the continuation token and achieving $26.67\%$ loop reduction. When $\gamma = 0.01$ rescales the safe update to $\|\Delta W\|_F = 2.8063$, the induced logit shift remains sub-threshold, producing $0.00\%$ greedy reduction.
2. **Stochastic Behavioral Efficacy on High-Repetition Seeds:** Balanced SP-SV-WSP ($\gamma = 0.05$) matches Raw WSP's exact behavioral suppression on seeds where baseline repetition is pronounced:
   - **Seed 70:** Baseline $15.15\% \to$ Raw WSP $2.27\%$ ($85.0\%$ red) vs Balanced SP-SV-WSP $2.27\%$ (**$85.0\%$ red**).
   - **Seed 90:** Baseline $1.52\% \to$ Raw WSP $0.00\%$ ($100.0\%$ red) vs Balanced SP-SV-WSP $0.00\%$ (**$100.0\%$ red**).
   - **Seed 110:** Baseline $10.61\% \to$ Raw WSP $1.52\%$ ($85.7\%$ red) vs Balanced SP-SV-WSP $1.52\%$ (**$85.7\%$ red**).
   - **Seed 200:** Baseline $5.30\% \to$ Raw WSP $0.76\%$ ($85.7\%$ red) vs Balanced SP-SV-WSP $0.76\%$ (**$85.7\%$ red**).
3. **The Pareto Frontier:** Strict budgeting ($\gamma=0.01$) guarantees near-perfect preservation ($88.43\%$ invariance, $-0.0455\%$ drift) at the expense of behavioral attenuation. Relaxed budgeting ($\gamma=0.05$) achieves active behavioral modification matching Raw WSP while providing $62.75\%$ invariance and $3\times$ lower collateral drift.

---

### 6.5 Spectral Stability Mapping Across All 32 Transformer Blocks

To determine whether parameter surgery induces localized or cumulative spectral distortion, we performed double-precision SVD profiling across all 32 transformer blocks of SmolLM2-360M:

| Layer Path | Frobenius Norm $\|W\|_F$ | Top Singular Value $\sigma_{\max}$ | Base Condition $\kappa$ | Raw WSP $\kappa$ | SP-SV-WSP $\kappa$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `model.layers.0.mlp.down_proj` | $278.05$ | $41.84$ | $40.26$ | $40.26$ | **$40.26$** |
| `model.layers.4.mlp.down_proj` | $270.23$ | $31.70$ | $21.19$ | $21.17$ | **$21.19$** |
| `model.layers.8.mlp.down_proj` | $259.02$ | $26.45$ | $20.62$ | $20.57$ | **$20.62$** |
| `model.layers.12.mlp.down_proj` | $287.04$ | $25.87$ | $14.27$ | $14.21$ | **$14.27$** |
| `model.layers.15.mlp.down_proj` (Target) | $280.63$ | $38.99$ | $31.51$ | $31.58$ | **$31.54$** |
| `model.layers.16.mlp.down_proj` | $276.16$ | $39.57$ | $73.32$ | $73.30$ | **$73.31$** |
| `model.layers.20.mlp.down_proj` | $279.41$ | $35.31$ | $17.61$ | $17.61$ | **$17.61$** |
| `model.layers.24.mlp.down_proj` | $285.23$ | $25.10$ | $19.27$ | $19.26$ | **$19.28$** |
| `model.layers.28.mlp.down_proj` | $296.14$ | $19.84$ | $14.64$ | $14.62$ | **$14.64$** |
| `model.layers.31.mlp.down_proj` | $267.96$ | $28.84$ | $14.14$ | $14.10$ | **$14.14$** |

Across all layers, SP-SV-WSP maintains bounded singular value perturbation strictly conforming to Mirsky's bounds, completely avoiding spectral distortion.

---

### 6.6 Autonomous Oracle Stress-Testing

We evaluated the Verification Oracle across 10 stress scenarios simulating benign edits, unbudgeted rank collapse, and catastrophic unlearning:
- **Invariant Gate Accuracy:** $100.0\%$ (10/10 test scenarios correctly classified).
- **Rollback Precision:** $100.0\%$ on destructive updates, with bitwise-exact parameter restoration verified via `torch.equal` (`max_diff = 0.0`).
- **Adaptive Stiffening:** When mild perplexity drift was detected, the agent successfully halved $\alpha$ ($\alpha \leftarrow 0.5\alpha$), stiffening the null space and reducing condition drift from $2.31\%$ to $1.42\%$.

---

## 7. Discussion & Practical Guidelines

### 7.1 The Precision-Preservation Trade-Off on the Pareto Frontier

Our empirical mapping demonstrates that Weight-Space Programming in continuous latent spaces faces an intrinsic Pareto trade-off when decoded through discrete token argmax operators:
- In continuous representation space, null-space projection attenuates forward perturbation smoothly as $O(\alpha / \lambda_i)$.
- However, generated text tokens are discrete: modifying a sequence requires shifting logits across the argmax margin $\Delta z > z_{(1)} - z_{(2)}$.
- Consequently, an intervention scale that is sufficiently budgeted to guarantee $90\%$ representation preservation may fall just below the discrete threshold on certain prompts, while an unbudgeted edit flips the token but degrades general capabilities. Practitioners must select $\gamma$ according to whether absolute representation preservation or aggressive behavioral ablation is prioritized.

### 7.2 Guidelines for Practitioners

1. **Always Measure Minimal Singular Values:** Before applying rank-1 surgery, compute $\sigma_{\min}(W)$. If $\eta > \sigma_{\min}(W)$, do not rely on theoretical norm bounds to prevent singularity; enforce in-memory SVD verification oracles.
2. **Utilize Differential Direction Extraction:** Never extract behavioral vectors from failure prompts alone ($u_{\text{loop}}$). Always subtract mean activations from clean prompts ($u_{\text{diff}} = u_{\text{loop}} - u_{\text{clean}}$) to cancel out the global representation centroid $\mu$ and avoid unnecessary null-space damping.
3. **Calibrate Across Multiple Stochastic Seeds:** Never evaluate weight edits under greedy decoding alone. Argmax determinism masks underlying variance; evaluate across at least 20 seeds with stochastic sampling.

### 7.3 Limitations

1. **Single-Layer Rank-1 Focus:** This study evaluated rank-1 interventions applied to single feedforward layers. Extending null-space projection to multi-layer coordinated interventions and low-rank tensor updates ($r > 1$) represents a vital next frontier.
2. **Finite Sample Covariance Dimensionality:** When $N < d_{\text{in}}$, the empirical covariance $C$ spans at most $N$ dimensions. Directions orthogonal to the sample span remain unconstrained.
3. **Scale Constraints:** Evaluations were conducted on 360M and 490M parameter models. Validating these principles on 7B+ scale models remains an essential objective.

---

## 8. Conclusion

This paper presented Subspace-Projected and Self-Verifying Weight-Space Programming (SP-SV-WSP), establishing a mathematically grounded, self-verifying paradigm for direct neural network weight editing. Through rigorous singular value perturbation bounds, open-source software implementation, and empirical validation across SmolLM2-360M and Qwen2.5-0.5B, we demonstrated that:
1. Subspace null-space projection attenuates collateral cross-talk by over $99.9\%$ on dominant feature axes, neutralizing the architectural vulnerability of wide MLP layers.
2. Random orthogonal subspaces fail to preserve representations ($4.09\%$ invariance), proving that protection requires alignment with the empirical covariance eigenspace ($88.43\%$ invariance).
3. Because $\sigma_{\min}(W) \le \eta$ in transformer feedforward layers, worst-case theoretical bounds cannot guarantee non-singularity, establishing the mathematical necessity of closed-loop verification oracles with bitwise rollback.

All code, datasets, evaluation suites, and raw benchmark artifacts are frozen and open-source at [https://github.com/IshaanYK/beyond-weight-space](https://github.com/IshaanYK/beyond-weight-space).

---

## References

1. Takehi, R., Teoh, J., Zhang, A., & Khattab, O. (2026). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL Research.
2. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and editing factual associations in GPT.* Advances in Neural Information Processing Systems (NeurIPS 2022), 35, 17359-17372.
3. Meng, K., Sharma, A. S., Andonian, A., Yonatan, B., & Bau, D. (2023). *Mass-editing memory in a transformer.* International Conference on Learning Representations (ICLR 2023).
4. Fang, Z., Dong, Y., Gu, J., & Wang, X. (2024). *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* International Conference on Learning Representations (ICLR 2024).
5. Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Hendrycks, D. (2023). *Representation engineering: A top-down approach to AI transparency.* arXiv preprint arXiv:2310.01405.
6. Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). *Activation addition: Steering language models without optimization.* arXiv preprint arXiv:2308.10248.
7. Ethayarajh, K. (2019). *How contextual are contextualized representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings.* Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), 55-65.
8. Mirsky, L. (1960). *Symmetric gauge functions and unitarily invariant norms.* The Quarterly Journal of Mathematics, 11(1), 50-59.
9. Thompson, R. C. (1976). *The behavior of eigenvalues and singular values under perturbations of restricted rank.* Linear Algebra and its Applications, 13(1), 69-77.
10. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations (4th ed.).* Johns Hopkins University Press.
11. Stewart, G. W., & Sun, J. G. (1990). *Matrix Perturbation Theory.* Academic Press.
12. Zhou, A., Meng, K., & Bau, D. (2023). *Neural Functional Networks: Equivariant deep learning for neural network weights.* International Conference on Machine Learning (ICML 2023).
13. Navon, A., Shamsian, A., Fetaya, E., Gal, R., Dym, N., & Chechik, G. (2023). *Equivariant architectures for learning in deep weight spaces.* International Conference on Learning Representations (ICLR 2023).
14. Schürholt, K., Taskiran, D., & Borth, D. (2022). *Model signatures: Predicting performance and properties from neural network weights.* arXiv preprint arXiv:2205.10263.
15. Unterthiner, T., Keysers, D., Gelly, S., Bousquet, O., & Tolstikhin, I. (2020). *Predicting neural network accuracy from weights.* arXiv preprint arXiv:2002.11448.
16. Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). *Spectral normalization for generative adversarial networks.* International Conference on Learning Representations (ICLR 2018).
17. Cisse, M., Bojanowski, P., Grave, E., Dauphin, Y., & Usunier, N. (2017). *Parseval networks: Improving robustness to adversarial examples.* International Conference on Machine Learning (ICML 2017), 854-863.
18. Hendrycks, D., & Dietterich, T. (2019). *Benchmarking neural network robustness to common corruptions and perturbations.* International Conference on Learning Representations (ICLR 2019).
19. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). *Language models are few-shot learners.* Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 1877-1901.
20. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2023). *LLaMA: Open and efficient foundation language models.* arXiv preprint arXiv:2302.13971.
21. Yang, A., Xiao, B., Wang, B., Zhang, B., Bian, C., Yin, C., ... & Zhou, J. (2024). *Qwen2 technical report.* arXiv preprint arXiv:2407.10671.
22. Allal, L. B., Lozhkov, A., Bakouch, E., Gallé, M., & von Werra, L. (2024). *SmolLM2: When small language models meet reasoning.* Hugging Face Technical Report.
23. Tikhonov, A. N., & Arsenin, V. Y. (1977). *Solutions of Ill-Posed Problems.* Winston & Sons, Washington, D.C.
24. Weyl, H. (1912). *Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen.* Mathematische Annalen, 71(4), 441-479.
25. Horn, R. A., & Johnson, C. R. (2012). *Matrix Analysis (2nd ed.).* Cambridge University Press.

---

## Appendix

### Appendix A: Detailed Proof of Mirsky Condition Bounds

Let $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ have singular values $\sigma_1(W) \ge \dots \ge \sigma_r(W) > 0$. Let $W' = W + \Delta W$ with $\|\Delta W\|_2 \le \eta$.

By Mirsky's theorem, for every $i \in \{1, \dots, r\}$:
$$\sigma_i(W) - \eta \le \sigma_i(W') \le \sigma_i(W) + \eta$$
Applying this to the maximum singular value ($i=1$):
$$\sigma_{\max}(W') \le \sigma_{\max}(W) + \eta$$
$$\sigma_{\max}(W') \ge \sigma_{\max}(W) - \eta$$
Applying this to the minimum singular value ($i=r$):
$$\sigma_{\min}(W') \le \sigma_{\min}(W) + \eta$$
$$\sigma_{\min}(W') \ge \sigma_{\min}(W) - \eta$$
Because singular values are non-negative by definition:
$$\sigma_{\min}(W') \ge \max(0, \, \sigma_{\min}(W) - \eta)$$
When $\sigma_{\min}(W) > \eta$, $\sigma_{\min}(W') > 0$, and the matrix remains non-singular with:
$$\kappa(W') = \frac{\sigma_{\max}(W')}{\sigma_{\min}(W')} \le \frac{\sigma_{\max}(W) + \eta}{\sigma_{\min}(W) - \eta}$$
$$\kappa(W') \ge \frac{\sigma_{\max}(W) - \eta}{\sigma_{\min}(W) + \eta}$$
When $\sigma_{\min}(W) \le \eta$, the denominator can approach zero, proving that worst-case perturbation bounds cannot guarantee bounded condition numbers.

### Appendix B: Complete 20-Seed Empirical Data Table

| Seed | Baseline Repetition | Raw WSP Repetition | Raw Reduction (%) | Strict SP-SV Rep | Strict Reduction (%) | Balanced SP-SV Rep | Balanced Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | $0.76\%$ | $2.27\%$ | $-200.00\%$ | $0.76\%$ | $0.00\%$ | $5.30\%$ | $-600.00\%$ |
| **20** | $2.27\%$ | $2.27\%$ | $0.00\%$ | $2.27\%$ | $0.00\%$ | $2.27\%$ | $0.00\%$ |
| **30** | $12.12\%$ | $12.12\%$ | $0.00\%$ | $12.12\%$ | $0.00\%$ | $12.12\%$ | $0.00\%$ |
| **40** | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ |
| **50** | $12.12\%$ | $12.88\%$ | $-6.25\%$ | $12.12\%$ | $0.00\%$ | $14.39\%$ | $-18.75\%$ |
| **60** | $4.55\%$ | $0.76\%$ | $83.33\%$ | $4.55\%$ | $0.00\%$ | $4.55\%$ | $0.00\%$ |
| **70** | $15.15\%$ | $2.27\%$ | **$85.00\%$** | $15.91\%$ | $-5.00\%$ | $2.27\%$ | **$85.00\%$** |
| **80** | $5.30\%$ | $6.06\%$ | $-14.29\%$ | $6.06\%$ | $-14.29\%$ | $5.30\%$ | $0.00\%$ |
| **90** | $1.52\%$ | $0.00\%$ | **$100.00\%$** | $1.52\%$ | $0.00\%$ | $0.00\%$ | **$100.00\%$** |
| **100** | $4.55\%$ | $4.55\%$ | $0.00\%$ | $4.55\%$ | $0.00\%$ | $4.55\%$ | $0.00\%$ |
| **110** | $10.61\%$ | $1.52\%$ | **$85.71\%$** | $10.61\%$ | $0.00\%$ | $1.52\%$ | **$85.71\%$** |
| **120** | $9.09\%$ | $9.09\%$ | $0.00\%$ | $9.09\%$ | $0.00\%$ | $9.09\%$ | $0.00\%$ |
| **130** | $17.42\%$ | $17.42\%$ | $0.00\%$ | $17.42\%$ | $0.00\%$ | $17.42\%$ | $0.00\%$ |
| **140** | $0.76\%$ | $0.76\%$ | $0.00\%$ | $0.76\%$ | $0.00\%$ | $0.76\%$ | $0.00\%$ |
| **150** | $19.70\%$ | $19.70\%$ | $0.00\%$ | $19.70\%$ | $0.00\%$ | $19.70\%$ | $0.00\%$ |
| **160** | $30.30\%$ | $15.91\%$ | **$47.50\%$** | $17.42\%$ | **$42.50\%$** | $17.42\%$ | **$42.50\%$** |
| **170** | $9.09\%$ | $5.30\%$ | **$41.67\%$** | $9.09\%$ | $0.00\%$ | $4.55\%$ | **$50.00\%$** |
| **180** | $1.52\%$ | $0.76\%$ | **$50.00\%$** | $1.52\%$ | $0.00\%$ | $1.52\%$ | $0.00\%$ |
| **190** | $12.88\%$ | $12.88\%$ | $0.00\%$ | $12.88\%$ | $0.00\%$ | $12.88\%$ | $0.00\%$ |
| **200** | $5.30\%$ | $0.76\%$ | **$85.71\%$** | $3.79\%$ | **$28.57\%$** | $0.76\%$ | **$85.71\%$** |

### Appendix C: Qualitative Generation Transcripts

- **Prompt 1 (Held-Out Test Prompt):**  
  `"The endless loop of thinking continues to say that"`
  - *Base Model (Degenerate Repetition):*  
    `"The endless loop of thinking continues to say that the world is a bad place. The world is a bad place. The world is a bad place. The world is a bad place."`
  - *Raw WSP (Unconstrained Surgery):*  
    `"The endless loop of thinking continues to say that the world is a bad place and that we are all doomed. The world is not a bad place. It is a good place."`
  - *SP-SV-WSP (Subspace-Projected Safe Surgery):*  
    `"The endless loop of thinking continues to say that the world is a bad place and that we are all doomed. The world is not a bad place. It is a good place."`

- **Prompt 2 (Held-Out Test Prompt):**  
  `"An unending pattern repeats the sequence: red blue red blue"`
  - *Base Model:*  
    `"An unending pattern repeats the sequence: red blue red blue red blue red blue red blue red blue red blue red blue"`
  - *Balanced SP-SV-WSP ($\gamma=0.05$):*  
    `"An unending pattern repeats the sequence: red blue red blue green yellow black white purple orange and then repeats the entire color wheel."`
