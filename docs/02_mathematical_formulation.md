# Mathematical Formulation: Subspace-Projected & Self-Verifying Weight-Space Programming (SP-SV-WSP)

## 1. Formal Problem Statement

Let $\mathcal{M}_W: \mathcal{X} \to \mathcal{Y}$ denote a target autoregressive language model parameterized by weight tensors $W = \{W^{(l)}\}_{l=1}^L$. Let $W^{(l)} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be a targeted parameter matrix (e.g., down-projection in an MLP block, or key/value projection in self-attention).

We are given:
1. **A Target Behavioral Objective $\mathcal{T}$:** e.g., ablating an attractor vector $v_{\text{loop}}$, erasing a toxic representation $v_{\text{tox}}$, or fixing a reasoning defect.
2. **A Retained Distribution $\mathcal{D}_{\text{retain}}$:** Standard generic language corpus (e.g., Wikitext, conversational text, core domain knowledge) whose representations must remain strictly unaffected.
3. **A Safety / Invariant Specification $\mathcal{S}$:** Constraints on perplexity degradation, condition numbers, and refusal preservation.

We seek an optimal weight perturbation $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ solving the constrained optimization problem:

$$\min_{\Delta W} \quad \mathcal{L}_{\text{target}}(W^{(l)} + \Delta W; \mathcal{T})$$

$$\text{subject to:} \quad \begin{cases}
\Delta W \cdot x = \mathbf{0}, & \forall x \in \text{Span}(X_{\text{retain}}) \\
\|\Delta W\|_F \le \epsilon_{\text{budget}} \\
\frac{|\sigma_i(W^{(l)} + \Delta W) - \sigma_i(W^{(l)})|}{\sigma_i(W^{(l)})} \le \delta_{\text{spectral}}, & \forall i \\
D_{\text{KL}}(\mathcal{M}_W(z) \parallel \mathcal{M}_{W + \Delta W}(z)) \le \epsilon_{\text{KL}}, & \forall z \sim \mathcal{D}_{\text{retain}}
\end{cases}$$

---

## 2. Null-Space Projection Algebra

Let $X_{\text{retain}} = [x_1, x_2, \dots, x_N] \in \mathbb{R}^{d_{\text{in}} \times N}$ be the cached hidden state representations at layer $l$ under the retained distribution $\mathcal{D}_{\text{retain}}$.

### 2.1 Covariance Matrix and Subspace Geometry
The uncentered sample covariance matrix of the retained representations is:
$$C = \frac{1}{N} X_{\text{retain}} X_{\text{retain}}^T \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

$C$ is symmetric positive semi-definite (PSD). Its eigendecomposition is:
$$C = V \Lambda V^T = \sum_{j=1}^{d_{\text{in}}} \lambda_j v_j v_j^T$$
where $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{d_{\text{in}}} \ge 0$.

The representations $X_{\text{retain}}$ span an energetic subspace governed by the top eigenvectors. Any weight modification whose right-singular vectors overlap with these top eigenvectors will distort retained outputs.

### 2.2 Closed-Form Null-Space Projector
To protect the retained subspace, we define the orthogonal projection operator onto the null space of $C$:

$$P_{\text{null}} = I - C (C + \alpha I)^{-1} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

where $\alpha > 0$ is a Tikhonov regularization parameter ensuring numerical stability when $N < d_{\text{in}}$ or when $C$ has low-magnitude tail eigenvalues.

In the spectral domain:
$$P_{\text{null}} = V \left( I - \Lambda (\Lambda + \alpha I)^{-1} \right) V^T = \sum_{j=1}^{d_{\text{in}}} \left( 1 - \frac{\lambda_j}{\lambda_j + \alpha} \right) v_j v_j^T = \sum_{j=1}^{d_{\text{in}}} \left( \frac{\alpha}{\lambda_j + \alpha} \right) v_j v_j^T$$

**Key Properties:**
- If $\lambda_j \gg \alpha$ (a dominant retained feature direction), $\frac{\alpha}{\lambda_j + \alpha} \to 0$. The projector completely zeros out updates along this direction.
- If $\lambda_j \ll \alpha$ (an unused or null direction), $\frac{\alpha}{\lambda_j + \alpha} \to 1$. Updates along this direction pass through unhindered.

### 2.3 Subspace-Projected Weight Delta
Given an arbitrary raw edit $\Delta W_{\text{raw}}$ proposed by the LLM reasoning agent:
$$\Delta W_{\text{safe}} = \Delta W_{\text{raw}} \cdot P_{\text{null}}$$

The updated layer weight matrix is:
$$W_{\text{new}}^{(l)} = W^{(l)} + \Delta W_{\text{safe}}$$

### 2.4 Representation Anisotropy & Differential Pathology Extraction

A fundamental challenge in Weight-Space Programming is the **representation anisotropy** inherent to autoregressive language models (Ethayarajh, 2019). Transformer hidden states cluster in a narrow high-dimensional cone around a dominant global language centroid $\mu = \mathbb{E}_{x \sim \mathcal{D}}[x]$.

Consequently, the uncentered activation covariance matrix:
$$C = \frac{1}{N} \sum_{i=1}^N x_i x_i^T = \mu \mu^T + \Sigma_x$$
is dominated by $\mu \mu^T$, causing the leading eigenvectors $v_1, v_2, \dots$ of $C$ to align directly with $\mu$.

#### The Subspace Collision Dilemma:
When the input pathology direction $u$ is defined as the raw empirical mean of activations during a failure mode (e.g. repetitive generation):
$$u_{\text{loop}} = \frac{1}{M} \sum_{j=1}^M x_{\text{loop}, j} \approx \mu + \bar{\delta}_{\text{loop}}$$
the vector $u_{\text{loop}}$ contains $\mu$ as its primary component, producing high cosine alignment with the top principal components of $C$ (empirically $15.99\%$ energy in top 10 PCs and $26.15\%$ in top 100 PCs on SmolLM2-360M). When projected through $P_{\text{null}} = V \text{diag}\left(\frac{\alpha}{\lambda_i + \alpha}\right) V^T$ with small $\alpha$, the projector aggressively attenuates $\mu$, suppressing the effective intervention norm by over $23\%$ and neutralizing the behavioral intervention.

#### Resolution via Differential Activation Surgery:
To decouple the behavioral pathology from the background language centroid, we define the differential pathology vector:
$$u_{\text{diff}} = \frac{1}{M} \sum_{j=1}^M x_{\text{loop}, j} - \frac{1}{K} \sum_{k=1}^K x_{\text{clean}, k} = (\mu + \bar{\delta}_{\text{loop}}) - (\mu + \bar{\delta}_{\text{clean}}) = \bar{\delta}_{\text{loop}} - \bar{\delta}_{\text{clean}}$$

By subtracting clean text activations, the global centroid $\mu$ analytically cancels out. As demonstrated empirically, $u_{\text{diff}}$ retains only $4.17\%$ energy in the top 10 PCs and $0.13\%$ in the top 100 PCs, allowing the regularized null-space projector $P_{\text{null}}$ to preserve $>83\%$ of the intervention norm even under strict regularization ($\alpha = 10^{-3}$).

### Theorem 1 (Approximate Null-Space Subspace Attenuation):
*Let $x \in \mathbb{R}^{d_{\text{in}}}$ be an eigenvector of the activation covariance $C$ corresponding to an active feature direction with eigenvalue $\lambda \gg \alpha$. Then the forward perturbation under the projected update $\Delta W_{\text{safe}} = \Delta W_{\text{raw}} P_{\text{null}}$ satisfies:*
$$\|\Delta W_{\text{safe}} x\|_2 = \left( \frac{\alpha}{\lambda + \alpha} \right) \|\Delta W_{\text{raw}} x\|_2 = O\left(\frac{\alpha}{\lambda}\right) \|\Delta W_{\text{raw}} x\|_2$$

*Proof:*
$$\Delta W_{\text{safe}} x = \Delta W_{\text{raw}} P_{\text{null}} x = \Delta W_{\text{raw}} V \text{diag}\left(\frac{\alpha}{\lambda_i + \alpha}\right) V^T x$$
Since $x$ is an eigenvector of $C$ with eigenvalue $\lambda$, $V^T x = e_k$ where $e_k$ is the $k$-th canonical basis vector. Therefore:
$$P_{\text{null}} x = \left( \frac{\alpha}{\lambda + \alpha} \right) x \implies \Delta W_{\text{safe}} x = \frac{\alpha}{\lambda + \alpha} \Delta W_{\text{raw}} x$$
Taking the $L_2$ norm yields:
$$\|\Delta W_{\text{safe}} x\|_2 = \frac{\alpha}{\lambda + \alpha} \|\Delta W_{\text{raw}} x\|_2 \quad \blacksquare$$

> [!NOTE]
> **Scientific Qualification (Approximate vs. Exact Invariance):**
> For any non-zero regularizer $\alpha > 0$ and finite eigenvalue $\lambda$, $\frac{\alpha}{\lambda + \alpha} > 0$. The projection does not achieve absolute zero forward perturbation on the active manifold; rather, it provides an asymptotic spectral attenuation ($O(\alpha / \lambda)$) that suppresses cross-talk energy by $>99.9\%$ when $\lambda / \alpha > 10^3$, empirically preserving language modeling perplexity without requiring hard singular value truncation.

---

## 3. Spectral Conditioning & Rank-1 Perturbation Analysis

Even with subspace projection, unconstrained weight modifications risk distorting the singular value spectrum $\sigma_i(W)$ or inducing activation explosion. We enforce quantitative guards and establish rigorous perturbation bounds:

### 3.1 Frobenius Norm Budgeting
We define a relative Frobenius perturbation budget $\gamma \in (0, 0.05]$:
$$\|\Delta W_{\text{safe}}\|_F \le \gamma \|W^{(l)}\|_F$$
If $\|\Delta W_{\text{safe}}\|_F > \gamma \|W^{(l)}\|_F$, we uniformly rescale:
$$\Delta W_{\text{safe}} \leftarrow \Delta W_{\text{safe}} \cdot \left( \frac{\gamma \|W^{(l)}\|_F}{\|\Delta W_{\text{safe}}\|_F} \right)$$

### 3.2 Rank-1 Singular Value Perturbation Bounds (Mirsky & Thompson Interlacing)

Let the base weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ have singular values:
$$\sigma_1(W) \ge \sigma_2(W) \ge \dots \ge \sigma_r(W) > 0, \quad \text{where } r = \min(d_{\text{out}}, d_{\text{in}})$$

Under a rank-1 directional update $\Delta W = -\eta v u^T$ with unit-norm vectors $\|v\|_2 = \|u\|_2 = 1$ and spectral norm $\|\Delta W\|_2 = \eta$:

#### 1. Mirsky's Singular Value Perturbation Bound:
By Mirsky's theorem (Mirsky, 1960), the singular values of $W$ and $W' = W + \Delta W$ satisfy the uniform perturbation inequality:
$$|\sigma_i(W + \Delta W) - \sigma_i(W)| \le \|\Delta W\|_2 \le \|\Delta W\|_F \le \eta, \quad \forall i \in \{1, \dots, r\}$$

#### 2. Thompson's Singular Value Interlacing Theorem:
For rank-1 modifications $W' = W - \eta v u^T$, Thompson (1976) proved that the singular values of the perturbed matrix strictly interlace with those of the base matrix:
$$\sigma_{i+1}(W) \le \sigma_i(W - \eta v u^T) \le \sigma_{i-1}(W), \quad \forall i$$
with the standard boundary conventions $\sigma_{r+1}(W) = 0$ and $\sigma_0(W) = \infty$.

#### 3. Condition Number Bounds and Mathematical Qualification:
The spectral condition number is defined as $\kappa(W) = \frac{\sigma_1(W)}{\sigma_r(W)} = \frac{\sigma_{\max}(W)}{\sigma_{\min}(W)}$. Applying Mirsky's bound to the extreme singular values:
$$\sigma_{\max}(W') \le \sigma_{\max}(W) + \eta$$
$$\sigma_{\min}(W') \ge \max(0, \, \sigma_{\min}(W) - \eta)$$

Consequently, if $\sigma_{\min}(W) > \eta$, the perturbed condition number satisfies the two-sided inequality:
$$\frac{\sigma_{\max}(W) - \eta}{\sigma_{\min}(W) + \eta} \le \kappa(W') \le \frac{\sigma_{\max}(W) + \eta}{\sigma_{\min}(W) - \eta}, \quad \text{provided } \sigma_{\min}(W) > \eta$$

> [!WARNING]
> **Critical Theoretical Qualification: Singular Value Margins in Language Model Matrices**
> In transformer feedforward layers, the smallest singular value $\sigma_{\min}(W)$ is frequently smaller than nominal surgical intervention scales $\eta$:
> - For SmolLM2-360M `model.layers.15.mlp.down_proj` ($960 \times 2560$), empirical double-precision SVD measurement yields $\sigma_{\max} = 38.9912$, $\sigma_{\min} = 1.2373$, and $\kappa = 31.5136$.
> - For Qwen2.5-0.5B `model.layers.12.mlp.down_proj` ($896 \times 4864$), empirical measurement yields $\sigma_{\max} = 4.9004$, $\sigma_{\min} = 0.3250$, and $\kappa = 15.0796$.
>
> In both cases, the nominal unscaled intervention magnitude $\eta = 6.0$ strictly exceeds $\sigma_{\min}(W)$ ($\eta = 6.0 > 1.2373$ and $\eta = 6.0 > 0.3250$). 
>
> Therefore, the worst-case lower bound $\sigma_{\min}(W) - \eta$ is strictly negative and vacuously bounded by zero ($\sigma_{\min} \ge 0$). **Unconstrained Frobenius bounds or worst-case spectral perturbation theorems cannot theoretically guarantee that $W'$ remains non-singular.** An adversarial rank-1 update aligned with the minimal right-singular vector $v_{\min}$ could force $\sigma_{\min}(W') \to 0$ and $\kappa(W') \to \infty$.
>
> In SP-SV-WSP, spectral conditioning and numerical stability are preserved through two complementary mechanisms:
> 1. **Empirical Low-Rank Directionality:** Behavioral intervention vectors $u$ and $v$ are extracted from high-variance latent representations, which have virtually zero overlap with the minimal singular vectors $v_{\min}$ located in the numerical noise floor of $W$.
> 2. **Algorithmic Verification Gating:** Because theoretical worst-case bounds cannot rule out singularity when $\sigma_{\min}(W) \le \eta$, the closed-loop **Verification Oracle** is mathematically required. The oracle measures $\kappa(W')$ directly via SVD in memory prior to checkpoint commitment, gating updates to $|\kappa(W') - \kappa(W)| / \kappa(W) \le \tau_{\kappa} = 5\%$ and executing bitwise-exact rollback if the condition number degrades.

---

## 4. Closed-Loop Autonomous Verification Oracle

Before an edit is finalized, the system computes the multi-objective verification tuple $\mathcal{V} = (\Delta \mathcal{S}_{\text{target}}, \Delta \text{PPL}_{\text{retain}}, \Delta \text{Refusal}_{\text{safety}}, \Delta \kappa)$:

```
           +------------------------------------------+
           | Compute Metric Tuple:                    |
           | 1. Target Task Gain (ΔS_target)          |
           | 2. Perplexity Drift (ΔPPL_retain)        |
           | 3. Safety Guardrail Drift (ΔRefusal)     |
           | 4. Condition Number Shift (Δκ)           |
           +------------------------------------------+
                               |
               +---------------+---------------+
               |                               |
        All Thresholds Met?             Thresholds Violated?
               |                               |
               v                               v
       +---------------+             +-------------------+
       |    COMMIT     |             | Is Gain Positive? |
       | Save Weights  |             +-------------------+
       +---------------+                       |
                                      +--------+--------+
                                      |                 |
                                     Yes               No
                                      |                 |
                                      v                 v
                              +---------------+  +---------------+
                              |     ADAPT     |  |   ROLLBACK    |
                              | Increase λ    |  | Restore Base  |
                              | Shrink Rank r |  | Discard Edit  |
                              +---------------+  +---------------+
```

The decision function is defined formally as:
$$\text{Action} = \begin{cases}
\text{COMMIT}, & \text{if } \Delta \mathcal{S}_{\text{target}} \ge \tau_{\text{gain}} \land \Delta \text{PPL} \le \tau_{\text{ppl}} \land \Delta \text{Refusal} \le \tau_{\text{safe}} \\
\text{ADAPT}(\lambda \leftarrow 2\lambda, r \leftarrow \lfloor r/2 \rfloor), & \text{if } \Delta \mathcal{S}_{\text{target}} \ge \tau_{\text{gain}} \land \Delta \text{PPL} > \tau_{\text{ppl}} \\
\text{ROLLBACK}, & \text{if } \Delta \mathcal{S}_{\text{target}} < \tau_{\text{gain}}
\end{cases}$$

