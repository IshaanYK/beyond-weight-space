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

### Theorem 1 (Guaranteed Invariance on Preserved Subspaces):
*Let $x \in \mathbb{R}^{d_{\text{in}}}$ be an eigenvector of $C$ with eigenvalue $\lambda \gg \alpha$. Then the perturbation on the forward output satisfies:*
$$\|\Delta W_{\text{safe}} x\|_2 = O\left(\frac{\alpha}{\lambda}\right) \|\Delta W_{\text{raw}} x\|_2 \approx 0$$
*Proof:*
$$\Delta W_{\text{safe}} x = \Delta W_{\text{raw}} P_{\text{null}} x = \Delta W_{\text{raw}} \left( \frac{\alpha}{\lambda + \alpha} \right) x = \frac{\alpha}{\lambda + \alpha} \Delta W_{\text{raw}} x$$
*As $\alpha / \lambda \to 0$, the perturbation vanishes identically. $\blacksquare$*

---

## 3. Spectral Conditioning & Frobenius Budgeting

Even with null-space projection, large parameter updates can degrade the condition number $\kappa(W)$ or cause internal numerical overflow. We enforce two quantitative guards:

### 3.1 Frobenius Norm Budgeting
We define the relative Frobenius perturbation budget $\gamma \in (0, 0.05]$:
$$\|\Delta W_{\text{safe}}\|_F \le \gamma \|W^{(l)}\|_F$$
If $\|\Delta W_{\text{safe}}\|_F > \gamma \|W^{(l)}\|_F$, we rescale:
$$\Delta W_{\text{safe}} \leftarrow \Delta W_{\text{safe}} \cdot \left( \frac{\gamma \|W^{(l)}\|_F}{\|\Delta W_{\text{safe}}\|_F} \right)$$

### 3.2 Singular Value Perturbation Bound (Weyl's Inequality)
By Weyl's theorem, the singular values of $W_{\text{new}} = W + \Delta W$ satisfy:
$$|\sigma_i(W_{\text{new}}) - \sigma_i(W)| \le \|\Delta W_{\text{safe}}\|_2 \le \|\Delta W_{\text{safe}}\|_F \le \gamma \|W\|_F$$
This guarantees that no singular value shifts by more than $\gamma \|W\|_F$, preserving the condition number and transmission dynamics of the transformer block.

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

