"""
NullSpaceProjector: Linear algebraic spectral attenuation engine for zero-interference weight editing.
Implements Tikhonov-regularized spectral attenuation operators to suppress cross-talk.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import torch

_UNSET: Any = object()


@dataclass
class ProjectionResult:
    """Detailed summary of the subspace projection operation."""
    delta_w_safe: torch.Tensor
    raw_frobenius_norm: float
    safe_frobenius_norm: float
    budget_rescaled: bool
    frobenius_budget_ratio: float
    projection_residual_error: float
    retained_invariance_score: float  # 1.0 = perfect null-space protection (0 disturbance)


class NullSpaceProjector:
    """
    Subspace projector implementing Tikhonov-regularized null-space spectral attenuation
    to guarantee invariant preservation for neural network weight programming.
    
    Mathematical Note:
    For finite alpha > 0, P_null = I - C(C + alpha I)^(-1) is a smooth spectral attenuation
    operator (P_null^2 != P_null), not an idempotent orthogonal projection matrix. It attenuates
    perturbations along covariance eigenvectors as alpha / (lambda_i + alpha) = O(alpha / lambda_i).
    """

    def __init__(self, alpha: float = 1e-3, gamma: Optional[float] = 0.02):
        """
        Args:
            alpha: Tikhonov regularization parameter controlling null-space stiffness.
                   Higher alpha softens the boundary; lower alpha enforces sharper null space.
            gamma: Maximum allowable relative Frobenius norm perturbation ||Delta W||_F / ||W||_F.
                   Pass None for unbudgeted edits.
        """
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def compute_covariance(activations: torch.Tensor, center: bool = False) -> torch.Tensor:
        """
        Compute the sample covariance matrix C = (1/N) X^T X.
        
        Args:
            activations: Representation matrix X of shape [N, d_in]
            center: Whether to center activations by subtracting their mean
            
        Returns:
            Covariance matrix C of shape [d_in, d_in]
        """
        if activations.ndim != 2:
            raise ValueError(f"Expected 2D activations tensor [N, d_in], got {activations.shape}")
        
        N, d_in = activations.shape
        X = activations.to(torch.float32)
        if center:
            X = X - torch.mean(X, dim=0, keepdim=True)
            
        # Uncentered outer product / N
        C = torch.matmul(X.T, X) / max(1, N)
        # Symmetrize for numerical exactness
        C = 0.5 * (C + C.T)
        return C

    def compute_projector(self, covariance: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """
        Compute the closed-form null-space projector:
            P_null = I - C (C + alpha * I)^(-1)
        Computed using symmetric eigendecomposition for optimal numerical stability:
            C = V Lambda V^T  ==>  P_null = V diag(alpha / (lambda_i + alpha)) V^T
            
        Args:
            covariance: Positive semi-definite covariance matrix [d_in, d_in]
            alpha: Optional override for regularization parameter
            
        Returns:
            P_null operator of shape [d_in, d_in]
        """
        alpha_val = alpha if alpha is not None else self.alpha
        
        # Symmetric eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        
        # Filter negative eigenvalues from floating point imprecision
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        
        # Spectral damping coefficients: alpha / (lambda_i + alpha)
        # When lambda_i >> alpha (retained feature direction), coefficient -> 0 (erased)
        # When lambda_i << alpha (null / unused direction), coefficient -> 1 (preserved)
        spectral_weights = alpha_val / (eigenvalues + alpha_val)
        
        # Reconstruct projector: V * diag(spectral_weights) * V^T
        P_null = torch.matmul(eigenvectors, torch.matmul(torch.diag(spectral_weights), eigenvectors.T))
        
        # Symmetrize projector
        P_null = 0.5 * (P_null + P_null.T)
        return P_null

    def project(
        self,
        delta_w_raw: torch.Tensor,
        covariance: torch.Tensor,
        base_w: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        gamma: Any = _UNSET,
        test_activations: Optional[torch.Tensor] = None,
    ) -> ProjectionResult:
        """
        Project candidate weight delta onto the null space of preserved representations
        and enforce spectral/Frobenius budget.
        
        Args:
            delta_w_raw: Unconstrained edit tensor [d_out, d_in]
            covariance: Covariance matrix of preserved representations [d_in, d_in]
            base_w: Pristine base weight tensor [d_out, d_in] for relative budget check
            alpha: Optional alpha override
            gamma: Optional gamma budget override. Pass None explicitly for unbudgeted edits.
            test_activations: Optional sample [N_test, d_in] to verify invariance empirically
            
        Returns:
            ProjectionResult with safe update tensor and verification statistics.
        """
        delta_raw = delta_w_raw.to(torch.float32)
        P_null = self.compute_projector(covariance, alpha=alpha)
        
        # Subspace-projected update: Delta W_safe = Delta W_raw * P_null
        delta_safe = torch.matmul(delta_raw, P_null)
        
        raw_norm = float(torch.norm(delta_raw, p='fro').item())
        safe_norm = float(torch.norm(delta_safe, p='fro').item())
        
        # Frobenius budgeting
        gamma_val = self.gamma if gamma is _UNSET else gamma
        rescaled = False
        budget_ratio = 1.0
        
        if base_w is not None and gamma_val is not None:
            base_norm = float(torch.norm(base_w.to(torch.float32), p='fro').item())
            max_allowed = gamma_val * base_norm
            budget_ratio = safe_norm / (base_norm + 1e-12)
            if safe_norm > max_allowed and safe_norm > 1e-12:
                scale_factor = max_allowed / safe_norm
                delta_safe = delta_safe * scale_factor
                safe_norm = float(torch.norm(delta_safe, p='fro').item())
                rescaled = True
                budget_ratio = safe_norm / (base_norm + 1e-12)
        elif base_w is not None:
            base_norm = float(torch.norm(base_w.to(torch.float32), p='fro').item())
            budget_ratio = safe_norm / (base_norm + 1e-12)

        # Invariance residual calculation
        residual_error = 0.0
        invariance_score = 1.0
        
        if test_activations is not None and test_activations.shape[0] > 0:
            X = test_activations.to(torch.float32)
            # Forward perturbation on test tokens: ||Delta W_safe * X^T||_F
            pert_safe = torch.matmul(delta_safe, X.T)
            pert_raw = torch.matmul(delta_raw, X.T)
            
            norm_safe_pert = torch.norm(pert_safe, p='fro').item()
            norm_raw_pert = torch.norm(pert_raw, p='fro').item()
            
            residual_error = norm_safe_pert / (norm_raw_pert + 1e-12)
            invariance_score = max(0.0, 1.0 - residual_error)
        else:
            # Theoretical null-space projection error: ||P_null * C||_F / ||C||_F
            prod = torch.matmul(P_null, covariance)
            residual_error = float(torch.norm(prod, p='fro').item() / (torch.norm(covariance, p='fro').item() + 1e-12))
            invariance_score = max(0.0, 1.0 - residual_error)

        return ProjectionResult(
            delta_w_safe=delta_safe,
            raw_frobenius_norm=raw_norm,
            safe_frobenius_norm=safe_norm,
            budget_rescaled=rescaled,
            frobenius_budget_ratio=budget_ratio,
            projection_residual_error=residual_error,
            retained_invariance_score=invariance_score,
        )

