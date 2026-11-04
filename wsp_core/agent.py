"""
CognitiveReasoningAgent: Autonomous agent loop executing diagnostic probing,
subspace projection, and adaptive multi-round self-verification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn

from .inspector import WeightInspector, SpectralProfile
from .projector import NullSpaceProjector, ProjectionResult
from .verifier import VerificationOracle, VerificationResult, VerificationDecision


@dataclass
class EditHypothesis:
    """Represents a proposed weight modification intent."""
    layer_path: str
    target_behavior_desc: str
    direction_method: str = "mean_diff"  # 'mean_diff' or 'pca_diff'
    scale_factor: float = 1.0
    alpha: float = 1e-3
    gamma: float = 0.02


@dataclass
class ProgrammingSessionReport:
    """Comprehensive log of the autonomous weight-space programming session."""
    hypothesis: EditHypothesis
    spectral_profile_before: SpectralProfile
    spectral_profile_after: Optional[SpectralProfile]
    projection_result: Optional[ProjectionResult]
    verification_history: List[VerificationResult] = field(default_factory=list)
    final_verdict: VerificationDecision = VerificationDecision.ROLLBACK
    status_message: str = ""


class CognitiveReasoningAgent:
    """
    Autonomous agent coordinating mechanistic diagnostics, null-space
    projected weight delta generation, and closed-loop verification.
    """

    def __init__(
        self,
        model: nn.Module,
        tau_gain: float = 0.15,
        tau_ppl_drift: float = 0.03,
        tau_condition_drift: float = 0.05,
        max_adaptation_rounds: int = 3,
    ):
        self.model = model
        self.inspector = WeightInspector(model)
        self.verifier = VerificationOracle(
            model=model,
            tau_gain=tau_gain,
            tau_ppl_drift=tau_ppl_drift,
            tau_condition_drift=tau_condition_drift,
        )
        self.max_adaptation_rounds = max_adaptation_rounds

    def execute_edit_session(
        self,
        hypothesis: EditHypothesis,
        target_acts: torch.Tensor,
        base_acts: torch.Tensor,
        retained_acts: torch.Tensor,
        target_eval_fn: Callable[[], float],
        retained_tokens: List[torch.Tensor],
        target_input_acts: Optional[torch.Tensor] = None,
    ) -> ProgrammingSessionReport:
        """
        Execute full autonomous programming cycle:
        1. Profile layer spectrum
        2. Extract directional pathology vector
        3. Formulate raw edit tensor
        4. Solve null-space projection
        5. Run closed-loop verification with adaptive feedback
        """
        layer_path = hypothesis.layer_path
        profile_before = self.inspector.profile_layer(layer_path)
        base_w = self.inspector.get_weight(layer_path)
        d_out, d_in = base_w.shape
        
        # 1. Compute covariance of preserved representations (must be in R^d_in)
        projector = NullSpaceProjector(alpha=hypothesis.alpha, gamma=hypothesis.gamma)
        C_retain = projector.compute_covariance(retained_acts)
        
        # 2. Extract contrastive vector for output space (v_dir in R^d_out)
        v_dir = self.inspector.extract_contrastive_vector(
            target_acts=target_acts,
            base_acts=base_acts,
            method=hypothesis.direction_method,
        )
        if len(v_dir) != d_out:
            raise ValueError(f"v_dir dimension ({len(v_dir)}) does not match layer output dimension ({d_out})")
        
        # 3. Formulate input direction u_dir in R^d_in
        if target_input_acts is not None:
            u_dir = torch.mean(target_input_acts, dim=0)
        else:
            # Use top right singular vector of C_retain or mean of retained representations
            u_dir = torch.mean(retained_acts, dim=0)
            
        norm_u = torch.norm(u_dir, p=2)
        if norm_u > 1e-12:
            u_dir = u_dir / norm_u
        else:
            u_dir = torch.ones(d_in) / (d_in ** 0.5)

        # Delta W has shape [d_out, d_in]
        delta_raw = - hypothesis.scale_factor * torch.outer(v_dir, u_dir)
        
        # 4. Multi-round adaptive verification loop
        current_alpha = hypothesis.alpha
        current_gamma = hypothesis.gamma
        history: List[VerificationResult] = []
        final_decision = VerificationDecision.ROLLBACK
        proj_result: Optional[ProjectionResult] = None
        
        for round_idx in range(self.max_adaptation_rounds):
            # Compute safe subspace projection
            proj_result = projector.project(
                delta_w_raw=delta_raw,
                covariance=C_retain,
                base_w=base_w,
                alpha=current_alpha,
                gamma=current_gamma,
                test_activations=retained_acts[:100] if len(retained_acts) > 0 else None,
            )
            
            # Verify in-memory via oracle
            v_res = self.verifier.verify_and_decide(
                layer_path=layer_path,
                delta_w=proj_result.delta_w_safe,
                target_eval_fn=target_eval_fn,
                retained_tokens=retained_tokens,
            )
            history.append(v_res)
            
            if v_res.decision == VerificationDecision.COMMIT:
                final_decision = VerificationDecision.COMMIT
                status = f"Successfully verified and committed in Round {round_idx + 1}."
                break
            elif v_res.decision == VerificationDecision.ADAPT:
                # Perplexity drifted or condition degraded: stiffen null-space projection & reduce budget
                # Note: In P_null = V diag(alpha / (lambda_i + alpha)) V^T, reducing alpha stiffens the null space
                # by reducing pass-through along active retained feature directions (lambda_i >> alpha).
                current_alpha = current_alpha * 0.5
                current_gamma = current_gamma * 0.7
                # Restore pristine layer before re-projecting in next round
                self.verifier.restore_layer(layer_path)
            else:  # ROLLBACK
                final_decision = VerificationDecision.ROLLBACK
                status = f"Hypothesis rejected in Round {round_idx + 1}: {v_res.rationale}"
                break
        else:
            status = f"Exceeded maximum adaptation rounds ({self.max_adaptation_rounds}). Layer restored to baseline."
            self.verifier.restore_layer(layer_path)

        profile_after = self.inspector.profile_layer(layer_path) if final_decision == VerificationDecision.COMMIT else None

        return ProgrammingSessionReport(
            hypothesis=hypothesis,
            spectral_profile_before=profile_before,
            spectral_profile_after=profile_after,
            projection_result=proj_result,
            verification_history=history,
            final_verdict=final_decision,
            status_message=status,
        )
