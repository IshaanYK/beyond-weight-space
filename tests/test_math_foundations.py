"""
test_math_foundations.py: Mathematical and unit tests for wsp_core.
Verifies Theorem 1 (Null-space invariance), spectral condition stability,
and closed-loop verification oracle rollback.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from wsp_core.inspector import WeightInspector, SpectralProfile
from wsp_core.projector import NullSpaceProjector, ProjectionResult
from wsp_core.verifier import VerificationOracle, VerificationDecision
from wsp_core.agent import CognitiveReasoningAgent, EditHypothesis, ProgrammingSessionReport


class TestSpectralInspector(unittest.TestCase):
    """Test SVD spectral diagnostics and condition number computations."""

    def setUp(self):
        # Create a mock 2-layer MLP
        torch.manual_seed(42)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.inspector = WeightInspector(self.mlp)

    def test_profile_layer_dimensions_and_singular_values(self):
        profile = self.inspector.profile_layer("0")
        self.assertEqual(profile.layer_name, "0")
        self.assertEqual(profile.shape, (128, 64))
        self.assertGreater(profile.frobenius_norm, 0.0)
        self.assertGreater(profile.spectral_norm, 0.0)
        self.assertGreater(profile.condition_number, 1.0)
        self.assertGreater(profile.effective_rank, 1.0)
        self.assertEqual(len(profile.top_singular_values), 10)

    def test_activation_hook_capture(self):
        self.inspector.register_activation_hook("0", capture_input=True)
        dummy_input = torch.randn(8, 64)
        _ = self.mlp(dummy_input)
        
        captured = self.inspector.get_captured_activations("0")
        self.assertEqual(captured.shape, (8, 64))
        self.inspector.clear_hooks()

    def test_contrastive_vector_extraction(self):
        target_acts = torch.randn(20, 64) + 2.5
        base_acts = torch.randn(20, 64) - 2.5
        
        v_mean = WeightInspector.extract_contrastive_vector(target_acts, base_acts, method="mean_diff")
        self.assertEqual(v_mean.shape, (64,))
        self.assertAlmostEqual(torch.norm(v_mean, p=2).item(), 1.0, places=5)
        
        v_pca = WeightInspector.extract_contrastive_vector(target_acts, base_acts, method="pca_diff")
        self.assertEqual(v_pca.shape, (64,))
        self.assertAlmostEqual(torch.norm(v_pca, p=2).item(), 1.0, places=5)


class TestNullSpaceProjector(unittest.TestCase):
    """Test Null-Space Projection algebra and Theorem 1."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_in = 64
        self.d_out = 128
        self.n_samples = 200

        # Generate synthetic retained representations spanning a low-dimensional active subspace
        # The first 16 dimensions have massive energy, remaining 48 have near-zero energy
        self.X_retain = torch.zeros(self.n_samples, self.d_in)
        self.X_retain[:, :16] = torch.randn(self.n_samples, 16) * 5.0
        self.X_retain[:, 16:] = torch.randn(self.n_samples, 48) * 0.01

        self.projector = NullSpaceProjector(alpha=1e-4, gamma=0.05)

    def test_covariance_is_symmetric_psd(self):
        C = self.projector.compute_covariance(self.X_retain)
        self.assertEqual(C.shape, (self.d_in, self.d_in))
        # Test symmetry: C = C^T
        sym_diff = torch.norm(C - C.T, p='fro').item()
        self.assertLess(sym_diff, 1e-6)
        
        # Test PSD: all eigenvalues >= -1e-6
        eigvals = torch.linalg.eigvalsh(C)
        self.assertTrue(torch.all(eigvals >= -1e-5))

    def test_theorem_1_null_space_invariance(self):
        """
        Verify Theorem 1: ||Delta W_safe * X_retain^T|| << ||Delta W_raw * X_retain^T||
        Null-space projection suppresses interference along dominant retained axes.
        """
        C = self.projector.compute_covariance(self.X_retain)
        
        # Create an unconstrained edit that aggressively perturbs the active retained directions
        delta_w_raw = torch.randn(self.d_out, self.d_in)
        
        # Unconstrained perturbation on retained tokens
        unconstrained_pert = torch.matmul(delta_w_raw, self.X_retain.T)
        norm_unconstrained = torch.norm(unconstrained_pert, p='fro').item()

        # Apply null space projection
        result = self.projector.project(
            delta_w_raw=delta_w_raw,
            covariance=C,
            test_activations=self.X_retain,
            alpha=1e-5
        )
        
        safe_pert = torch.matmul(result.delta_w_safe, self.X_retain.T)
        norm_safe = torch.norm(safe_pert, p='fro').item()

        # The safe perturbation MUST be significantly suppressed relative to unconstrained
        suppression_ratio = norm_safe / norm_unconstrained
        self.assertLess(suppression_ratio, 0.05, f"Suppression ratio was {suppression_ratio}, expected < 0.05")
        self.assertGreater(result.retained_invariance_score, 0.95)

    def test_frobenius_norm_budgeting(self):
        base_w = torch.randn(self.d_out, self.d_in)
        base_norm = torch.norm(base_w, p='fro').item()
        
        # Raw delta with small eigenvalue suppression so safe_norm exceeds budget
        huge_delta = torch.randn(self.d_out, self.d_in) * 2.0
        # Empty covariance -> P_null is identity
        C_empty = torch.zeros(self.d_in, self.d_in)
        
        result = self.projector.project(
            delta_w_raw=huge_delta,
            covariance=C_empty,
            base_w=base_w,
            gamma=0.02
        )
        
        self.assertTrue(result.budget_rescaled)
        self.assertLessEqual(result.safe_frobenius_norm, 0.02 * base_norm + 1e-5)


class TestVerificationOracle(unittest.TestCase):
    """Test closed-loop verification, invariant gates, and rollback mechanism."""

    def setUp(self):
        torch.manual_seed(42)
        # Mock mini LM: Embedding(32, 64) -> Linear(64, 32)
        self.model = nn.Sequential(
            nn.Embedding(32, 64),
            nn.Linear(64, 32)
        )
        self.oracle = VerificationOracle(
            self.model,
            tau_gain=0.10,
            tau_ppl_drift=0.02,
            tau_condition_drift=0.05
        )

    def test_backup_and_restore(self):
        original_w = self.model[1].weight.detach().clone()
        self.oracle.backup_layer("1")
        
        # Corrupt weights
        with torch.no_grad():
            self.model[1].weight.add_(torch.ones_like(self.model[1].weight) * 99.0)
            
        corrupted_w = self.model[1].weight.detach().clone()
        self.assertFalse(torch.allclose(original_w, corrupted_w))
        
        # Restore
        self.oracle.restore_layer("1")
        restored_w = self.model[1].weight.detach().clone()
        self.assertTrue(torch.allclose(original_w, restored_w))

    def test_rollback_on_failed_gain(self):
        original_w = self.model[1].weight.detach().clone()
        
        # Target evaluation function returns a non-improving gain
        def failing_task():
            return 0.01  # Below tau_gain
            
        delta_w = torch.randn_like(self.model[1].weight) * 0.001
        dummy_tokens = [torch.randint(0, 32, (4, 10))]
        
        result = self.oracle.verify_and_decide(
            layer_path="1",
            delta_w=delta_w,
            target_eval_fn=failing_task,
            retained_tokens=dummy_tokens
        )
        
        self.assertEqual(result.decision, VerificationDecision.ROLLBACK)
        # Verify layer was automatically rolled back
        current_w = self.model[1].weight.detach().clone()
        self.assertTrue(torch.allclose(original_w, current_w))


class TestCognitiveReasoningAgent(unittest.TestCase):
    """Test full end-to-end autonomous weight-space programming session."""

    def setUp(self):
        torch.manual_seed(42)
        # Mock mini LM: Embedding(32, 64) -> Linear(64, 32)
        self.model = nn.Sequential(
            nn.Embedding(32, 64),
            nn.Linear(64, 32)
        )
        self.agent = CognitiveReasoningAgent(
            model=self.model,
            tau_gain=0.10,
            tau_ppl_drift=0.05,
            tau_condition_drift=0.10,
            max_adaptation_rounds=2
        )

    def test_successful_edit_session_commit(self):
        # Target evaluation function simulates successful task improvement
        call_count = 0
        def simulated_task():
            nonlocal call_count
            call_count += 1
            return 0.10 if call_count == 1 else 0.50  # Base: 0.10, Edited: 0.50 (+400% gain)

        # Mock activations
        target_acts = torch.randn(50, 32) + 2.0
        base_acts = torch.randn(50, 32) - 1.0
        retained_acts = torch.randn(100, 64)
        dummy_tokens = [torch.randint(0, 32, (4, 10))]

        hypothesis = EditHypothesis(
            layer_path="1",
            target_behavior_desc="Ablate pathological attractor in output projection",
            scale_factor=0.01,
            alpha=1e-3,
            gamma=0.02
        )

        report = self.agent.execute_edit_session(
            hypothesis=hypothesis,
            target_acts=target_acts,
            base_acts=base_acts,
            retained_acts=retained_acts,
            target_eval_fn=simulated_task,
            retained_tokens=dummy_tokens
        )

        self.assertEqual(report.final_verdict, VerificationDecision.COMMIT)
        self.assertIsNotNone(report.projection_result)
        self.assertIsNotNone(report.spectral_profile_after)
        self.assertGreater(len(report.verification_history), 0)


if __name__ == '__main__':
    unittest.main()
