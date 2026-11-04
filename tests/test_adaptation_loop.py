"""
tests/test_adaptation_loop.py:
Verification of the adaptive closed-loop feedback engine in CognitiveReasoningAgent.
Tests that an initially loose projection (causing perplexity drift or condition shift) triggers ADAPT,
stiffens alpha (alpha = alpha * 0.5), re-projects, and adapts across multiple rounds.
"""

import pytest
import torch
import torch.nn as nn
from wsp_core.agent import CognitiveReasoningAgent, EditHypothesis
from wsp_core.verifier import VerificationDecision, VerificationResult
from wsp_core.projector import NullSpaceProjector


def test_complete_adaptive_feedback_loop():
    torch.manual_seed(42)
    vocab_size = 32
    d_in = 64
    d_out = vocab_size
    
    # Language model with Embedding and Linear head
    model = nn.Sequential(
        nn.Embedding(vocab_size, d_in),
        nn.Linear(d_in, d_out, bias=False)
    )
    
    agent = CognitiveReasoningAgent(
        model=model,
        tau_gain=0.10,
        tau_ppl_drift=0.002,
        tau_condition_drift=0.05,
        max_adaptation_rounds=3
    )
    
    target_acts = torch.randn(40, d_out) + 1.0
    base_acts = torch.randn(40, d_out) - 1.0
    retained_acts = torch.randn(100, d_in) * 2.0
    retained_tokens = [torch.randint(0, vocab_size, (4, 16)) for _ in range(3)]
    
    call_count = 0
    def target_eval():
        nonlocal call_count
        call_count += 1
        return 0.10 if call_count == 1 else 0.40
        
    hypothesis = EditHypothesis(
        layer_path="1",
        target_behavior_desc="Adaptive Loop Verification",
        scale_factor=0.3,
        alpha=0.2,
        gamma=0.20
    )
    
    report = agent.execute_edit_session(
        hypothesis=hypothesis,
        target_acts=target_acts,
        base_acts=base_acts,
        retained_acts=retained_acts,
        target_eval_fn=target_eval,
        retained_tokens=retained_tokens
    )
    
    assert len(report.verification_history) >= 1
    assert report.final_verdict in [VerificationDecision.COMMIT, VerificationDecision.ROLLBACK]


def test_forced_multi_round_adaptation():
    """Explicitly verify that spectral shift exceeding threshold triggers ADAPT and re-evaluates across rounds."""
    torch.manual_seed(123)
    vocab_size = 32
    d_in = 64
    d_out = vocab_size
    
    model = nn.Sequential(
        nn.Embedding(vocab_size, d_in),
        nn.Linear(d_in, d_out, bias=False)
    )
    
    # Set tau_condition_drift=0.01 (tight: 1% tolerance) so Round 1's ~2.3% shift triggers ADAPT
    agent = CognitiveReasoningAgent(
        model=model,
        tau_gain=0.10,
        tau_ppl_drift=0.05,
        tau_condition_drift=0.01,
        max_adaptation_rounds=3
    )
    
    target_acts = torch.randn(40, d_out) + 1.0
    base_acts = torch.randn(40, d_out) - 1.0
    retained_acts = torch.randn(100, d_in) * 2.0
    retained_tokens = [torch.randint(0, vocab_size, (4, 16)) for _ in range(3)]
    
    call_count = 0
    def target_eval():
        nonlocal call_count
        call_count += 1
        return 0.10 if call_count == 1 else 0.50
        
    hypothesis = EditHypothesis(
        layer_path="1",
        target_behavior_desc="Forced Multi-Round Adaptation",
        scale_factor=0.5,
        alpha=1.0,  # Loose initial alpha
        gamma=0.20
    )
    
    report = agent.execute_edit_session(
        hypothesis=hypothesis,
        target_acts=target_acts,
        base_acts=base_acts,
        retained_acts=retained_acts,
        target_eval_fn=target_eval,
        retained_tokens=retained_tokens
    )
    
    print(f"\n[+] Forced Adaptation Test:")
    print(f"    Rounds Executed: {len(report.verification_history)}")
    for i, res in enumerate(report.verification_history):
        print(f"    Round {i+1}: {res.decision.value} (Cond Drift: {res.condition_drift_rel*100:.3f}%)")
        
    assert len(report.verification_history) > 1, "Agent did not trigger adaptation round!"
    assert report.verification_history[0].decision == VerificationDecision.ADAPT


def test_alpha_stiffening_reduces_forward_perturbation():
    """Verify mathematically that alpha_new = alpha * 0.5 strictly reduces forward perturbation on active features."""
    torch.manual_seed(42)
    d_in = 32
    d_out = 16
    
    X = torch.randn(100, d_in)
    projector = NullSpaceProjector()
    C = projector.compute_covariance(X)
    
    delta_raw = torch.randn(d_out, d_in)
    
    alpha_loose = 0.1
    alpha_stiff = alpha_loose * 0.5
    
    res_loose = projector.project(delta_raw, C, alpha=alpha_loose, test_activations=X)
    res_stiff = projector.project(delta_raw, C, alpha=alpha_stiff, test_activations=X)
    
    pert_loose = torch.norm(torch.matmul(res_loose.delta_w_safe, X.T), p='fro').item()
    pert_stiff = torch.norm(torch.matmul(res_stiff.delta_w_safe, X.T), p='fro').item()
    
    assert pert_stiff < pert_loose, f"Stiffening alpha did not reduce perturbation: {pert_stiff} >= {pert_loose}"
    print(f"\n[+] Alpha stiffening verified: Perturbation dropped from {pert_loose:.6f} to {pert_stiff:.6f}")
