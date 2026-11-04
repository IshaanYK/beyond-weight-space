"""
VerificationOracle: Closed-loop multi-objective verification and autonomous rollback oracle.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np


class VerificationDecision(Enum):
    COMMIT = "COMMIT"
    ADAPT = "ADAPT"
    ROLLBACK = "ROLLBACK"


@dataclass
class VerificationResult:
    decision: VerificationDecision
    target_score_base: float
    target_score_edited: float
    target_gain: float
    perplexity_base: float
    perplexity_edited: float
    perplexity_drift_rel: float
    spectral_condition_base: float
    spectral_condition_edited: float
    condition_drift_rel: float
    rationale: str


class VerificationOracle:
    """
    Autonomous verification oracle that benchmarks target task gain,
    retained perplexity preservation, and spectral stability before
    committing or rolling back weight-space programming modifications.
    """

    def __init__(
        self,
        model: nn.Module,
        tau_gain: float = 0.15,
        tau_ppl_drift: float = 0.03,
        tau_condition_drift: float = 0.05,
    ):
        """
        Args:
            model: Target language model or transformer module
            tau_gain: Minimum required relative gain on target task (e.g. +15%)
            tau_ppl_drift: Maximum acceptable relative increase in perplexity (e.g. +3%)
            tau_condition_drift: Maximum acceptable relative shift in condition number (e.g. +5%)
        """
        self.model = model
        self.tau_gain = tau_gain
        self.tau_ppl_drift = tau_ppl_drift
        self.tau_condition_drift = tau_condition_drift
        self._backups: Dict[str, torch.Tensor] = {}

    def get_layer(self, layer_path: str) -> nn.Module:
        """Resolve a submodule by path."""
        module = self.model
        for part in layer_path.split('.'):
            module = getattr(module, part)
        return module

    def backup_layer(self, layer_path: str) -> None:
        """Store pristine clone of layer weights in CPU memory."""
        layer = self.get_layer(layer_path)
        if not hasattr(layer, 'weight') or layer.weight is None:
            raise ValueError(f"Layer '{layer_path}' has no weight attribute.")
        self._backups[layer_path] = layer.weight.detach().clone().cpu()

    def restore_layer(self, layer_path: str) -> None:
        """Restore pristine layer weights from memory backup."""
        if layer_path not in self._backups:
            raise KeyError(f"No backup found for layer '{layer_path}'.")
        layer = self.get_layer(layer_path)
        with torch.no_grad():
            layer.weight.copy_(self._backups[layer_path].to(layer.weight.device))

    def clear_backup(self, layer_path: Optional[str] = None) -> None:
        """Clear cached weight backups."""
        if layer_path:
            self._backups.pop(layer_path, None)
        else:
            self._backups.clear()

    def apply_delta(self, layer_path: str, delta_w: torch.Tensor) -> None:
        """Apply a weight modification in-place after ensuring a backup exists."""
        if layer_path not in self._backups:
            self.backup_layer(layer_path)
            
        layer = self.get_layer(layer_path)
        with torch.no_grad():
            target_dtype = layer.weight.dtype
            target_device = layer.weight.device
            layer.weight.add_(delta_w.to(dtype=target_dtype, device=target_device))

    @staticmethod
    def compute_condition_number(weight: torch.Tensor) -> float:
        """Compute condition number sigma_max / sigma_min for a weight matrix."""
        w = weight.detach().cpu().to(torch.float32)
        if w.ndim > 2:
            w = w.view(w.shape[0], -1)
        s = torch.linalg.svdvals(w).numpy()
        if len(s) == 0 or s[-1] < 1e-12:
            return float('inf')
        return float(s[0] / s[-1])

    @staticmethod
    def detect_repetition_loops(token_ids: List[int], n_gram_window: int = 4) -> float:
        """
        Calculate token repetition score: fraction of repeating n-grams.
        0.0 = completely unique, 1.0 = degenerate loop.
        Defaults to n_gram_window=4 to support both short (e.g. 25 tokens) and long generations.
        """
        if len(token_ids) < n_gram_window * 2:
            effective_window = max(2, len(token_ids) // 4)
            if len(token_ids) < effective_window * 2:
                return 0.0
        else:
            effective_window = n_gram_window
            
        seen = set()
        duplicates = 0
        total_ngrams = len(token_ids) - effective_window + 1
        
        for i in range(total_ngrams):
            ngram = tuple(token_ids[i:i + effective_window])
            if ngram in seen:
                duplicates += 1
            else:
                seen.add(ngram)
                
        return float(duplicates / max(1, total_ngrams))

    @staticmethod
    def detect_loop_incidence(token_ids: List[int], threshold: float = 0.15, n_gram_window: int = 4) -> bool:
        """
        Binary indicator of whether generation exhibits a pathological loop.
        Returns True if repetition score exceeds threshold.
        """
        score = VerificationOracle.detect_repetition_loops(token_ids, n_gram_window=n_gram_window)
        return score >= threshold

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        token_batches: List[torch.Tensor],
        loss_fn: Optional[Callable] = None
    ) -> float:
        """
        Evaluate average token perplexity over cached input token sequences.
        token_batches contains tensors of shape [batch_size, seq_len].
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        cross_entropy = loss_fn or nn.CrossEntropyLoss(reduction='sum')
        device = next(self.model.parameters()).device
        
        for batch in token_batches:
            input_ids = batch.to(device)
            if input_ids.shape[1] < 2:
                continue
                
            labels = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            outputs = self.model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            vocab_size = logits.shape[-1]
            loss = cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            
            num_tokens = labels.numel()
            total_loss += loss.item()
            total_tokens += num_tokens
            
        if total_tokens == 0:
            return 1.0
            
        avg_neg_log_lik = total_loss / total_tokens
        return float(np.exp(min(50.0, avg_neg_log_lik)))

    def verify_and_decide(
        self,
        layer_path: str,
        delta_w: torch.Tensor,
        target_eval_fn: Callable[[], float],
        retained_tokens: List[torch.Tensor],
    ) -> VerificationResult:
        """
        Execute full closed-loop verification pipeline:
        1. Measure base metrics
        2. Apply delta in-memory
        3. Measure edited metrics
        4. Render verdict (COMMIT / ADAPT / ROLLBACK)
        5. Automatically rollback if decision is ROLLBACK.
        """
        layer = self.get_layer(layer_path)
        base_w = layer.weight.detach().clone().cpu()
        cond_base = self.compute_condition_number(base_w)
        
        # 1. Measure pristine baseline
        score_base = target_eval_fn()
        ppl_base = self.evaluate_perplexity(retained_tokens)
        
        # 2. Apply delta
        self.apply_delta(layer_path, delta_w)
        
        # 3. Measure post-edit metrics
        score_edited = target_eval_fn()
        ppl_edited = self.evaluate_perplexity(retained_tokens)
        edited_w = layer.weight.detach().clone().cpu()
        cond_edited = self.compute_condition_number(edited_w)
        
        # Relative calculations
        target_gain = (score_edited - score_base) / max(1e-12, abs(score_base)) if score_base != 0 else (score_edited - score_base)
        ppl_drift = (ppl_edited - ppl_base) / max(1e-12, ppl_base)
        cond_drift = abs(cond_edited - cond_base) / max(1e-12, cond_base)
        
        # 4. Verification Decision
        if target_gain < self.tau_gain:
            decision = VerificationDecision.ROLLBACK
            rationale = (f"Target gain ({target_gain*100:.2f}%) fell below threshold "
                         f"({self.tau_gain*100:.2f}%). Rolling back.")
            self.restore_layer(layer_path)
            self.clear_backup(layer_path)
        elif ppl_drift > self.tau_ppl_drift:
            decision = VerificationDecision.ADAPT
            rationale = (f"Target gain passed (+{target_gain*100:.2f}%), but retained perplexity drifted "
                         f"by +{ppl_drift*100:.2f}% (tolerance: {self.tau_ppl_drift*100:.2f}%). Recommend stiffer null-space projection.")
        elif cond_drift > self.tau_condition_drift:
            decision = VerificationDecision.ADAPT
            rationale = (f"Target gain passed, but condition number shifted by "
                         f"{cond_drift*100:.2f}% (tolerance: {self.tau_condition_drift*100:.2f}%). Spectral distortion detected.")
        else:
            decision = VerificationDecision.COMMIT
            rationale = (f"All invariants satisfied: Target Gain = +{target_gain*100:.2f}%, "
                         f"PPL Drift = +{ppl_drift*100:.2f}%, Condition Drift = {cond_drift*100:.2f}%. Ready to commit.")
            self.clear_backup(layer_path)

        return VerificationResult(
            decision=decision,
            target_score_base=score_base,
            target_score_edited=score_edited,
            target_gain=target_gain,
            perplexity_base=ppl_base,
            perplexity_edited=ppl_edited,
            perplexity_drift_rel=ppl_drift,
            spectral_condition_base=cond_base,
            spectral_condition_edited=cond_edited,
            condition_drift_rel=cond_drift,
            rationale=rationale,
        )

