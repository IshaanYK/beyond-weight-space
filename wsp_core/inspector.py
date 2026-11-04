"""
WeightInspector: Diagnostic profiler for reading and analyzing weight tensors and activations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import numpy as np


@dataclass
class SpectralProfile:
    """Spectral characteristics of a weight tensor."""
    layer_name: str
    shape: Tuple[int, ...]
    frobenius_norm: float
    spectral_norm: float  # Top singular value sigma_1
    min_singular_value: float
    condition_number: float
    effective_rank: float
    top_singular_values: List[float]
    energy_top_10_percent: float


class WeightInspector:
    """
    Diagnostic profiler for reading neural network weights and activations
    without performing backpropagation or computing gradients.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._cached_activations: Dict[str, List[torch.Tensor]] = {}

    def get_layer(self, layer_path: str) -> nn.Module:
        """Retrieve a specific submodule by its dot-separated path."""
        module = self.model
        for part in layer_path.split('.'):
            if not hasattr(module, part):
                raise AttributeError(f"Module {type(module).__name__} has no attribute '{part}' (path: {layer_path})")
            module = getattr(module, part)
        return module

    def get_weight(self, layer_path: str) -> torch.Tensor:
        """Extract a clone of a layer's weight tensor in float32 on CPU."""
        module = self.get_layer(layer_path)
        if not hasattr(module, 'weight') or module.weight is None:
            raise ValueError(f"Layer '{layer_path}' does not possess a weight tensor.")
        return module.weight.detach().cpu().to(torch.float32)

    def profile_layer(self, layer_path: str, top_k: int = 10) -> SpectralProfile:
        """
        Compute comprehensive SVD spectral profile for a targeted layer weight matrix.
        Handles 2D linear matrices directly and flattens higher-rank tensors if needed.
        """
        w = self.get_weight(layer_path)
        if w.ndim > 2:
            w = w.view(w.shape[0], -1)

        frob_norm = torch.norm(w, p='fro').item()
        
        # SVD computation via torch.linalg.svd (svdvals for singular values)
        s = torch.linalg.svdvals(w)
        s_np = s.numpy()
        
        sigma_max = float(s_np[0]) if len(s_np) > 0 else 0.0
        # Filter for non-zero minimum singular value for numerical stability
        eps = 1e-12
        positive_s = s_np[s_np > eps]
        sigma_min = float(positive_s[-1]) if len(positive_s) > 0 else eps
        condition_number = float(sigma_max / sigma_min) if sigma_min > 0 else float('inf')

        # Effective rank via spectral entropy: exp(-sum p_i log p_i)
        s_sum = np.sum(s_np)
        if s_sum > eps:
            p = s_np / s_sum
            p = p[p > eps]
            entropy = -np.sum(p * np.log(p))
            eff_rank = float(np.exp(entropy))
        else:
            eff_rank = 1.0

        # Energy in top 10% of singular values
        k_10 = max(1, int(len(s_np) * 0.1))
        energy_top_10 = float(np.sum(s_np[:k_10] ** 2) / (np.sum(s_np ** 2) + eps))

        return SpectralProfile(
            layer_name=layer_path,
            shape=tuple(w.shape),
            frobenius_norm=frob_norm,
            spectral_norm=sigma_max,
            min_singular_value=sigma_min,
            condition_number=condition_number,
            effective_rank=eff_rank,
            top_singular_values=[float(x) for x in s_np[:top_k]],
            energy_top_10_percent=energy_top_10,
        )

    def register_activation_hook(self, layer_path: str, capture_input: bool = True) -> None:
        """
        Register a non-destructive forward hook on the target layer
        to capture representations during evaluation passes.
        """
        module = self.get_layer(layer_path)
        if layer_path not in self._cached_activations:
            self._cached_activations[layer_path] = []

        def hook_fn(_mod, inp, out):
            tensor_to_cache = inp[0] if capture_input else out
            # Store detached tensor on CPU
            self._cached_activations[layer_path].append(tensor_to_cache.detach().cpu().to(torch.float32))

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def clear_hooks(self) -> None:
        """Remove all forward hooks and clear activation cache."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._cached_activations.clear()

    def get_captured_activations(
        self,
        layer_path: str,
        flatten_tokens: bool = True,
        pooling: Optional[str] = None
    ) -> torch.Tensor:
        """
        Retrieve accumulated activation tensors across evaluated prompt batches.
        
        Args:
            layer_path: Path to the probed layer
            flatten_tokens: If True, flattens each (batch, seq, d) into (batch * seq, d)
                            before concatenating, supporting variable sequence lengths.
            pooling: Optional token pooling ('last' token or 'mean' over sequence).
            
        Returns:
            Concatenated activation tensor of shape [Total_Tokens, d] or [Total_Batches, d]
        """
        if layer_path not in self._cached_activations or not self._cached_activations[layer_path]:
            raise RuntimeError(f"No activations captured for layer '{layer_path}'. Ensure hook was registered and model was run.")
        
        tensors = self._cached_activations[layer_path]
        processed = []
        for t in tensors:
            if t.ndim == 3:
                if pooling == "last":
                    processed.append(t[:, -1, :])
                elif pooling == "mean":
                    processed.append(torch.mean(t, dim=1))
                elif flatten_tokens:
                    processed.append(t.view(-1, t.shape[-1]))
                else:
                    processed.append(t)
            elif t.ndim == 2:
                processed.append(t)
            else:
                processed.append(t.view(-1, t.shape[-1]))
                
        return torch.cat(processed, dim=0)

    @staticmethod
    def extract_contrastive_vector(
        target_acts: torch.Tensor,
        base_acts: torch.Tensor,
        method: str = "mean_diff"
    ) -> torch.Tensor:
        """
        Extract the directional intervention vector separating pathological vs. desired behavior.
        
        Args:
            target_acts: Activations during target/pathology state (e.g. repetitive loop) [N_t, d]
            base_acts: Activations during standard baseline state [N_b, d]
            method: 'mean_diff' (Delta of centroids) or 'pca_diff' (Top component of difference matrix)
        
        Returns:
            Normalized directional vector v in R^d
        """
        mu_target = torch.mean(target_acts, dim=0)
        mu_base = torch.mean(base_acts, dim=0)
        diff = mu_target - mu_base

        if method == "mean_diff":
            norm = torch.norm(diff, p=2)
            if norm < 1e-12:
                return torch.zeros_like(diff)
            return diff / norm
        elif method == "pca_diff":
            # Difference of centered matrices
            centered_target = target_acts - mu_target
            centered_base = base_acts - mu_base
            stacked = torch.cat([centered_target, -centered_base], dim=0)
            # SVD to find top variance direction
            _, _, vh = torch.linalg.svd(stacked, full_matrices=False)
            top_dir = vh[0]
            # Align sign with mean difference
            if torch.dot(top_dir, diff) < 0:
                top_dir = -top_dir
            return top_dir / torch.norm(top_dir, p=2)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'mean_diff' or 'pca_diff'.")

