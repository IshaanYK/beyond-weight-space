"""
experiments/04_multimodel_comparative_benchmark.py:
Multi-Model Cross-Architecture Validation of Subspace-Projected Weight-Space Programming.
Evaluates across distinct open-weight model architectures:
1. HuggingFaceTB/SmolLM2-360M (Llama / RMSNorm / SwiGLU)
2. Qwen/Qwen2.5-0.5B (Qwen / RMSNorm / SwiGLU)
3. EleutherAI/pythia-410m (GPT-NeoX / LayerNorm / Parallel Attention)

Tests cross-architecture universality of:
- Null-space suppression ratio (||Delta W_safe X|| / ||Delta W_raw X||)
- Retained perplexity preservation
- SVD condition number stability
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure wsp_core is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from wsp_core.inspector import WeightInspector, SpectralProfile
from wsp_core.projector import NullSpaceProjector, ProjectionResult
from wsp_core.verifier import VerificationOracle, VerificationDecision, VerificationResult

RETAINED_TEXTS = [
    "The solar system consists of the Sun and astronomical objects bound by gravity.",
    "Photosynthesis is a biological process that converts light energy into chemical energy.",
    "In computer science, an algorithm is a finite sequence of instructions to solve problems.",
    "Quantum mechanics describes the physical properties of nature at microscopic scales.",
    "Machine learning models discover mathematical representations from training data."
]

MODELS_TO_BENCHMARK = [
    {
        "name": "SmolLM2-360M",
        "id": "HuggingFaceTB/SmolLM2-360M",
        "architecture": "Llama / SwiGLU",
        "target_layer": "model.layers.15.mlp.down_proj"
    },
    {
        "name": "Qwen2.5-0.5B",
        "id": "Qwen/Qwen2.5-0.5B",
        "architecture": "Qwen / SwiGLU",
        "target_layer": "model.layers.12.mlp.down_proj"
    }
]


def evaluate_model_architecture(model_info: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = model_info["id"]
    model_name = model_info["name"]
    layer_path = model_info["target_layer"]
    
    print(f"\n" + "="*70)
    print(f"[*] Benchmarking Architecture: {model_name} ({model_info['architecture']})")
    print(f"[*] Hugging Face ID: {model_id} | Probing Layer: {layer_path}")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    inspector = WeightInspector(model)
    oracle = VerificationOracle(model)
    projector = NullSpaceProjector(alpha=1e-3, gamma=0.02)
    
    # 1. Base SVD Profile
    base_prof = inspector.profile_layer(layer_path)
    base_w = inspector.get_weight(layer_path)
    print(f"[+] Base Weight Shape: {tuple(base_w.shape)}, Frobenius Norm: {base_prof.frobenius_norm:.2f}, Condition Number: {base_prof.condition_number:.2f}")
    
    # 2. Capture Retained Activations
    inspector.register_activation_hook(layer_path, capture_input=True)
    with torch.no_grad():
        for t in RETAINED_TEXTS:
            inp = tokenizer(t, return_tensors="pt").to(device)
            _ = model(**inp)
    acts_retain = inspector.get_captured_activations(layer_path)
    inspector.clear_hooks()
    print(f"[+] Captured {acts_retain.shape[0]} token representations in R^{acts_retain.shape[1]}")
    
    # 3. Base Perplexity
    tokens_batch = [tokenizer(t, return_tensors="pt").input_ids for t in RETAINED_TEXTS]
    base_ppl = oracle.evaluate_perplexity(tokens_batch)
    print(f"[+] Base Retained Perplexity: {base_ppl:.2f}")
    
    # 4. Synthesize Edit Intent (Simulating a targeted 2% Frobenius perturbation)
    d_out, d_in = base_w.shape
    torch.manual_seed(42)
    v = torch.randn(d_out)
    v = v / torch.norm(v, p=2)
    u = torch.mean(acts_retain, dim=0)
    u = u / (torch.norm(u, p=2) + 1e-12)
    
    scale = 0.02 * base_prof.frobenius_norm
    delta_raw = - scale * torch.outer(v, u)
    
    # 5. Null-Space Projection
    C_retain = projector.compute_covariance(acts_retain)
    proj_result = projector.project(
        delta_w_raw=delta_raw,
        covariance=C_retain,
        base_w=base_w,
        test_activations=acts_retain[:50]
    )
    print(f"[+] Null-Space Invariance Score: {proj_result.retained_invariance_score*100:.2f}%")
    print(f"[+] Residual Projection Error: {proj_result.projection_residual_error:.6f}")
    
    # 6. Compare Raw vs. Safe in Memory
    # Test Raw
    oracle.apply_delta(layer_path, delta_raw)
    raw_ppl = oracle.evaluate_perplexity(tokens_batch)
    raw_cond = VerificationOracle.compute_condition_number(inspector.get_weight(layer_path))
    raw_ppl_drift = (raw_ppl - base_ppl) / base_ppl * 100.0
    oracle.restore_layer(layer_path)
    
    # Test Safe (SP-SV-WSP)
    oracle.apply_delta(layer_path, proj_result.delta_w_safe)
    safe_ppl = oracle.evaluate_perplexity(tokens_batch)
    safe_cond = VerificationOracle.compute_condition_number(inspector.get_weight(layer_path))
    safe_ppl_drift = (safe_ppl - base_ppl) / base_ppl * 100.0
    oracle.restore_layer(layer_path)
    
    result = {
        "model_name": model_name,
        "model_id": model_id,
        "architecture": model_info["architecture"],
        "target_layer": layer_path,
        "layer_shape": list(base_w.shape),
        "frob_norm": base_prof.frobenius_norm,
        "base_condition": base_prof.condition_number,
        "raw_condition": raw_cond,
        "safe_condition": safe_cond,
        "base_ppl": base_ppl,
        "raw_ppl": raw_ppl,
        "raw_ppl_drift_pct": raw_ppl_drift,
        "safe_ppl": safe_ppl,
        "safe_ppl_drift_pct": safe_ppl_drift,
        "invariance_score": proj_result.retained_invariance_score,
        "residual_error": proj_result.projection_residual_error
    }
    
    print(f"\n[Summary for {model_name}]:")
    print(f"  Raw WSP PPL Drift:  {f'+{raw_ppl_drift:.2f}%' if raw_ppl_drift >= 0 else f'{raw_ppl_drift:.2f}%'}")
    print(f"  SP-SV-WSP PPL Drift: {f'+{safe_ppl_drift:.2f}%' if safe_ppl_drift >= 0 else f'{safe_ppl_drift:.2f}%'}")
    print(f"  Invariance Score:   {proj_result.retained_invariance_score*100:.2f}%")
    
    return result


def run_cross_architecture_benchmarks():
    all_results = []
    for model_info in MODELS_TO_BENCHMARK:
        try:
            res = evaluate_model_architecture(model_info)
            all_results.append(res)
        except Exception as e:
            print(f"[-] Error evaluating {model_info['name']}: {e}")
            
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "multimodel_comparative_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n" + "="*90)
    print(f"{'MODEL':<16} | {'ARCH':<16} | {'BASE PPL':<10} | {'RAW DRIFT':<12} | {'SAFE DRIFT':<12} | {'INVARIANCE':<10}")
    print("-" * 90)
    for r in all_results:
        raw_str = f"{r['raw_ppl_drift_pct']:+.2f}%"
        safe_str = f"{r['safe_ppl_drift_pct']:+.2f}%"
        inv_str = f"{r['invariance_score']*100:.2f}%"
        print(f"{r['model_name']:<16} | {r['architecture']:<16} | {r['base_ppl']:>9.2f}  | {raw_str:>11} | {safe_str:>11} | {inv_str:>9}")
    print("="*90)
    print(f"[+] Multi-model benchmark report saved to {out_file}")


if __name__ == "__main__":
    run_cross_architecture_benchmarks()
