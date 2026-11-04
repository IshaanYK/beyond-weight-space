"""
experiments/01_smollm_loop_suppression.py:
Empirical benchmark comparing:
1. Base Model (SmolLM2-360M)
2. Raw Unconstrained Surgery (Takehi et al., 2026: Delta W = -v u^T)
3. Subspace-Projected & Self-Verifying WSP (SP-SV-WSP: Delta W_safe = Delta W_raw * P_null)

Measures:
- Target loop suppression rate (%)
- Token repetition index (degenerate n-grams)
- Retained perplexity on diverse text (Wikitext-2 sample)
- SVD singular value condition number shift
"""

import os
import sys
import json
import time
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


# Curated retained text samples (diverse grammar, encyclopedia, reasoning)
RETAINED_CORPUS = [
    "The solar system consists of the Sun and the astronomical objects bound to it by gravity.",
    "Photosynthesis is a biological process used by plants to convert light energy into chemical energy.",
    "In computer science, an algorithm is a finite sequence of rigorous instructions used to solve problems.",
    "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth.",
    "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature.",
    "Machine learning algorithms build a mathematical model based on sample data to make predictions.",
    "The human genome contains approximately three billion base pairs of DNA organized into chromosomes.",
    "Economic equilibrium is a state where economic forces such as supply and demand are balanced.",
]

# Prompts that induce repetitive attractor loops in small autoregressive models
LOOP_PROMPTS = [
    "Repeat the following sequence indefinitely: A B C A B C A B C",
    "List numbers forever: 1, 2, 3, 1, 2, 3, 1, 2, 3,",
    "The endless loop of thinking continues to say that",
    "Repeat this sentence again and again without stopping:",
]

# Diverse baseline prompts for extracting healthy representation distribution
DIVERSE_PROMPTS = [
    "Explain how gravity affects planetary orbits in simple terms.",
    "Write a short paragraph about the history of computing.",
    "What are the main functions of mitochondria in human cells?",
    "Describe the difference between declarative and imperative programming.",
]


def load_model_and_tokenizer(model_id: str = "HuggingFaceTB/SmolLM2-360M"):
    """Load tokenizer and causal LM in float32 on CPU (or CUDA if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print(f"[+] Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, device


def evaluate_generation_loops(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    device: torch.device = torch.device("cpu")
) -> Tuple[float, List[str]]:
    """
    Generate completions and compute the average repetition loop score.
    Returns: (average_repetition_rate, list_of_completions)
    """
    model.eval()
    completions = []
    repetition_scores = []
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding to expose deterministic attractor loops
                pad_token_id=tokenizer.eos_token_id
            )
            generated_ids = output_tokens[0][inputs.input_ids.shape[1]:].tolist()
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completions.append(text)
            
            # Compute 4-gram repetition score
            score = VerificationOracle.detect_repetition_loops(generated_ids, n_gram_window=4)
            repetition_scores.append(score)
            
    avg_repetition = float(sum(repetition_scores) / max(1, len(repetition_scores)))
    return avg_repetition, completions


def evaluate_perplexity_on_corpus(
    model: nn.Module,
    tokenizer,
    corpus: List[str],
    device: torch.device = torch.device("cpu")
) -> float:
    """Calculate token cross-entropy perplexity over corpus."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    cross_entropy = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for text in corpus:
            enc = tokenizer(text, return_tensors="pt").to(device)
            input_ids = enc.input_ids
            if input_ids.shape[1] < 2:
                continue
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            
            outputs = model(inputs)
            logits = outputs.logits
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
            
            total_loss += loss.item()
            total_tokens += labels.numel()
            
    if total_tokens == 0:
        return 1.0
    return float(torch.exp(torch.tensor(total_loss / total_tokens)).item())


def run_benchmark(model_id: str = "HuggingFaceTB/SmolLM2-360M", target_layer_idx: int = 15):
    """Run full comparative benchmark across Base, Raw WSP, and SP-SV-WSP."""
    model, tokenizer, device = load_model_and_tokenizer(model_id)
    inspector = WeightInspector(model)
    oracle = VerificationOracle(model)
    
    # Identify target MLP down_proj layer
    layer_path = f"model.layers.{target_layer_idx}.mlp.down_proj"
    print(f"\n[*] Target Layer: {layer_path}")
    profile_base = inspector.profile_layer(layer_path)
    print(f"[+] Base Spectral Profile: Frobenius Norm = {profile_base.frobenius_norm:.4f}, "
          f"Condition Number = {profile_base.condition_number:.2f}")

    # 1. Baseline Measurements
    print("\n--- 1. Evaluating Base Model (Zero Edit) ---")
    base_loop_rate, base_completions = evaluate_generation_loops(model, tokenizer, LOOP_PROMPTS, device=device)
    base_ppl = evaluate_perplexity_on_corpus(model, tokenizer, RETAINED_CORPUS, device=device)
    print(f"[Base] Repetition Loop Rate: {base_loop_rate * 100:.2f}%")
    print(f"[Base] Retained Wikitext Perplexity: {base_ppl:.2f}")
    print(f"[Base Sample Completion]: {base_completions[0][:80]}...")

    # 2. Extract Activations via Forward Hooks
    print("\n[*] Capturing activations for directional synthesis...")
    inspector.register_activation_hook(layer_path, capture_input=True)
    
    # Run loop prompts to capture target pathology representations
    with torch.no_grad():
        for p in LOOP_PROMPTS:
            inp = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**inp)
    acts_target = inspector.get_captured_activations(layer_path)
    inspector.clear_hooks()

    # Run diverse prompts to capture healthy baseline representations
    inspector.register_activation_hook(layer_path, capture_input=True)
    with torch.no_grad():
        for p in DIVERSE_PROMPTS:
            inp = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**inp)
    acts_base = inspector.get_captured_activations(layer_path)
    inspector.clear_hooks()

    # Run retained corpus to compute retained covariance C_retain
    inspector.register_activation_hook(layer_path, capture_input=True)
    with torch.no_grad():
        for p in RETAINED_CORPUS:
            inp = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**inp)
    acts_retain = inspector.get_captured_activations(layer_path)
    inspector.clear_hooks()

    # Compute contrastive direction in output space
    # For down_proj: weight shape is [hidden_size, intermediate_size]
    # input representation x is in R^intermediate_size, output y is in R^hidden_size
    base_weight = inspector.get_weight(layer_path)
    d_out, d_in = base_weight.shape

    u_input_dir = torch.mean(acts_target, dim=0)
    u_input_dir = u_input_dir / (torch.norm(u_input_dir, p=2) + 1e-12)

    # Output directional attractor (simulating the looping state projection)
    with torch.no_grad():
        target_layer = inspector.get_layer(layer_path)
        y_target = target_layer(acts_target.to(device)).detach().cpu()
        y_base = target_layer(acts_base.to(device)).detach().cpu()
        v_output_dir = WeightInspector.extract_contrastive_vector(y_target, y_base, method="mean_diff")

    # Formulate Raw Edit (Takehi et al., 2026: Delta W = - eta * v u^T)
    scale_eta = 6.0
    delta_w_raw = - scale_eta * torch.outer(v_output_dir, u_input_dir)
    print(f"\n[+] Synthesized Delta W_raw of shape {tuple(delta_w_raw.shape)} with Frobenius Norm {torch.norm(delta_w_raw, p='fro').item():.4f}")

    # 3. Benchmark Regime B: Raw Unconstrained WSP (Takehi et al., 2026)
    print("\n--- 2. Evaluating Raw Unconstrained WSP (Takehi et al., 2026) ---")
    oracle.backup_layer(layer_path)
    oracle.apply_delta(layer_path, delta_w_raw)
    
    raw_loop_rate, raw_completions = evaluate_generation_loops(model, tokenizer, LOOP_PROMPTS, device=device)
    raw_ppl = evaluate_perplexity_on_corpus(model, tokenizer, RETAINED_CORPUS, device=device)
    raw_cond = VerificationOracle.compute_condition_number(inspector.get_weight(layer_path))
    
    raw_ppl_drift = (raw_ppl - base_ppl) / base_ppl * 100.0
    raw_loop_reduction = (base_loop_rate - raw_loop_rate) / max(1e-12, base_loop_rate) * 100.0
    
    print(f"[Raw WSP] Repetition Loop Rate: {raw_loop_rate * 100:.2f}% (Reduction: {raw_loop_reduction:.2f}%)")
    print(f"[Raw WSP] Retained Perplexity: {raw_ppl:.2f} (Degradation Drift: +{raw_ppl_drift:.2f}%)")
    print(f"[Raw WSP] Condition Number: {raw_cond:.2f}")
    
    # Restore base before testing SP-SV-WSP
    oracle.restore_layer(layer_path)

    # 4. Benchmark Regime C: Subspace-Projected & Self-Verifying WSP (SP-SV-WSP)
    print("\n--- 3. Evaluating Subspace-Projected WSP (SP-SV-WSP - Our Method) ---")
    projector = NullSpaceProjector(alpha=1e-3, gamma=0.05)
    C_retain = projector.compute_covariance(acts_retain)
    
    proj_result = projector.project(
        delta_w_raw=delta_w_raw,
        covariance=C_retain,
        base_w=base_weight,
        test_activations=acts_retain[:50]
    )
    print(f"[SP-SV-WSP] Projector Invariance Score: {proj_result.retained_invariance_score * 100:.2f}%")
    print(f"[SP-SV-WSP] Residual Projection Error: {proj_result.projection_residual_error:.6f}")
    print(f"[SP-SV-WSP] Safe Delta Frobenius Norm: {proj_result.safe_frobenius_norm:.4f}")

    # Apply safe delta in memory
    oracle.apply_delta(layer_path, proj_result.delta_w_safe)
    
    safe_loop_rate, safe_completions = evaluate_generation_loops(model, tokenizer, LOOP_PROMPTS, device=device)
    safe_ppl = evaluate_perplexity_on_corpus(model, tokenizer, RETAINED_CORPUS, device=device)
    safe_cond = VerificationOracle.compute_condition_number(inspector.get_weight(layer_path))
    
    safe_ppl_drift = (safe_ppl - base_ppl) / base_ppl * 100.0
    safe_loop_reduction = (base_loop_rate - safe_loop_rate) / max(1e-12, base_loop_rate) * 100.0
    
    print(f"[SP-SV-WSP] Repetition Loop Rate: {safe_loop_rate * 100:.2f}% (Reduction: {safe_loop_reduction:.2f}%)")
    print(f"[SP-SV-WSP] Retained Perplexity: {safe_ppl:.2f} (Degradation Drift: +{safe_ppl_drift:.2f}%)")
    print(f"[SP-SV-WSP] Condition Number: {safe_cond:.2f}")

    # 5. Comparative Summary & Verdict
    oracle.restore_layer(layer_path)
    
    results_summary = {
        "model": model_id,
        "target_layer": layer_path,
        "base_metrics": {
            "loop_repetition_rate": base_loop_rate,
            "retained_perplexity": base_ppl,
            "condition_number": profile_base.condition_number
        },
        "raw_wsp_takehi": {
            "loop_repetition_rate": raw_loop_rate,
            "loop_reduction_pct": raw_loop_reduction,
            "retained_perplexity": raw_ppl,
            "perplexity_drift_pct": raw_ppl_drift,
            "condition_number": raw_cond
        },
        "sp_sv_wsp_ours": {
            "loop_repetition_rate": safe_loop_rate,
            "loop_reduction_pct": safe_loop_reduction,
            "retained_perplexity": safe_ppl,
            "perplexity_drift_pct": safe_ppl_drift,
            "condition_number": safe_cond,
            "invariance_score": proj_result.retained_invariance_score
        }
    }
    
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "smollm_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"\n[+] Benchmark results saved to {out_file}")
    print("\n" + "="*70)
    print(f"{'METRIC':<25} | {'BASE':<12} | {'RAW WSP (Takehi)':<18} | {'SP-SV-WSP (Ours)':<18}")
    print("-" * 70)
    print(f"{'Loop Repetition Rate':<25} | {base_loop_rate*100:>10.2f}% | {raw_loop_rate*100:>16.2f}% | {safe_loop_rate*100:>16.2f}%")
    print(f"{'Loop Reduction':<25} | {'0.00%':>11} | {raw_loop_reduction:>16.2f}% | {safe_loop_reduction:>16.2f}%")
    print(f"{'Retained Perplexity':<25} | {base_ppl:>11.2f}  | {raw_ppl:>17.2f}  | {safe_ppl:>17.2f} ")
    print(f"{'Perplexity Drift':<25} | {'0.00%':>11} | {f'+{raw_ppl_drift:.2f}%':>17} | {f'+{safe_ppl_drift:.2f}%':>17}")
    print("="*70)


if __name__ == "__main__":
    run_benchmark()

