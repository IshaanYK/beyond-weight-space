"""
experiments/test_autonomous_agent_wsp.py
Investigating the Autonomous Agent (replacing GPT Astra) for SP-SV-WSP:
Closed-loop margin calibration and null-space projection on SmolLM2-360M.
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set root
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
from wsp_core.projector import NullSpaceProjector

def get_down_proj(model, layer_idx=15):
    return model.model.layers[layer_idx].mlp.down_proj

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    device = torch.device('cpu')
    model_id = 'HuggingFaceTB/SmolLM2-360M'
    print(f"[*] Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    model.eval()

    # 1. Collect clean training activations for C_train
    print("[*] Collecting activation covariance on training passages...", flush=True)
    train_texts = [
        "The solar system consists of the Sun and the astronomical objects bound to it by gravitational attraction.",
        "General relativity describes gravitation as a geometric property of spacetime curved by mass and energy.",
        "Quantum mechanics indicates that particles exhibit wave-particle duality and probabilistic state collapse.",
        "Photosynthesis is a biochemical process where chlorophyll absorbs photon energy to synthesize glucose.",
        "An algorithm is an unambiguous sequence of computational instructions designed to solve a decision problem.",
        "Relational database management systems enforce ACID properties to guarantee transactional reliability.",
        "The Renaissance was a fervent cultural and intellectual movement originating in fourteenth-century Italy.",
        "Epistemology investigates the nature, sources, scope, and foundational justification of human knowledge.",
        "Syntax governs the combinatorial grammatical rules that organize lexical tokens into coherent phrases.",
        "Monetary policy manages the money supply and interest rates to achieve sustainable economic expansion."
    ]

    target_layer_idx = 15
    down_proj = get_down_proj(model, target_layer_idx)
    d_out, d_in = down_proj.weight.shape # [960, 2560]
    print(f"[+] Layer {target_layer_idx} down_proj weight shape: {down_proj.weight.shape}")

    # Hook to capture input to down_proj (shape: [N, d_in]) and output (shape: [N, d_out])
    captured_inputs = []
    captured_outputs = []
    def hook_fn(module, inp, out):
        captured_inputs.append(inp[0].detach().reshape(-1, d_in))
        captured_outputs.append(out.detach().reshape(-1, d_out))

    hook_handle = down_proj.register_forward_hook(hook_fn)

    # Forward pass on train texts
    with torch.no_grad():
        for t in train_texts:
            enc = tok(t, return_tensors='pt').to(device)
            model(**enc)

    X_train = torch.cat(captured_inputs, dim=0) # [N_tokens, d_in]
    print(f"[+] Captured {X_train.shape[0]} clean training activation vectors in R^{d_in}")
    captured_inputs.clear()
    captured_outputs.clear()

    # Compute C_train and null-space projector
    projector = NullSpaceProjector(alpha=1e-3)
    C_train = projector.compute_covariance(X_train)
    eigs, V = torch.linalg.eigh(C_train)
    print(f"[+] C_train eigenvalues: min={eigs.min().item():.6e}, median={eigs.median().item():.6e}, max={eigs.max().item():.6e}")
    P_null_001 = projector.compute_projector(C_train, alpha=1e-3)
    P_null_01 = projector.compute_projector(C_train, alpha=1e-2)

    # 2. Test prompt that triggers repetition / loop
    prompt = "The sequence continues: 1 2 3 4 1 2 3 4 1 2 3"
    print(f"\n[*] Probing model on target prompt: '{prompt}'", flush=True)
    input_ids = tok(prompt, return_tensors='pt').input_ids.to(device)
    
    # Generate baseline
    with torch.no_grad():
        base_out = model.generate(input_ids, max_new_tokens=25, do_sample=False)
    base_text = tok.decode(base_out[0])
    print(f"[+] Baseline generation:\n    {base_text}\n")

    # Capture down_proj input and output on the looping prompt
    with torch.no_grad():
        out_logits = model(input_ids).logits
    loop_in = captured_inputs[-1] # [seq_len, d_in]
    loop_out = captured_outputs[-1] # [seq_len, d_out]
    hook_handle.remove()

    # The last token's activation
    u_last = loop_in[-1] # [d_in]
    v_last = loop_out[-1] # [d_out]
    
    # Check top-5 logits at last position
    logits_last = out_logits[0, -1]
    top5_vals, top5_ids = torch.topk(logits_last, 5)
    print("[+] Top-5 predicted next tokens at prompt end:")
    for rank, (val, idx) in enumerate(zip(top5_vals, top5_ids)):
        print(f"    #{rank+1}: token_id={idx.item()} ('{tok.decode([idx.item()])}') | logit={val.item():.4f}")

    top_tok = top5_ids[0].item()
    second_tok = top5_ids[1].item()
    logit_gap = (top5_vals[0] - top5_vals[1]).item()
    print(f"[+] Logit gap to flip token: {logit_gap:.4f} between '{tok.decode([top_tok])}' and '{tok.decode([second_tok])}'")

    # 3. Analyze u_last projection onto null space
    norm_u = torch.norm(u_last).item()
    u_proj_001 = torch.matmul(P_null_001, u_last)
    u_proj_01 = torch.matmul(P_null_01, u_last)
    norm_p001 = torch.norm(u_proj_001).item()
    norm_p01 = torch.norm(u_proj_01).item()
    
    print(f"\n[*] Null-Space Spectral Analysis of Target Vector u:")
    print(f"    ||u||_2           = {norm_u:.4f}")
    print(f"    ||P_null(1e-3) u|| = {norm_p001:.4f} (ratio: {norm_p001/norm_u:.4f})")
    print(f"    ||P_null(1e-2) u|| = {norm_p01:.4f} (ratio: {norm_p01/norm_u:.4f})")

    # Check alignment with C_train:
    energy_in_train = float(torch.dot(u_last, torch.matmul(C_train, u_last)).item()) / (norm_u**2 * eigs.max().item())
    print(f"    Normalized energy in C_train principal space: {energy_in_train:.6f}")

    # Check forward perturbation on X_train for different update scalings:
    # We want Delta W = - eta * outer(v_dir, u_null)
    # Compare:
    # 1. Raw WSP: - 1.0 * outer(v_last, u_last) / ||u_last||^2
    # 2. Strict SP-SV-WSP (un-normalized): - 1.0 * outer(v_last, P_null u_last) / ||u_last||^2
    # 3. Normalized Margin-Calibrated SP-SV-WSP: - beta * outer(v_last, u_proj / ||u_proj||^2)
    
    print("\n" + "=" * 80)
    print("EXPERIMENTAL INTERVENTIONS COMPARISON:")
    print("=" * 80)

    orig_weight = down_proj.weight.detach().clone()

    interventions = {
        "Raw WSP (eta=1.0)": - 1.0 * torch.outer(v_last, u_last) / (norm_u**2 + 1e-12),
        "Raw WSP (eta=3.0)": - 3.0 * torch.outer(v_last, u_last) / (norm_u**2 + 1e-12),
        "Strict SP-SV-WSP (un-normalized, alpha=1e-3)": - 1.0 * torch.outer(v_last, u_proj_001) / (norm_u**2 + 1e-12),
        "Strict SP-SV-WSP (un-normalized, alpha=1e-2)": - 1.0 * torch.outer(v_last, u_proj_01) / (norm_u**2 + 1e-12),
        "Normalized SP-SV-WSP (beta=1.0, alpha=1e-3)": - 1.0 * torch.outer(v_last, u_proj_001) / (norm_p001**2 + 1e-12),
        "Normalized SP-SV-WSP (beta=2.0, alpha=1e-3)": - 2.0 * torch.outer(v_last, u_proj_001) / (norm_p001**2 + 1e-12),
        "Normalized SP-SV-WSP (beta=3.0, alpha=1e-3)": - 3.0 * torch.outer(v_last, u_proj_001) / (norm_p001**2 + 1e-12),
    }

    # Held out validation passage to measure perplexity drift:
    val_text = "Calculus provides mathematical methods for modeling dynamic continuous change through differentiation and integration."
    val_ids = tok(val_text, return_tensors='pt').input_ids.to(device)
    
    # Base val loss
    with torch.no_grad():
        base_val_loss = model(val_ids, labels=val_ids).loss.item()
        base_val_ppl = math.exp(base_val_loss)
    print(f"Pristine Base Held-Out PPL: {base_val_ppl:.4f}\n")

    for name, delta_w in interventions.items():
        with torch.no_grad():
            # Apply delta
            down_proj.weight.copy_(orig_weight + delta_w)
            
            # 1. Target generation
            gen_out = model.generate(input_ids, max_new_tokens=25, do_sample=False)
            gen_text = tok.decode(gen_out[0])[len(prompt):].strip().replace('\n', ' ')
            
            # 2. Target logit gap
            new_logits = model(input_ids).logits[0, -1]
            new_top = torch.topk(new_logits, 2)
            top_tok_new = new_top.indices[0].item()
            new_gap = (new_top.values[0] - new_top.values[1]).item()
            
            # 3. Held-out PPL
            val_loss = model(val_ids, labels=val_ids).loss.item()
            val_ppl = math.exp(val_loss)
            ppl_drift = (val_ppl - base_val_ppl) / base_val_ppl * 100.0
            
            # 4. Retention Invariance on X_train: ||Delta W X^T||_F / ||W X^T||_F
            pert = torch.matmul(delta_w, X_train.T)
            orig_out = torch.matmul(orig_weight, X_train.T)
            rel_pert = (torch.norm(pert, 'fro') / (torch.norm(orig_out, 'fro') + 1e-12)).item() * 100.0
            inv_pct = max(0.0, 100.0 - rel_pert)
            
            # Restore
            down_proj.weight.copy_(orig_weight)

        print(f"--- {name} ---")
        print(f"  Generated text: '{gen_text}'")
        print(f"  Top token now : '{tok.decode([top_tok_new])}' (Margin: {new_gap:.2f})")
        print(f"  Held-out PPL  : {val_ppl:.4f} (Drift: {ppl_drift:+.3f}%)")
        print(f"  Invariance    : {inv_pct:.2f}% (Rel Perturbation: {rel_pert:.4f}%)")
        print(f"  ||Delta W||_F : {torch.norm(delta_w, 'fro').item():.4f}")
        print()

if __name__ == '__main__':
    main()
