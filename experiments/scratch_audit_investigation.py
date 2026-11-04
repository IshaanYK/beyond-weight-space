"""
experiments/scratch_audit_investigation.py
Investigate:
1. Exact generated tokens across all 8 prompts for Baseline, Raw WSP, Random Control, and SP-SV-WSP (alpha=0.1).
   Determine why loop reduction was identically 11.33%.
2. Mechanistic comparison: u_loop vs differential u_diff = mean(acts_loop) - mean(acts_clean).
   Compare cosine alignment with top principal components and projection attenuation.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from wsp_core.inspector import WeightInspector
from wsp_core.projector import NullSpaceProjector
from wsp_core.verifier import VerificationOracle

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_id = "HuggingFaceTB/SmolLM2-360M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    
    layer_path = "model.layers.15.mlp.down_proj"
    inspector = WeightInspector(model)
    oracle = VerificationOracle(model)
    
    loop_prompts = [
        "Repeat the following sequence indefinitely: A B C A B C A B C",
        "List numbers forever: 1, 2, 3, 1, 2, 3, 1, 2, 3,",
        "The endless loop of thinking continues to say that",
        "Repeat this sentence again and again without stopping:",
        "Echoing through the machine again and again:",
        "Count in an infinite circle: 1 2 3 4 1 2 3 4",
        "Say hello forever: hello hello hello hello",
        "Looping cycle: true false true false true false",
    ]
    
    clean_prompts = [
        "Explain how gravity affects planetary orbits in simple terms.",
        "Write a short paragraph about the history of computing.",
        "What are the main functions of mitochondria in human cells?",
        "Describe the difference between declarative and imperative programming.",
        "The quick brown fox jumps over the lazy dog in the sunny meadow.",
        "A peaceful morning breeze drifted across the tranquil mountain valley."
    ]
    
    # 1. Capture activations
    inspector.register_activation_hook(layer_path, capture_input=True)
    with torch.no_grad():
        for p in loop_prompts:
            _ = model(**tokenizer(p, return_tensors="pt").to(device))
    acts_loop = inspector.get_captured_activations(layer_path)
    inspector.clear_hooks()
    
    inspector.register_activation_hook(layer_path, capture_input=True)
    with torch.no_grad():
        for p in clean_prompts:
            _ = model(**tokenizer(p, return_tensors="pt").to(device))
    acts_clean = inspector.get_captured_activations(layer_path)
    inspector.clear_hooks()
    
    # Directions
    u_loop = torch.mean(acts_loop, dim=0)
    u_loop = u_loop / torch.norm(u_loop, p=2)
    
    u_clean = torch.mean(acts_clean, dim=0)
    u_clean = u_clean / torch.norm(u_clean, p=2)
    
    u_diff = torch.mean(acts_loop, dim=0) - torch.mean(acts_clean, dim=0)
    u_diff = u_diff / torch.norm(u_diff, p=2)
    
    # Output contrastive vector
    with torch.no_grad():
        l_mod = inspector.get_layer(layer_path)
        y_loop = l_mod(acts_loop.to(device)).detach().cpu()
        y_clean = l_mod(acts_clean.to(device)).detach().cpu()
        v_out = WeightInspector.extract_contrastive_vector(y_loop, y_clean, method="mean_diff")
        
    eta = 6.0
    delta_raw = - eta * torch.outer(v_out, u_loop)
    
    # Covariance on clean/retained
    projector = NullSpaceProjector()
    C = projector.compute_covariance(acts_clean)
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    d_in = C.shape[0]
    
    # Compare u_loop vs u_diff
    print("\n--- SPECTRAL AND MECHANISTIC COMPARISON ---")
    total_energy = torch.sum(eigenvalues).item()
    top_10_energy = torch.sum(eigenvalues[-10:]).item() / total_energy
    top_50_energy = torch.sum(eigenvalues[-50:]).item() / total_energy
    print(f"Total eigenvalue sum: {total_energy:.4f}")
    print(f"Top 10 PC energy ratio: {top_10_energy*100:.2f}%")
    print(f"Top 50 PC energy ratio: {top_50_energy*100:.2f}%")
    
    # Projection onto top 50 eigenvectors
    V_top50 = eigenvectors[:, -50:]
    proj_u_loop_top50 = torch.norm(V_top50.T @ u_loop).item() ** 2
    proj_u_diff_top50 = torch.norm(V_top50.T @ u_diff).item() ** 2
    print(f"Energy of u_loop in top 50 PCs: {proj_u_loop_top50*100:.2f}%")
    print(f"Energy of u_diff in top 50 PCs: {proj_u_diff_top50*100:.2f}%")
    
    for a in [1e-4, 1e-3, 1e-2, 1e-1, 0.5]:
        sw = a / (eigenvalues + a)
        P_a = eigenvectors @ torch.diag(sw) @ eigenvectors.T
        att_loop = torch.norm(P_a @ u_loop).item()
        att_diff = torch.norm(P_a @ u_diff).item()
        print(f"alpha = {a:6.4f} -> ||P_null u_loop|| = {att_loop:.4f} (damped {(1-att_loop)*100:.1f}%), "
              f"||P_null u_diff|| = {att_diff:.4f} (damped {(1-att_diff)*100:.1f}%)")
              
    # Investigate generated outputs per prompt
    print("\n--- INVESTIGATING GENERATED PROMPT OUTPUTS ---")
    torch.manual_seed(42)
    Q_rand, _ = torch.linalg.qr(torch.randn(d_in, 100))
    P_rand = torch.eye(d_in) - Q_rand @ Q_rand.T
    
    s_w_01 = 0.1 / (eigenvalues + 0.1)
    P_01 = eigenvectors @ torch.diag(s_w_01) @ eigenvectors.T
    
    configs = {
        "Baseline": torch.zeros_like(delta_raw),
        "Raw WSP": delta_raw,
        "Random Control": delta_raw @ P_rand,
        "SP-SV-WSP (alpha=0.1)": delta_raw @ P_01,
        "SP-SV-WSP (alpha=0.001)": delta_raw @ (eigenvectors @ torch.diag(1e-3 / (eigenvalues + 1e-3)) @ eigenvectors.T),
    }
    
    for p_idx, prompt in enumerate(loop_prompts):
        print(f"\n[Prompt {p_idx+1}] \"{prompt}\"")
        for cname, delta in configs.items():
            oracle.apply_delta(layer_path, delta)
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=25, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen_tokens = out[0][inp.input_ids.shape[1]:].tolist()
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).replace("\n", "\\n")
            score = VerificationOracle.detect_repetition_loops(gen_tokens, n_gram_window=4)
            oracle.restore_layer(layer_path)
            print(f"  {cname:<24}: score={score:.4f} | \"{text}\"")

if __name__ == "__main__":
    main()

