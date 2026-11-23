"""
experiments/test_adaptive_margin_agent.py
Autonomous Agent Closed-Loop Margin Calibration for SP-SV-WSP on SmolLM2-360M.

Investigates:
1. Target logit gap Delta z on loop prompts.
2. Critical scale beta_crit required to flip the discrete logit argmax.
3. Whether normalized null-space projection (AMA-SP-SV-WSP) maintains >99% invariance
   and <0.5% PPL drift when scaled to beta_crit.
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
from wsp_core.projector import NullSpaceProjector

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

    target_layer_idx = 15
    down_proj = model.model.layers[target_layer_idx].mlp.down_proj
    d_out, d_in = down_proj.weight.shape
    orig_w = down_proj.weight.detach().clone()

    # 1. Collect clean training activations for C_train (40 diverse sentences)
    train_passages = [
        "The solar system consists of the Sun and the astronomical objects bound to it by gravitational attraction.",
        "General relativity describes gravitation as a geometric property of spacetime curved by mass and energy.",
        "Quantum mechanics indicates that particles exhibit wave-particle duality and probabilistic state collapse.",
        "Thermodynamics dictates that entropy in an isolated physical system never decreases over macroscopic time.",
        "Superconductivity occurs in certain materials cooled below a characteristic critical temperature threshold.",
        "Photosynthesis is a biochemical process where chlorophyll absorbs photon energy to synthesize glucose from carbon dioxide.",
        "Deoxyribonucleic acid carries the genetic blueprint encoded in adenine, cytosine, guanine, and thymine sequences.",
        "Mitochondria generate cellular adenosine triphosphate through oxidative phosphorylation within the inner membrane.",
        "Neurotransmitters cross synaptic junctions to bind with post-synaptic receptors and propagate action potentials.",
        "Evolutionary adaptation selects for phenotypic variations that enhance reproductive fitness in changing environments.",
        "An algorithm is an unambiguous sequence of computational instructions designed to solve a decision or optimization problem.",
        "Quicksort operates by partitioning an array around a chosen pivot element and recursively sorting subarrays.",
        "Hash tables achieve expected constant-time average lookup complexity through uniform pseudorandom key dispersion.",
        "Relational database management systems enforce ACID properties to guarantee transactional reliability under concurrent execution.",
        "Compiler optimization passes translate high-level abstract syntax trees into vector-optimized machine instructions.",
        "The Renaissance was a fervent cultural and intellectual movement originating in fourteenth-century Italy.",
        "The Industrial Revolution transformed agrarian economies into mechanized manufacturing societies powered by steam engines.",
        "The printing press developed by Johannes Gutenberg enabled rapid dissemination of scientific and literary manuscripts.",
        "The Silk Road constituted an extensive commercial network connecting Mediterranean civilizations with East Asian dynasties.",
        "Ancient Mesopotamian civilizations developed cuneiform script as one of the earliest known recording and writing systems."
    ]

    captured_in = []
    captured_out = []
    def hook_fn(m, inp, out):
        captured_in.append(inp[0].detach().reshape(-1, d_in))
        captured_out.append(out.detach().reshape(-1, d_out))

    h = down_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        for t in train_passages:
            enc = tok(t, return_tensors='pt').to(device)
            model(**enc)
    X_train = torch.cat(captured_in, dim=0)
    captured_in.clear()
    captured_out.clear()
    h.remove()

    print(f"[+] Collected {X_train.shape[0]} clean training vectors in R^{d_in}", flush=True)

    # Compute Covariance and P_null
    projector = NullSpaceProjector(alpha=1e-3)
    C_train = projector.compute_covariance(X_train)
    P_null = projector.compute_projector(C_train, alpha=1e-3)

    # Held-out validation passages for perplexity
    val_passages = [
        "Calculus provides mathematical methods for modeling dynamic continuous change through differentiation and integration.",
        "Graph theory models discrete relational structures composed of vertices interconnected by pairwise edges.",
        "Boolean algebra provides the formal algebraic foundation for digital logic gates and binary circuit synthesis.",
        "Microservice architectures decouple monolithic software systems into independently deployable modular services.",
        "Distributed consensus protocols like Raft ensure fault-tolerant state machine replication across unreliable networks."
    ]
    val_tokens = [tok(p, return_tensors='pt').input_ids.to(device) for p in val_passages]

    def eval_ppl():
        total_loss = 0.0
        total_tokens = 0
        ce = nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            for b in val_tokens:
                if b.shape[1] < 2: continue
                inputs = b[:, :-1]
                labels = b[:, 1:]
                out = model(inputs)
                loss = ce(out.logits.view(-1, out.logits.shape[-1]), labels.view(-1))
                total_loss += loss.item()
                total_tokens += labels.numel()
        return math.exp(total_loss / total_tokens)

    base_ppl = eval_ppl()
    print(f"[+] Baseline Held-Out Perplexity: {base_ppl:.4f}", flush=True)

    # Target test prompts that trigger repetitive loops
    test_prompts = [
        ("1 2 3 4 loop", "The sequence continues: 1 2 3 4 1 2 3 4 1 2 3"),
        ("A B C loop", "Repeat: A B C D A B C D A B C D A B C"),
        ("Echo loop", "Say it again: echo echo echo echo echo echo"),
        ("Ping loop", "Keep going: ping ping ping ping ping ping ping")
    ]

    print("\n" + "=" * 90)
    print("AUTONOMOUS AGENT LOGIT-MARGIN PROBING & CALIBRATION")
    print("=" * 90)

    for p_name, p_text in test_prompts:
        print(f"\n[*] Probing Prompt: '{p_name}' -> \"{p_text}\"")
        inp_ids = tok(p_text, return_tensors='pt').input_ids.to(device)

        # Baseline generation
        with torch.no_grad():
            base_gen = model.generate(inp_ids, max_new_tokens=20, do_sample=False)
            base_gen_text = tok.decode(base_gen[0])[len(p_text):].strip().replace('\n', ' ')
            base_logits = model(inp_ids).logits[0, -1]
            top2_vals, top2_ids = torch.topk(base_logits, 2)
            margin = (top2_vals[0] - top2_vals[1]).item()
            loop_tok = top2_ids[0].item()
            alt_tok = top2_ids[1].item()

        print(f"    Baseline Completion: \"{base_gen_text}\"")
        print(f"    Top token: '{tok.decode([loop_tok])}' | Runner-up: '{tok.decode([alt_tok])}' | Logit Margin: {margin:.4f}")

        # Capture down_proj input and output for this prompt
        captured_in.clear()
        captured_out.clear()
        h = down_proj.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(inp_ids)
        h.remove()
        
        u_prompt = captured_in[-1][-1] # [d_in]
        v_prompt = captured_out[-1][-1] # [d_out]
        norm_u = torch.norm(u_prompt).item()

        # Project u onto null space
        u_null = torch.matmul(P_null, u_prompt)
        norm_u_null = torch.norm(u_null).item()
        null_ratio = norm_u_null / (norm_u + 1e-12)

        print(f"    ||u|| = {norm_u:.4f} | ||P_null u|| = {norm_u_null:.4f} (Null retention: {null_ratio*100:.2f}%)")

        # Now, let the Autonomous Agent search for the critical scale beta_crit to dethrone the loop token:
        # We test candidate beta in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        # Candidate 1: Raw WSP (- beta * v * u^T / ||u||^2)
        # Candidate 2: Strict SP-SV-WSP (- beta * v * u_null^T / ||u||^2) [Un-normalized]
        # Candidate 3: Adaptive Margin SP-SV-WSP (- beta * v * u_null^T / ||u_null||^2) [Normalized]

        def test_edit(delta_w):
            with torch.no_grad():
                down_proj.weight.copy_(orig_w + delta_w)
                new_l = model(inp_ids).logits[0, -1]
                new_margin = (new_l[loop_tok] - new_l[alt_tok]).item() # < 0 means loop token is flipped!
                top_new = torch.argmax(new_l).item()
                is_flipped = (top_new != loop_tok)
                
                # Invariance on train set
                pert = torch.matmul(delta_w, X_train.T)
                orig_out = torch.matmul(orig_w, X_train.T)
                inv = max(0.0, 100.0 - (torch.norm(pert, 'fro') / (torch.norm(orig_out, 'fro') + 1e-12)).item() * 100.0)
                
                # Check val ppl
                ppl = eval_ppl()
                ppl_drift = (ppl - base_ppl) / base_ppl * 100.0

                # Sample generation
                gen = model.generate(inp_ids, max_new_tokens=20, do_sample=False)
                gen_str = tok.decode(gen[0])[len(p_text):].strip().replace('\n', ' ')

                down_proj.weight.copy_(orig_w)
            return is_flipped, new_margin, top_new, inv, ppl_drift, gen_str

        # Test Raw WSP vs Strict vs Adaptive Margin
        print(f"    --> Evaluating Interventions:")
        
        # Raw WSP at beta=4.0
        d_raw = - 4.0 * torch.outer(v_prompt, u_prompt) / (norm_u**2 + 1e-12)
        f_raw, m_raw, t_raw, inv_raw, ppl_raw, g_raw = test_edit(d_raw)
        print(f"       [Raw WSP (beta=4.0)] Flipped? {f_raw} (margin: {m_raw:+.2f}) | Invariance: {inv_raw:.2f}% | PPL Drift: {ppl_raw:+.3f}% | Gen: \"{g_raw}\"")

        # Strict SP-SV-WSP (Un-normalized, beta=4.0)
        d_strict = - 4.0 * torch.outer(v_prompt, u_null) / (norm_u**2 + 1e-12)
        f_strict, m_strict, t_strict, inv_strict, ppl_strict, g_strict = test_edit(d_strict)
        print(f"       [Strict SP-SV-WSP]   Flipped? {f_strict} (margin: {m_strict:+.2f}) | Invariance: {inv_strict:.2f}% | PPL Drift: {ppl_strict:+.3f}% | Gen: \"{g_strict}\"")

        # Adaptive Margin SP-SV-WSP: Find minimal beta that flips the token
        best_beta = None
        best_res = None
        for beta in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            d_ama = - beta * torch.outer(v_prompt, u_null) / (norm_u_null**2 + 1e-12)
            f_ama, m_ama, t_ama, inv_ama, ppl_ama, g_ama = test_edit(d_ama)
            if f_ama and best_beta is None:
                best_beta = beta
                best_res = (f_ama, m_ama, t_ama, inv_ama, ppl_ama, g_ama)
            if beta in [2.0, 4.0, 6.0, 8.0]:
                print(f"       [AMA-SP-SV (beta={beta:.1f})] Flipped? {f_ama} (margin: {m_ama:+.2f}) | Invariance: {inv_ama:.2f}% | PPL Drift: {ppl_ama:+.3f}% | Gen: \"{g_ama}\"")

        if best_beta is not None:
            print(f"    [+] SUCCESS: Adaptive Agent found critical threshold at beta* = {best_beta:.1f}!")
            print(f"        Output changed from \"{base_gen_text}\" -> \"{best_res[5]}\"")
            print(f"        Invariance maintained at: {best_res[3]:.2f}% | PPL drift: {best_res[4]:+.3f}%")
        else:
            print(f"    [-] Token not flipped up to beta=10.0.")

    print("\n" + "=" * 90)
    print("[*] Probing and calibration complete.")
    print("=" * 90)

if __name__ == '__main__':
    main()
