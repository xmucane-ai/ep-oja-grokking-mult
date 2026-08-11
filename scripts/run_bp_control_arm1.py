#!/usr/bin/env python3
"""run_bp_control_arm1.py -- SPEC_BP_CONTROL_ARM1_v1.1 (task t_49f44438).

BP reference/falsifier arm on the SAME L=6 genuine-3D substrate as the EP arm.
Same forward pass (forward_init), same initialization (byte-identical via
make_engine from run_basis_swap_v13.py), but AdamW + cross-entropy replaces
the EP contrastive update.

STE CONFOUND DIRECTIONALITY (spec §6.1 A1):
  The straight-through estimator gives BP a FULL-RANK gradient through the hard
  gate, while EP's update is GATED by the same threshold (neurons below
  threshold receive no contrastive update). This asymmetry FAVORS BP — BP gets
  strictly more gradient information per step. Therefore any accuracy match is
  CONSERVATIVE FOR EP: EP matches despite a weaker gradient path. The paper
  must state this directionality explicitly.

LOSS/BATCH (spec §3.2, §6.2 B2):
  BP arm: cross-entropy loss, full-batch (all 2163 train pairs/step).
  EP arm: PC energy (squared-error), mini-batch (128/step).
  These differences are INTENTIONAL design choices (the BP arm is the field's
  yardstick, not a matched-loss control). The comparison table keeps loss and
  batch columns EXPLICIT (§6.2 B2).

P-COMPLIANCE (spec §5):
  This arm is a reference/falsifier, explicitly EXEMPT from P1/P3/P4/P5
  (govern the EP+Oja learning track only). Subject to P6 (genuine 3D: same
  x,y,z substrate, cKDTree connectivity) and P7 (L=6 ≥ 2). P2 satisfied by
  forward-pass divisive normalization. P8 partial (calibrated thresholds FROZEN,
  no adaptive update — BP uses them as fixed nonlinearity).

NOT a revival of the REJECTED c3_dynamic/AdamW line (§7): that was an old engine
(different forward pass, one-hot input, claimed to be local). This is the SAME
L=6 engine as C2 (matched architecture), honestly labeled as BP reference.

NUMERICS (spec §6): N=1536, in_dim=104, P=53, g=2, E_mult. AdamW lr=1e-3
wd=1.0, beta1=0.9, beta2=0.999, grad_clip=1.0, full-batch 3000 steps, 10 seeds.

Usage:
  python -u run_bp_control_arm1.py --phase main --seeds 0-9 --steps 3000
  python -u run_bp_control_arm1.py --phase tcliff_bp --seeds 0,4,9 --steps 3000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

# Import shared data/engine construction from run_basis_swap_v13
from run_basis_swap_v13 import (
    make_mult_data, make_add_data, to_onehot, make_engine,
    P_MOD, K_FREQ, IN_DIM, CHANCE, G_PRIM,
    HIDDEN, SHEET_SIZE, T_INF, L_LAMINAR,
)
from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE

# ── BP-specific constants (spec §3.2) ──
BP_LR = 1e-3
BP_WD = 1.0
BP_BETA1 = 0.9
BP_BETA2 = 0.999
BP_GRAD_CLIP = 1.0
BP_STEPS = 3000
EVAL_EVERY = 100


# ================================================================
# STE (Straight-Through Estimator) phi_norm — differentiable
# ================================================================
def bp_phi_norm(u, thresholds_l, inhib_mask, k_target, beta):
    """Differentiable phi_norm with STE for the hard gate.

    SAME forward pass as AblationCortex.phi_norm (exact):
      s = softplus(beta*(u - theta)) / beta
      gate = (u > theta).float()
      s = s * gate; clamp(0,5)
      local_sum = s @ inhib_mask; x = s / clamp(local_sum/k_target, min=1)

    DIFFERENT backward (STE): the hard gate's gradient is replaced by identity,
    so gradients flow through ALL neurons' softplus activations (full-rank),
    not just above-threshold ones. This is the STE confound (§6.1 A1):
    BP gets full-rank gradient, EP's update is gated → accuracy match is
    conservative for EP.

    The divisive normalization (matmul, division, clamp) is naturally
    differentiable — gradients flow through it without STE.
    """
    # softplus activation (C^2 smooth, differentiable)
    s_raw = F.softplus(beta * (u - thresholds_l.unsqueeze(0))) / beta

    # Hard gate (non-differentiable) — STE: forward exact, backward identity
    gate = (u > thresholds_l.unsqueeze(0)).float()
    s_gated = s_raw * gate  # exact forward
    # STE trick: s_ste has forward=s_gated, backward=d/d(s_raw) as if gate=1
    s_ste = s_gated.detach() + s_raw - s_raw.detach()

    s_ste = s_ste.clamp(0, 5.0)

    # P2: LOCAL divisive normalization (same as engine, differentiable)
    local_sum = s_ste @ inhib_mask
    pool_ratio = local_sum / k_target
    denom = torch.clamp(pool_ratio, min=1.0)
    x_norm = s_ste / denom

    return x_norm


def bp_forward(model, X, W_lin, W_prod, W_ff_list, W_out):
    """Differentiable forward pass matching engine.forward_init exactly (with STE).

    Uses the engine's FIXED structure: conn indices, ff_masks, inhib_masks,
    thresholds, k_target, s_L. Only the weight tensors are trainable autograd
    leaves.

    STE confound: BP gets full-rank gradient through the hard gate.
    """
    # Layer 0: dendritic product encoder (Delta 1, proven EPNet form)
    cv = X[:, model.conn]  # [B, N, k_conn]
    u0 = (cv * W_lin.unsqueeze(0)).sum(dim=2)  # [B, N]
    pv = cv[:, :, model.pi] * cv[:, :, model.pj]  # [B, N, n_pairs]
    u0 = u0 + (pv * W_prod.unsqueeze(0)).sum(dim=2)

    x_list = [bp_phi_norm(u0, model.thresholds[0], model.inhib_masks_raw[0],
                          model.k_target[0], model.beta)]

    # Layers 1..L-1: residual + W_ff, muPC depth scaling
    for l in range(model.L - 1):
        u = x_list[l] + model.s_L * (x_list[l] @ W_ff_list[l])
        x_list.append(bp_phi_norm(u, model.thresholds[l + 1],
                                  model.inhib_masks_raw[l + 1],
                                  model.k_target[l + 1], model.beta))

    # Readout
    yhat = x_list[model.L - 1] @ W_out
    return x_list, yhat


def bp_predict(model, X, W_lin, W_prod, W_ff_list, W_out):
    """Test-time prediction (no_grad, uses exact forward — STE forward is exact)."""
    with torch.no_grad():
        _, yhat = bp_forward(model, X, W_lin, W_prod, W_ff_list, W_out)
        return yhat.argmax(dim=-1)


def bp_evaluate(model, X, Y, W_lin, W_prod, W_ff_list, W_out):
    """Test accuracy."""
    preds = bp_predict(model, X, W_lin, W_prod, W_ff_list, W_out)
    return float((preds == Y).float().mean().item())


# ================================================================
# Per-area gradient diagnostics (the BP analog of ε_l)
# ================================================================
def compute_per_layer_grad_norms(model, X, Yoh, W_lin, W_prod, W_ff_list, W_out):
    """Gradient norms per layer — the BP analog of EP's per-area ε_l (§10 Q4).

    For BP, dh (contrastive signal) is not meaningful (no clamped/free phases).
    Instead we report ||∂L/∂x_l|| per layer as the gradient magnitude that
    reaches each area — the BP credit-assignment analog.
    """
    model_zero_grads = None  # we'll capture intermediates
    x_list, yhat = bp_forward(model, X, W_lin, W_prod, W_ff_list, W_out)
    loss = F.cross_entropy(yhat, Yoh.argmax(dim=-1))

    grad_norms = []
    for l in range(model.L):
        if x_list[l].requires_grad:
            grad = torch.autograd.grad(loss, x_list[l], retain_graph=True,
                                       allow_unused=True)[0]
            if grad is not None:
                B = X.shape[0]
                grad_norms.append(float(grad.norm().item()) / max(B, 1))
            else:
                grad_norms.append(0.0)
        else:
            grad_norms.append(0.0)

    # Also weight gradient norms per matrix
    w_grad_norms = {
        'W_lin': float(W_lin.grad.norm().item()) if W_lin.grad is not None else 0.0,
        'W_prod': float(W_prod.grad.norm().item()) if W_prod.grad is not None else 0.0,
        'W_out': float(W_out.grad.norm().item()) if W_out.grad is not None else 0.0,
    }
    for l, w in enumerate(W_ff_list):
        w_grad_norms[f'W_ff[{l}]'] = float(w.grad.norm().item()) if w.grad is not None else 0.0

    # Firing rates per area (same metric as EP, from forward pass)
    firing_rates = []
    hoyer_vals = []
    for l in range(model.L):
        xa = x_list[l].detach()
        fr = float((xa > 1e-6).float().mean().item())
        firing_rates.append(fr)
        mean_act = xa.abs().mean(dim=0)
        l1 = mean_act.sum() + 1e-12
        l2 = mean_act.norm() + 1e-12
        hoy = (np.sqrt(model.N) - float(l1) / float(l2)) / (np.sqrt(model.N) - 1)
        hoyer_vals.append(max(0.0, min(1.0, hoy)))

    return {
        'grad_norms_per_layer': grad_norms,
        'weight_grad_norms': w_grad_norms,
        'firing_rates': firing_rates,
        'hoyer': hoyer_vals,
        'loss': float(loss.item()),
    }


# ================================================================
# Single-seed BP runner
# ================================================================
def run_bp_seed(seed, task_label, steps, t_inf=T_INF, eval_every=EVAL_EVERY,
                out_dir=None):
    """Run one seed of the BP control arm.

    task_label: 'mult' or 'add'
    t_inf: T parameter (INERT for BP — forward_init doesn't use T; only stored
           for the T-cliff null-result verification per §4.3).
    """
    t0 = time.time()
    col_seed = seed * 100 + 0  # matches proven l6eqcap/C2 seed mapping

    # ── Data (identical to C2/EP arm) ──
    if task_label == 'mult':
        Xtr, Ytr, Xte, Yte = make_mult_data(seed=42, train_fraction=0.80)
    else:
        Xtr, Ytr, Xte, Yte = make_add_data(seed=42, train_fraction=0.80)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh_tr = to_onehot(Ytr, P_MOD)
    n_train = len(Xtr)
    n_test = len(Xte)

    # ── Engine construction (byte-identical to EP arm via make_engine) ──
    # T_inference is set but INERT for BP (forward_init doesn't use T).
    model = make_engine(seed=col_seed, n_layers=L_LAMINAR, stabilization=True)
    # Override T for T-cliff verification (only affects stored self.T, not forward)
    model.T = int(t_inf)
    model.calibrate_thresholds(Xtr[:200])

    # ── Create autograd-trainable weight clones (from engine's init) ──
    # These are byte-identical copies of the EP arm's initial weights.
    W_lin = model.W_lin.clone().detach().requires_grad_(True)
    W_prod = model.W_prod.clone().detach().requires_grad_(True)
    W_ff_list = [w.clone().detach().requires_grad_(True) for w in model.W_ff]
    W_out = model.W_out.clone().detach().requires_grad_(True)

    # ── AdamW optimizer (spec §3.2) ──
    params = [W_lin, W_prod] + W_ff_list + [W_out]
    optimizer = torch.optim.AdamW(params, lr=BP_LR, weight_decay=BP_WD,
                                  betas=(BP_BETA1, BP_BETA2))

    history = []
    grok_step = None
    best_acc = 0.0
    final_diag = None

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        # Full-batch forward pass (spec §3.2: full-batch = grokking convention)
        x_list, yhat = bp_forward(model, Xtr, W_lin, W_prod, W_ff_list, W_out)
        loss = F.cross_entropy(yhat, Ytr)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, BP_GRAD_CLIP)
        optimizer.step()

        # Apply 3D connectivity masks to W_ff (project back to substrate topology)
        with torch.no_grad():
            for l in range(model.L - 1):
                W_ff_list[l].mul_(model.ff_masks[l])

        # ── Evaluation ──
        if step % eval_every == 0 or step == 1:
            test_acc = bp_evaluate(model, Xte, Yte, W_lin, W_prod, W_ff_list, W_out)
            train_acc = bp_evaluate(model, Xtr[:500], Ytr[:500], W_lin, W_prod,
                                    W_ff_list, W_out)
            best_acc = max(best_acc, test_acc)
            if test_acc >= 0.9 and grok_step is None:
                grok_step = step
            history.append({'step': step, 'test_acc': test_acc,
                            'train_acc': train_acc, 'loss': float(loss.item())})

            if step % 500 == 0 or step == eval_every or step == steps:
                elapsed = time.time() - t0
                print(f"  [BP s{seed} T={t_inf}] step {step:5d}: test={test_acc:.3f} "
                      f"train={train_acc:.3f} best={best_acc:.3f} loss={loss:.4f} "
                      f"[{elapsed:.0f}s]", flush=True)

    # ── Final per-area diagnostics ──
    with torch.no_grad():
        pass  # need grad for diagnostics
    optimizer.zero_grad()
    final_diag = compute_per_layer_grad_norms(
        model, Xte[:256], to_onehot(Yte[:256], P_MOD),
        W_lin, W_prod, W_ff_list, W_out)
    print(f"  [BP s{seed} T={t_inf}] FINAL DIAG:")
    print(f"    grad_norms/layer: {[f'{g:.4f}' for g in final_diag['grad_norms_per_layer']]}")
    print(f"    firing_rates:     {[f'{f:.3f}' for f in final_diag['firing_rates']]}")
    print(f"    hoyer:            {[f'{h:.2f}' for h in final_diag['hoyer']]}")
    print(f"    loss:             {final_diag['loss']:.4f}")

    # ── Headline metrics (same schema as EP arm) ──
    W = 5
    test_accs = [h['test_acc'] for h in history]
    window_avg = float(np.mean(test_accs[-W:])) if len(test_accs) >= W else \
        float(np.mean(test_accs)) if test_accs else 0.0
    final_acc = test_accs[-1] if test_accs else 0.0
    dt = time.time() - t0

    result = {
        'arm': 'BP',
        'optimizer': 'AdamW',
        'loss_fn': 'cross_entropy',
        'lr': BP_LR, 'weight_decay': BP_WD,
        'batch_size': 'full-batch',
        'ste': True,  # STE confound documented (§6.1 A1)
        'seed': seed, 'task': task_label,
        'L': L_LAMINAR, 'N': HIDDEN, 'sheet_size': SHEET_SIZE,
        't_inf': t_inf,  # INERT for BP (no relaxation in forward)
        'effective_T': t_inf,
        'steps': steps, 'n_train': n_train, 'n_test': n_test,
        'final_test_acc': final_acc,
        'best_test_acc': best_acc,
        'window_avg_acc': window_avg,
        'grok_step': grok_step,
        'chance': CHANCE,
        'time': dt,
        'history': history,
        'final_diag': final_diag,
    }
    return result


# ================================================================
# Main
# ================================================================
def parse_seeds(seed_str):
    seeds = []
    for part in seed_str.replace(',', ' ').split():
        if '-' in part:
            lo, hi = part.split('-')
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    ap = argparse.ArgumentParser(
        description='SPEC_BP_CONTROL_ARM1_v1.1 — BP reference/falsifier control arm')
    ap.add_argument('--phase', default='main',
                    choices=['main', 'tcliff_bp', 'ep_mult_tcliff_banking', 'all'],
                    help='main: 10-seed BP vs EP comparison; '
                         'tcliff_bp: BP T-cliff null (3 seeds × 3 T); '
                         'ep_mult_tcliff_banking: EP mult T=1/T=5 (§8.1 step 7b)')
    ap.add_argument('--task', default='mult', choices=['mult', 'add'],
                    help='MULT (E_mult) or ADD (E_add) task')
    ap.add_argument('--seeds', type=str, default='0-9')
    ap.add_argument('--steps', type=int, default=BP_STEPS)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    # Output paths
    out_dir = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs'))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SPEC_BP_CONTROL_ARM1_v1.1 — BP Reference/Falsifier Control Arm")
    print(f"{'='*70}")
    print(f"  Role:       LABELED reference/falsifier (NOT learning track)")
    print(f"  Exempt:     P1/P3/P4/P5 (govern EP learning track)")
    print(f"  Subject:    P6 (genuine 3D), P7 (L={L_LAMINAR} >= 2)")
    print(f"  STE:        full-rank gradient (§6.1 A1 — favors BP, conservative for EP)")
    print(f"  Loss/batch: CE / full-batch (vs EP: energy / 128 mini-batch)")
    print(f"  Optimizer:  AdamW lr={BP_LR} wd={BP_WD} clip={BP_GRAD_CLIP}")
    print(f"  Task:       {args.task.upper()} (E_{'mult' if args.task == 'mult' else 'add'})")
    print(f"  L={L_LAMINAR}, N={HIDDEN}, sheet={SHEET_SIZE}, T_inf={T_INF}")
    print(f"  Steps:      {args.steps}, Seeds: {seeds}")
    print(f"  Chance:     {CHANCE:.4f}")
    print(f"  Device:     {DEVICE}")
    print(f"{'='*70}\n", flush=True)

    if args.phase in ('main', 'all'):
        # ── Main: 10-seed BP reference arm ──
        print(f"\n{'='*70}")
        print(f"  PHASE: MAIN — 10-seed BP reference arm ({args.task.upper()})")
        print(f"{'='*70}\n")

        bp_results = []
        for seed in seeds:
            print(f"--- BP seed {seed} ({args.task}) ---")
            r = run_bp_seed(seed, args.task, args.steps, t_inf=T_INF, out_dir=out_dir)
            verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                       "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
            print(f"  => BP s{seed}: WINDOW={r['window_avg_acc']:.4f} "
                  f"FINAL={r['final_test_acc']:.4f} BEST={r['best_test_acc']:.4f} "
                  f"| {verdict} | {r['time']:.0f}s\n", flush=True)
            bp_results.append(r)

        # Summary
        windows = [r['window_avg_acc'] for r in bp_results]
        finals = [r['final_test_acc'] for r in bp_results]
        bests = [r['best_test_acc'] for r in bp_results]
        groks = [r['grok_step'] for r in bp_results if r['grok_step'] is not None]
        n_win = sum(1 for w in windows if w >= 0.90)

        config = {
            'spec': 'SPEC_BP_CONTROL_ARM1_v1.1',
            'task': args.task,
            'engine': 'AblationCortexOpt (ablation_cortex_v14_1.py)',
            'forward_pass': 'forward_init only (no EP relaxation)',
            'P_compliance': 'Exempt P1/P3/P4/P5; Subject P6/P7; P2 via divnorm; P8 partial (frozen thresholds)',
            'ste_confound': 'STE gives BP full-rank gradient (§6.1 A1) — favors BP, conservative for EP',
            'loss_fn': 'cross_entropy (vs EP: PC energy squared-error)',
            'batch_protocol': 'full-batch 2163 (vs EP: mini-batch 128)',
            'optimizer': 'AdamW',
            'lr': BP_LR, 'weight_decay': BP_WD,
            'grad_clip': BP_GRAD_CLIP,
            'L': L_LAMINAR, 'N': HIDDEN, 'sheet_size': SHEET_SIZE,
            'P': P_MOD, 'g': G_PRIM, 'k_freq': K_FREQ, 'in_dim': IN_DIM,
            'steps': args.steps, 't_inf': T_INF,
            'chance': CHANCE,
        }

        summary = {
            'n_seeds': len(bp_results),
            'window_avg_mean': float(np.mean(windows)),
            'window_avg_std': float(np.std(windows)),
            'final_mean': float(np.mean(finals)),
            'final_std': float(np.std(finals)),
            'best_mean': float(np.mean(bests)),
            'grok_rate_window': f"{n_win}/{len(bp_results)}",
            'mean_grok_step': float(np.mean(groks)) if groks else None,
            'chance': CHANCE,
        }

        # Per-area diagnostic averages
        if bp_results and bp_results[0].get('final_diag'):
            diag_keys = ['grad_norms_per_layer', 'firing_rates', 'hoyer']
            diag_avg = {}
            for key in diag_keys:
                vals = [r['final_diag'][key] for r in bp_results
                        if r.get('final_diag', {}).get(key)]
                if vals:
                    diag_avg[key] = [float(np.mean([v[i] for v in vals if i < len(v)]))
                                     for i in range(max(len(v) for v in vals))]
            summary['per_area_diagnostics_avg'] = diag_avg

        output_data = {'config': config, 'results': bp_results, 'summary': summary}

        main_name = args.output or os.path.join(
            out_dir, f'bp_control_arm1_{args.task}_main_L{L_LAMINAR}_N{HIDDEN}_T{T_INF}.json')
        with open(main_name, 'w') as f:
            json.dump(output_data, f, indent=1, default=str)

        print(f"\n{'='*70}")
        print(f"  SUMMARY: BP {args.task.upper()} (L={L_LAMINAR}, {len(bp_results)} seeds)")
        print(f"{'='*70}")
        print(f"  *** WINDOW-AVG (W=5) — HEADLINE ***")
        print(f"  Window-avg: mean={summary['window_avg_mean']:.3f} "
              f"±{summary['window_avg_std']:.3f}")
        print(f"  Grok (window>=0.90): {summary['grok_rate_window']}")
        print(f"  Final acc:  mean={summary['final_mean']:.3f} "
              f"±{summary['final_std']:.3f}")
        if groks:
            print(f"  Mean grok_step: {summary['mean_grok_step']:.0f}")
        print(f"  Per-seed window: {[f'{a:.3f}' for a in windows]}")
        print(f"  Per-seed final:  {[f'{a:.3f}' for a in finals]}")
        if 'per_area_diagnostics_avg' in summary:
            d = summary['per_area_diagnostics_avg']
            if 'grad_norms_per_layer' in d:
                print(f"  Grad norms/layer: {[f'{g:.4f}' for g in d['grad_norms_per_layer']]}")
            if 'firing_rates' in d:
                print(f"  Firing rates:     {[f'{f:.3f}' for f in d['firing_rates']]}")
        print(f"  Chance: {CHANCE:.4f}")
        print(f"  Results: {main_name}")

    if args.phase in ('tcliff_bp', 'all'):
        # ── T-cliff sub-experiment (a): BP null result, 3 seeds × 3 T ──
        tcliff_seeds = parse_seeds(args.seeds) if args.phase == 'tcliff_bp' else [0, 4, 9]
        # Use representative seeds if not explicitly specified
        if args.phase == 'all':
            tcliff_seeds = [0, 4, 9]
        t_values = [1, 5, 10]

        print(f"\n{'='*70}")
        print(f"  PHASE: T-CLIFF BP (null result) — {len(tcliff_seeds)} seeds × {len(t_values)} T")
        print(f"  Expected: NO T-dependence (BP forward_init doesn't use T)")
        print(f"{'='*70}\n")

        tcliff_results = {}
        for t_val in t_values:
            tcliff_results[t_val] = []
            for seed in tcliff_seeds:
                print(f"--- BP T-cliff T={t_val} seed {seed} ({args.task}) ---")
                r = run_bp_seed(seed, args.task, args.steps, t_inf=t_val, out_dir=out_dir)
                tcliff_results[t_val].append(r)
                print(f"  => BP T={t_val} s{seed}: WINDOW={r['window_avg_acc']:.4f} "
                      f"FINAL={r['final_test_acc']:.4f}\n", flush=True)

        tcliff_summary = {}
        for t_val in t_values:
            windows = [r['window_avg_acc'] for r in tcliff_results[t_val]]
            tcliff_summary[t_val] = {
                'window_avg_mean': float(np.mean(windows)),
                'window_avg_std': float(np.std(windows)),
                'per_seed': windows,
            }

        tcliff_name = os.path.join(
            out_dir, f'bp_control_arm1_{args.task}_tcliff_L{L_LAMINAR}_N{HIDDEN}.json')
        with open(tcliff_name, 'w') as f:
            json.dump({'config': config if args.phase == 'all' else {
                          'spec': 'SPEC_BP_CONTROL_ARM1_v1.1',
                          'phase': 'tcliff_bp',
                          'task': args.task,
                          'ste': True,
                       },
                       'results': tcliff_results,
                       'summary': tcliff_summary,
                       'note': 'T-cliff null result: BP forward_init does not use T. '
                               'Differences are RNG stochasticity only.'},
                      f, indent=1, default=str)

        print(f"\n  T-CLIFF BP SUMMARY:")
        for t_val in t_values:
            s = tcliff_summary[t_val]
            print(f"    T={t_val:2d}: window_mean={s['window_avg_mean']:.3f} "
                  f"±{s['window_avg_std']:.3f} per_seed={[f'{x:.3f}' for x in s['per_seed']]}")
        print(f"  Results: {tcliff_name}")

    if args.phase in ('ep_mult_tcliff_banking', 'all'):
        # ── EP mult T-cliff banking (§8.1 step 7b) ──
        # C2 config: gamma_W=0.5, gamma_alpha=0.25, T_decay=1500, 3000 steps
        # T=10 already banked; this banks T=1 and T=5
        bank_seeds = [0, 4, 9] if args.phase == 'all' else parse_seeds(args.seeds)
        bank_t_values = [1, 5]  # T=10 already banked as C2

        print(f"\n{'='*70}")
        print(f"  PHASE: EP MULT T-CLIFF BANKING (§8.1 step 7b)")
        print(f"  {len(bank_seeds)} seeds × T={bank_t_values}")
        print(f"  C2 config: gamma_W=0.5, gamma_alpha=0.25, T_decay=1500")
        print(f"  Until banked, paper claim scoped to ADD-task provenance (C7)")
        print(f"{'='*70}\n")

        # Need to set T_INF globally for run_single_seed to pick up
        import run_basis_swap_v13 as rbs
        bank_results = {}
        for t_val in bank_t_values:
            rbs.T_INF = t_val  # override global for this T
            bank_results[t_val] = []
            for seed in bank_seeds:
                print(f"--- EP MULT T={t_val} seed {seed} (BANKING) ---")
                r = rbs.run_single_seed(
                    seed, 'mult-stab', args.steps,
                    eval_every=EVAL_EVERY, gate_every=500,
                    train_fraction=0.80, stabilization=True)
                bank_results[t_val].append(r)
                print(f"  => EP MULT T={t_val} s{seed}: WINDOW={r['window_avg_acc']:.4f} "
                      f"FINAL={r['final_test_acc']:.4f} BEST={r['best_test_acc']:.4f}\n",
                      flush=True)
        rbs.T_INF = T_INF  # restore

        bank_summary = {}
        for t_val in bank_t_values:
            windows = [r['window_avg_acc'] for r in bank_results[t_val]]
            bank_summary[t_val] = {
                'window_avg_mean': float(np.mean(windows)),
                'window_avg_std': float(np.std(windows)),
                'per_seed': windows,
            }

        bank_name = os.path.join(
            out_dir, f'ep_mult_tcliff_banking_L{L_LAMINAR}_N{HIDDEN}.json')
        with open(bank_name, 'w') as f:
            json.dump({'config': {
                          'spec': 'SPEC_BP_CONTROL_ARM1_v1.1 §8.1 step 7b',
                          'phase': 'ep_mult_tcliff_banking',
                          'task': 'mult',
                          'engine': 'AblationCortexOpt (EP contrastive)',
                          'gamma_W': 0.5, 'gamma_alpha': 0.25, 'T_decay': 1500,
                          'note': 'T=10 already banked as C2. This banks T=1 and T=5.',
                       },
                       'results': bank_results,
                       'summary': bank_summary},
                      f, indent=1, default=str)

        print(f"\n  EP MULT T-CLIFF BANKING SUMMARY:")
        for t_val in bank_t_values:
            s = bank_summary[t_val]
            print(f"    T={t_val}: window_mean={s['window_avg_mean']:.3f} "
                  f"±{s['window_avg_std']:.3f} per_seed={[f'{x:.3f}' for x in s['per_seed']]}")
        print(f"  Results: {bank_name}")


if __name__ == '__main__':
    main()
