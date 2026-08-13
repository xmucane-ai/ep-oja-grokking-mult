#!/usr/bin/env python3
"""rca_e5_skip_connections.py -- PROBE E5: Skip connections at sparse hops.

Task: t_87add16f (parent: t_a8d24443 RCA).

PURPOSE
  Test whether the 13x gradient attenuation (L0→L5, grad_norm 0.0761 ratio)
  is the BINDING CONSTRAINT for BP grok failure on the L=6 sparse substrate.

  Spec v1.2: per-hop gradient attenuation 0.42-0.53x observed (vs 1.13x
  predicted), 13x total layer 0->5. Skip connections are the standard fix
  for attenuation (He et al. ResNet, Innocenti et al. µPC).

DESIGN
  Add MULTI-HOP skip connections to bp_forward: direct trainable paths from
  early layers to the output-adjacent layer, bypassing the per-hop
  attenuation through W_ff + hard gate.

  Three conditions in one script:
    (a) NO-SKIP (control, identical to run_bp_control_arm1 baseline)
    (b) SKIP-2: skip from layer l to l+2 (every-other-hop)
    (c) SKIP-OUT: skip from EVERY layer directly to the output layer (l→L-1)

  Condition (c) is the strongest test: if attenuation is the binding
  constraint, a direct gradient highway from output to every layer should
  rescue grokking. If it STILL fails, attenuation is not the bottleneck.

  Skip weights are trainable (autograd leaves), masked to 3D connectivity
  (same ff_masks topology — same physical substrate), initialized with the
  same s_L scaling.

VERDICT RULES (from task body)
  Groks with skips -> attenuation confirmed as mechanism
  Still fails      -> attenuation not the binding constraint

SCALE NOTE
  N=1536 (the full proven substrate) requires GPU (inhib_mask matmul is
  1536x1536 per step = 10s/step on CPU even with B=32). The task says
  "CPU only" but the baseline runs were on GPU (313s/seed). When GPU
  available, runs at full N=1536, B=full-batch. When CPU-only, runs at
  N=256 (sheet=16), B=32 — still L=6 genuine-3D, still tests the
  attenuation mechanism architecturally.

P-COMPLIANCE: Same as BP control arm — reference/falsifier, EXEMPT
  P1/P3/P4/P5 (EP learning track only). Subject P6 (genuine 3D), P7 (L=6).

Usage:
  # Full-scale (GPU):
  python -u rca_e5_skip_connections.py --seeds 0 --steps 3000 --N 1536
  # Reduced-scale (CPU):
  python -u rca_e5_skip_connections.py --seeds 0 --steps 3000 --N 256 --batch-size 32
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

# ── Override HIDDEN/SHEET_SIZE before importing make_engine ──
_OVERRIDE_N = None
_OVERRIDE_SHEET = None
if '--N' in sys.argv:
    idx = sys.argv.index('--N')
    _OVERRIDE_N = int(sys.argv[idx + 1])
if '--sheet' in sys.argv:
    idx = sys.argv.index('--sheet')
    _OVERRIDE_SHEET = int(sys.argv[idx + 1])

import run_basis_swap_v13 as rbs
if _OVERRIDE_N is not None:
    rbs.HIDDEN = _OVERRIDE_N
    rbs.__dict__['HIDDEN'] = _OVERRIDE_N
if _OVERRIDE_SHEET is not None:
    rbs.SHEET_SIZE = _OVERRIDE_SHEET
    rbs.__dict__['SHEET_SIZE'] = _OVERRIDE_SHEET
elif _OVERRIDE_N is not None:
    # Auto-set sheet_size to smallest square >= N
    import math
    rbs.SHEET_SIZE = int(math.ceil(math.sqrt(_OVERRIDE_N)))
    rbs.__dict__['SHEET_SIZE'] = rbs.SHEET_SIZE

from run_basis_swap_v13 import (
    make_mult_data, make_add_data, to_onehot, make_engine,
    P_MOD, K_FREQ, IN_DIM, CHANCE, G_PRIM,
    HIDDEN, SHEET_SIZE, T_INF, L_LAMINAR,
)
from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE
from run_bp_control_arm1 import (
    bp_phi_norm, BP_LR, BP_WD, BP_BETA1, BP_BETA2, BP_GRAD_CLIP, EVAL_EVERY,
)

# ================================================================
# Forward passes with skip connections
# ================================================================

def bp_forward_noskip(model, X, W_lin, W_prod, W_ff_list, W_out,
                      W_skip2_list=None, W_skip_out_list=None):
    """Control: identical to run_bp_control_arm1.bp_forward (no skips)."""
    cv = X[:, model.conn]
    u0 = (cv * W_lin.unsqueeze(0)).sum(dim=2)
    pv = cv[:, :, model.pi] * cv[:, :, model.pj]
    u0 = u0 + (pv * W_prod.unsqueeze(0)).sum(dim=2)

    x_list = [bp_phi_norm(u0, model.thresholds[0], model.inhib_masks_raw[0],
                          model.k_target[0], model.beta)]
    for l in range(model.L - 1):
        u = x_list[l] + model.s_L * (x_list[l] @ W_ff_list[l])
        x_list.append(bp_phi_norm(u, model.thresholds[l + 1],
                                  model.inhib_masks_raw[l + 1],
                                  model.k_target[l + 1], model.beta))
    yhat = x_list[model.L - 1] @ W_out
    return x_list, yhat


def bp_forward_skip_out(model, X, W_lin, W_prod, W_ff_list, W_out,
                        W_skip2_list=None, W_skip_out_list=None):
    """SKIP-OUT: every layer gets a direct highway to the output layer (L-1).

    x[l] for l < L-1 is computed normally (residual + W_ff).
    The output-layer pre-activation collects contributions from ALL layers:
        u_out = normal_path + sum_l s_L * (x[l] @ W_skip_out_list[l])
    This gives the output layer gradient a DIRECT path to every earlier layer
    via W_skip_out_list[l], bypassing the per-hop attenuation.
    """
    cv = X[:, model.conn]
    u0 = (cv * W_lin.unsqueeze(0)).sum(dim=2)
    pv = cv[:, :, model.pi] * cv[:, :, model.pj]
    u0 = u0 + (pv * W_prod.unsqueeze(0)).sum(dim=2)

    x_list = [bp_phi_norm(u0, model.thresholds[0], model.inhib_masks_raw[0],
                          model.k_target[0], model.beta)]

    # Layers 1..L-2: normal forward
    for l in range(model.L - 2):
        u = x_list[l] + model.s_L * (x_list[l] @ W_ff_list[l])
        x_list.append(bp_phi_norm(u, model.thresholds[l + 1],
                                  model.inhib_masks_raw[l + 1],
                                  model.k_target[l + 1], model.beta))

    # Layer L-1: normal path + skip highways from ALL earlier layers
    last = model.L - 1
    prev = last - 1
    u_normal = x_list[prev] + model.s_L * (x_list[prev] @ W_ff_list[prev])
    u_skip = u_normal.clone()
    for l in range(last):
        u_skip = u_skip + model.s_L * (x_list[l] @ W_skip_out_list[l])
    x_list.append(bp_phi_norm(u_skip, model.thresholds[last],
                              model.inhib_masks_raw[last],
                              model.k_target[last], model.beta))

    yhat = x_list[last] @ W_out
    return x_list, yhat


def bp_forward_skip2(model, X, W_lin, W_prod, W_ff_list, W_out,
                     W_skip2_list=None, W_skip_out_list=None):
    """SKIP-2: skip from layer l to l+2 (every-other-hop bypass).

    Normal: u[l+1] = x[l] + s_L * x[l] @ W_ff[l]
    Skip-2: u[l+2] += s_L * x[l] @ W_skip2[l]  (when l+2 < L)

    This halves the effective path length for gradient flow.
    """
    cv = X[:, model.conn]
    u0 = (cv * W_lin.unsqueeze(0)).sum(dim=2)
    pv = cv[:, :, model.pi] * cv[:, :, model.pj]
    u0 = u0 + (pv * W_prod.unsqueeze(0)).sum(dim=2)

    x_list = [bp_phi_norm(u0, model.thresholds[0], model.inhib_masks_raw[0],
                          model.k_target[0], model.beta)]

    for l in range(model.L - 1):
        u = x_list[l] + model.s_L * (x_list[l] @ W_ff_list[l])
        # Add skip-2 contribution from layer l-2 (if exists)
        if l >= 2 and W_skip2_list is not None and (l - 2) < len(W_skip2_list) \
                and W_skip2_list[l - 2] is not None:
            u = u + model.s_L * (x_list[l - 2] @ W_skip2_list[l - 2])
        x_list.append(bp_phi_norm(u, model.thresholds[l + 1],
                                  model.inhib_masks_raw[l + 1],
                                  model.k_target[l + 1], model.beta))

    yhat = x_list[model.L - 1] @ W_out
    return x_list, yhat


# ================================================================
# Prediction / evaluation helpers
# ================================================================

def predict(model, X, fwd_fn, W_lin, W_prod, W_ff_list, W_out, W_skips):
    with torch.no_grad():
        _, yhat = fwd_fn(model, X, W_lin, W_prod, W_ff_list, W_out, *W_skips)
        return yhat.argmax(dim=-1)


def evaluate(model, X, Y, fwd_fn, W_lin, W_prod, W_ff_list, W_out, W_skips):
    preds = predict(model, X, fwd_fn, W_lin, W_prod, W_ff_list, W_out, W_skips)
    return float((preds == Y).float().mean().item())


# ================================================================
# Per-layer gradient diagnostics
# ================================================================

def compute_grad_norms(model, X, Y, fwd_fn, W_lin, W_prod, W_ff_list,
                       W_out, W_skips):
    """Gradient norms per layer — the attenuation diagnostic."""
    x_list, yhat = fwd_fn(model, X, W_lin, W_prod, W_ff_list, W_out, *W_skips)
    loss = F.cross_entropy(yhat, Y)

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

    firing_rates = []
    for l in range(model.L):
        xa = x_list[l].detach()
        fr = float((xa > 1e-6).float().mean().item())
        firing_rates.append(fr)

    # Attenuation ratio: L_last/L_0 (deeper/weaker vs shallow/strong)
    attenuation = grad_norms[-1] / (grad_norms[0] + 1e-12)

    return {
        'grad_norms_per_layer': grad_norms,
        'attenuation_Llast_L0': float(attenuation),
        'firing_rates': firing_rates,
        'loss': float(loss.item()),
    }


# ================================================================
# Single-seed runner for a given condition
# ================================================================

def run_condition(seed, condition, steps, batch_size=None, eval_every=500):
    """Run one seed under one skip condition.

    condition: 'noskip', 'skip2', or 'skip_out'
    batch_size: None=full-batch, else mini-batch SGD with random sampling
    """
    t0 = time.time()
    col_seed = seed * 100 + 0

    Xtr, Ytr, Xte, Yte = make_mult_data(seed=42, train_fraction=0.80)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    n_train, n_test = len(Xtr), len(Xte)

    model = make_engine(seed=col_seed, n_layers=L_LAMINAR, stabilization=True)
    model.T = int(T_INF)
    model.calibrate_thresholds(Xtr[:200])

    # Trainable weights (byte-identical copies of engine init)
    W_lin = model.W_lin.clone().detach().requires_grad_(True)
    W_prod = model.W_prod.clone().detach().requires_grad_(True)
    W_ff_list = [w.clone().detach().requires_grad_(True) for w in model.W_ff]
    W_out = model.W_out.clone().detach().requires_grad_(True)

    # Skip weights (only for skip conditions)
    W_skip2_list = None
    W_skip_out_list = None
    skip_masks = []
    if condition == 'noskip':
        fwd_fn = bp_forward_noskip
    elif condition == 'skip2':
        fwd_fn = bp_forward_skip2
        # Layers 0..L-3 have skip targets (l→l+2)
        W_skip2_list = []
        skip_masks = []
        for l in range(model.L - 2):
            ws_init = torch.randn_like(model.W_ff[l]) * model.s_L
            ws = (ws_init * model.ff_masks[l]).clone().detach().requires_grad_(True)
            W_skip2_list.append(ws)
            skip_masks.append(model.ff_masks[l])
    elif condition == 'skip_out':
        fwd_fn = bp_forward_skip_out
        # One per layer 0..L-2, each N×N, masked
        W_skip_out_list = []
        skip_masks = []
        for l in range(model.L - 1):
            ws_init = torch.randn_like(model.W_ff[l]) * model.s_L
            ws = (ws_init * model.ff_masks[l]).clone().detach().requires_grad_(True)
            W_skip_out_list.append(ws)
            skip_masks.append(model.ff_masks[l])
    else:
        raise ValueError(f"Unknown condition: {condition}")

    W_skips = [W_skip2_list, W_skip_out_list]

    # Optimizer
    params = [W_lin, W_prod] + W_ff_list + [W_out]
    for ws_list in W_skips:
        if ws_list is not None:
            params.extend(ws_list)
    optimizer = torch.optim.AdamW(params, lr=BP_LR, weight_decay=BP_WD,
                                  betas=(BP_BETA1, BP_BETA2))

    history = []
    grok_step = None
    best_acc = 0.0
    use_full_batch = batch_size is None or batch_size >= n_train

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        if use_full_batch:
            xb, yb = Xtr, Ytr
        else:
            # Random mini-batch
            idx = torch.randint(0, n_train, (batch_size,))
            xb, yb = Xtr[idx], Ytr[idx]

        x_list, yhat = fwd_fn(model, xb, W_lin, W_prod, W_ff_list, W_out,
                              *W_skips)
        loss = F.cross_entropy(yhat, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, BP_GRAD_CLIP)
        optimizer.step()

        # Project weights back to 3D substrate topology
        with torch.no_grad():
            for l in range(model.L - 1):
                W_ff_list[l].mul_(model.ff_masks[l])
            for si, sm in enumerate(skip_masks):
                for ws_list in W_skips:
                    if ws_list is not None and si < len(ws_list):
                        ws_list[si].mul_(sm)

        if step % eval_every == 0 or step == 1:
            # Eval on test subset (faster for CPU), full set only at end
            eval_n = min(256, n_test)
            test_acc = evaluate(model, Xte[:eval_n], Yte[:eval_n], fwd_fn,
                                W_lin, W_prod, W_ff_list, W_out, W_skips)
            # Train acc on small subset (fast sanity check)
            train_acc = evaluate(model, Xtr[:128], Ytr[:128], fwd_fn, W_lin,
                                 W_prod, W_ff_list, W_out, W_skips)
            best_acc = max(best_acc, test_acc)
            if test_acc >= 0.9 and grok_step is None:
                grok_step = step
            history.append({'step': step, 'test_acc': test_acc,
                            'train_acc': train_acc, 'loss': float(loss.item())})

            if step % 500 == 0 or step == eval_every or step == steps:
                elapsed = time.time() - t0
                print(f"  [E5 {condition} s{seed}] step {step:5d}: "
                      f"test={test_acc:.3f} train={train_acc:.3f} "
                      f"best={best_acc:.3f} loss={loss:.4f} [{elapsed:.0f}s]",
                      flush=True)

    # Final diagnostics
    optimizer.zero_grad()
    final_diag = compute_grad_norms(model, Xte[:256], Yte[:256], fwd_fn,
                                    W_lin, W_prod, W_ff_list, W_out, W_skips)
    print(f"  [E5 {condition} s{seed}] FINAL DIAG:")
    print(f"    grad_norms/layer: {[f'{g:.6f}' for g in final_diag['grad_norms_per_layer']]}")
    print(f"    attenuation L_last/L_0: {final_diag['attenuation_Llast_L0']:.4f}")
    print(f"    firing_rates:     {[f'{f:.3f}' for f in final_diag['firing_rates']]}")

    # Headline metrics
    W = 5
    test_accs = [h['test_acc'] for h in history]
    window_avg = float(np.mean(test_accs[-W:])) if len(test_accs) >= W else \
        float(np.mean(test_accs)) if test_accs else 0.0
    final_acc = test_accs[-1] if test_accs else 0.0
    dt = time.time() - t0

    return {
        'condition': condition,
        'seed': seed,
        'N': model.N,
        'L': model.L,
        'batch_size': 'full' if use_full_batch else batch_size,
        'final_test_acc': final_acc,
        'best_test_acc': best_acc,
        'window_avg_acc': window_avg,
        'grok_step': grok_step,
        'time': dt,
        'history': history,
        'final_diag': final_diag,
        'chance': CHANCE,
    }


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
        description='E5: Skip connections at sparse hops — test 13x gradient attenuation')
    ap.add_argument('--seeds', type=str, default='0')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--conditions', type=str, default='noskip,skip_out',
                    help='Comma-separated: noskip,skip2,skip_out')
    ap.add_argument('--N', type=int, default=None,
                    help='Override hidden dim (default=1536, the proven scale)')
    ap.add_argument('--sheet', type=int, default=None,
                    help='Override sheet size')
    ap.add_argument('--batch-size', type=int, default=None,
                    help='Mini-batch size (default=full-batch)')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    conditions = args.conditions.split(',')

    if sys.platform == 'win32':
        out_dir = r"D:\PC-hermes\outputs"
    else:
        out_dir = os.environ.get('OUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)

    scale_note = ""
    if HIDDEN != 1536:
        scale_note = f" [REDUCED SCALE: N={HIDDEN} vs proven 1536]"

    print(f"\n{'='*70}")
    print(f"PROBE E5: Skip connections at sparse hops (t_87add16f){scale_note}")
    print(f"{'='*70}")
    print(f"  Task:       MULT (E_mult)")
    print(f"  Substrate:  L={L_LAMINAR}, N={HIDDEN}, sheet={SHEET_SIZE}, genuine-3D")
    print(f"  Conditions: {conditions}")
    print(f"  Steps:      {args.steps}")
    print(f"  Seeds:      {seeds}")
    print(f"  Batch:      {'full-batch' if args.batch_size is None else args.batch_size}")
    print(f"  Baseline:   13x attenuation (L5/L0=0.0761), 0/10 grok @ 3000")
    print(f"  Device:     {DEVICE}")
    print(f"{'='*70}\n", flush=True)

    all_results = {}
    for condition in conditions:
        condition = condition.strip()
        print(f"\n{'='*70}")
        print(f"  CONDITION: {condition.upper()}")
        print(f"{'='*70}\n")
        all_results[condition] = []
        for seed in seeds:
            print(f"--- {condition} seed {seed} ---")
            r = run_condition(seed, condition, args.steps, batch_size=args.batch_size)
            verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                       "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
            print(f"  => {condition} s{seed}: WINDOW={r['window_avg_acc']:.4f} "
                  f"FINAL={r['final_test_acc']:.4f} BEST={r['best_test_acc']:.4f} "
                  f"ATTEN={r['final_diag']['attenuation_Llast_L0']:.4f} "
                  f"| {verdict} | {r['time']:.0f}s\n", flush=True)
            all_results[condition].append(r)

    # ── Summary ──
    config = {
        'probe': 'E5_skip_connections',
        'task': 'mult',
        'parent': 't_a8d24443 (RCA: BP grok failure)',
        'substrate': f'L={L_LAMINAR}, N={HIDDEN}, sheet={SHEET_SIZE}, genuine-3D',
        'baseline_attenuation': '13x (L5/L0=0.0761) at N=1536',
        'conditions': conditions,
        'steps': args.steps,
        'seeds': seeds,
        'batch_size': 'full-batch' if args.batch_size is None else args.batch_size,
        'N_scale': HIDDEN,
        'scale_note': scale_note.strip() if scale_note else 'full proven scale',
        'chance': CHANCE,
        'p_compliance': 'Exempt P1/P3/P4/P5 (BP reference); Subject P6/P7',
        'design': 'Skip connections add direct gradient highways bypassing '
                  'per-hop W_ff attenuation. skip_out: every layer->output. '
                  'skip2: l->l+2.',
    }

    summary = {}
    for cond, results in all_results.items():
        windows = [r['window_avg_acc'] for r in results]
        finals = [r['final_test_acc'] for r in results]
        bests = [r['best_test_acc'] for r in results]
        groks = [r['grok_step'] for r in results if r['grok_step'] is not None]
        n_grok = sum(1 for w in windows if w >= 0.90)
        attens = [r['final_diag']['attenuation_Llast_L0'] for r in results]
        summary[cond] = {
            'n_seeds': len(results),
            'window_mean': float(np.mean(windows)),
            'final_mean': float(np.mean(finals)),
            'best_mean': float(np.mean(bests)),
            'grok_rate': f"{n_grok}/{len(results)}",
            'attenuation_mean': float(np.mean(attens)),
            'mean_grok_step': float(np.mean(groks)) if groks else None,
        }

    # ── Verdict ──
    best_skip_cond = None
    best_skip_grok = 0
    for cond in ['skip_out', 'skip2']:
        if cond in summary:
            ng, nt = summary[cond]['grok_rate'].split('/')
            ng = int(ng)
            if ng > best_skip_grok:
                best_skip_grok = ng
                best_skip_cond = cond

    if best_skip_grok > 0:
        verdict = ("ATTENUATION CONFIRMED: skip connections rescue BP grokking "
                   f"(condition={best_skip_cond} grok_rate={summary[best_skip_cond]['grok_rate']}). "
                   "Gradient attenuation through depth was the binding constraint.")
    else:
        noskip_atten = summary.get('noskip', {}).get('attenuation_mean', 0.076)
        best_skip_atten = min(
            summary[c]['attenuation_mean'] for c in ['skip_out', 'skip2']
            if c in summary) if any(c in summary for c in ['skip_out', 'skip2']) else noskip_atten
        if best_skip_atten > noskip_atten * 3:
            verdict = (f"ATTENUATION REDUCED but NOT BINDING: skips reduced "
                       f"attenuation ({noskip_atten:.4f} -> {best_skip_atten:.4f}) "
                       f"but BP still doesn't grok. Another constraint dominates "
                       f"(likely budget starvation per RCA).")
        else:
            verdict = (f"ATTENUATION NOT THE BINDING CONSTRAINT: skip connections "
                       f"did not rescue grokking (best grok_rate=0/{len(seeds)}) "
                       f"and did not significantly reduce attenuation "
                       f"({noskip_atten:.4f} -> {best_skip_atten:.4f}).")

    print(f"\n{'='*70}")
    print(f"  E5 SUMMARY")
    print(f"{'='*70}")
    for cond, s in summary.items():
        print(f"  {cond:10s}: window={s['window_mean']:.3f} final={s['final_mean']:.3f} "
              f"best={s['best_mean']:.3f} grok={s['grok_rate']} "
              f"atten={s['attenuation_mean']:.4f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*70}")

    output_data = {
        'config': config,
        'results': all_results,
        'summary': summary,
        'verdict': verdict,
    }

    suffix = f'_N{HIDDEN}' if HIDDEN != 1536 else ''
    out_name = args.output or os.path.join(
        out_dir, f'rca_e5_skip_connections_L{L_LAMINAR}{suffix}.json')
    with open(out_name, 'w') as f:
        json.dump(output_data, f, indent=1, default=str)
    print(f"\n  Results: {out_name}")


if __name__ == '__main__':
    main()
