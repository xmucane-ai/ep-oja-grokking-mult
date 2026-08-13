#!/usr/bin/env python3
"""E4: WD SWEEP on the L=6 substrate BP arm — wd in {0, 0.1, 1.0}.

RCA t_a8d24443, experiment E4.
Isolates the weight-decay interaction (candidate cause #5).

The BP arm uses wd=1.0 (grokking standard on dense nets). On the sparse 3D
substrate the wd:gradient ratio is inflated ~O(1/sparsity). E4 sweeps wd.

If wd=0 or wd=0.1 GROKS while wd=1.0 fails: wd interaction is a real factor.
If all fail: wd is not the primary cause.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

SCRIPTS_DIR = r"D:\PC-hermes\scripts"
sys.path.insert(0, SCRIPTS_DIR)
from run_basis_swap_v13 import (make_mult_data, make_engine, P_MOD, IN_DIM,
                                CHANCE, L_LAMINAR, HIDDEN, T_INF)
from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE
from run_bp_control_arm1 import bp_forward, bp_evaluate

BP_LR = 1e-3
BP_GRAD_CLIP = 1.0


def run_seed(seed, steps, wd, eval_every=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    col_seed = seed * 100 + 0
    Xtr, Ytr, Xte, Yte = make_mult_data(seed=42, train_fraction=0.80)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)

    model = make_engine(seed=col_seed, n_layers=L_LAMINAR, stabilization=True)
    model.calibrate_thresholds(Xtr[:200])

    W_lin = model.W_lin.clone().detach().requires_grad_(True)
    W_prod = model.W_prod.clone().detach().requires_grad_(True)
    W_ff_list = [w.clone().detach().requires_grad_(True) for w in model.W_ff]
    W_out = model.W_out.clone().detach().requires_grad_(True)
    params = [W_lin, W_prod] + W_ff_list + [W_out]
    opt = torch.optim.AdamW(params, lr=BP_LR, weight_decay=wd,
                            betas=(0.9, 0.999))

    history = []
    grok_step = None
    best = 0.0
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad()
        x_list, yhat = bp_forward(model, Xtr, W_lin, W_prod, W_ff_list, W_out)
        loss = F.cross_entropy(yhat, Ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, BP_GRAD_CLIP)
        opt.step()
        with torch.no_grad():
            for l in range(model.L - 1):
                W_ff_list[l].mul_(model.ff_masks[l])
        if step % eval_every == 0 or step == 1:
            test_acc = bp_evaluate(model, Xte, Yte, W_lin, W_prod, W_ff_list, W_out)
            train_acc = bp_evaluate(model, Xtr[:500], Ytr[:500], W_lin, W_prod, W_ff_list, W_out)
            best = max(best, test_acc)
            if test_acc >= 0.9 and grok_step is None:
                grok_step = step
            history.append({'step': step, 'test_acc': test_acc, 'train_acc': train_acc})
    dt = time.time() - t0
    W = 5
    test_accs = [h['test_acc'] for h in history]
    window_avg = float(np.mean(test_accs[-W:])) if len(test_accs) >= W else float(np.mean(test_accs))
    return {'seed': seed, 'final_test_acc': test_accs[-1], 'best_test_acc': best,
            'window_avg_acc': window_avg, 'grok_step': grok_step, 'time': dt,
            'history': history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0-9')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--wd', type=float, default=1.0)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.replace(',', ' ').split():
        if '-' in part:
            lo, hi = part.split('-'); seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))

    print(f"E4 WD SWEEP on L=6 substrate: wd={args.wd} steps={args.steps} "
          f"seeds={seeds} device={DEVICE}", flush=True)
    results = []
    for seed in seeds:
        r = run_seed(seed, args.steps, args.wd)
        verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                   "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
        print(f"  s{seed}: WINDOW={r['window_avg_acc']:.3f} FINAL={r['final_test_acc']:.3f} "
              f"BEST={r['best_test_acc']:.3f} | {verdict} | {r['time']:.0f}s", flush=True)
        results.append(r)

    windows = [r['window_avg_acc'] for r in results]
    finals = [r['final_test_acc'] for r in results]
    groks = [r['grok_step'] for r in results if r['grok_step'] is not None]
    n_win = sum(1 for w in windows if w >= 0.90)
    summary = {'n_seeds': len(results),
               'window_avg_mean': float(np.mean(windows)),
               'window_avg_std': float(np.std(windows)),
               'final_mean': float(np.mean(finals)),
               'grok_rate_window': f"{n_win}/{len(results)}",
               'mean_grok_step': float(np.mean(groks)) if groks else None,
               'chance': CHANCE}
    print(f"SUMMARY: window={summary['window_avg_mean']:.3f}±{summary['window_avg_std']:.3f} "
          f"grok={summary['grok_rate_window']} final={summary['final_mean']:.3f}")

    out = args.output or os.path.join(r"D:\PC-hermes\outputs",
        f"rca_e4_wd{args.wd}_S{args.steps}.json")
    with open(out, 'w') as f:
        json.dump({'config': vars(args), 'results': results, 'summary': summary}, f, indent=1, default=str)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
