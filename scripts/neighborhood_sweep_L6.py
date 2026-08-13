#!/usr/bin/env python3
"""neighborhood_sweep_L6.py — SPEC_NEIGHBORHOOD_SWEEP_L6_v1.0

Decisive single-variable test of the user's neighborhood hypothesis:
  mult mod-53 = add under the discrete-log relabeling; the relabeling is
  a GLOBAL permutation needing a neighborhood large enough to hold it.

ONE variable: firing rate / active-set size (target_rate ∈ {0.10, 0.20, 0.35}).
Input: one-hot 106-dim (2P), NO Fourier, NO learned encoder.
Controls: add-sanity at each rate, C·C spreading, chance, test∩train=∅.

Harness changes (exactly two, on the proven CC3 harness):
  1. Input feature builder: Fourier 104 → one-hot 106 (spec §4.1)
  2. target_rate parameterization through make_column (spec §4.2)

NO mechanism changes (no W_enc, no block-tiling, no op-cue, no head-swap).
All hyperparameters UNCHANGED from the accepted v1.4 baseline.

Constitution: P1–P8 all PASS (see spec §9).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE

# ── Constants (identical to tbt_laminar_opt.py, except IN_DIM) ──
P_MOD = 53
IN_DIM = 2 * P_MOD          # ★ 106: one-hot 2P (spec §4.1)
CHANCE = 1.0 / P_MOD
N_COL = 3
L_LAMINAR = 6
HIDDEN_PER_LAYER = 512
SHEET_SIZE = 23
T_INF = 10
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001
BATCH = 128


# ════════════════════════════════════════════════════════════════════
# CHANGE 1 (spec §4.1): one-hot input feature builder
# ════════════════════════════════════════════════════════════════════
def make_onehot_data(task, n_train=2247, n_test=562, seed=42):
    """One-hot 106-dim input (2P), task ∈ {'add','mult'}. Disjoint train/test."""
    rng = np.random.RandomState(seed)
    aa = np.repeat(np.arange(P_MOD), P_MOD)
    bb = np.tile(np.arange(P_MOD), P_MOD)
    cc = (aa + bb) % P_MOD if task == 'add' else (aa * bb) % P_MOD
    X = np.zeros((P_MOD * P_MOD, 2 * P_MOD), dtype=np.float32)
    X[np.arange(P_MOD * P_MOD), aa] = 1.0          # a-token one-hot (cols 0..52)
    X[np.arange(P_MOD * P_MOD), P_MOD + bb] = 1.0  # b-token one-hot (cols 53..105)
    Y = cc.astype(np.int64)
    perm = rng.permutation(P_MOD * P_MOD)
    Xtr, Ytr = X[perm[:n_train]], Y[perm[:n_train]]
    Xte, Yte = X[perm[n_train:n_train + n_test]], Y[perm[n_train:n_train + n_test]]
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def to_onehot(Y, n_classes):
    Yoh = torch.zeros(len(Y), n_classes, device=DEVICE)
    Yoh[torch.arange(len(Y)), Y] = 1.0
    return Yoh


# ════════════════════════════════════════════════════════════════════
# CHANGE 2 (spec §4.2): target_rate parameterization
# ════════════════════════════════════════════════════════════════════
def make_column_rate(seed, target_rate=0.10, n_layers=L_LAMINAR,
                     hidden=HIDDEN_PER_LAYER):
    """make_column with target_rate parameterized (spec §4.2).

    Identical to tbt_laminar_opt.py:make_column except:
    - in_dim=IN_DIM=106 (one-hot, not Fourier 104)
    - target_rate is a parameter, not hardcoded 0.10
    All other hyperparameters UNCHANGED.
    """
    net = AblationCortexOpt(
        in_dim=IN_DIM, hidden_dim=hidden, out_dim=P_MOD, n_layers=n_layers,
        sheet_size=SHEET_SIZE,
        target_rate=target_rate, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
    )
    return net


# ════════════════════════════════════════════════════════════════════
# CC3 Ensemble on one-hot data (based on MultEnsembleCC3)
# ════════════════════════════════════════════════════════════════════
class OnehotEnsembleCC3:
    """CC3 ensemble: 3 columns, full view, simple average, target_rate parameterized."""

    def __init__(self, seed=0, target_rate=0.10, task='mult',
                 n_layers=L_LAMINAR, hidden=HIDDEN_PER_LAYER):
        self.seed = seed
        self.target_rate = target_rate
        self.task = task
        self.n_layers = n_layers
        self.hidden = hidden
        self.columns = []
        for c in range(N_COL):
            net = make_column_rate(seed=seed * 100 + c, target_rate=target_rate,
                                   n_layers=n_layers, hidden=hidden)
            self.columns.append(net)
        # Calibrate thresholds on one-hot training data
        Xtr, _, _, _ = make_onehot_data(task)
        Xtr = Xtr.to(DEVICE)
        for c in range(N_COL):
            self.columns[c].calibrate_thresholds(Xtr[:200])

    def predict(self, X):
        with torch.no_grad():
            y_list = []
            for c in range(N_COL):
                x, _ = self.columns[c].forward_init(X)
                y = x[self.columns[c].L - 1] @ self.columns[c].W_out
                y_list.append(y)
            y_avg = sum(y_list) / len(y_list)
            return y_avg.argmax(dim=-1)

    def evaluate(self, X, Y):
        return float((self.predict(X) == Y).float().mean().item())

    def train_step(self, X, Yoh, return_gates=False):
        for c in range(N_COL):
            self.columns[c].train_step(X, Yoh, return_gates=return_gates)

    def lightweight_diagnostics(self, Xte_sample):
        """Firing rates + hoyer from forward_init (no full inference)."""
        diag = {}
        firing_rates_all = []
        hoyer_all = []
        for c in range(N_COL):
            col = self.columns[c]
            with torch.no_grad():
                x0, _ = col.forward_init(Xte_sample)
                for l in range(col.L):
                    xa = x0[l]
                    mean_act = xa.abs().mean(dim=0)
                    l1 = mean_act.sum() + 1e-12
                    l2 = mean_act.norm() + 1e-12
                    hoy = (np.sqrt(col.N) - float(l1) / float(l2)) / \
                          (np.sqrt(col.N) - 1)
                    hoy = max(0.0, min(1.0, hoy))
                    fr = float((xa > 1e-6).float().mean().item())
                    if len(firing_rates_all) <= l:
                        firing_rates_all.append([])
                        hoyer_all.append([])
                    firing_rates_all[l].append(fr)
                    hoyer_all[l].append(hoy)
        diag['firing_rates'] = [float(np.mean(fr)) for fr in firing_rates_all]
        diag['hoyer'] = [float(np.mean(h)) for h in hoyer_all]
        return diag

    def full_diagnostics(self, Xte_sample, Yte_sample):
        """Full per-layer diagnostics: dh, eps_a, gates, weights, prod_share."""
        Yoh = to_onehot(Yte_sample, P_MOD)
        diag_per_col = []
        for c in range(N_COL):
            col = self.columns[c]
            with torch.no_grad():
                result = col.infer(Xte_sample, Yoh, return_gates=True)
            diag_per_col.append(result.get('gate_log', {}))

        agg = {}
        for key in ['hoyer', 'firing_rates', 'eps_a_norms_clamped',
                    'eps_a_norms_free', 'dh_norms', 'lam_norms',
                    'gate1', 'gate1d',
                    'gate2_min', 'gate2_mean', 'gate2_contrastive_mean',
                    'W_ff_norms', 'W_ff_spectral_norm',
                    'd_dependent_contrastive', 'contrastive_bias_part',
                    'contrastive_bias', 'contrastive_d_dep',
                    'threshold_norms', 'energy',
                    'gate2_per_layer', 'gate2_contrastive_per_layer']:
            vals = []
            for c_diag in diag_per_col:
                if key in c_diag:
                    vals.append(c_diag[key])
            if vals:
                if isinstance(vals[0], list):
                    agg[key] = [float(np.mean([v[i] for v in vals
                                               if i < len(v)]))
                                for i in range(len(vals[0]))]
                else:
                    agg[key] = float(np.mean(vals))

        # prod_share (spec §10): product pathway share of dendritic output
        prod_shares = []
        for c in range(N_COL):
            col = self.columns[c]
            with torch.no_grad():
                cv = Xte_sample[:, col.conn]
                pv = cv[:, :, col.pi] * cv[:, :, col.pj]
                lin_contrib = (cv * col.W_lin.unsqueeze(0)).sum(dim=2)
                prod_contrib = (pv * col.W_prod.unsqueeze(0)).sum(dim=2)
                lin_norm = float(lin_contrib.norm().item())
                prod_norm = float(prod_contrib.norm().item())
                prod_shares.append(prod_norm / (lin_norm + prod_norm + 1e-8))
        agg['prod_share'] = float(np.mean(prod_shares))

        return agg


# ════════════════════════════════════════════════════════════════════
# C·C spreading control (spec §8)
# ════════════════════════════════════════════════════════════════════
def cc_spreading_control(Xtr_np, Ytr_np, Xte_np, Yte_np, k=5):
    """C·C spreading baseline: k-NN in input space, majority vote on labels.

    With one-hot input, cosine sim = (1[a=a']+1[b=b'])/2.
    Neighbors share a or b — modular regularities don't correlate with
    shared operands, so this plateaus near chance.
    """
    # Cosine similarity (one-hot has norm sqrt(2), so sim = dot/2)
    sims = Xte_np @ Xtr_np.T / 2.0  # [n_test, n_train]
    preds = np.zeros(len(Xte_np), dtype=np.int64)
    for i in range(len(Xte_np)):
        top_k_idx = np.argsort(sims[i])[-k:]
        labels = Ytr_np[top_k_idx]
        preds[i] = np.bincount(labels, minlength=P_MOD).argmax()
    acc = float((preds == Yte_np).mean())
    return acc


# ════════════════════════════════════════════════════════════════════
# Run a single arm
# ════════════════════════════════════════════════════════════════════
def run_arm(task, target_rate, seed, steps, eval_every=100,
            diag_every=500, verbose=True):
    t0 = time.time()
    Xtr, Ytr, Xte, Yte = make_onehot_data(task)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)

    model = OnehotEnsembleCC3(seed=seed, target_rate=target_rate, task=task)
    rng = np.random.RandomState(seed)

    history = []
    grok_step = None
    best = 0.0
    diag_snapshots = []

    for step in range(1, steps + 1):
        idx = rng.randint(0, len(Xtr), BATCH)
        model.train_step(Xtr[idx], Yoh[idx])

        if step % eval_every == 0 or step == 1:
            acc = model.evaluate(Xte, Yte)
            best = max(best, acc)
            if acc >= 0.9 and grok_step is None:
                grok_step = step
            entry = {'step': step, 'acc': acc, 'best': best}
            history.append(entry)

            # Full diagnostics at intervals
            if step % diag_every == 0 or step == 1 or step == steps:
                full_diag = model.full_diagnostics(Xte[:64], Yte[:64])
                entry['firing_rates'] = full_diag.get('firing_rates', [])
                entry['dh_norms'] = full_diag.get('dh_norms', [])
                entry['gate2_mean'] = full_diag.get('gate2_mean', 0)
                entry['gate2_contrastive_mean'] = full_diag.get(
                    'gate2_contrastive_mean', 0)
                entry['eps_a_norms_clamped'] = full_diag.get(
                    'eps_a_norms_clamped', [])
                entry['hoyer'] = full_diag.get('hoyer', [])
                entry['threshold_norms'] = full_diag.get('threshold_norms', [])
                entry['W_ff_spectral_norm'] = full_diag.get(
                    'W_ff_spectral_norm', [])
                entry['prod_share'] = full_diag.get('prod_share', 0)
                entry['energy'] = full_diag.get('energy', 0)
                diag_snapshots.append({'step': step, **full_diag})
            else:
                # Lightweight: firing rates only
                lw = model.lightweight_diagnostics(Xte[:64])
                entry['firing_rates'] = lw.get('firing_rates', [])

            if verbose and (step % 500 == 0 or step == eval_every):
                fr = entry.get('firing_rates', [0])
                fr_str = ' '.join(f'{f:.3f}' for f in fr[:L_LAMINAR])
                dh = entry.get('dh_norms', [])
                dh_str = ' '.join(f'{d:.4f}' for d in dh[:L_LAMINAR]) if dh else ''
                g2 = entry.get('gate2_mean', 0)
                print(f"  [{task} r{target_rate:.2f} s{seed}] step {step:5d}: "
                      f"acc={acc:.3f} best={best:.3f} g2={g2:.3f} "
                      f"fr=[{fr_str}] dh=[{dh_str}] "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    # Final full diagnostics
    final_diag = model.full_diagnostics(Xte[:64], Yte[:64])

    dt = time.time() - t0
    return {
        'task': task, 'target_rate': target_rate, 'seed': seed,
        'steps': steps,
        'final_acc': history[-1]['acc'] if history else 0.0,
        'best_acc': best, 'grok_step': grok_step,
        'history': history,
        'diag_snapshots': diag_snapshots,
        'final_diagnostics': final_diag,
        'time_s': dt,
    }


# ════════════════════════════════════════════════════════════════════
# Main: sweep driver
# ════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description='SPEC_NEIGHBORHOOD_SWEEP_L6_v1.0 — firing-rate sweep')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--mult_steps', type=int, default=3000)
    ap.add_argument('--add_steps', type=int, default=3000)
    ap.add_argument('--rates', type=float, nargs='+',
                    default=[0.10, 0.20, 0.35])
    ap.add_argument('--tasks', type=str, nargs='+', default=['mult', 'add'])
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--diag_every', type=int, default=500)
    ap.add_argument('--output', default=None)
    ap.add_argument('--cc_k', type=int, default=5,
                    help='k for C·C spreading control')
    args = ap.parse_args()

    out_dir = os.environ.get('OUTPUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        out_dir, 'neighborhood_sweep_L6.json')
    progress_path = os.path.join(out_dir, 'neighborhood_sweep_PROGRESS.json')

    print(f"NEIGHBORHOOD SWEEP L6 (SPEC v1.0)")
    print(f"  IN_DIM={IN_DIM} (one-hot 2P={2*P_MOD})")
    print(f"  Rates: {args.rates}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Mult steps: {args.mult_steps}, Add steps: {args.add_steps}")
    print(f"  L={L_LAMINAR}, N={HIDDEN_PER_LAYER}, N_COL={N_COL}")
    print(f"  Device: {DEVICE}")
    print(f"  Chance: {CHANCE:.4f}")
    print(f"  Output: {output_path}")
    print(f"  Progress: {progress_path}")
    print()

    # ── C·C spreading control (computed once, same for all rates) ──
    cc_results = {}
    for task in args.tasks:
        Xtr, Ytr, Xte, Yte = make_onehot_data(task)
        Xtr_np = Xtr.numpy()
        Ytr_np = Ytr.numpy()
        Xte_np = Xte.numpy()
        Yte_np = Yte.numpy()
        cc_acc = cc_spreading_control(Xtr_np, Ytr_np, Xte_np, Yte_np,
                                      k=args.cc_k)
        cc_results[task] = cc_acc
        print(f"  C·C spreading control ({task}, k={args.cc_k}): "
              f"{cc_acc:.4f} (chance={CHANCE:.4f})", flush=True)

    results = {
        'spec': 'SPEC_NEIGHBORHOOD_SWEEP_L6_v1.0',
        'config': {
            'IN_DIM': IN_DIM, 'P_MOD': P_MOD, 'L': L_LAMINAR,
            'N': HIDDEN_PER_LAYER, 'N_COL': N_COL,
            'BATCH': BATCH, 'T_INF': T_INF,
            'rates': args.rates, 'seeds': args.seeds,
            'mult_steps': args.mult_steps, 'add_steps': args.add_steps,
        },
        'cc_spreading_control': cc_results,
        'chance': CHANCE,
        'arms': {},
    }

    # ── Sweep: rate × task × seed ──
    for rate in args.rates:
        for task in args.tasks:
            steps = args.mult_steps if task == 'mult' else args.add_steps
            arm_key = f"{task}_rate{rate:.2f}"
            results['arms'][arm_key] = []

            for seed in args.seeds:
                print(f"\n{'='*60}")
                print(f"  ARM: {arm_key} seed={seed} steps={steps}")
                print(f"{'='*60}", flush=True)

                r = run_arm(task, rate, seed, steps,
                            eval_every=args.eval_every,
                            diag_every=args.diag_every)
                verdict = "GROK" if r['final_acc'] >= 0.9 else \
                          ("LEARN" if r['best_acc'] >= 0.3 else "CHANCE")
                print(f"  [DONE] {arm_key} s{seed}: "
                      f"final={r['final_acc']:.3f} best={r['best_acc']:.3f} "
                      f"grok={r['grok_step']} ({verdict}) "
                      f"({r['time_s']:.0f}s)", flush=True)

                results['arms'][arm_key].append(r)

                # ── Incremental JSON write ──
                try:
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=1)
                except Exception:
                    pass

                # ── Progress file ──
                try:
                    prog = {
                        'arm': arm_key, 'seed': seed,
                        'step': steps, 'final_acc': r['final_acc'],
                        'best_acc': r['best_acc'],
                        'grok_step': r['grok_step'],
                        'elapsed_s': r['time_s'],
                        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    }
                    with open(progress_path, 'w') as pf:
                        json.dump(prog, pf)
                except Exception:
                    pass

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: NEIGHBORHOOD SWEEP L6 (one-hot {IN_DIM}-dim)")
    print(f"{'='*70}")
    print(f"  Chance: {CHANCE:.4f}")
    for task in args.tasks:
        cc = cc_results.get(task, 0)
        print(f"  C·C control ({task}): {cc:.4f}")
    print()
    print(f"{'Arm':<22} {'FinalAcc':>10} {'Best':>8} {'Grok':>6} "
          f"{'dh[L-1]':>10} {'g2_mean':>10} {'fr_mean':>10}")
    print("-" * 80)
    for rate in args.rates:
        for task in args.tasks:
            arm_key = f"{task}_rate{rate:.2f}"
            arm_results = results['arms'].get(arm_key, [])
            if not arm_results:
                continue
            finals = [r['final_acc'] for r in arm_results]
            bests = [r['best_acc'] for r in arm_results]
            n_grok = sum(1 for r in arm_results if r['grok_step'] is not None)
            med_final = float(np.median(finals))
            med_best = float(np.median(bests))
            # Get final diagnostics from last seed
            last_diag = arm_results[-1].get('final_diagnostics', {})
            dh = last_diag.get('dh_norms', [0])
            dh_last = dh[-1] if dh else 0
            g2 = last_diag.get('gate2_mean', 0)
            fr = last_diag.get('firing_rates', [0])
            fr_mean = float(np.mean(fr)) if fr else 0
            print(f"{arm_key:<22} {med_final:>10.3f} {med_best:>8.3f} "
                  f"{n_grok}/{len(arm_results):<4} {dh_last:>10.4f} "
                  f"{g2:>10.3f} {fr_mean:>10.4f}")

    # ── Discriminator ──
    print(f"\n  DISCRIMINATOR (mult-vs-ρ curve):")
    for rate in args.rates:
        arm_key = f"mult_rate{rate:.2f}"
        arm_results = results['arms'].get(arm_key, [])
        if arm_results:
            finals = [r['final_acc'] for r in arm_results]
            med = float(np.median(finals))
            n_grok = sum(1 for r in arm_results if r['grok_step'] is not None)
            print(f"    mult @ ρ={rate:.2f}: median_final={med:.3f} "
                  f"grok={n_grok}/{len(arm_results)}")

    print(f"\n  Results: {output_path}")


if __name__ == '__main__':
    main()
