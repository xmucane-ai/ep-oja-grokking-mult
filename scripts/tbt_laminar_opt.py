#!/usr/bin/env python3
"""tbt_laminar_opt.py -- Speed-optimized tbt_laminar (math-identical).

Card t_0b58d520: uses AblationCortexOpt for GPU-native homeostatic update,
residual reuse in ALM cycle. Vectorized Scheme C voting.

User requirement (2026-08-07): intermediate output for observability.
1. Per-seed JSON: results written to output file after EACH seed completes.
2. Progress JSON: outputs/tbt_laminar_PROGRESS.json updated every eval.
3. [DONE] line printed after each seed with final acc + grok step.

ALL HYPERPARAMETERS UNCHANGED. Bit-identical trajectory.
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
sys.path.insert(0, r"D:\PC-hermes\scripts")

from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE

# ── Data constants (identical) ──
P_MOD = 53
K_FREQ = 26
IN_DIM = 4 * K_FREQ
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


def make_data(n_train=2247, n_test=562, seed=42):
    rng = np.random.RandomState(seed)
    aa = np.repeat(np.arange(P_MOD), P_MOD)
    bb = np.tile(np.arange(P_MOD), P_MOD)
    cc = (aa + bb) % P_MOD
    freqs = np.arange(1, K_FREQ + 1, dtype=np.float32)
    ta = 2.0 * np.pi * np.outer(aa, freqs) / P_MOD
    tb = 2.0 * np.pi * np.outer(bb, freqs) / P_MOD
    X = np.empty((P_MOD * P_MOD, IN_DIM), dtype=np.float32)
    X[:, 0::4] = np.cos(ta)
    X[:, 1::4] = np.sin(ta)
    X[:, 2::4] = np.cos(tb)
    X[:, 3::4] = np.sin(tb)
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


def make_column(seed, n_layers=L_LAMINAR, hidden=HIDDEN_PER_LAYER):
    """Same as original but uses AblationCortexOpt."""
    net = AblationCortexOpt(
        in_dim=IN_DIM, hidden_dim=hidden, out_dim=P_MOD, n_layers=n_layers,
        sheet_size=SHEET_SIZE,
        target_rate=0.10, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
    )
    return net


def column_positions_3d(n_col=3, seed=7):
    rng = np.random.RandomState(seed)
    return [(rng.rand(), rng.rand(), rng.rand()) for _ in range(n_col)]


def lateral_weights_3d(positions, sigma=0.5):
    n = len(positions)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.sqrt(sum((positions[i][k] - positions[j][k]) ** 2
                                for k in range(3)))
                w[i, j] = float(np.exp(-d ** 2 / (2 * sigma ** 2)))
    row_sums = w.sum(axis=1, keepdims=True)
    w = w / np.maximum(row_sums, 1e-8)
    return w


class LaminarEnsemble3D:
    """3 genuine-3D laminar columns + partial views + vectorized Scheme C voting."""

    def __init__(self, condition, seed=0, n_layers=L_LAMINAR,
                 hidden=HIDDEN_PER_LAYER):
        assert condition in ('CC3', 'FV', 'SS', 'HS')
        self.condition = condition
        self.seed = seed
        self.n_layers = n_layers
        self.hidden = hidden
        self.voting = condition in ('FV', 'SS', 'HS')
        self.split = 'full' if condition in ('CC3', 'FV') else \
                     ('soft' if condition == 'SS' else 'hard')

        self.positions = column_positions_3d(N_COL, seed=seed + 42)
        self.w_lat = torch.from_numpy(
            lateral_weights_3d(self.positions).astype(np.float32)).to(DEVICE)

        rng_masks = np.random.RandomState(seed + 1000)

        self.columns = []
        self.masks = []
        for c in range(N_COL):
            net = make_column(seed=seed * 100 + c,
                              n_layers=n_layers, hidden=hidden)
            self.columns.append(net)
            if self.split == 'full':
                m = np.ones(IN_DIM, dtype=np.float32)
            elif self.split == 'soft':
                m = (rng_masks.rand(IN_DIM) < 0.65).astype(np.float32)
            else:
                m = np.zeros(IN_DIM, dtype=np.float32)
                lo = c * IN_DIM // N_COL
                hi = (c + 1) * IN_DIM // N_COL
                m[lo:hi] = 1.0
            self.masks.append(torch.from_numpy(m).to(DEVICE))

        Xtr, _, _, _ = make_data()
        Xtr = Xtr.to(DEVICE)
        for c in range(N_COL):
            self.columns[c].calibrate_thresholds(Xtr[:200] * self.masks[c])

    def _view(self, X, c):
        return X * self.masks[c]

    def column_outputs(self, X):
        with torch.no_grad():
            y_list = []
            for c in range(N_COL):
                x, _ = self.columns[c].forward_init(self._view(X, c))
                y = x[self.columns[c].L - 1] @ self.columns[c].W_out
                y_list.append(y)
            return y_list

    def predict(self, X):
        y_list = self.column_outputs(X)
        if not self.voting:
            y_avg = sum(y_list) / len(y_list)
            return y_avg.argmax(dim=-1)

        # VECTORIZED Scheme C voting (einsum replaces Python double-loop)
        log_probs = torch.stack([
            torch.log_softmax(y, dim=-1) for y in y_list
        ], dim=0)  # [N_COL, B, n_classes]

        n = len(y_list)
        w = self.w_lat  # [N_COL, N_COL]

        # Build effective weight matrix: w_eff[c,c2] = w[c,c2] if c!=c2, else 1.0
        w_eff = w.clone()
        w_eff.fill_diagonal_(1.0)
        w_sums = w_eff.sum(dim=1)  # [N_COL]

        # For each column c: weighted_sum[c] = sum_c2 w_eff[c,c2] * log_probs[c2]
        # w_eff [C, C], log_probs [C, B, K] → output [C, B, K]
        weighted_sum = torch.einsum('ij,jbk->ibk', w_eff, log_probs)  # [C, B, K]
        # Normalize by w_sums per column
        consensus_per_col = weighted_sum / (w_sums.unsqueeze(1).unsqueeze(2) + 1e-8)
        # Average over columns
        consensus_logp = consensus_per_col.sum(dim=0) / n  # [B, C]

        return consensus_logp.argmax(dim=-1)

    def evaluate(self, X, Y):
        return float((self.predict(X) == Y).float().mean().item())

    def train_step(self, X, Yoh, return_gates=False):
        gate_logs = []
        for c in range(N_COL):
            gl = self.columns[c].train_step(
                self._view(X, c), Yoh, return_gates=return_gates)
            gate_logs.append(gl)
        return gate_logs

    def diagnostics(self, Xte):
        diag = {'per_column': []}
        for c in range(N_COL):
            col = self.columns[c]
            x_view = self._view(Xte, c)
            with torch.no_grad():
                x0, pre_acts = col.forward_init(x_view)
                gate_log = {}
                gate_log['firing_rates'] = []
                gate_log['hoyer'] = []
                for l in range(col.L):
                    xa = x0[l]
                    mean_act = xa.abs().mean(dim=0)
                    l1 = mean_act.sum() + 1e-12
                    l2 = mean_act.norm() + 1e-12
                    hoy = (np.sqrt(col.N) - float(l1) / float(l2)) / \
                          (np.sqrt(col.N) - 1)
                    hoy = max(0.0, min(1.0, hoy))
                    gate_log['hoyer'].append(hoy)
                    fr = float((xa > 1e-6).float().mean().item())
                    gate_log['firing_rates'].append(fr)
            diag['per_column'].append(gate_log)
        agg = {}
        for key in ['hoyer', 'firing_rates']:
            vals = [c_diag.get(key, []) for c_diag in diag['per_column']]
            if vals and isinstance(vals[0], list):
                agg[key] = [float(np.mean([v[i] for v in vals if i < len(v)]))
                            for i in range(len(vals[0]))]
        diag['aggregate'] = agg
        return diag

    def full_diagnostics(self, Xte, Yte):
        diag = {'per_column': []}
        Yoh = to_onehot(Yte, P_MOD)
        for c in range(N_COL):
            col = self.columns[c]
            x_view = self._view(Xte, c)
            with torch.no_grad():
                gl = col.infer(x_view[:64], Yoh[:64], return_gates=True)
            diag['per_column'].append(gl.get('gate_log', {}))
        agg = {}
        for key in ['hoyer', 'firing_rates', 'eps_a_norms_clamped',
                    'dh_norms', 'lam_norms', 'gate1', 'gate1d',
                    'gate2_min', 'gate2_mean', 'gate2_contrastive_mean',
                    'W_ff_norms', 'W_ff_spectral_norm',
                    'd_dependent_contrastive', 'contrastive_bias_part',
                    'contrastive_d_dep', 'contrastive_bias']:
            vals = []
            for c_diag in diag['per_column']:
                if key in c_diag:
                    v = c_diag[key]
                    if isinstance(v, list):
                        vals.append(v)
                    else:
                        vals.append(v)
            if vals:
                if isinstance(vals[0], list):
                    agg[key] = [float(np.mean([v[i] for v in vals
                                               if i < len(v)]))
                                for i in range(len(vals[0]))]
                else:
                    agg[key] = float(np.mean(vals))
        return agg


def run_condition(condition, seed, steps=3000, eval_every=100, verbose=True,
                  n_layers=L_LAMINAR, hidden=HIDDEN_PER_LAYER,
                  progress_path=None):
    t0 = time.time()
    Xtr, Ytr, Xte, Yte = make_data()
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)

    model = LaminarEnsemble3D(condition, seed=seed,
                              n_layers=n_layers, hidden=hidden)
    rng = np.random.RandomState(seed)

    history = []
    grok_step = None
    best = 0.0
    diag_log = []

    for step in range(1, steps + 1):
        idx = rng.randint(0, len(Xtr), BATCH)
        model.train_step(Xtr[idx], Yoh[idx], return_gates=False)

        if step % eval_every == 0 or step == 1:
            acc = model.evaluate(Xte, Yte)
            best = max(best, acc)
            if acc >= 0.9 and grok_step is None:
                grok_step = step
            history.append((step, acc))

            diag = model.diagnostics(Xte[:128])
            agg = diag['aggregate']
            diag_entry = {
                'step': step, 'ens_acc': acc,
                'hoyer': agg.get('hoyer', []),
                'firing_rates': agg.get('firing_rates', []),
            }
            diag_log.append(diag_entry)

            if verbose and (step % 500 == 0 or step == eval_every):
                hoy_mean = np.mean(agg.get('hoyer', [0]))
                fr = agg.get('firing_rates', [0])
                fr_str = ' '.join(f'{f:.3f}' for f in fr[:n_layers])
                print(f"  [{condition} s{seed}] step {step:5d}: "
                      f"ens={acc:.3f} hoyer={hoy_mean:.3f} "
                      f"fr=[{fr_str}] "
                      f"[{time.time()-t0:.0f}s]", flush=True)

            # ── Write lightweight progress file for the poller ──
            if progress_path:
                try:
                    prog = {
                        'condition': condition, 'seed': seed,
                        'step': step, 'acc': acc, 'best': best,
                        'grok_step': grok_step,
                        'elapsed_s': time.time() - t0,
                        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    }
                    with open(progress_path, 'w') as pf:
                        json.dump(prog, pf)
                except Exception:
                    pass  # never let progress write crash the run

    if verbose:
        print(f"  [{condition} s{seed}] computing final credit-propagation "
              f"profile...", flush=True)
    full_diag = model.full_diagnostics(Xte, Yte)
    if full_diag:
        diag_log.append({'step': steps, 'type': 'full_credit_diag',
                         **full_diag})
        if verbose:
            g1 = full_diag.get('gate1', 0)
            g1d = full_diag.get('gate1d', 0)
            g2min = full_diag.get('gate2_min', 0)
            g2mean = full_diag.get('gate2_mean', 0)
            eps_a = full_diag.get('eps_a_norms_clamped', [])
            dh = full_diag.get('dh_norms', [])
            lam = full_diag.get('lam_norms', [])
            hoy = full_diag.get('hoyer', [])
            fr = full_diag.get('firing_rates', [])
            print(f"  [{condition} s{seed}] CREDIT PROFILE:")
            print(f"    Gate1 (||eps_a[1]||/||eps_a[L-1]||) = {g1:.3f} "
                  f"(>=0.3 = ballistic)")
            print(f"    Gate1d (||d[1]||/||d[L-1]||)        = {g1d:.3f}")
            print(f"    Gate2 (cos(eps_a, -BP), min/mean)   = {g2min:.3f}/{g2mean:.3f}")
            g2c = full_diag.get('gate2_contrastive_mean', 'N/A')
            d_dep = full_diag.get('d_dependent_contrastive',
                    full_diag.get('contrastive_d_dep', []))
            cbias = full_diag.get('contrastive_bias_part',
                    full_diag.get('contrastive_bias', []))
            wff_sn = full_diag.get('W_ff_spectral_norm', [])
            wff_fn = full_diag.get('W_ff_norms', [])
            print(f"    Gate2_contrastive (cos(dEps,-BP))   = {g2c}")
            print(f"    d_dependent_contrastive (L-B metric) = "
                  f"{[f'{d:.6f}' for d in d_dep] if isinstance(d_dep, list) else d_dep}")
            print(f"    contrastive_bias_part (diagnostic)  = "
                  f"{[f'{b:.6f}' for b in cbias] if isinstance(cbias, list) else cbias}")
            print(f"    W_ff spectral norms: {wff_sn}")
            print(f"    W_ff Frobenius norms: "
                  f"{[f'{w:.3f}' for w in wff_fn] if isinstance(wff_fn, list) else wff_fn}")
            print(f"    Hoyer per layer:  {[f'{h:.2f}' for h in hoy]}")
            print(f"    Firing per layer: {[f'{f:.3f}' for f in fr]}")
            print(f"    eps_a norms:      {[f'{e:.4f}' for e in eps_a]}")
            print(f"    dh norms:         {[f'{d:.4f}' for d in dh]}")
            print(f"    lambda norms:     {[f'{l:.4f}' for l in lam]}")

    dt = time.time() - t0
    return {
        'condition': condition, 'seed': seed, 'steps': steps,
        'final_ens_acc': history[-1][1] if history else 0.0,
        'best_ens_acc': best, 'grok_step': grok_step,
        'n_col': N_COL, 'hidden_per_layer': hidden, 'L_col': n_layers,
        'total_hidden': hidden * n_layers,
        'split': model.split, 'voting': model.voting,
        'time': dt,
        'diag_log': diag_log,
    }


def main():
    ap = argparse.ArgumentParser(
        description='Brain-faithful TBT: L=6-8 laminar sparse 3D columns (OPTIMIZED)')
    ap.add_argument('--condition', nargs='+', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--layers', type=int, default=L_LAMINAR)
    ap.add_argument('--hidden', type=int, default=HIDDEN_PER_LAYER)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    if args.condition is None or args.condition == ['all']:
        conds = ['CC3', 'FV', 'SS', 'HS']
    else:
        conds = args.condition

    out_dir = r"D:\PC-hermes\outputs"
    os.makedirs(out_dir, exist_ok=True)
    base_name = args.output or os.path.join(
        out_dir, f'tbt_laminar_L{args.layers}_N{args.hidden}_opt.json')
    progress_path = os.path.join(out_dir, 'tbt_laminar_PROGRESS.json')

    print(f"TBT Laminar Experiment OPTIMIZED (L={args.layers}, N={args.hidden}/layer)")
    print(f"Conditions: {conds}")
    print(f"Seeds: {args.seeds}")
    print(f"Steps: {args.steps}")
    print(f"Device: {DEVICE}")
    print(f"Total hidden per column: {args.hidden * args.layers}")
    print(f"Sheet size: {SHEET_SIZE} ({SHEET_SIZE**2} capacity)")
    print(f"Output: {base_name}")
    print(f"Progress: {progress_path}")

    results = {}
    for cond in conds:
        print(f"\n{'=' * 70}\n  CONDITION: {cond}\n{'=' * 70}", flush=True)
        results[cond] = []
        for seed in args.seeds:
            print(f"\n  --- {cond} seed {seed} ---", flush=True)
            r = run_condition(cond, seed, steps=args.steps,
                              eval_every=args.eval_every,
                              n_layers=args.layers, hidden=args.hidden,
                              progress_path=progress_path)
            verdict = "GROK" if r['final_ens_acc'] >= 0.9 else \
                      ("LEARN" if r['best_ens_acc'] >= 0.3 else "CHANCE")
            print(f"  [DONE] {cond} s{seed}: final={r['final_ens_acc']:.3f} "
                  f"best={r['best_ens_acc']:.3f} grok={r['grok_step']} "
                  f"[{verdict}] ({r['time']:.0f}s)", flush=True)
            results[cond].append(r)

            # ── Write cumulative results JSON after EACH seed ──
            # (not just at the end — gives visibility mid-run)
            try:
                with open(base_name, 'w') as f:
                    json.dump(results, f, indent=1)
            except Exception:
                pass  # never let JSON write crash the run

    print(f"\n{'=' * 80}\n  SUMMARY: TBT Laminar OPTIMIZED (L={args.layers})\n{'=' * 80}")
    print(f"{'Cond':<8} {'EnsAcc':>8} {'Best':>8} {'Grok':>8} {'Hoyer':>8} "
          f"{'FiringRate':>12}")
    print("-" * 80)
    for cond in conds:
        if cond not in results:
            continue
        rs = results[cond]
        n = len(rs)
        ens = np.median([r['final_ens_acc'] for r in rs])
        best = np.median([r['best_ens_acc'] for r in rs])
        n_grok = sum(1 for r in rs if r['grok_step'] is not None)
        hoy_vals = []
        fr_vals = []
        for r in rs:
            if r['diag_log']:
                dl = r['diag_log'][-1]
                hoy_vals.append(np.mean(dl.get('hoyer', [0])))
                fr_vals.append(np.mean(dl.get('firing_rates', [0])))
        hoy_med = np.median(hoy_vals) if hoy_vals else 0.0
        fr_med = np.median(fr_vals) if fr_vals else 0.0
        print(f"{cond:<8} {ens:>8.3f} {best:>8.3f} {n_grok:>2}/{n:<2}    "
              f"{hoy_med:>8.3f} {fr_med:>12.4f}")

    # Final JSON already written per-seed; just confirm path
    print(f"\nResults (per-seed incremental writes): {base_name}")


if __name__ == '__main__':
    main()
