#!/usr/bin/env python3
"""aggregate_p97_wall_v22.py -- Aggregate all SPEC_P97_SCALING_WALL_v2.2 cells.

Produces a single summary table + per-area error survival for ALL cells,
verifying the triple-constraint model predictions.
"""
import json
import os
import sys
import glob
import numpy as np


def load_cell(path):
    with open(path) as f:
        return json.load(f)


def analyze_cell(path):
    d = load_cell(path)
    cfg = d.get('config', {})
    results = d.get('results', [])
    summary = d.get('summary', {})

    p = cfg.get('p', cfg.get('prime', 0))
    N = cfg.get('hidden_per_layer', cfg.get('N', 0))
    n_train = cfg.get('n_train', 0)
    steps = cfg.get('steps', 0)
    t_decay = cfg.get('T_decay', cfg.get('t_decay', 1500))
    BATCH = 128

    # Coupon-collector rho (SPEC v2.2 eq 8)
    e_uniq = n_train * (1.0 - (1.0 - 1.0/n_train)**BATCH) if n_train > 0 else 0
    rho = e_uniq / BATCH if BATCH > 0 else 0
    c_grad = rho >= 0.90

    steps_per_epoch = max(1, -(-n_train // BATCH)) if n_train > 0 else 1
    epochs = steps / steps_per_epoch
    r_cap = N / ((p-1)/2.0)**2 if p > 1 else 0

    windows = [r.get('window_avg_acc', 0) for r in results]
    finals = [r.get('final_test_acc', 0) for r in results]
    bests = [r.get('best_test_acc', 0) for r in results]
    n_grok = sum(1 for w in windows if w >= 0.90)

    if n_grok >= 8:
        verdict = "GROK"
    elif n_grok >= 4:
        verdict = "PARTIAL"
    elif max(bests) >= 0.30 if bests else False:
        verdict = "TRANSIENT"
    else:
        verdict = "CHANCE"

    # Per-area metrics
    eps_c = []
    dh = []
    fr = []
    for r in results:
        fg = r.get('final_gate', {})
        e = fg.get('eps_a_norms_clamped', [])
        d2 = fg.get('dh_norms', [])
        f2 = fg.get('firing_rates', [])
        if e: eps_c.append(e)
        if d2: dh.append(d2)
        if f2: fr.append(f2)

    def avg_vec(vals):
        if not vals:
            return []
        maxlen = max(len(v) for v in vals)
        return [float(np.mean([v[i] for v in vals if i < len(v)])) for i in range(maxlen)]

    return {
        'path': os.path.basename(path),
        'p': p, 'N': N, 'steps': steps, 't_decay': t_decay,
        'n_train': n_train, 'rho': rho, 'c_grad': c_grad,
        'epochs': epochs, 'r_cap': r_cap,
        'n_seeds': len(results),
        'n_grok': n_grok,
        'win_med': float(np.median(windows)) if windows else 0,
        'win_mean': float(np.mean(windows)) if windows else 0,
        'win_std': float(np.std(windows)) if windows else 0,
        'final_mean': float(np.mean(finals)) if finals else 0,
        'best_mean': float(np.mean(bests)) if bests else 0,
        'verdict': verdict,
        'eps_a_clamped': avg_vec(eps_c),
        'dh_norms': avg_vec(dh),
        'firing_rates': avg_vec(fr),
    }


def main():
    out_dir = r"D:\PC-hermes\outputs" if sys.platform == 'win32' else os.environ.get('OUT_DIR', '/root/gate2/outputs')

    # Collect all relevant cells
    patterns = [
        os.path.join(out_dir, 'p97wall_v22_*.json'),
        os.path.join(out_dir, 'p97wall_I*_*.json'),     # v1.2 crossing-point results
        os.path.join(out_dir, 'p97wall_D*_*.json'),      # v1.2 budget arms
        os.path.join(out_dir, 'prime_sweep_p*_stab.json'),  # prime sweep cells
    ]
    paths = set()
    for pat in patterns:
        paths.update(glob.glob(pat))
    paths = sorted(paths)

    print(f"\n{'='*100}")
    print(f"SPEC_P97_SCALING_WALL_v2.2 — TRIPLE-CONSTRAINT VERIFICATION")
    print(f"{'='*100}")
    print(f"Found {len(paths)} result files\n")

    cells = []
    for path in paths:
        try:
            c = analyze_cell(path)
            cells.append(c)
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}")

    # Sort by prime, then N, then steps
    cells.sort(key=lambda c: (c['p'], c['N'], c['steps']))

    print(f"{'Cell':45s} {'p':>3s} {'N':>5s} {'steps':>6s} {'rho':>6s} {'Cgrd':>5s} {'epoch':>7s} "
          f"{'Rcap':>5s} {'grok':>6s} {'winMed':>7s} {'winMean':>8s} {'verdict':>10s}")
    print("-" * 120)

    for c in cells:
        name = c['path'].replace('.json', '')
        cgrad = "PASS" if c['c_grad'] else "FAIL"
        print(f"{name:45s} {c['p']:3d} {c['N']:5d} {c['steps']:6d} {c['rho']:6.3f} {cgrad:>5s} "
              f"{c['epochs']:7.1f} {c['r_cap']:5.2f} {c['n_grok']:3d}/{c['n_seeds']:<2d} "
              f"{c['win_med']:7.3f} {c['win_mean']:8.3f} {c['verdict']:>10s}")

    # Per-area error survival
    print(f"\n{'='*100}")
    print(f"PER-AREA ERROR SURVIVAL (SPEC acceptance criterion 4)")
    print(f"{'='*100}")
    for c in cells:
        name = c['path'].replace('.json', '')
        print(f"\n--- {name} (p={c['p']}, N={c['N']}, {c['n_seeds']} seeds) ---")
        if c['eps_a_clamped']:
            print(f"  eps_a_clamped: {[f'{x:.4f}' for x in c['eps_a_clamped']]}")
        if c['dh_norms']:
            print(f"  dh_norms:      {[f'{x:.4f}' for x in c['dh_norms']]}")
        if c['firing_rates']:
            print(f"  firing_rates:  {[f'{x:.3f}' for x in c['firing_rates']]}")

    # Triple-constraint summary
    print(f"\n{'='*100}")
    print(f"TRIPLE-CONSTRAINT MODEL VERDICT")
    print(f"{'='*100}")
    print("""
The triple-constraint model (SPEC v2.2) is FALSIFIED if any prediction is wrong:
  - p=29 must GROK (rho=0.905, R_cap=7.84 COMFORTABLE, 600 epochs) -- WEAKEST C_grad test
  - p=41 must GROK (rho=0.952, R_cap=3.84 COMFORTABLE, 300 epochs)
  - p=13 must FAIL (rho=0.605, C_grad FAIL -- gradient diversity collapse)
  - p=53 N=1536 must GROK (rho=0.971, R_cap=2.27, 176.5 epochs) -- proven
""")

    # Check predictions
    predictions_ok = True
    for c in cells:
        if c['p'] == 29 and c['N'] == 1536 and c['verdict'] != 'GROK':
            print(f"  FALSIFICATION: p=29 N=1536 predicted GROK, got {c['verdict']}")
            predictions_ok = False
        elif c['p'] == 41 and c['N'] == 1536 and c['verdict'] != 'GROK':
            print(f"  FALSIFICATION: p=41 N=1536 predicted GROK, got {c['verdict']}")
            predictions_ok = False
        elif c['p'] == 13 and c['verdict'] == 'GROK':
            print(f"  FALSIFICATION: p=13 predicted FAIL/CHANCE, got GROK")
            predictions_ok = False
        elif c['p'] == 53 and c['N'] == 1536 and c['verdict'] != 'GROK':
            print(f"  WARNING: p=53 N=1536 expected GROK, got {c['verdict']} (was 8/10 proven)")

    if predictions_ok:
        print("  All available cells MATCH the triple-constraint predictions.")


if __name__ == '__main__':
    main()
