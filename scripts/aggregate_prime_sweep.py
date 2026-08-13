#!/usr/bin/env python3
"""aggregate_prime_sweep.py — aggregate prime-scaling sweep results into summary.

SPEC_PRIME_SCALING_SWEEP_ARM2_v1.1 §7.2: aggregated summary table.
Reads all prime_sweep_p{p}_N{N}_stab.json files and produces
outputs/prime_sweep_summary.json with per-cell metrics.
"""
import json
import os
import sys
import numpy as np


def load_cell(fn):
    with open(fn) as f:
        d = json.load(f)
    results = d.get('results', [])
    config = d.get('config', {})
    if not results:
        return None
    windows = [r['window_avg_acc'] for r in results]
    finals = [r['final_test_acc'] for r in results]
    bests = [r['best_test_acc'] for r in results]
    grok_steps = [r.get('grok_step') for r in results if r.get('grok_step')]

    n_win = sum(1 for w in windows if w >= 0.90)
    n_best = sum(1 for b in bests if b >= 0.90)

    # Per-area error survival (averaged across seeds, from final_gate)
    eps_keys = ['eps_a_norms_clamped', 'eps_a_norms_free', 'dh_norms',
                'firing_rates', 'hoyer']
    per_area = {}
    for key in eps_keys:
        vals = []
        for r in results:
            fg = r.get('final_gate', {})
            v = fg.get(key, [])
            if v:
                vals.append(v)
        if vals:
            avg = [float(np.mean([v[i] for v in vals if i < len(v)]))
                   for i in range(max(len(v) for v in vals))]
            per_area[key] = avg

    return {
        'p': config.get('p'),
        'N': config.get('hidden_per_layer'),
        'in_dim': config.get('in_dim'),
        'g': config.get('g'),
        'k_freq': config.get('k_freq'),
        'n_train': config.get('n_train'),
        'n_pairs': config.get('n_pairs'),
        'epochs': config.get('epochs_at_3000_steps'),
        'steps_per_epoch': config.get('steps_per_epoch'),
        'n_seeds': len(results),
        'grok_rate_window': f"{n_win}/{len(results)}",
        'grok_rate_best': f"{n_best}/{len(results)}",
        'mean_window_avg': float(np.mean(windows)),
        'mean_final': float(np.mean(finals)),
        'mean_best': float(np.mean(bests)),
        'median_window_avg': float(np.median(windows)),
        'median_final': float(np.median(finals)),
        'std_window_avg': float(np.std(windows)),
        'std_final': float(np.std(finals)),
        'mean_grok_step': float(np.mean(grok_steps)) if grok_steps else None,
        'chance': config.get('chance'),
        'per_area_error': per_area,
        'per_seed_window': [round(w, 3) for w in windows],
        'per_seed_final': [round(f, 3) for f in finals],
        'per_seed_best': [round(b, 3) for b in bests],
    }


def main():
    out_dir = os.environ.get('OUT_DIR', '.')
    if sys.platform == 'win32':
        out_dir = r"D:\PC-hermes\outputs"

    cells = {}
    # Find all prime_sweep files
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith('prime_sweep_p') and fn.endswith('_stab.json') and 'summary' not in fn:
            path = os.path.join(out_dir, fn)
            cell = load_cell(path)
            if cell:
                key = f"p{cell['p']}_N{cell['N']}"
                cells[key] = cell
                print(f"Loaded {key}: grok_window={cell['grok_rate_window']}, "
                      f"mean_window={cell['mean_window_avg']:.3f}, "
                      f"mean_final={cell['mean_final']:.3f}")

    if not cells:
        print("No prime sweep results found!")
        return

    summary = {
        'spec': 'SPEC_PRIME_SCALING_SWEEP_ARM2_v1.1',
        'task_id': 't_7a5baaf6',
        'headline_metric': 'window_avg_acc (W=5, threshold 0.90)',
        'cells': cells,
    }

    out_path = os.path.join(out_dir, 'prime_sweep_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"\nSummary written to {out_path}")

    # Print table
    print(f"\n{'='*90}")
    print(f"  PRIME SCALING SWEEP SUMMARY (SPEC_PRIME_SCALING_SWEEP_ARM2_v1.1)")
    print(f"{'='*90}")
    print(f"  {'Cell':>6s} {'p':>4s} {'N':>5s} {'IN_DIM':>6s} {'grok_win':>9s} "
          f"{'grok_best':>10s} {'mean_win':>8s} {'mean_final':>10s} {'mean_best':>9s} "
          f"{'epochs':>7s}")
    for key, c in sorted(cells.items(), key=lambda x: (x[1]['p'], x[1]['N'])):
        print(f"  {key:>6s} {c['p']:4d} {c['N']:5d} {c['in_dim']:6d} "
              f"{c['grok_rate_window']:>9s} {c['grok_rate_best']:>10s} "
              f"{c['mean_window_avg']:8.3f} {c['mean_final']:10.3f} "
              f"{c['mean_best']:9.3f} {c['epochs']:7.1f}")
    print(f"{'='*90}")

    # Per-area error survival
    print(f"\n  PER-AREA ERROR SURVIVAL (averaged across seeds, final gate):")
    for key, c in sorted(cells.items(), key=lambda x: (x[1]['p'], x[1]['N'])):
        print(f"\n  {key} (p={c['p']}, N={c['N']}):")
        for ek, ev in c.get('per_area_error', {}).items():
            print(f"    {ek:25s}: {[f'{x:.4f}' for x in ev]}")


if __name__ == '__main__':
    main()
