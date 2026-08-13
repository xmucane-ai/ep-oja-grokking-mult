#!/usr/bin/env python3
"""run_fresh_seed_confirm.py -- SPEC_ARM4_FRESH_SEED_CONFIRM_v1.2 (task t_3a269dea).

Held-out-seed falsification test: does the P8-derived C+D stabilization
schedule (gamma_W=0.5, gamma_alpha=0.25, T_decay=1500) produce sustained
grok on 5 TRULY FRESH seeds {100, 101, 102, 103, 104} that the engine
has NEVER seen?

This runner uses the FROZEN C2 engine + data generation verbatim from
run_basis_swap_v13.py. The ONLY change is the seeds and the output JSON
schema (which adds the spec-required verdict block).

PASS CRITERION (BINDING): >= 4/5 fresh seeds achieve final_test_acc >= 0.90.
INTERPRETATION: a pass means "NOT FALSIFIED" — the P8 constants are not
seed-fitted. It does NOT mean "confirmed" (n=5 has insufficient power).

CONTROLS:
  1. add-sanity (E_add → ADD, mod-53) on same seeds — MUST grok. Proves
     seeds aren't pathological. If add-sanity fails, the seeds are bad.
  2. C·C spreading baseline = mult floor 0.0427; chance = 1/53 = 0.0189.

COMPUTE LANE: 3060 container via run_slot.sh (cap 2). 2060 OFF-LIMITS.

DELIVERABLE: fresh_seed_confirm_100-104_L6_N1536_T10.json
  - committed to outputs/ in the same change as this script
  - per-seed: final_test_acc, best_test_acc, window_avg_acc, grok_step,
    sustained_grok, per-area error survival (dh, eps at every layer)

Usage:
  # Full experiment: mult-stab (5 fresh seeds) + add-sanity control (5 seeds)
  python run_fresh_seed_confirm.py --seeds 100-104 --steps 3000

  # Just the headline run
  python run_fresh_seed_confirm.py --seeds 100-104 --steps 3000 --no-control
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

# Import the FROZEN C2 functions — do NOT reimplement
from run_basis_swap_v13 import (
    make_mult_data, make_add_data, to_onehot, make_engine,
    run_single_seed as c2_run_single_seed,
    P_MOD, K_FREQ, IN_DIM, CHANCE, G_PRIM,
    HIDDEN, SHEET_SIZE, T_INF, ETA_W, ETA_OUT, ETA_THETA, BATCH, L_LAMINAR,
    GAMMA_W, GAMMA_ALPHA, T_DECAY,
)

# ================================================================
# Seed parser (same logic as C2 runner)
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


# ================================================================
# ARM4 wrapper — runs C2 frozen runner, formats ARM4 output schema
# ================================================================
def run_arm4(seeds, steps, task, eval_every=100, gate_every=500,
             progress_path=None, stabilization=True):
    """Run the C2 frozen single-seed runner for each seed.

    Returns list of per-seed results with ARM4-required fields.
    """
    results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ({task}) ---", flush=True)
        r = c2_run_single_seed(
            seed, task, steps,
            eval_every=eval_every, gate_every=gate_every,
            train_fraction=0.80, progress_path=progress_path,
            stabilization=stabilization,
        )
        verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                   "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
        print(f"  => FINAL={r['final_test_acc']:.4f} "
              f"WINDOW_AVG={r['window_avg_acc']:.4f} "
              f"BEST={r['best_test_acc']:.4f} "
              f"GROK_STEP={r['grok_step']} "
              f"| {verdict} | {r['time']:.0f}s", flush=True)
        results.append(r)
    return results


# ================================================================
# Build the ARM4 verdict + config block per spec §6
# ================================================================
def build_arm4_output(mult_results, control_results, seeds, steps,
                      run_control):
    """Assemble the spec-required JSON schema."""
    # ── Per-seed results (mult-stab) ──
    formatted_results = []
    for r in mult_results:
        formatted_results.append({
            'seed': r['seed'],
            'final_test_acc': r['final_test_acc'],
            'best_test_acc': r['best_test_acc'],
            'window_avg_acc': r['window_avg_acc'],
            'grok_step': r['grok_step'],       # CRITICAL DIAGNOSTIC (§2.3)
            'sustained_grok': r['sustained_grok'],
            'time': r['time'],
            'history': r['history'],
            'final_gate': r.get('final_gate', {}),
            'gate_snapshots': r.get('gate_snapshots', []),
            'schedule_log': r.get('schedule_log', []),
        })

    # ── Verdict (§6 schema) ──
    n_ge90 = sum(1 for r in mult_results if r['final_test_acc'] >= 0.90)
    n_window_ge90 = sum(1 for r in mult_results if r['window_avg_acc'] >= 0.90)
    grok_steps = [r['grok_step'] for r in mult_results if r['grok_step'] is not None]

    verdict = {
        'n_ge90': n_ge90,
        'n_window_avg_ge90': n_window_ge90,
        'pass': n_ge90 >= 4,
        'criterion': '>= 4 of 5 fresh seeds final >= 0.90 '
                     '(interpretation: NOT FALSIFIED, not confirmed)',
        'grok_steps': grok_steps,
        'grok_step_range_c2': [1400, 1800],
        'grok_step_in_c2_window': all(
            1400 <= gs <= 1800 for gs in grok_steps) if grok_steps else None,
    }

    # ── Config (§6 schema) ──
    config = {
        'spec': 'SPEC_ARM4_FRESH_SEED_CONFIRM_v1.2',
        'task': 'mult-stab',
        'arm': 'Arm4 held-out-seed confirm',
        'engine': 'ablation_cortex_v14_1.py AblationCortex (via AblationCortexOpt)',
        'stabilization': True,
        'gamma_W': GAMMA_W,
        'gamma_alpha': GAMMA_ALPHA,
        'T_decay': T_DECAY,
        'P': P_MOD, 'g': G_PRIM, 'K_freq': K_FREQ, 'in_dim': IN_DIM,
        'hidden_per_layer': HIDDEN, 'n_layers': L_LAMINAR,
        'sheet_size': SHEET_SIZE,
        't_inf': T_INF, 'steps': steps,
        'seeds': seeds,
        'prior_seeds_excluded': '{0-39, 42-44} ∪ big12 — disjoint from {100-104} '
                                 '(max prior int=44, min fresh=100, 56-wide moat). '
                                 'Machine-verified across 249 prior JSON files.',
        'train_fraction': 0.8, 'batch': BATCH,
        'eta_W': ETA_W, 'eta_out': ETA_OUT, 'eta_theta': ETA_THETA,
        'k_conn': 8, 'target_rate': 0.1,
        'chance': CHANCE,
        'constants_origin': 'P8 ratio derivation (SPEC_BASIS_SWAP_v1.3 §4.5b, '
                            'commit b8c6196) — NOT seed-fitted; T_decay '
                            'empirically calibrated to C2 grok-onset window',
        'eval_window_W': 5,
        'headline_metric': 'final_test_acc (gate); window_avg_acc (W=5, report)',
    }

    # ── Summary stats ──
    finals = [r['final_test_acc'] for r in mult_results]
    windows = [r['window_avg_acc'] for r in mult_results]
    bests = [r['best_test_acc'] for r in mult_results]
    summary = {
        'n_seeds': len(mult_results),
        'final_mean': float(np.mean(finals)),
        'final_median': float(np.median(finals)),
        'final_std': float(np.std(finals)),
        'window_avg_mean': float(np.mean(windows)),
        'window_avg_median': float(np.median(windows)),
        'window_avg_std': float(np.std(windows)),
        'best_mean': float(np.mean(bests)),
        'best_std': float(np.std(bests)),
        'n_final_ge_090': n_ge90,
        'n_window_avg_ge_090': n_window_ge90,
        'grok_rate_final': f"{n_ge90}/{len(mult_results)}",
        'grok_rate_window': f"{n_window_ge90}/{len(mult_results)}",
        'chance': CHANCE,
        'cc_spreading_floor': 0.0427,  # C·C spreading baseline
    }

    output = {
        'config': config,
        'results': formatted_results,
        'summary': summary,
        'verdict': verdict,
    }

    # ── Control results (add-sanity) ──
    if run_control and control_results:
        control_formatted = []
        for r in control_results:
            control_formatted.append({
                'seed': r['seed'],
                'final_test_acc': r['final_test_acc'],
                'best_test_acc': r['best_test_acc'],
                'window_avg_acc': r['window_avg_acc'],
                'grok_step': r['grok_step'],
                'sustained_grok': r['sustained_grok'],
                'time': r['time'],
            })
        control_summary = {
            'task': 'add-sanity (E_add → ADD, mod-53)',
            'purpose': 'Prove seeds {100-104} are not pathological. '
                       'Add must grok on the same seeds.',
            'n_seeds': len(control_results),
            'n_grok': sum(1 for r in control_results
                          if r['window_avg_acc'] >= 0.90),
            'final_mean': float(np.mean(
                [r['final_test_acc'] for r in control_results])),
        }
        output['control_add_sanity'] = {
            'results': control_formatted,
            'summary': control_summary,
        }

    return output


# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser(
        description='SPEC_ARM4_FRESH_SEED_CONFIRM_v1.2 — '
                    'held-out-seed confirmatory run')
    ap.add_argument('--seeds', type=str, default='100-104',
                    help='Fresh seed spec (default: 100-104)')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--gate_every', type=int, default=500)
    ap.add_argument('--output', default=None,
                    help='Output JSON path (default: auto)')
    ap.add_argument('--no-control', action='store_true',
                    help='Skip add-sanity control run')
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    run_control = not args.no_control

    # Output paths
    if sys.platform == 'win32':
        out_dir = r"D:\PC-hermes\outputs"
    else:
        out_dir = os.environ.get('OUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)

    default_name = 'fresh_seed_confirm_100-104_L6_N1536_T10.json'
    out_path = args.output or os.path.join(out_dir, default_name)
    progress_path = os.path.join(out_dir, 'fresh_seed_confirm_PROGRESS.json')

    print(f"\n{'='*70}")
    print(f"SPEC_ARM4_FRESH_SEED_CONFIRM_v1.2 — Fresh Seed Confirmatory Run")
    print(f"{'='*70}")
    print(f"  Task:       mult-stab (E_mult → MULT, mod-{P_MOD})")
    print(f"  Seeds:      {seeds}  (FRESH — disjoint from all prior runs)")
    print(f"  Stabiliz.:  C+D (gamma_W={GAMMA_W}, gamma_alpha={GAMMA_ALPHA}, "
          f"T_decay={T_DECAY})")
    print(f"  L={L_LAMINAR}, N={HIDDEN}/layer, T_inf={T_INF}, sheet={SHEET_SIZE}")
    print(f"  Steps:      {args.steps}")
    print(f"  Chance:     {CHANCE:.4f}")
    print(f"  C·C floor:  0.0427")
    print(f"  Pass:       >= 4/5 fresh seeds final >= 0.90")
    print(f"  Control:    {'add-sanity (same seeds)' if run_control else 'SKIPPED'}")
    print(f"  Device:     {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  Output:     {out_path}")
    print(f"{'='*70}\n", flush=True)

    # ── P6/P7 guards ──
    assert HIDDEN <= SHEET_SIZE ** 2, \
        f"P6 VIOLATION: hidden={HIDDEN} > sheet^2={SHEET_SIZE**2}"
    assert L_LAMINAR >= 2, f"P7 VIOLATION: L={L_LAMINAR} < 2"

    # ── MAIN RUN: mult-stab on fresh seeds ──
    t0 = time.time()
    print(f">>> PHASE 1: mult-stab on fresh seeds {seeds}", flush=True)
    mult_results = run_arm4(
        seeds, args.steps, 'mult-stab',
        eval_every=args.eval_every, gate_every=args.gate_every,
        progress_path=progress_path,
    )
    t_mult = time.time() - t0

    # ── CONTROL: add-sanity on same seeds ──
    control_results = []
    if run_control:
        print(f"\n>>> PHASE 2: add-sanity control on same seeds {seeds}", flush=True)
        t1 = time.time()
        control_results = run_arm4(
            seeds, args.steps, 'add-sanity',
            eval_every=args.eval_every, gate_every=args.gate_every,
            progress_path=progress_path,
        )
        t_control = time.time() - t1
    else:
        t_control = 0.0

    # ── Build and save output ──
    output = build_arm4_output(mult_results, control_results, seeds,
                               args.steps, run_control)

    output['timing'] = {
        'mult_stab_total_s': t_mult,
        'control_total_s': t_control,
        'total_s': t_mult + t_control,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=1, default=str)
    print(f"\n  Saved to {out_path}", flush=True)

    # ── Print verdict ──
    v = output['verdict']
    s = output['summary']
    print(f"\n{'='*70}")
    print(f"  ARM4 VERDICT (mult-stab, {len(seeds)} fresh seeds)")
    print(f"{'='*70}")
    print(f"  PASS CRITERION: >= 4/5 final >= 0.90")
    print(f"  n_final >= 0.90: {v['n_ge90']}/{len(seeds)}")
    print(f"  n_window_avg >= 0.90: {v['n_window_avg_ge90']}/{len(seeds)}")
    print(f"  RESULT: {'PASS (NOT FALSIFIED)' if v['pass'] else 'FAIL'}")
    print(f"  ---")
    print(f"  Final acc:   mean={s['final_mean']:.3f} "
          f"median={s['final_median']:.3f} ±{s['final_std']:.3f}")
    print(f"  Window avg:  mean={s['window_avg_mean']:.3f} "
          f"median={s['window_avg_median']:.3f} ±{s['window_avg_std']:.3f}")
    print(f"  Best acc:    mean={s['best_mean']:.3f} ±{s['best_std']:.3f}")
    per_seed_final = [f"{r['final_test_acc']:.3f}" for r in mult_results]
    per_seed_window = [f"{r['window_avg_acc']:.3f}" for r in mult_results]
    print(f"  Per-seed final:  {per_seed_final}")
    print(f"  Per-seed window: {per_seed_window}")
    print(f"  Grok steps:  {v['grok_steps']}")
    print(f"  C2 window:   {v['grok_step_range_c2']}")

    if run_control:
        cs = output['control_add_sanity']['summary']
        print(f"\n  CONTROL (add-sanity): {cs['n_grok']}/{cs['n_seeds']} grok "
              f"(mean final={cs['final_mean']:.3f})")
        print(f"  {'SEEDS OK' if cs['n_grok'] >= 4 else 'WARNING: seeds may be pathological'}")

    print(f"  Chance:     {CHANCE:.4f}")
    print(f"  C·C floor:  0.0427")
    print(f"  Total time: {t_mult + t_control:.0f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
