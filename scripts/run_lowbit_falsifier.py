#!/usr/bin/env python3
"""run_lowbit_falsifier.py -- SPEC_LOWBIT_TRAINING_FALSIFIER_v1.1 (commit 0fadc72).

L=6 main-engine low-bit training falsifier (int4 floor test). Ports the proven
CiM quantization recipe (BFP per-neuron scale + STE-by-construction + latent
fp32 master + two-stage LR/wd) onto the L=6 stabilized main engine
(ablation_cortex_v14_1.py, FROZEN commit 87c7250, local RMSProp, C+D
stabilization). The EP contrastive learning rule is UNTOUCHED.

ARMS (spec §11.2), all on L=6 laminar 3D cortex, F-1 split (80/20 disjoint),
10 seeds, modular mult p=53 (E_mult discrete-log encoder):
  R0  fp32  two-stage wd  C+D ON  (G3 parity ref)
  R1  fp32  constant wd=1.0 C+D ON (G3 baseline)
  R4  int4  per-neuron BFP two-stage wd C+D ECO SR  (G1 main arm)
  R4c int4  per-neuron BFP constant wd=1.0 C+D ECO SR (two-stage wd control)
  R4t int4  per-tensor BFP two-stage wd C+D ECO SR   (BFP control)
  R3  int3  per-neuron BFP two-stage wd C+D ECO SR   (G2 floor test)
  R3x int3  per-neuron BFP two-stage wd_0=2.0 C+D ECO SR (G2 aggressive)

METRICS (spec §11.3): test, grok count, conc, dc_frac, grok_epoch,
vanish_terminal, dh per area, SR-noise-vs-contrastive (Var(dh) R4 vs R0).

GATES (spec §11.4): G1 (R4 groks >=8/10, conc>=0.80, dc_frac<=0.15),
G2 (R3 groks >=5/10), G3 (R0 groks >=8/10).

COMPUTE: 3060 container via run_slot.sh (ONLY compute; local RTX2060 OFF-LIMITS).

Usage:
  python run_lowbit_falsifier.py --arms R0 R1 R4 R4c --seeds 0-9 --steps 3000
  python run_lowbit_falsifier.py --arms R4 --seeds 0-9 --steps 10000
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

from quantized_ablation_cortex import QuantizedAblationCortex, DEVICE

# ── Constants (proven C2 config from run_basis_swap_v13.py) ──
P_MOD = 53
K_FREQ = 26
IN_DIM = 4 * K_FREQ  # 104
CHANCE = 1.0 / P_MOD
G_PRIM = 2  # primitive root mod 53

HIDDEN = 1536
SHEET_SIZE = 40
T_INF = 10
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001
BATCH = 128
L_LAMINAR = 6

# C+D stabilization (proven)
GAMMA_W = 0.5
GAMMA_ALPHA = 0.25
T_DECAY = 1500

# Two-stage wd schedule (spec §7.4)
WD_0 = 1.0
T_DECAY_WD = T_DECAY  # both transitions fire at 1500 (spec §4.3 keeps C+D as-is)


# ================================================================
# Discrete log table for E_mult
# ================================================================
def build_dlog_table(p=53, g=2):
    dlog = {}
    val = 1
    for exp in range(p - 1):
        dlog[val] = exp
        val = (val * g) % p
    assert len(dlog) == p - 1, f"dlog table incomplete: {len(dlog)}/{p-1}"
    return dlog


DLOG = build_dlog_table(P_MOD, G_PRIM)


def make_mult_data(n_train=2163, n_test=541, seed=42, train_fraction=0.80):
    """E_mult: multiplicative character-basis features for modular multiplication."""
    rng = np.random.RandomState(seed)
    vals = np.arange(1, P_MOD)
    aa = np.repeat(vals, P_MOD - 1)
    bb = np.tile(vals, P_MOD - 1)
    cc = (aa * bb) % P_MOD

    dlog_a = np.array([DLOG[a] for a in aa], dtype=np.float64)
    dlog_b = np.array([DLOG[b] for b in bb], dtype=np.float64)

    freqs = np.arange(1, K_FREQ + 1, dtype=np.float64)
    ta = 2.0 * np.pi * np.outer(dlog_a, freqs) / (P_MOD - 1)
    tb = 2.0 * np.pi * np.outer(dlog_b, freqs) / (P_MOD - 1)

    X = np.empty((len(aa), IN_DIM), dtype=np.float32)
    X[:, 0::4] = np.cos(ta).astype(np.float32)
    X[:, 1::4] = np.sin(ta).astype(np.float32)
    X[:, 2::4] = np.cos(tb).astype(np.float32)
    X[:, 3::4] = np.sin(tb).astype(np.float32)
    Y = cc.astype(np.int64)

    total = len(aa)
    n_tr = int(total * train_fraction)
    n_te = total - n_tr
    perm = rng.permutation(total)
    Xtr, Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
    Xte, Yte = X[perm[n_tr:]], Y[perm[n_tr:]]
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def to_onehot(Y, n_classes):
    Yoh = torch.zeros(len(Y), n_classes, device=DEVICE)
    Yoh[torch.arange(len(Y)), Y] = 1.0
    return Yoh


# ================================================================
# Arm definitions (spec §11.2)
# ================================================================
ARMS = {
    'R0':  dict(precision='fp32', bfp='neuron', wd_schedule='two_stage', wd_0=WD_0, eco=False, rounding='nearest'),
    'R1':  dict(precision='fp32', bfp='neuron', wd_schedule='constant', wd_0=WD_0, eco=False, rounding='nearest'),
    'R4':  dict(precision='int4', bfp='neuron', wd_schedule='two_stage', wd_0=WD_0, eco=True,  rounding='stochastic'),
    'R4c': dict(precision='int4', bfp='neuron', wd_schedule='constant', wd_0=WD_0, eco=True,  rounding='stochastic'),
    'R4t': dict(precision='int4', bfp='tensor', wd_schedule='two_stage', wd_0=WD_0, eco=True,  rounding='stochastic'),
    'R3':  dict(precision='int3', bfp='neuron', wd_schedule='two_stage', wd_0=WD_0, eco=True,  rounding='stochastic'),
    'R3x': dict(precision='int3', bfp='neuron', wd_schedule='two_stage', wd_0=2.0, eco=True,  rounding='stochastic'),
}


def make_engine(seed, arm_cfg):
    """Create QuantizedAblationCortex with the proven C2 config + C+D + quant wrapper."""
    kwargs = dict(
        in_dim=IN_DIM, hidden_dim=HIDDEN, out_dim=P_MOD, n_layers=L_LAMINAR,
        sheet_size=SHEET_SIZE,
        target_rate=0.10, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.0, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
        gamma_W=GAMMA_W, gamma_alpha=GAMMA_ALPHA, T_decay=T_DECAY,
        alpha_theta_0=0.05,
        # Quant wrapper
        precision=arm_cfg['precision'], bfp=arm_cfg['bfp'],
        wd_schedule=arm_cfg['wd_schedule'], wd_0=arm_cfg['wd_0'],
        eco=arm_cfg['eco'], rounding=arm_cfg['rounding'],
        T_decay_wd=T_DECAY_WD,
    )
    return QuantizedAblationCortex(**kwargs)


# ================================================================
# Single-seed runner
# ================================================================
def run_single_seed(seed, arm, steps, eval_every=100, gate_every=500,
                    train_fraction=0.80, progress_path=None):
    t0 = time.time()
    col_seed = seed * 100 + 0  # matches proven l6eqcap seed mapping

    Xtr, Ytr, Xte, Yte = make_mult_data(seed=42, train_fraction=train_fraction)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)

    arm_cfg = ARMS[arm]
    model = make_engine(seed=col_seed, arm_cfg=arm_cfg)
    model.calibrate_thresholds(Xtr[:200])

    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0
    gate_log_list = []
    schedule_log = []
    vanish_log = []
    dh_log = []  # per-area dh at each eval step (for SR-noise-vs-contrastive)

    # Fixed eval batch for dh measurement (same inputs each step -> Var(dh) is
    # across mini-batches/training trajectory, isolating SR noise).
    dh_eval_X = Xte[:64]
    dh_eval_Yoh = to_onehot(Yte[:64], P_MOD)

    for step in range(1, steps + 1):
        idx = rng.randint(0, len(Xtr), BATCH)
        do_gate = (step % gate_every == 0) or (step == 1) or (step == steps)
        model.train_step(Xtr[idx], Yoh[idx], return_gates=do_gate)

        if step % eval_every == 0 or step == 1:
            acc = model.evaluate(Xte, Yte)
            train_acc = model.evaluate(Xtr[:500], Ytr[:500])
            best = max(best, acc)
            if acc >= 0.9 and grok_step is None:
                grok_step = step
            history.append({'step': step, 'test_acc': acc, 'train_acc': train_acc})

            # Per-area dh (spec §11.3) + vanish
            dh = model.dh_per_area(dh_eval_X, dh_eval_Yoh)
            dh_log.append({'step': step, 'dh': dh})
            vanish_log.append({'step': step, 'vanish': model._last_vanish[0]})

            last_gate = model.last_gates if do_gate else {}
            if do_gate and last_gate:
                gate_entry = {'step': step, **last_gate}
                gate_log_list.append(gate_entry)

            schedule_log.append({
                'step': step,
                'eta_W_eff': float(model._eta_W_eff),
                'alpha_theta_eff': float(model._alpha_theta_eff),
            })

            if step % 500 == 0 or step == eval_every or step == steps:
                elapsed = time.time() - t0
                fr = last_gate.get('firing_rates', []) if last_gate else []
                dh_str = ' '.join(f'{d:.4f}' for d in dh)
                print(f"  [s{seed}] step {step:5d}: test={acc:.3f} train={train_acc:.3f} "
                      f"best={best:.3f} dh=[{dh_str}] vanish={model._last_vanish[0]:.3f} "
                      f"ηW={model._eta_W_eff:.5f} [{elapsed:.0f}s]", flush=True)

        if progress_path:
            try:
                prog = {
                    'seed': seed, 'step': step,
                    'acc': history[-1]['test_acc'] if history else 0,
                    'best': best, 'grok_step': grok_step,
                    'elapsed_s': time.time() - t0,
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }
                with open(progress_path, 'w') as pf:
                    json.dump(prog, pf)
            except Exception:
                pass

    # ── Final full gate diagnostic ──
    print(f"  [s{seed}] computing final credit-propagation profile...", flush=True)
    with torch.no_grad():
        full_diag = model.infer(Xte[:64], to_onehot(Yte[:64], P_MOD), return_gates=True)
    final_gate = full_diag.get('gate_log', {})

    # ── Fourier concentration + DC fraction (spec §11.3) ──
    conc, dc_frac = model.fourier_conc()

    # ── Headline metrics ──
    W = 5
    test_accs = [h['test_acc'] for h in history]
    window_avg = float(np.mean(test_accs[-W:])) if len(test_accs) >= W else (
        float(np.mean(test_accs)) if test_accs else 0.0)
    final_acc = history[-1]['test_acc'] if history else 0.0
    sustained_grok = window_avg >= 0.90

    # SR-noise-vs-contrastive: Var(dh_l) across eval steps (mini-batches)
    dh_arrays = {l: [e['dh'][l] for e in dh_log] for l in range(L_LAMINAR)}
    dh_var = [float(np.var(dh_arrays[l])) if dh_arrays[l] else 0.0
              for l in range(L_LAMINAR)]
    dh_mean = [float(np.mean(dh_arrays[l])) if dh_arrays[l] else 0.0
               for l in range(L_LAMINAR)]

    dt = time.time() - t0
    result = {
        'seed': seed, 'arm': arm, 'L': L_LAMINAR, 'N': HIDDEN,
        'sheet_size': SHEET_SIZE, 't_inf': T_INF,
        'steps': steps, 'n_train': len(Xtr), 'n_test': len(Xte),
        'train_fraction': train_fraction,
        'precision': arm_cfg['precision'], 'bfp': arm_cfg['bfp'],
        'wd_schedule': arm_cfg['wd_schedule'], 'wd_0': arm_cfg['wd_0'],
        'eco': arm_cfg['eco'], 'rounding': arm_cfg['rounding'],
        'gamma_W': GAMMA_W, 'gamma_alpha': GAMMA_ALPHA, 'T_decay': T_DECAY,
        'final_test_acc': final_acc, 'best_test_acc': best,
        'window_avg_acc': window_avg, 'sustained_grok': sustained_grok,
        'grok_step': grok_step, 'chance': CHANCE, 'time': dt,
        'conc': conc, 'dc_frac': dc_frac,
        'vanish_terminal': model._last_vanish[0],
        'dh_per_area_final': dh_log[-1]['dh'] if dh_log else [],
        'dh_var_per_area': dh_var,
        'dh_mean_per_area': dh_mean,
        'history': history,
        'gate_snapshots': gate_log_list,
        'final_gate': final_gate,
        'schedule_log': schedule_log,
        'vanish_log': vanish_log,
        'dh_log': dh_log,
    }
    return result


def parse_seeds(seed_str):
    if seed_str is None:
        return list(range(10))
    seeds = []
    for part in seed_str.replace(',', ' ').split():
        if '-' in part:
            lo, hi = part.split('-')
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser(
        description='SPEC_LOWBIT_TRAINING_FALSIFIER_v1.1 — int4/int3 floor test on L=6 EP+Oja')
    ap.add_argument('--arms', type=str, default='R0 R1 R4 R4c',
                    help="space-separated arm names (default: base probe R0 R1 R4 R4c)")
    ap.add_argument('--seeds', type=str, default='0-9')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--gate_every', type=int, default=500)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    arms = args.arms.split()
    seeds = parse_seeds(args.seeds)

    # P6/P7 legality
    assert HIDDEN <= SHEET_SIZE ** 2, f"P6 VIOLATION: {HIDDEN} > {SHEET_SIZE**2}"
    assert L_LAMINAR >= 2, f"P7 VIOLATION: L={L_LAMINAR} < 2"

    if sys.platform == 'win32':
        out_dir = r"D:\PC-hermes\outputs"
    else:
        out_dir = os.environ.get('OUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)

    arm_tag = '_'.join(arms)
    default_name = f'lowbit_falsifier_{arm_tag}_L{L_LAMINAR}_N{HIDDEN}_T{T_INF}_S{args.steps}.json'
    base_name = args.output or os.path.join(out_dir, default_name)
    progress_path = os.path.join(out_dir, f'lowbit_falsifier_{arm_tag}_PROGRESS.json')

    config = {
        'spec': 'SPEC_LOWBIT_TRAINING_FALSIFIER_v1.1 (commit 0fadc72)',
        'engine': 'ablation_cortex_v14_1.py AblationCortex (via QuantizedAblationCortex)',
        'arms': {a: ARMS[a] for a in arms},
        'P': P_MOD, 'g': G_PRIM, 'K_freq': K_FREQ, 'in_dim': IN_DIM,
        'hidden_per_layer': HIDDEN, 'sheet_size': SHEET_SIZE,
        't_inf': T_INF, 'n_layers': L_LAMINAR,
        'steps': args.steps, 'seeds': seeds,
        'batch': BATCH, 'eta_W': ETA_W, 'eta_out': ETA_OUT, 'eta_theta': ETA_THETA,
        'k_conn': 8, 'target_rate': 0.10,
        'gamma_W': GAMMA_W, 'gamma_alpha': GAMMA_ALPHA, 'T_decay': T_DECAY,
        'T_decay_wd': T_DECAY_WD, 'wd_0': WD_0,
        'chance': CHANCE,
        'eval_every': args.eval_every, 'gate_every': args.gate_every,
        'eval_window_W': 5,
        'headline_metric': 'window_avg_acc (W=5)',
    }

    print(f"\n{'='*70}")
    print(f"SPEC_LOWBIT_TRAINING_FALSIFIER_v1.1 — int4/int3 floor test on L=6 EP+Oja")
    print(f"{'='*70}")
    print(f"  Arms:     {arms}")
    print(f"  L={L_LAMINAR}, N={HIDDEN}/layer, T_inf={T_INF}, sheet={SHEET_SIZE}")
    print(f"  Steps:    {args.steps}")
    print(f"  Seeds:    {seeds}")
    print(f"  C+D:      γ_W={GAMMA_W}, γ_α={GAMMA_ALPHA}, T_decay={T_DECAY}")
    print(f"  wd:       wd_0={WD_0}, T_decay_wd={T_DECAY_WD}")
    print(f"  Chance:   {CHANCE:.4f}")
    print(f"  Device:   {DEVICE}")
    print(f"  Output:   {base_name}")
    print(f"{'='*70}\n", flush=True)

    all_results = {}
    for arm in arms:
        print(f"\n{'='*70}")
        print(f"  ARM {arm}: {ARMS[arm]}")
        print(f"{'='*70}")
        results = []
        for seed in seeds:
            print(f"--- Seed {seed} ({arm}) ---")
            r = run_single_seed(seed, arm, args.steps,
                                eval_every=args.eval_every, gate_every=args.gate_every,
                                progress_path=progress_path)
            verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                       "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
            print(f"  => WINDOW_AVG={r['window_avg_acc']:.4f} FINAL={r['final_test_acc']:.4f} "
                  f"BEST={r['best_test_acc']:.4f} conc={r['conc']:.3f} dc={r['dc_frac']:.3f} "
                  f"| {verdict} | {r['time']:.0f}s\n", flush=True)
            results.append(r)

            # Write cumulative results after each seed
            try:
                windows = [x['window_avg_acc'] for x in results]
                finals = [x['final_test_acc'] for x in results]
                bests = [x['best_test_acc'] for x in results]
                concs = [x['conc'] for x in results]
                dc_fracs = [x['dc_frac'] for x in results]
                vanishes = [x['vanish_terminal'] for x in results]
                n_win = sum(1 for w in windows if w >= 0.90)
                summary = {
                    'n_seeds': len(results),
                    'window_avg_mean': float(np.mean(windows)),
                    'window_avg_median': float(np.median(windows)),
                    'window_avg_std': float(np.std(windows)),
                    'final_mean': float(np.mean(finals)),
                    'final_std': float(np.std(finals)),
                    'best_mean': float(np.mean(bests)),
                    'best_std': float(np.std(bests)),
                    'n_window_avg_ge_090': n_win,
                    'grok_rate_window': f"{n_win}/{len(results)}",
                    'conc_mean': float(np.mean(concs)),
                    'dc_frac_mean': float(np.mean(dc_fracs)),
                    'vanish_mean': float(np.mean(vanishes)),
                    'chance': CHANCE,
                }
                all_results[arm] = {'config': config, 'results': results, 'summary': summary}
                with open(base_name, 'w') as f:
                    json.dump(all_results, f, indent=1, default=str)
            except Exception as e:
                print(f"  WARNING: failed to save results: {e}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: SPEC_LOWBIT_TRAINING_FALSIFIER_v1.1")
    print(f"{'='*70}")
    print(f"  {'arm':>5} | {'grok':>6} | {'win_mean':>9} | {'final':>7} | {'best':>7} | "
          f"{'conc':>6} | {'dc':>6} | {'vanish':>7} | {'grok_ep':>8}")
    for arm in arms:
        s = all_results[arm]['summary']
        results = all_results[arm]['results']
        grok_eps = [r['grok_step'] for r in results if r['grok_step'] is not None]
        ge = str(int(np.median(grok_eps))) if grok_eps else "-"
        van = f"{s.get('vanish_mean', 0):.3f}" if 'vanish_mean' in s else "-"
        print(f"  {arm:>5} | {s['grok_rate_window']:>6} | {s['window_avg_mean']:9.3f} | "
              f"{s['final_mean']:7.3f} | {s['best_mean']:7.3f} | {s['conc_mean']:6.3f} | "
              f"{s['dc_frac_mean']:6.3f} | {van:>7} | {ge:>8}")

    # ── Gate verdicts (spec §11.4) ──
    print(f"\n  GATES:")
    r4 = all_results.get('R4', {}).get('summary', {})
    r3 = all_results.get('R3', {}).get('summary', {})
    r0 = all_results.get('R0', {}).get('summary', {})
    r1 = all_results.get('R1', {}).get('summary', {})
    if r4:
        g1 = (r4.get('n_window_avg_ge_090', 0) >= 8 and
              r4.get('conc_mean', 0) >= 0.80 and r4.get('dc_frac_mean', 1) <= 0.15)
        print(f"  G1 (R4 int4 groks >=8/10, conc>=0.80, dc<=0.15): "
              f"{'PASS' if g1 else 'FAIL'} — grok {r4.get('grok_rate_window')}, "
              f"conc {r4.get('conc_mean', 0):.3f}, dc {r4.get('dc_frac_mean', 1):.3f}")
    if r3:
        g2 = r3.get('n_window_avg_ge_090', 0) >= 5
        print(f"  G2 (R3 int3 groks >=5/10): {'PASS' if g2 else 'FAIL (expected: int4 is floor)'} "
              f"— grok {r3.get('grok_rate_window')}")
    if r0 and r1:
        g3 = r0.get('n_window_avg_ge_090', 0) >= 8
        print(f"  G3 (R0 fp32 two-stage groks >=8/10): {'PASS' if g3 else 'FAIL'} "
              f"— R0 grok {r0.get('grok_rate_window')}, R1 grok {r1.get('grok_rate_window')}")

    # ── Per-area error survival (THE measurement, spec §11.3) ──
    print(f"\n  PER-AREA ERROR SURVIVAL (final gate, averaged over seeds):")
    eps_keys = ['eps_a_norms_clamped', 'eps_a_norms_free', 'dh_norms',
                'lam_norms', 'firing_rates', 'hoyer', 'threshold_norms']
    for arm in arms:
        results = all_results[arm]['results']
        print(f"  --- {arm} ---")
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
                print(f"    {key:25s}: {[f'{x:.4f}' for x in avg]}")

    # ── SR-noise-vs-contrastive (spec §11.3) ──
    if 'R4' in all_results and 'R0' in all_results:
        print(f"\n  SR-NOISE-VS-CONTRASTIVE (Var(dh) across mini-batches):")
        for arm in ['R0', 'R4']:
            results = all_results[arm]['results']
            dh_var = np.mean([r['dh_var_per_area'] for r in results], axis=0)
            dh_mean = np.mean([r['dh_mean_per_area'] for r in results], axis=0)
            print(f"    {arm}: dh_var={[f'{v:.6f}' for v in dh_var]}")
            print(f"         dh_mean={[f'{m:.4f}' for m in dh_mean]}")

    print(f"\n  Results: {base_name}")


if __name__ == '__main__':
    main()
