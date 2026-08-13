#!/usr/bin/env python3 -u
"""run_p97_scaling_wall_v22.py -- SPEC_P97_SCALING_WALL_v2.2 (task t_92034a55).

The p=29/41/53/97 prime sweep testing the C_epoch x C_cap x C_grad
triple-constraint scaling-wall falsifier.

Sweep primes:
  p=29  (g=2, k_freq=14): grok_step~440, 3000 steps, N=1536  [I1 crossing-point]
  p=41  (g=6, k_freq=20): grok_step~880, 3000 steps, N=1536  [I2 crossing-point]
  p=53  (g=2, k_freq=26): proven 8/10 grok at N=1536          [B2 replication]
  p=97  (g=5, k_freq=48): budget arms D1-D4
  p=13  (g=2, k_freq=6):  FAIL control (0/10, C_grad FAIL)    [A1/A2 control]

p=97 budget arms (SPEC v2.2 §4):
  D1:  5000 steps, N=1536, sheet=40, T_DECAY=1500 (no decay, S<5500)
  D2:  8000 steps, N=1536, sheet=40, T_DECAY=5500 (decay post-grok)
  D3: 10237 steps, N=1536, sheet=40, T_DECAY=5500 (epoch-equivalent of p=53)
  D4: 10237 steps, N=2048, sheet=46, T_DECAY=5500 (capacity arm, 46x46=2116)

p=17/p=19 are NON-OPERATIVE (not in sweep set) — do NOT run them.

Engine: ablation_cortex_v14_1_opt.py AblationCortexOpt (FROZEN at 87c7250).
This runner is an extension of run_prime_scaling_sweep.py with:
  - p=29 (g=2) and p=41 (g=6) added to PRIMES_AND_GENERATORS
  - CLI args: --t-decay (default 1500), --sheet-size (default 40)
  - Budget arm presets via --arm D1/D2/D3/D4

ACCEPTANCE CRITERIA (BINDING, from card t_92034a55):
1. PASSES only if it GROKS: held-out acc -> ~1.0, beating strong control,
   improving with data.
2. dh/eps survival are NECESSARY but NOT SUFFICIENT.
3. 10+ seeds, equal step counts across conditions, no hardcoded seeds.
4. Report per-area error (eps_l) survival — THE measurement.

COMPUTE LANE: 3060 container (training-container via run_slot.sh).
The 2060 local GPU is OFF-LIMITS (user rule 2026-08-09).

Usage:
  # Crossing-point: p=29
  python run_p97_scaling_wall_v22.py --prime 29 --hidden 1536 --seeds 0-9 --steps 3000

  # Crossing-point: p=41
  python run_p97_scaling_wall_v22.py --prime 41 --hidden 1536 --seeds 0-9 --steps 3000

  # p=97 budget arm D1 (clean no-decay probe)
  python run_p97_scaling_wall_v22.py --prime 97 --arm D1 --seeds 0-9

  # p=97 budget arm D4 (capacity arm, N=2048, sheet=46)
  python run_p97_scaling_wall_v22.py --prime 97 --arm D4 --seeds 0-9

  # p=13 FAIL control
  python run_p97_scaling_wall_v22.py --prime 13 --hidden 1536 --seeds 0-9 --steps 3000
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

# ── Engine hyperparameters (FROZEN — identical to C2 proven config) ──
T_INF = 10
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001  # DEAD param (kept for engine API compat)
BATCH = 128
L_LAMINAR = 6

# SPEC v1.3 §4.5c: C+D stabilization (FROZEN at proven values)
GAMMA_W = 0.5
GAMMA_ALPHA = 0.25

# Per-prime primitive roots (SPEC v2.2 §4.3, verified computationally)
# CRITICAL: g=6 for p=41 (g=2 is NOT primitive mod 41, order 20 not 40)
PRIMES_AND_GENERATORS = {13: 2, 29: 2, 41: 6, 53: 2, 97: 5}

# p=97 budget arms (SPEC v2.2 §4.1/§4.2)
BUDGET_ARMS = {
    'D1': dict(steps=5000,  hidden=1536, sheet_size=40, t_decay=1500),
    'D2': dict(steps=8000,  hidden=1536, sheet_size=40, t_decay=5500),
    'D3': dict(steps=10237, hidden=1536, sheet_size=40, t_decay=5500),
    'D4': dict(steps=10237, hidden=2048, sheet_size=46, t_decay=5500),
}


# ================================================================
# Primitive-root guard (SPEC §4.3 — BINDING)
# ================================================================
def check_primitive_root(g, p):
    """Assert g is a primitive root mod p (order(g,p) == p-1).

    Uses stdlib only (avoids sympy dependency on container).
    """
    order, val = 0, 1
    while True:
        order += 1
        val = (val * g) % p
        if val == 1:
            break
        if order > p:  # safety
            break
    assert order == p - 1, \
        f"g={g} is NOT a primitive root mod {p} (order {order} != {p - 1}). " \
        f"Silent correctness bug if used for build_dlog_table."
    return True


# ================================================================
# Discrete log table for E_mult (generalized per prime)
# ================================================================
def build_dlog_table(p, g):
    """Build discrete log table: dlog[a] = log_g(a) mod (p-1) for a in {1..p-1}."""
    check_primitive_root(g, p)  # SPEC §4.3 binding guard
    dlog = {}
    val = 1
    for exp in range(p - 1):
        dlog[val] = exp
        val = (val * g) % p
    assert len(dlog) == p - 1, f"dlog table incomplete: {len(dlog)}/{p - 1}"
    return dlog


# ================================================================
# Data generation (generalized E_mult per prime)
# ================================================================
def make_mult_data(p, g, k_freq, in_dim, seed=42, train_fraction=0.80):
    """E_mult: multiplicative character-basis features for modular multiplication.

    For each (a,b), a,b in {1..p-1}: target = (a*b) mod p.
    Excludes a=0, b=0 (no discrete log). (p-1)^2 pairs.
    """
    rng = np.random.RandomState(seed)
    dlog = build_dlog_table(p, g)

    vals = np.arange(1, p)  # [1, 2, ..., p-1]
    aa = np.repeat(vals, p - 1)
    bb = np.tile(vals, p - 1)
    cc = (aa * bb) % p

    # Discrete logs
    dlog_a = np.array([dlog[a] for a in aa], dtype=np.float64)
    dlog_b = np.array([dlog[b] for b in bb], dtype=np.float64)

    freqs = np.arange(1, k_freq + 1, dtype=np.float64)
    # omega_k(a) = 2*pi*k*log_g(a)/(p-1)  [p-1 in denominator!]
    ta = 2.0 * np.pi * np.outer(dlog_a, freqs) / (p - 1)
    tb = 2.0 * np.pi * np.outer(dlog_b, freqs) / (p - 1)

    X = np.empty((len(aa), in_dim), dtype=np.float32)
    X[:, 0::4] = np.cos(ta).astype(np.float32)
    X[:, 1::4] = np.sin(ta).astype(np.float32)
    X[:, 2::4] = np.cos(tb).astype(np.float32)
    X[:, 3::4] = np.sin(tb).astype(np.float32)

    Y = cc.astype(np.int64)

    total = len(aa)
    n_tr = int(total * train_fraction)
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
# Engine construction
# ================================================================
def make_engine(in_dim, hidden_dim, out_dim, seed, sheet_size, t_decay,
                n_layers=L_LAMINAR):
    """Create AblationCortexOpt with the EXACT proven C2 config + C+D schedule.

    t_decay: SPEC v2.2 §4.4 — D1 uses 1500 (no decay), D2/D3/D4 use 5500
    (post-grok decay for p=97).
    sheet_size: SPEC v2.2 §4.2 — D4 uses 46 (46x46=2116 >= 2048).
    """
    kwargs = dict(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers,
        sheet_size=sheet_size,
        target_rate=0.10, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
        # C+D stabilization (SPEC v1.3 §4.5c / v2.2 §4.4)
        gamma_W=GAMMA_W,
        gamma_alpha=GAMMA_ALPHA,
        T_decay=t_decay,
        alpha_theta_0=0.05,
    )
    return AblationCortexOpt(**kwargs)


# ================================================================
# Single-seed runner
# ================================================================
def run_single_seed(prime, g, k_freq, in_dim, hidden, sheet_size, t_decay,
                    seed, steps, eval_every=100, gate_every=500,
                    train_fraction=0.80, progress_path=None):
    """Run one seed of the p97 scaling-wall sweep."""
    t0 = time.time()
    col_seed = seed * 100 + 0  # matches proven l6eqcap seed mapping

    # ── Data generation ──
    Xtr, Ytr, Xte, Yte = make_mult_data(prime, g, k_freq, in_dim,
                                         seed=42, train_fraction=train_fraction)

    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, prime)

    n_train = len(Xtr)
    n_test = len(Xte)

    # ── Engine ──
    model = make_engine(in_dim=in_dim, hidden_dim=hidden, out_dim=prime,
                        sheet_size=sheet_size, t_decay=t_decay,
                        seed=col_seed)
    model.calibrate_thresholds(Xtr[:200])

    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0
    gate_log_list = []
    schedule_log = []

    for step in range(1, steps + 1):
        idx = rng.randint(0, n_train, BATCH)
        do_gate = (step % gate_every == 0) or (step == 1) or (step == steps)
        model.train_step(Xtr[idx], Yoh[idx], return_gates=do_gate)

        if step % eval_every == 0 or step == 1:
            acc = model.evaluate(Xte, Yte)
            train_acc = model.evaluate(Xtr[:500], Ytr[:500])
            best = max(best, acc)
            if acc >= 0.9 and grok_step is None:
                grok_step = step
            history.append({'step': step, 'test_acc': acc,
                            'train_acc': train_acc})

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
                g1 = last_gate.get('gate1', 0) if last_gate else 0
                g2 = last_gate.get('gate2_min', 0) if last_gate else 0
                dh = last_gate.get('dh_norms', []) if last_gate else []
                eps_c = last_gate.get('eps_a_norms_clamped', []) if last_gate else []
                fr_str = ' '.join(f'{f:.2f}' for f in fr)
                dh_str = ' '.join(f'{d:.4f}' for d in dh)
                eps_str = ' '.join(f'{e:.4f}' for e in eps_c)
                ew = model._eta_W_eff
                at = model._alpha_theta_eff
                print(f"  [s{seed}] step {step:5d}: test={acc:.3f} train={train_acc:.3f} "
                      f"best={best:.3f} G1={g1:.2f} G2={g2:+.2f} "
                      f"fr=[{fr_str}] dh=[{dh_str}] eps=[{eps_str}] "
                      f"\u03b7W={ew:.5f} \u03b1={at:.5f} [{elapsed:.0f}s]", flush=True)

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
        full_diag = model.infer(Xte[:64], to_onehot(Yte[:64], prime),
                                return_gates=True)
    final_gate = full_diag.get('gate_log', {})

    if final_gate:
        g1 = final_gate.get('gate1', 0)
        g1d = final_gate.get('gate1d', 0)
        g2min = final_gate.get('gate2_min', 0)
        g2mean = final_gate.get('gate2_mean', 0)
        eps_a = final_gate.get('eps_a_norms_clamped', [])
        eps_a_free = final_gate.get('eps_a_norms_free', [])
        dh = final_gate.get('dh_norms', [])
        hoy = final_gate.get('hoyer', [])
        fr = final_gate.get('firing_rates', [])
        energy = final_gate.get('energy', 0)
        print(f"  [s{seed}] FINAL CREDIT PROFILE:")
        print(f"    Gate1 = {g1:.3f}, Gate1d = {g1d:.3f}")
        print(f"    Gate2 (min/mean) = {g2min:.3f}/{g2mean:.3f}")
        print(f"    eps_a clamped = {[f'{e:.4f}' for e in eps_a]}")
        print(f"    eps_a free    = {[f'{e:.4f}' for e in eps_a_free]}")
        print(f"    dh norms      = {[f'{d:.4f}' for d in dh]}")
        print(f"    Hoyer         = {[f'{h:.2f}' for h in hoy]}")
        print(f"    Firing        = {[f'{f:.3f}' for f in fr]}")
        print(f"    Energy        = {energy:.4f}")

    # ── Compute headline metrics (§4.3/4.4) ──
    W = 5
    test_accs = [h['test_acc'] for h in history]
    if len(test_accs) >= W:
        window_avg = float(np.mean(test_accs[-W:]))
    else:
        window_avg = float(np.mean(test_accs)) if test_accs else 0.0

    checkpoint_best = float(max(test_accs)) if test_accs else 0.0
    sustained_grok = window_avg >= 0.90
    final_acc = history[-1]['test_acc'] if history else 0.0
    dt = time.time() - t0

    result = {
        'seed': seed,
        'L': L_LAMINAR, 'N': hidden, 'sheet_size': sheet_size,
        'p': prime, 'g': g, 'k_freq': k_freq, 'in_dim': in_dim,
        't_inf': T_INF, 'effective_T': T_INF,
        't_decay': t_decay,
        'steps': steps, 'n_train': n_train, 'n_test': n_test,
        'train_fraction': train_fraction,
        'final_test_acc': final_acc,
        'best_test_acc': best,
        'window_avg_acc': window_avg,        # §4.4 SOLE HEADLINE
        'checkpoint_best_acc': checkpoint_best,  # §4.3 DIAGNOSTIC ONLY
        'sustained_grok': sustained_grok,
        'grok_step': grok_step,
        'chance': 1.0 / prime, 'time': dt,
        'history': history,
        'gate_snapshots': gate_log_list,
        'final_gate': final_gate,
        'schedule_log': schedule_log,
    }
    return result


def parse_seeds(seed_str):
    """Parse '0-9' or '0 1 2 3' or '0-4,7,9'."""
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
        description='SPEC_P97_SCALING_WALL_v2.2 — triple-constraint scaling-wall sweep')
    ap.add_argument('--prime', type=int, required=True,
                    choices=[13, 29, 41, 53, 97],
                    help='Prime p for modular multiplication')
    ap.add_argument('--hidden', type=int, default=1536,
                    help='Hidden units per layer (N)')
    ap.add_argument('--sheet-size', type=int, default=40,
                    help='Cortical sheet size (P6: hidden <= sheet^2). D4 uses 46.')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--t-decay', type=int, default=1500,
                    help='C+D decay period. D2/D3/D4 use 5500 (SPEC v2.2 §4.4).')
    ap.add_argument('--arm', default=None,
                    choices=['D1', 'D2', 'D3', 'D4'],
                    help='p=97 budget arm preset (overrides steps/hidden/sheet/t_decay)')
    ap.add_argument('--seeds', type=str, default='0-9',
                    help="seed spec: '0-9' or '0 1 2' or '0-4,7'")
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--gate-every', type=int, default=500)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    prime = args.prime
    g = PRIMES_AND_GENERATORS[prime]
    k_freq = (prime - 1) // 2
    in_dim = 2 * (prime - 1)
    chance = 1.0 / prime
    seeds = parse_seeds(args.seeds)

    # ── Budget arm preset (SPEC v2.2 §4.1/§4.2) ──
    arm_tag = ''
    if args.arm:
        assert prime == 97, f"--arm only valid for p=97, got p={prime}"
        arm = BUDGET_ARMS[args.arm]
        args.steps = arm['steps']
        args.hidden = arm['hidden']
        args.sheet_size = arm['sheet_size']
        args.t_decay = arm['t_decay']
        arm_tag = f"_{args.arm}"

    hidden = args.hidden
    sheet_size = args.sheet_size
    t_decay = args.t_decay

    # ── Primitive-root guard (SPEC §4.3 BINDING) ──
    print(f"\n  Primitive-root guard: checking g={g} mod p={prime}...")
    check_primitive_root(g, prime)
    print(f"  \u2713 g={g} is primitive root mod {prime} (order={prime - 1})")

    # P6 legality check
    assert hidden <= sheet_size ** 2, (
        f"P6 VIOLATION: hidden_dim={hidden} > sheet_size^2={sheet_size ** 2}")
    # P7 check
    assert L_LAMINAR >= 2, f"P7 VIOLATION: L={L_LAMINAR} < 2"

    # Epoch confound (SPEC v2.2 §2.4)
    n_pairs = (prime - 1) ** 2
    n_train = int(n_pairs * 0.80)
    steps_per_epoch = max(1, (n_train + BATCH - 1) // BATCH)
    epochs = args.steps / steps_per_epoch

    # C_grad coverage (SPEC v2.2 §2.1, eq 8)
    e_unique = n_train * (1.0 - (1.0 - 1.0 / n_train) ** BATCH)
    rho = e_unique / BATCH

    # R_cap (SPEC v2.2 §2.1, eq 5b)
    r_cap = hidden / ((prime - 1) / 2.0) ** 2

    # Output paths
    if sys.platform == 'win32':
        out_dir = r"D:\PC-hermes\outputs"
    else:
        out_dir = os.environ.get('OUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)

    default_name = f'p97wall_v22_p{prime}_N{hidden}_T{t_decay}{arm_tag}.json'
    base_name = args.output or os.path.join(out_dir, default_name)
    progress_path = os.path.join(out_dir, f'p97wall_v22_p{prime}_N{hidden}_PROGRESS.json')

    # Config block
    config = {
        'spec': 'SPEC_P97_SCALING_WALL_v2.2',
        'task_id': 't_92034a55',
        'engine': 'ablation_cortex_v14_1_opt.py AblationCortexOpt (FROZEN C2 @ 87c7250)',
        'arm': args.arm,
        'p': prime, 'g': g, 'k_freq': k_freq, 'in_dim': in_dim,
        'hidden_per_layer': hidden, 'sheet_size': sheet_size,
        't_inf': T_INF, 'effective_T': T_INF,
        'n_layers': L_LAMINAR,
        'steps': args.steps,
        'seeds': seeds,
        'train_fraction': 0.80,
        'batch': BATCH,
        'eta_W': ETA_W, 'eta_out': ETA_OUT,
        'gamma_W': GAMMA_W, 'gamma_alpha': GAMMA_ALPHA, 'T_decay': t_decay,
        'alpha_theta_0': 0.05,
        'k_conn': 8, 'target_rate': 0.10,
        'chance': chance,
        'eval_every': args.eval_every, 'gate_every': args.gate_every,
        'eval_window_W': 5,
        'headline_metric': 'window_avg_acc (W=5, §4.4)',
        'grok_threshold': 0.90,
        # Triple-constraint analysis (SPEC v2.2 §2.1)
        'n_pairs': n_pairs, 'n_train': n_train,
        'steps_per_epoch': steps_per_epoch,
        'epochs_at_steps': round(epochs, 1),
        'e_unique_per_batch': round(e_unique, 1),
        'rho_coverage': round(rho, 3),
        'rho_min': 0.90,
        'c_grad_pass': rho >= 0.90,
        'r_cap': round(r_cap, 2),
        'K_epoch_sustained': 176.5,
        'K_epoch_transition': 88.0,
        'c_epoch_pass': epochs >= 176.5,
    }

    print(f"\n{'=' * 70}")
    print(f"SPEC_P97_SCALING_WALL_v2.2 — Triple-Constraint Scaling-Wall Sweep")
    print(f"{'=' * 70}")
    print(f"  Prime:      p={prime} (g={g}, K_freq={k_freq}, IN_DIM={in_dim})")
    print(f"  Hidden:     N={hidden}/layer, sheet={sheet_size} ({sheet_size}x{sheet_size}={sheet_size ** 2})")
    print(f"  Depth:      L={L_LAMINAR}")
    if args.arm:
        print(f"  Arm:        {args.arm} (steps={args.steps}, T_decay={t_decay})")
    print(f"  Stabiliz.:  C+D (\u03b3_W={GAMMA_W}, \u03b3_\u03b1={GAMMA_ALPHA}, T_decay={t_decay})")
    print(f"  Steps:      {args.steps}")
    print(f"  Seeds:      {seeds}")
    print(f"  Batch:      {BATCH}")
    print(f"  Chance:     {chance:.4f}")
    print(f"  Device:     {DEVICE}")
    print(f"  Headline:   window_avg_acc (W=5, threshold 0.90)")
    print(f"  --- Triple-constraint analysis (§2.1) ---")
    print(f"  n_pairs:    {n_pairs}")
    print(f"  n_train:    {n_train}")
    print(f"  steps/epoch: {steps_per_epoch}")
    print(f"  epochs:     {epochs:.1f}")
    print(f"  E[unique]:  {e_unique:.1f} / batch")
    print(f"  rho:        {rho:.3f} ({'PASS' if rho >= 0.90 else 'FAIL'} \u2265 0.90)")
    print(f"  R_cap:      {r_cap:.2f}")
    print(f"  Output:     {base_name}")
    print(f"{'=' * 70}\n", flush=True)

    results = []

    for seed in seeds:
        print(f"--- Seed {seed} (p={prime}, N={hidden}, T_decay={t_decay}) ---")
        r = run_single_seed(prime, g, k_freq, in_dim, hidden, sheet_size, t_decay,
                            seed, args.steps,
                            eval_every=args.eval_every, gate_every=args.gate_every,
                            train_fraction=0.80, progress_path=progress_path)
        verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                   "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
        print(f"  => WINDOW_AVG={r['window_avg_acc']:.4f} "
              f"FINAL={r['final_test_acc']:.4f} BEST={r['best_test_acc']:.4f} "
              f"| {verdict} | {r['time']:.0f}s\n", flush=True)
        results.append(r)

        # Write cumulative results after EACH seed
        try:
            windows = [x['window_avg_acc'] for x in results]
            finals = [x['final_test_acc'] for x in results]
            bests = [x['best_test_acc'] for x in results]
            n_window_ge90 = sum(1 for w in windows if w >= 0.90)
            n_best_ge90 = sum(1 for b in bests if b >= 0.90)
            summary = {
                'n_seeds': len(results),
                'window_avg_mean': float(np.mean(windows)),
                'window_avg_median': float(np.median(windows)),
                'window_avg_std': float(np.std(windows)),
                'final_mean': float(np.mean(finals)),
                'final_std': float(np.std(finals)),
                'best_mean': float(np.mean(bests)),
                'best_std': float(np.std(bests)),
                'n_window_avg_ge_090': n_window_ge90,
                'n_best_ge_090': n_best_ge90,
                'grok_rate_window': f"{n_window_ge90}/{len(results)}",
                'chance': chance,
            }
            out_data = {'config': config, 'results': results, 'summary': summary}
            with open(base_name, 'w') as f:
                json.dump(out_data, f, indent=1, default=str)
        except Exception as e:
            print(f"  WARNING: failed to save results: {e}")

    # ── Summary ──
    windows = [r['window_avg_acc'] for r in results]
    finals = [r['final_test_acc'] for r in results]
    bests = [r['best_test_acc'] for r in results]
    n_win = sum(1 for w in windows if w >= 0.90)
    n_best = sum(1 for b in bests if b >= 0.90)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: p={prime}, N={hidden} (L={L_LAMINAR}, "
          f"{len(results)} seeds, {args.steps} steps)")
    print(f"{'=' * 70}")
    print(f"  *** WINDOW-AVG (W=5) — SOLE HEADLINE (§4.4) ***")
    print(f"  Window-avg: median={np.median(windows):.3f} "
          f"mean={np.mean(windows):.3f} \u00b1{np.std(windows):.3f}")
    print(f"  Grok (window>=0.90): {n_win}/{len(results)}")
    print(f"  ---")
    print(f"  Final acc:  median={np.median(finals):.3f} mean={np.mean(finals):.3f}"
          f" \u00b1{np.std(finals):.3f}")
    print(f"  Best acc:   median={np.median(bests):.3f} mean={np.mean(bests):.3f}"
          f" \u00b1{np.std(bests):.3f}  [DIAGNOSTIC ONLY]")
    print(f"  Best>=0.90: {n_best}/{len(results)}  [DIAGNOSTIC ONLY]")
    print(f"  Per-seed window: {[f'{a:.3f}' for a in windows]}")
    print(f"  Per-seed final:  {[f'{a:.3f}' for a in finals]}")
    print(f"  Chance: {chance:.4f}")
    print(f"  Epochs: {epochs:.1f} (steps/epoch={steps_per_epoch})")
    print(f"  rho: {rho:.3f} | R_cap: {r_cap:.2f} | T_decay: {t_decay}")

    if n_win >= 8:
        verdict = "GROK (sustained)"
    elif n_win >= 4:
        verdict = "PARTIAL"
    elif n_best >= 1:
        verdict = "TRANSIENT"
    else:
        verdict = "CHANCE"
    print(f"  VERDICT: {verdict}")

    # Per-area epsilon summary (THE measurement, §4 criterion 4)
    if results and results[0].get('final_gate'):
        print(f"\n  PER-AREA ERROR SURVIVAL (final gate, averaged over seeds):")
        eps_keys = ['eps_a_norms_clamped', 'eps_a_norms_free', 'dh_norms',
                    'lam_norms', 'firing_rates', 'hoyer', 'threshold_norms']
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
        print()

    print(f"  Results: {base_name}")


if __name__ == '__main__':
    main()
