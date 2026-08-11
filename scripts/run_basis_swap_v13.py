#!/usr/bin/env python3
"""run_basis_swap_v13.py -- SPEC_BASIS_SWAP_AND_STABILIZATION_v1.3 (task t_16f0236f).

The decisive test: E_mult (multiplicative-character basis) + C+D stabilization
(gamma_W=0.5, gamma_alpha=0.25, T_decay=1500) at (L=6, N=1536, T=10, 10 seeds).

SPEC v1.3 §0: the matched ADD-vs-MULT experiment showed E_add→MULT = 0/10 grok.
This spec swaps the input basis to E_mult (group-theoretic discrete-log encoder)
which makes the dendritic product layer linearize modular multiplication (§2.1).
The C+D stabilization (§4.5/§4.5b) addresses the end-of-run decay seen in both
ADD (6/10) and composed-mult (4/10) arms.

ACCEPTANCE CRITERIA (binding, from card t_16f0236f):
1. PASSES only if it GROKS: held-out acc → ~1.0, beating strong control, improving with data.
2. dh/eps survival are NECESSARY but NOT SUFFICIENT.
3. 10+ seeds, equal step counts, no hardcoded seeds.
4. Report per-area error (ε_l) survival — THE measurement.

HEADLINE METRIC: eval-window average (W=5, §4.4) — the mean test accuracy over
the last 5 evaluation steps.  This is the SOLE headline (§4.3: checkpoint-at-best
is diagnostic-only because it selects on the test set).

Engine: ablation_cortex_v14_1.py AblationCortex (via AblationCortexOpt,
math-identical speed subclass) — C+D schedule params added per §4.5/§4.5b.

# COMPUTE LANE: containerized GPU slot system.

Usage:
  # Main experiment: E_mult → MULT with C+D stabilization
  python run_basis_swap_v13.py --task mult-stab --seeds 0-9 --steps 3000

  # Data-scaling curve (3 seeds × 4 fractions)
  python run_basis_swap_v13.py --task mult-stab --seeds 0-2 --steps 3000 --train-fractions 0.25 0.50 0.80

  # No-stabilization control (C+D OFF — shows the decay)
  python run_basis_swap_v13.py --task mult-nostab --seeds 0-9 --steps 3000
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

# ── Constants ──
P_MOD = 53
K_FREQ = 26
IN_DIM = 4 * K_FREQ  # 104
CHANCE = 1.0 / P_MOD
G_PRIM = 2  # primitive root mod 53 (ord(2)=52=P-1, verified)

# Engine hyperparameters (EXACT proven config from l6eqcap_T10_10seeds)
HIDDEN = 1536
SHEET_SIZE = 40
T_INF = 10
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001
BATCH = 128
L_LAMINAR = 6

# SPEC v1.3 §4.5c: C+D stabilization protocol
GAMMA_W = 0.5       # Mechanism C: weight step-decay
GAMMA_ALPHA = 0.25  # Mechanism D: threshold EMA decay (MUST differ from gamma_W)
T_DECAY = 1500      # Step-decay period


# ================================================================
# Discrete log table for E_mult
# ================================================================
def build_dlog_table(p=53, g=2):
    """Build discrete log table: dlog[a] = log_g(a) mod (p-1) for a in {1..p-1}."""
    dlog = {}
    val = 1
    for exp in range(p - 1):
        dlog[val] = exp
        val = (val * g) % p
    assert len(dlog) == p - 1, f"dlog table incomplete: {len(dlog)}/{p-1}"
    return dlog


DLOG = build_dlog_table(P_MOD, G_PRIM)


# ================================================================
# Data generation
# ================================================================
def make_mult_data(n_train=2163, n_test=541, seed=42, train_fraction=0.80):
    """E_mult: multiplicative character-basis features for modular multiplication.

    For each (a,b), a,b in {1..52}: target = (a*b) mod 53.
    Excludes a=0, b=0 (no discrete log). (P-1)^2 = 2704 pairs.
    Phase function: omega_k(a) = 2*pi*k*log_g(a)/(P-1).

    SPEC §2.1: period is P-1, NOT P.
    SPEC §5: train_fraction controls the data-scaling curve.
    """
    rng = np.random.RandomState(seed)
    vals = np.arange(1, P_MOD)  # [1, 2, ..., 52]
    aa = np.repeat(vals, P_MOD - 1)
    bb = np.tile(vals, P_MOD - 1)
    cc = (aa * bb) % P_MOD

    # Discrete logs: log_g(a) for each a
    dlog_a = np.array([DLOG[a] for a in aa], dtype=np.float64)
    dlog_b = np.array([DLOG[b] for b in bb], dtype=np.float64)

    freqs = np.arange(1, K_FREQ + 1, dtype=np.float64)
    # omega_k(a) = 2*pi*k*log_g(a)/(P-1)  [P-1 in denominator, NOT P!]
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


def make_add_data(n_train=2247, n_test=562, seed=42, train_fraction=0.80):
    """E_add: additive Fourier features for modular addition (sanity control).

    IDENTICAL to proven l6eqcap make_data.
    P^2 = 2809 pairs (a,b in {0..52}), target = (a+b) mod P.
    """
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
    total = len(Y)
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
def make_engine(seed, n_layers, stabilization=True):
    """Create AblationCortexOpt with the EXACT proven config + C+D schedule.

    SPEC v1.3 §4.5c: stabilization=True activates C+D (gamma_W=0.5,
    gamma_alpha=0.25, T_decay=1500). stabilization=False = proven baseline
    (no schedule) — used as the no-stab control arm.
    """
    kwargs = dict(
        in_dim=IN_DIM, hidden_dim=HIDDEN, out_dim=P_MOD, n_layers=n_layers,
        sheet_size=SHEET_SIZE,
        target_rate=0.10, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
    )
    if stabilization:
        # SPEC v1.3 §4.5c: C+D with MISMATCHED gamma (gamma_alpha != gamma_W)
        kwargs['gamma_W'] = GAMMA_W
        kwargs['gamma_alpha'] = GAMMA_ALPHA
        kwargs['T_decay'] = T_DECAY
        kwargs['alpha_theta_0'] = 0.05  # the hardcoded alpha from v14.1
    return AblationCortexOpt(**kwargs)


# ================================================================
# Single-seed runner
# ================================================================
def run_single_seed(seed, task, steps, eval_every=100, gate_every=500,
                    train_fraction=0.80, progress_path=None,
                    stabilization=True):
    """Run one seed of the basis-swap + stabilization experiment.

    task: 'mult-stab', 'mult-nostab', 'add-sanity'
    """
    t0 = time.time()
    col_seed = seed * 100 + 0  # matches proven l6eqcap seed mapping

    # ── Data generation ──
    if task.startswith('mult'):
        Xtr, Ytr, Xte, Yte = make_mult_data(
            seed=42, train_fraction=train_fraction)
        task_label = 'mult'
    elif task.startswith('add'):
        Xtr, Ytr, Xte, Yte = make_add_data(
            seed=42, train_fraction=train_fraction)
        task_label = 'add'
    else:
        raise ValueError(f"Unknown task: {task}")

    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)

    n_train = len(Xtr)
    n_test = len(Xte)

    # ── Engine ──
    model = make_engine(seed=col_seed, n_layers=L_LAMINAR,
                        stabilization=stabilization)
    model.calibrate_thresholds(Xtr[:200])

    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0
    gate_log_list = []
    schedule_log = []  # track C+D effective rates

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

            # Track C+D schedule state
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
                      f"ηW={ew:.5f} α={at:.5f} [{elapsed:.0f}s]", flush=True)

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
        full_diag = model.infer(Xte[:64], to_onehot(Yte[:64], P_MOD),
                                return_gates=True)
    final_gate = full_diag.get('gate_log', {})

    if final_gate:
        g1 = final_gate.get('gate1', 0)
        g1d = final_gate.get('gate1d', 0)
        g2min = final_gate.get('gate2_min', 0)
        g2mean = final_gate.get('gate2_mean', 0)
        d_dep = final_gate.get('d_dependent_contrastive', [])
        cbias = final_gate.get('contrastive_bias_part', [])
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

    # ── Compute headline metrics ──
    # §4.4: eval-window average (W=5) — SOLE headline
    W = 5
    test_accs = [h['test_acc'] for h in history]
    if len(test_accs) >= W:
        window_avg = float(np.mean(test_accs[-W:]))
    else:
        window_avg = float(np.mean(test_accs)) if test_accs else 0.0

    # §4.3: checkpoint-at-best — DIAGNOSTIC ONLY
    checkpoint_best = float(max(test_accs)) if test_accs else 0.0

    # Sustained grok: did it reach >=0.9 and STAY there in the window?
    sustained_grok = window_avg >= 0.90

    final_acc = history[-1]['test_acc'] if history else 0.0
    dt = time.time() - t0
    result = {
        'seed': seed, 'task': task, 'task_label': task_label,
        'L': L_LAMINAR, 'N': HIDDEN, 'sheet_size': SHEET_SIZE,
        't_inf': T_INF, 'effective_T': T_INF,
        'steps': steps, 'n_train': n_train, 'n_test': n_test,
        'train_fraction': train_fraction,
        'stabilization': stabilization,
        'gamma_W': GAMMA_W if stabilization else None,
        'gamma_alpha': GAMMA_ALPHA if stabilization else None,
        'T_decay': T_DECAY if stabilization else None,
        'final_test_acc': final_acc,
        'best_test_acc': best,
        'window_avg_acc': window_avg,        # §4.4 SOLE HEADLINE
        'checkpoint_best_acc': checkpoint_best,  # §4.3 DIAGNOSTIC ONLY
        'sustained_grok': sustained_grok,
        'grok_step': grok_step,
        'chance': CHANCE, 'time': dt,
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
    global HIDDEN, SHEET_SIZE, T_INF
    ap = argparse.ArgumentParser(
        description='SPEC_BASIS_SWAP_AND_STABILIZATION_v1.3 — basis swap + C+D stabilization')
    ap.add_argument('--task', default='mult-stab',
                    choices=['mult-stab', 'mult-nostab', 'add-sanity'],
                    help='mult-stab: E_mult→MULT with C+D; '
                         'mult-nostab: E_mult→MULT no schedule (control); '
                         'add-sanity: E_add→ADD (sanity)')
    ap.add_argument('--seeds', type=str, default='0-9',
                    help="seed spec: '0-9' or '0 1 2' or '0-4,7'")
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--gate_every', type=int, default=500)
    ap.add_argument('--output', default=None)
    ap.add_argument('--hidden-per-layer', type=int, default=None)
    ap.add_argument('--sheet-size', type=int, default=None)
    ap.add_argument('--t-inf', type=int, default=None)
    ap.add_argument('--train-fractions', type=float, nargs='+', default=None,
                    help='Data-scaling curve: run at these train fractions '
                         '(e.g. 0.25 0.50 0.80). Default: 0.80 only.')
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    if args.hidden_per_layer is not None:
        HIDDEN = args.hidden_per_layer
    if args.sheet_size is not None:
        SHEET_SIZE = args.sheet_size
    if args.t_inf is not None:
        T_INF = args.t_inf

    train_fractions = args.train_fractions if args.train_fractions else [0.80]

    # P6 legality check
    assert HIDDEN <= SHEET_SIZE ** 2, (
        f"P6 VIOLATION: hidden_dim={HIDDEN} > sheet_size^2={SHEET_SIZE**2}")

    stabilization = args.task.endswith('-stab')
    # P7 check
    assert L_LAMINAR >= 2, f"P7 VIOLATION: L={L_LAMINAR} < 2"

    # Output paths
    if sys.platform == 'win32':
        out_dir = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    else:
        out_dir = os.environ.get('OUT_DIR', os.path.join(os.path.dirname(__file__), '..', 'outputs'))
    os.makedirs(out_dir, exist_ok=True)

    stab_tag = 'stab' if stabilization else 'nostab'
    frac_tag = f"_frac{int(train_fractions[0]*100)}" if len(train_fractions) == 1 else "_sweep"
    default_name = f'basis_swap_v13_{args.task}{frac_tag}_L{L_LAMINAR}_N{HIDDEN}_T{T_INF}.json'
    base_name = args.output or os.path.join(out_dir, default_name)
    progress_path = os.path.join(out_dir, f'basis_swap_v13_{args.task}_PROGRESS.json')

    # Config block
    config = {
        'spec': 'SPEC_BASIS_SWAP_AND_STABILIZATION_v1.3',
        'task': args.task,
        'engine': 'ablation_cortex_v14_1.py AblationCortex (via AblationCortexOpt)',
        'stabilization': stabilization,
        'gamma_W': GAMMA_W if stabilization else None,
        'gamma_alpha': GAMMA_ALPHA if stabilization else None,
        'T_decay': T_DECAY if stabilization else None,
        'alpha_theta_0': 0.05,
        'P': P_MOD, 'g': G_PRIM, 'K_freq': K_FREQ, 'in_dim': IN_DIM,
        'hidden_per_layer': HIDDEN, 'sheet_size': SHEET_SIZE,
        't_inf': T_INF, 'effective_T': T_INF,
        'n_layers': L_LAMINAR,
        'steps': args.steps,
        'seeds': seeds,
        'train_fractions': train_fractions,
        'batch': BATCH,
        'eta_W': ETA_W, 'eta_out': ETA_OUT, 'eta_theta': ETA_THETA,
        'k_conn': 8, 'target_rate': 0.10,
        'chance': CHANCE,
        'eval_every': args.eval_every, 'gate_every': args.gate_every,
        'eval_window_W': 5,
        'headline_metric': 'window_avg_acc (W=5, §4.4)',
    }

    print(f"\n{'='*70}")
    print(f"SPEC_BASIS_SWAP_AND_STABILIZATION_v1.3 — Basis Swap + C+D Stabilization")
    print(f"{'='*70}")
    print(f"  Task:       {args.task}")
    print(f"  Encoder:    {'E_mult (discrete-log)' if 'mult' in args.task else 'E_add (additive Fourier)'}")
    print(f"  Stabiliz.:  {stabilization} (γ_W={GAMMA_W if stabilization else 'N/A'}, "
          f"γ_α={GAMMA_ALPHA if stabilization else 'N/A'}, T_decay={T_DECAY if stabilization else 'N/A'})")
    print(f"  L={L_LAMINAR}, N={HIDDEN}/layer, T_inf={T_INF}, sheet={SHEET_SIZE}")
    print(f"  Steps:      {args.steps}")
    print(f"  Seeds:      {seeds}")
    print(f"  Frac:       {train_fractions}")
    print(f"  Batch:      {BATCH}")
    print(f"  Chance:     {CHANCE:.4f}")
    print(f"  Device:     {DEVICE}")
    print(f"  Headline:   window_avg_acc (W=5, §4.4)")
    print(f"  Output:     {base_name}")
    print(f"{'='*70}\n", flush=True)

    all_results = {}  # keyed by train_fraction

    for frac in train_fractions:
        if len(train_fractions) > 1:
            print(f"\n{'='*70}")
            print(f"  TRAIN FRACTION = {frac:.2f} (n_train={int(2704*frac) if 'mult' in args.task else int(2809*frac)})")
            print(f"{'='*70}")

        results = []
        for seed in seeds:
            print(f"--- Seed {seed} ({args.task}, frac={frac:.2f}) ---")
            r = run_single_seed(
                seed, args.task, args.steps,
                eval_every=args.eval_every, gate_every=args.gate_every,
                train_fraction=frac, progress_path=progress_path,
                stabilization=stabilization)
            verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                       "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
            print(f"  => WINDOW_AVG={r['window_avg_acc']:.4f} "
                  f"FINAL={r['final_test_acc']:.4f} BEST={r['best_test_acc']:.4f} "
                  f"| {verdict} | {r['time']:.0f}s\n", flush=True)
            results.append(r)

            # Write cumulative results after EACH seed
            try:
                finals = [x['final_test_acc'] for x in results]
                windows = [x['window_avg_acc'] for x in results]
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
                    'chance': CHANCE,
                }
                all_results[frac] = {'config': config, 'results': results,
                                     'summary': summary}
                # Write single-fraction or multi-fraction output
                out_data = all_results if len(train_fractions) > 1 else all_results[frac]
                with open(base_name, 'w') as f:
                    json.dump(out_data, f, indent=1, default=str)
            except Exception as e:
                print(f"  WARNING: failed to save results: {e}")

    # ── Summary ──
    for frac, data in all_results.items():
        results = data['results']
        windows = [r['window_avg_acc'] for r in results]
        finals = [r['final_test_acc'] for r in results]
        bests = [r['best_test_acc'] for r in results]
        n_win = sum(1 for w in windows if w >= 0.90)
        n_best = sum(1 for b in bests if b >= 0.90)

        print(f"\n{'='*70}")
        print(f"  SUMMARY: {args.task} frac={frac:.2f} (L={L_LAMINAR}, "
              f"{len(results)} seeds, {args.steps} steps)")
        print(f"{'='*70}")
        print(f"  *** WINDOW-AVG (W=5) — SOLE HEADLINE (§4.4) ***")
        print(f"  Window-avg: median={np.median(windows):.3f} "
              f"mean={np.mean(windows):.3f} ±{np.std(windows):.3f}")
        print(f"  Grok (window>=0.90): {n_win}/{len(results)}")
        print(f"  ---")
        print(f"  Final acc:  median={np.median(finals):.3f} mean={np.mean(finals):.3f}"
              f" ±{np.std(finals):.3f}")
        print(f"  Best acc:   median={np.median(bests):.3f} mean={np.mean(bests):.3f}"
              f" ±{np.std(bests):.3f}  [DIAGNOSTIC ONLY — §4.3]")
        print(f"  Best>=0.90: {n_best}/{len(results)}  [DIAGNOSTIC ONLY]")
        print(f"  Per-seed window: {[f'{a:.3f}' for a in windows]}")
        print(f"  Per-seed final:  {[f'{a:.3f}' for a in finals]}")
        print(f"  Chance: {CHANCE:.4f}")

        # Verdict
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

    # Data-scaling curve summary
    if len(train_fractions) > 1:
        print(f"\n{'='*70}")
        print(f"  DATA-SCALING CURVE (does grokking improve with data?)")
        print(f"{'='*70}")
        print(f"  {'Frac':>6s} {'n_train':>8s} {'window_med':>11s} {'window_mean':>12s} "
              f"{'grok_rate':>10s}")
        for frac in sorted(all_results.keys()):
            data = all_results[frac]
            s = data['summary']
            n_tr = s.get('n_train', int((2704 if 'mult' in args.task else 2809) * frac))
            print(f"  {frac:6.2f} {n_tr:8d} {s['window_avg_median']:11.3f} "
                  f"{s['window_avg_mean']:12.3f} {s['grok_rate_window']:>10s}")
        print()

    print(f"  Results: {base_name}")


if __name__ == '__main__':
    main()
