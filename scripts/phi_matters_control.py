#!/usr/bin/env python3 -u
"""phi_matters_control.py -- PHI-MATTERS CONTROL at scale (task t_aee4395a).

THE REVIEWER-EXISTENTIAL CONTROL: does the EC alphabet actually matter on the
real L=6 engine, or is any distributed code of similar statistics sufficient?

Banked toy result: shuffled φ = real φ (0.753 vs 0.742) on CFG. If this holds
at scale on the real engine, the ENTIRE deliver-then-select EC story collapses
— the EC characters are decorative.

4 ARMS (only input encoding changes; engine is IDENTICAL across all arms):
  (1) real_phi:    additive Fourier characters ω_k(a)=2πka/P  [CORRECT basis]
  (2) shuffled_phi: same matrix, rows permuted by fixed π      [structure destroyed]
  (3) random_phi:  iid uniform[-1,1], same shape               [no structure]
  (4) onehot:      identity matrix (no EC)                      [no frequency]

TASK: mod-add (a+b) mod 53, p=53
ENGINE: AblationCortexOpt L=6, N=1536, sheet=40, T=10, C+D stabilization
  (proven config from run_basis_swap_v13.py: 9-10/10 grok on mod-mult)
STEPS: 2000, SEEDS: 10, BATCH: 128

PREDICTION if EC matters: real φ >> shuffled ≥ random ≈ onehot
PREDICTION if EC decorative: real ≈ shuffled >> random ≈ onehot

Constitution: P1-P8 all PASS on this engine (validated in living_ec_exp0_real).
COMPUTE: 3060 container via run_slot.sh + higgs_venv python.

Usage:
  python phi_matters_control.py --arms real shuffled random onehot \
      --seeds 10 --steps 2000
  python phi_matters_control.py --arms real --seeds 2 --steps 200  # smoke
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

# ================================================================
# Constants
# ================================================================
P_MOD = 53
K_FREQ = 26
D_CHAR = 2 * K_FREQ  # 52 (cos/sin pairs per element)
IN_DIM_FOURIER = 2 * D_CHAR  # 104 (two operands interleaved)
IN_DIM_ONEHOT = 2 * P_MOD  # 106 (one-hot for each operand)
CHANCE = 1.0 / P_MOD

# Engine hyperparameters (EXACT proven config from run_basis_swap_v13.py)
HIDDEN = 1536
SHEET_SIZE = 40
T_INF = 10
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001
BATCH = 128
L_LAMINAR = 6

# C+D stabilization (proven 9/10 grok)
GAMMA_W = 0.5
GAMMA_ALPHA = 0.25
T_DECAY = 1500


# ================================================================
# EC code construction — the 4 arms
# ================================================================
def build_real_phi():
    """Additive Fourier character basis for Z_53.

    W_enc[a] = [cos(2πka/P), sin(2πka/P)] for k=1..26, a=0..52.
    This is the CORRECT character basis for modular addition.
    Shape: [P, D_CHAR] = [53, 52].
    """
    W = np.zeros((P_MOD, D_CHAR), dtype=np.float32)
    freqs = np.arange(1, K_FREQ + 1, dtype=np.float32)
    for a in range(P_MOD):
        angles = 2.0 * np.pi * freqs * a / P_MOD
        W[a, 0::2] = np.cos(angles)
        W[a, 1::2] = np.sin(angles)
    return W


def build_shuffled_phi(perm_seed=42):
    """Same values as real φ, but rows permuted by a FIXED permutation.

    Element a gets the encoding of element π(a). This destroys the
    element↔character correspondence while preserving the SET of encoding
    vectors (same marginal statistics).

    The permutation is FIXED across all cortex seeds (perm_seed=42) so the
    only source of variance is cortex initialization, not the permutation.
    """
    W_real = build_real_phi()
    rng = np.random.RandomState(perm_seed)
    perm = rng.permutation(P_MOD)
    W_shuffled = W_real[perm].copy()
    return W_shuffled


def build_random_phi(seed=123):
    """iid uniform[-1,1] noise, same shape as real φ.

    No algebraic structure whatsoever. Same marginal scale as real φ
    (Fourier features are in [-1,1]).
    """
    rng = np.random.RandomState(seed)
    W = rng.uniform(-1.0, 1.0, size=(P_MOD, D_CHAR)).astype(np.float32)
    return W


def build_onehot_phi():
    """Identity matrix — one-hot encoding, no EC, no frequency structure.

    Shape: [P, P] = [53, 53].
    """
    return np.eye(P_MOD, dtype=np.float32)


# ================================================================
# Data generation — EC code → cortex input
# ================================================================
def make_modadd_data_fourier(W_enc, train_fraction=0.80, seed=42):
    """Modular addition data using a Fourier-style EC code.

    For each (a,b), a,b in {0..52}: target = (a+b) mod 53.
    EC encoding: phi(a) = W_enc[a], phi(b) = W_enc[b].
    Interleaved format: [cos_a, sin_a, cos_b, sin_b] × 26 freqs.
    IN_DIM = 104.

    Works for real_phi, shuffled_phi, random_phi (all D_CHAR=52).
    """
    rng = np.random.RandomState(seed)
    aa = np.repeat(np.arange(P_MOD), P_MOD)
    bb = np.tile(np.arange(P_MOD), P_MOD)
    cc = (aa + bb) % P_MOD

    phi_a = W_enc[aa]  # [P*P, D_CHAR]
    phi_b = W_enc[bb]  # [P*P, D_CHAR]

    X = np.empty((P_MOD * P_MOD, IN_DIM_FOURIER), dtype=np.float32)
    X[:, 0::4] = phi_a[:, 0::2]  # cos_a at each freq
    X[:, 1::4] = phi_a[:, 1::2]  # sin_a at each freq
    X[:, 2::4] = phi_b[:, 0::2]  # cos_b at each freq
    X[:, 3::4] = phi_b[:, 1::2]  # sin_b at each freq

    Y = cc.astype(np.int64)
    total = len(Y)
    n_tr = int(total * train_fraction)
    perm = rng.permutation(total)
    Xtr, Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
    Xte, Yte = X[perm[n_tr:]], Y[perm[n_tr:]]
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def make_modadd_data_onehot(train_fraction=0.80, seed=42):
    """Modular addition data with one-hot encoding (no EC).

    For each (a,b): X = [onehot(a), onehot(b)], IN_DIM=106.
    """
    rng = np.random.RandomState(seed)
    aa = np.repeat(np.arange(P_MOD), P_MOD)
    bb = np.tile(np.arange(P_MOD), P_MOD)
    cc = (aa + bb) % P_MOD

    n = P_MOD * P_MOD
    X = np.zeros((n, IN_DIM_ONEHOT), dtype=np.float32)
    X[np.arange(n), aa] = 1.0
    X[np.arange(n), P_MOD + bb] = 1.0

    Y = cc.astype(np.int64)
    total = len(Y)
    n_tr = int(total * train_fraction)
    perm = rng.permutation(total)
    Xtr, Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
    Xte, Yte = X[perm[n_tr:]], Y[perm[n_tr:]]
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def to_onehot(Y, n_classes=P_MOD):
    Yoh = torch.zeros(len(Y), n_classes, device=DEVICE)
    Yoh[torch.arange(len(Y)), Y] = 1.0
    return Yoh


# ================================================================
# Engine construction (EXACT proven config from run_basis_swap_v13.py)
# ================================================================
def make_engine(seed, n_layers=L_LAMINAR, in_dim=IN_DIM_FOURIER):
    """Create AblationCortexOpt with the EXACT proven config + C+D schedule."""
    return AblationCortexOpt(
        in_dim=in_dim, hidden_dim=HIDDEN, out_dim=P_MOD, n_layers=n_layers,
        sheet_size=SHEET_SIZE,
        target_rate=0.10, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
        # C+D stabilization (proven 9/10 grok)
        gamma_W=GAMMA_W, gamma_alpha=GAMMA_ALPHA, T_decay=T_DECAY,
        alpha_theta_0=0.05,
    )


# ================================================================
# Single-seed runner
# ================================================================
def run_single_seed(seed, arm_name, W_enc, is_onehot, steps=2000,
                    eval_every=100, gate_every=500, verbose=True):
    """Run one seed: EC code → REAL L=6 cortex grok test (mod-add)."""
    t0 = time.time()
    col_seed = seed * 100 + 0  # matches proven seed mapping

    if is_onehot:
        Xtr, Ytr, Xte, Yte = make_modadd_data_onehot(train_fraction=0.80, seed=42)
        in_dim = IN_DIM_ONEHOT
    else:
        Xtr, Ytr, Xte, Yte = make_modadd_data_fourier(W_enc, train_fraction=0.80, seed=42)
        in_dim = IN_DIM_FOURIER

    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)

    n_train = len(Xtr)

    model = make_engine(seed=col_seed, n_layers=L_LAMINAR, in_dim=in_dim)
    model.calibrate_thresholds(Xtr[:200])

    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0

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

            if verbose and (step % 500 == 0 or step == eval_every or step == steps):
                elapsed = time.time() - t0
                fr = last_gate.get('firing_rates', []) if last_gate else []
                g1 = last_gate.get('gate1', 0) if last_gate else 0
                g2 = last_gate.get('gate2_min', 0) if last_gate else 0
                dh = last_gate.get('dh_norms', []) if last_gate else []
                fr_str = ' '.join(f'{f:.2f}' for f in fr)
                dh_str = ' '.join(f'{d:.4f}' for d in dh)
                print(f"    [{arm_name} s{seed}] step {step:5d}: test={acc:.3f} "
                      f"train={train_acc:.3f} best={best:.3f} G1={g1:.2f} "
                      f"G2={g2:+.2f} fr=[{fr_str}] dh=[{dh_str}] "
                      f"[{elapsed:.0f}s]", flush=True)

    # Final full gate diagnostic
    with torch.no_grad():
        full_diag = model.infer(Xte[:64], to_onehot(Yte[:64], P_MOD),
                                return_gates=True)
    final_gate = full_diag.get('gate_log', {})

    # Eval-window average (W=5)
    W = 5
    test_accs = [h['test_acc'] for h in history]
    if len(test_accs) >= W:
        window_avg = float(np.mean(test_accs[-W:]))
    else:
        window_avg = float(np.mean(test_accs)) if test_accs else 0.0

    final_acc = history[-1]['test_acc'] if history else 0.0
    dt = time.time() - t0

    return {
        'seed': seed, 'arm': arm_name,
        'final_test_acc': final_acc,
        'best_test_acc': best,
        'window_avg_acc': window_avg,
        'grok_step': grok_step,
        'sustained_grok': window_avg >= 0.90,
        'time': dt,
        'final_gate': final_gate,
    }


# ================================================================
# MAIN
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
        description='PHI-MATTERS CONTROL: does the EC alphabet matter at scale?')
    ap.add_argument('--arms', nargs='+',
                    default=['real', 'shuffled', 'random', 'onehot'],
                    choices=['real', 'shuffled', 'random', 'onehot'])
    ap.add_argument('--seeds', type=str, default='0-9')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    out_dir = os.environ.get('OUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        out_dir, 'phi_matters_control_results.json')
    progress_path = os.path.join(out_dir, 'phi_matters_control_PROGRESS.json')

    t_start = time.time()

    print("=" * 78)
    print("PHI-MATTERS CONTROL AT SCALE (task t_aee4395a)")
    print("Does the EC alphabet actually matter on the real L=6 engine?")
    print("=" * 78)
    print(f"  Engine:  AblationCortexOpt (L={L_LAMINAR}, N={HIDDEN}, "
          f"sheet={SHEET_SIZE}x{SHEET_SIZE})")
    print(f"           phi_norm hard gate 10% firing, k_conn=8, EP contrastive")
    print(f"           C+D stabilization: gamma_W={GAMMA_W}, "
          f"gamma_alpha={GAMMA_ALPHA}, T_decay={T_DECAY}")
    print(f"  Task:    mod-add (a+b) mod {P_MOD}")
    print(f"  T_inf:   {T_INF}")
    print(f"  Steps:   {args.steps}")
    print(f"  Seeds:   {seeds}")
    print(f"  Arms:    {args.arms}")
    print(f"  Chance:  {CHANCE:.4f}")
    print(f"  Device:  {DEVICE}")
    print(f"  Output:  {output_path}")
    print("=" * 78, flush=True)

    # ── Step 1: Build EC codes ──
    print("\n--- Step 1: Build EC codes ---")

    ec_codes = {}
    if 'real' in args.arms:
        W_real = build_real_phi()
        ec_codes['real'] = (W_real, False)
        print(f"  real_phi:     shape={W_real.shape}, "
              f"range=[{W_real.min():.3f}, {W_real.max():.3f}]")

    if 'shuffled' in args.arms:
        W_shuffled = build_shuffled_phi(perm_seed=42)
        ec_codes['shuffled'] = (W_shuffled, False)
        # Verify shuffled has same values as real
        W_real = build_real_phi()
        assert np.allclose(np.sort(W_real.ravel()), np.sort(W_shuffled.ravel())), \
            "Shuffled should have same values as real"
        print(f"  shuffled_phi: shape={W_shuffled.shape}, "
              f"range=[{W_shuffled.min():.3f}, {W_shuffled.max():.3f}] "
              f"(verified: same values, permuted rows)")

    if 'random' in args.arms:
        W_random = build_random_phi(seed=123)
        ec_codes['random'] = (W_random, False)
        print(f"  random_phi:   shape={W_random.shape}, "
              f"range=[{W_random.min():.3f}, {W_random.max():.3f}]")

    if 'onehot' in args.arms:
        W_onehot = build_onehot_phi()
        ec_codes['onehot'] = (W_onehot, True)
        print(f"  onehot:       shape={W_onehot.shape} "
              f"(identity, IN_DIM={IN_DIM_ONEHOT})")

    # ── Step 2: Run cortex grok test for each arm ──
    all_results = {}

    for arm_name in args.arms:
        W_enc, is_onehot = ec_codes[arm_name]
        print(f"\n{'='*60}")
        print(f"  ARM: {arm_name.upper()} ({len(seeds)} seeds × {args.steps} steps)")
        print(f"{'='*60}", flush=True)

        seed_results = []
        for seed in seeds:
            r = run_single_seed(
                seed, arm_name, W_enc, is_onehot,
                steps=args.steps, eval_every=args.eval_every,
                verbose=True)
            verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                       "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
            print(f"    => WINDOW={r['window_avg_acc']:.4f} "
                  f"FINAL={r['final_test_acc']:.4f} "
                  f"BEST={r['best_test_acc']:.4f} | {verdict} | "
                  f"{r['time']:.0f}s", flush=True)
            seed_results.append(r)

            # Update progress file
            try:
                prog = {
                    'arm': arm_name, 'seed': seed,
                    'window': r['window_avg_acc'],
                    'final': r['final_test_acc'],
                    'best': r['best_test_acc'],
                    'verdict': verdict,
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }
                with open(progress_path, 'w') as pf:
                    json.dump(prog, pf)
            except Exception:
                pass

        # Arm summary
        windows = [r['window_avg_acc'] for r in seed_results]
        bests = [r['best_test_acc'] for r in seed_results]
        n_grok_window = sum(1 for w in windows if w >= 0.90)
        n_grok_best = sum(1 for b in bests if b >= 0.90)

        all_results[arm_name] = {
            'grok_rate_window': n_grok_window,
            'grok_rate_best': n_grok_best,
            'n_seeds': len(seeds),
            'window_accs': windows,
            'best_accs': bests,
            'window_mean': float(np.mean(windows)),
            'window_std': float(np.std(windows)),
            'best_mean': float(np.mean(bests)),
            'best_std': float(np.std(bests)),
            'seed_results': [{k: v for k, v in r.items() if k != 'final_gate'}
                             for r in seed_results],
        }

        # Per-area diagnostics (average across seeds)
        avg_eps = np.mean([r['final_gate'].get('eps_a_norms_clamped', [0]*L_LAMINAR)
                           for r in seed_results if r.get('final_gate')], axis=0).tolist()
        avg_dh = np.mean([r['final_gate'].get('dh_norms', [0]*L_LAMINAR)
                          for r in seed_results if r.get('final_gate')], axis=0).tolist()
        avg_fr = np.mean([r['final_gate'].get('firing_rates', [0]*L_LAMINAR)
                          for r in seed_results if r.get('final_gate')], axis=0).tolist()
        all_results[arm_name]['per_area_avg'] = {
            'eps_a_norms_clamped': avg_eps,
            'dh_norms': avg_dh,
            'firing_rates': avg_fr,
        }

        print(f"\n  --- {arm_name.upper()} SUMMARY ---")
        print(f"  Grok rate (window≥0.9): {n_grok_window}/{len(seeds)}")
        print(f"  Grok rate (best≥0.9):   {n_grok_best}/{len(seeds)}")
        print(f"  Window mean: {np.mean(windows):.4f} ± {np.std(windows):.4f}")
        print(f"  Best mean:   {np.mean(bests):.4f} ± {np.std(bests):.4f}")
        print(f"  Windows: {[round(w,3) for w in windows]}")
        print(f"  Bests:   {[round(b,3) for b in bests]}", flush=True)

        # Save intermediate results after each arm
        try:
            output = {
                'config': {
                    'engine': 'AblationCortexOpt',
                    'L': L_LAMINAR, 'N': HIDDEN, 'sheet_size': SHEET_SIZE,
                    'T_inf': T_INF, 'batch': BATCH,
                    'task': 'mod-add (a+b) mod 53',
                    'steps': args.steps, 'seeds': seeds,
                    'chance': CHANCE,
                    'stabilization': f'C+D (gamma_W={GAMMA_W}, '
                                     f'gamma_alpha={GAMMA_ALPHA}, '
                                     f'T_decay={T_DECAY})',
                    'arms': args.arms,
                },
                'results': all_results,
                'elapsed_total': time.time() - t_start,
            }
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=1, default=str)
        except Exception as e:
            print(f"  [WARN] Failed to save results: {e}")

    # ── Final comparison table ──
    print(f"\n{'='*78}")
    print("PHI-MATTERS CONTROL — FINAL COMPARISON")
    print(f"{'='*78}")
    print(f"{'Arm':<16} {'Grok(W)':<10} {'Grok(B)':<10} {'Win Mean':<12} "
          f"{'Best Mean':<12} {'eps_a[1]':<10} {'dh[L-1]':<10}")
    print("-" * 78)
    for arm_name in args.arms:
        r = all_results[arm_name]
        pa = r.get('per_area_avg', {})
        eps1 = pa.get('eps_a_norms_clamped', [0]*L_LAMINAR)
        dhs = pa.get('dh_norms', [0]*L_LAMINAR)
        eps1_val = eps1[1] if len(eps1) > 1 else 0
        dhL_val = dhs[-1] if dhs else 0
        print(f"{arm_name:<16} {r['grok_rate_window']}/{r['n_seeds']:<8} "
              f"{r['grok_rate_best']}/{r['n_seeds']:<8} "
              f"{r['window_mean']:<12.4f} {r['best_mean']:<12.4f} "
              f"{eps1_val:<10.4f} {dhL_val:<10.4f}")
    print(f"{'='*78}")

    # Verdict
    if 'real' in all_results and 'shuffled' in all_results:
        real_mean = all_results['real']['window_mean']
        shuf_mean = all_results['shuffled']['window_mean']
        diff = real_mean - shuf_mean
        print(f"\nVERDICT: real φ mean = {real_mean:.4f}, "
              f"shuffled φ mean = {shuf_mean:.4f}, "
              f"Δ = {diff:+.4f}")
        if abs(diff) < 0.05:
            print("  => shuffled ≈ real: EC IS DECORATIVE (the alphabet doesn't matter)")
        elif diff > 0.10:
            print("  => real >> shuffled: EC MATTERS (the alphabet is load-bearing)")
        else:
            print("  => ambiguous: small difference, needs more seeds")
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
