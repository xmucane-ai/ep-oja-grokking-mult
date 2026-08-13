#!/usr/bin/env python3
"""firing_rate_ablation_L6.py — Firing-rate ablation sweep on L=6 cortex.

CARD t_5312445a: does grokking depend on the exact 10% firing rate?
Sweep target_rate = {0.05, 0.10, 0.25, 0.50, 1.0(dense)} on the REAL engine
(AblationCortexOpt, L=6, N=1536, sheet_size=40, mod-MULT p=53, T=10,
C+D stabilization). 10 seeds each, 2000 steps.

If grok is robust across 5-25% → solid claim.
If it only works at exactly 10% → reviewer attack surface (hyperparameter fluke).

The ONLY variable swept is target_rate. Everything else is the EXACT proven
10/10 grok config from living_ec_exp0_real.py (frozen EC arm).

Constitution: P1-P8 all pass (same engine, same mechanisms, only firing rate
differs). target_rate=1.0 disables the hard gate (all neurons fire → dense).

DEPLOY ON 3060 CONTAINER via run_slot.sh.
"""
import sys
import os
import time
import json
import argparse

import numpy as np
import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE

# ================================================================
# Constants (EXACT proven config from living_ec_exp0_real.py)
# ================================================================
P_MOD = 53
N_EC = P_MOD - 1       # 52 (group order of Z_53*)
K_FREQ = 26
IN_DIM = 4 * K_FREQ    # 104 — Fourier feature dim
CHANCE = 1.0 / P_MOD

# EC parameters (SPEC §1)
G_PRIM = 2
S_A = [(2**k) % P_MOD for k in range(1, 6)]  # {2,4,8,16,32}

# Cortex parameters (EXACT proven config)
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
# Discrete log table + character matrix (frozen EC = E_mult)
# ================================================================
def build_dlog_table(p=53, g=2):
    dlog = {}
    val = 1
    for exp in range(p - 1):
        dlog[val] = exp
        val = (val * g) % p
    assert len(dlog) == p - 1
    return dlog


DLOG = build_dlog_table(P_MOD, G_PRIM)


def build_character_matrix():
    """E_mult: N×d character matrix. Block k: [cos(2πkj/N), sin(2πkj/N)]."""
    K = N_EC // 2
    d = 2 * K
    E = np.zeros((N_EC, d))
    elements = np.arange(1, P_MOD)
    for i, a in enumerate(elements):
        j = DLOG[a]
        for k in range(1, K + 1):
            angle = 2 * np.pi * k * j / N_EC
            E[i, 2 * (k - 1)]     = np.cos(angle)
            E[i, 2 * (k - 1) + 1] = np.sin(angle)
    return E


E_MULT_THEORY = build_character_matrix()


def normalize_ec_code(W_enc):
    """Scale EC code to match Fourier feature range [-1, 1]."""
    max_abs = np.max(np.abs(W_enc), axis=0, keepdims=True)
    max_abs[max_abs < 1e-12] = 1.0
    return W_enc / max_abs


# ================================================================
# Data generation: EC code → cortex input (mod-MULT p=53)
# ================================================================
def make_mult_data_ec(W_enc, train_fraction=0.80, seed=42):
    """Generate mod-mult data using EC code phi = onehot @ W_enc."""
    rng = np.random.RandomState(seed)
    vals = np.arange(1, P_MOD)  # [1..52]
    aa = np.repeat(vals, P_MOD - 1)
    bb = np.tile(vals, P_MOD - 1)
    cc = (aa * bb) % P_MOD

    phi_a = W_enc[aa - 1]
    phi_b = W_enc[bb - 1]

    X = np.empty((len(aa), IN_DIM), dtype=np.float32)
    X[:, 0::4] = phi_a[:, 0::2].astype(np.float32)
    X[:, 1::4] = phi_a[:, 1::2].astype(np.float32)
    X[:, 2::4] = phi_b[:, 0::2].astype(np.float32)
    X[:, 3::4] = phi_b[:, 1::2].astype(np.float32)

    Y = cc.astype(np.int64)
    total = len(aa)
    perm = rng.permutation(total)
    n_tr = int(total * train_fraction)
    Xtr, Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
    Xte, Yte = X[perm[n_tr:]], Y[perm[n_tr:]]
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def to_onehot(Y, n_classes=P_MOD):
    Yoh = torch.zeros(len(Y), n_classes, device=DEVICE)
    Yoh[torch.arange(len(Y)), Y] = 1.0
    return Yoh


# ================================================================
# Engine construction with parameterized target_rate
# ================================================================
def make_engine(seed, target_rate, n_layers=L_LAMINAR):
    """Create AblationCortexOpt with target_rate parameterized."""
    return AblationCortexOpt(
        in_dim=IN_DIM, hidden_dim=HIDDEN, out_dim=P_MOD, n_layers=n_layers,
        sheet_size=SHEET_SIZE,
        target_rate=target_rate, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
        gamma_W=GAMMA_W, gamma_alpha=GAMMA_ALPHA, T_decay=T_DECAY,
        alpha_theta_0=0.05,
    )


# ================================================================
# Single-seed runner
# ================================================================
def run_single_seed(seed, target_rate, W_enc, steps=2000,
                    eval_every=100, gate_every=500, verbose=True):
    """Run one seed: mod-mult p=53 grok test at given target_rate."""
    t0 = time.time()

    Xtr, Ytr, Xte, Yte = make_mult_data_ec(W_enc, train_fraction=0.80, seed=42)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)

    model = make_engine(seed=seed, target_rate=target_rate)
    model.calibrate_thresholds(Xtr[:200])

    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0
    gate_log_list = []

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
            history.append({'step': step, 'test_acc': acc,
                            'train_acc': train_acc})

            last_gate = model.last_gates if do_gate else {}
            if do_gate and last_gate:
                gate_log_list.append({'step': step, **last_gate})

            if verbose and (step % 500 == 0 or step == eval_every or step == steps):
                elapsed = time.time() - t0
                fr = last_gate.get('firing_rates', []) if last_gate else []
                g1 = last_gate.get('gate1', 0) if last_gate else 0
                g2 = last_gate.get('gate2_min', 0) if last_gate else 0
                dh = last_gate.get('dh_norms', []) if last_gate else []
                fr_str = ' '.join(f'{f:.2f}' for f in fr)
                dh_str = ' '.join(f'{d:.4f}' for d in dh)
                print(f"    [r{target_rate:.2f} s{seed}] step {step:5d}: "
                      f"test={acc:.3f} train={train_acc:.3f} best={best:.3f} "
                      f"G1={g1:.2f} G2={g2:+.2f} fr=[{fr_str}] "
                      f"dh=[{dh_str}] [{elapsed:.0f}s]", flush=True)

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
        'seed': seed,
        'target_rate': target_rate,
        'final_test_acc': final_acc,
        'best_test_acc': best,
        'window_avg_acc': window_avg,
        'grok_step': grok_step,
        'sustained_grok': window_avg >= 0.90,
        'time': dt,
        'final_gate': final_gate,
    }


# ================================================================
# MAIN: sweep driver
# ================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Firing-rate ablation sweep on L=6 cortex (t_5312445a)')
    ap.add_argument('--rates', type=float, nargs='+',
                    default=[0.05, 0.10, 0.25, 0.50, 1.0])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--gate_every', type=int, default=500)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    t_start = time.time()

    print("=" * 78)
    print("FIRING-RATE ABLATION SWEEP (L=6 CORTEX)")
    print("Card t_5312445a: does grokking depend on the exact 10% rate?")
    print("=" * 78)
    print(f"  Engine:    AblationCortexOpt (L=6, N={HIDDEN}, sheet={SHEET_SIZE})")
    print(f"  Task:      mod-MULT p={P_MOD}")
    print(f"  EC:        frozen E_mult character basis (proven)")
    print(f"  Stabilize: C+D (gamma_W={GAMMA_W}, gamma_alpha={GAMMA_ALPHA}, "
          f"T_decay={T_DECAY})")
    print(f"  Rates:     {args.rates}")
    print(f"  Seeds:     {args.seeds} per rate")
    print(f"  Steps:     {args.steps}")
    print(f"  Chance:    {CHANCE:.4f}")
    print(f"  Device:    {DEVICE}")
    print("=" * 78, flush=True)

    # Compute frozen EC code
    print("\n--- Computing frozen EC (E_mult) ---")
    W_enc = normalize_ec_code(E_MULT_THEORY.copy())
    print(f"  W_enc shape: {W_enc.shape}")

    out_dir = os.environ.get('OUTPUT_DIR',
                             os.environ.get('OUT_DIR', '/root/gate2/outputs'))
    os.makedirs(out_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        out_dir, 'firing_rate_ablation_L6.json')
    progress_path = os.path.join(out_dir, 'firing_rate_ablation_PROGRESS.json')

    results = {
        'experiment': 'firing_rate_ablation_L6',
        'card': 't_5312445a',
        'config': {
            'engine': 'AblationCortexOpt',
            'L': L_LAMINAR, 'N': HIDDEN, 'sheet_size': SHEET_SIZE,
            'T_inf': T_INF, 'batch': BATCH,
            'task': 'mod-mult', 'P': P_MOD,
            'steps': args.steps, 'seeds_per_rate': args.seeds,
            'rates': args.rates,
            'stabilization': {'gamma_W': GAMMA_W,
                              'gamma_alpha': GAMMA_ALPHA,
                              'T_decay': T_DECAY},
            'chance': CHANCE,
        },
        'arms': {},
    }

    # ── Sweep: rate × seed ──
    for rate in args.rates:
        arm_key = f"rate{rate:.2f}"
        results['arms'][arm_key] = []
        print(f"\n{'='*60}")
        print(f"  ARM: target_rate={rate:.2f} ({args.seeds} seeds, "
              f"{args.steps} steps)")
        print(f"{'='*60}", flush=True)

        for seed in range(args.seeds):
            r = run_single_seed(
                seed, rate, W_enc, steps=args.steps,
                eval_every=args.eval_every, gate_every=args.gate_every,
                verbose=True)
            verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                       "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
            print(f"    => [DONE] r{rate:.2f} s{seed}: "
                  f"window={r['window_avg_acc']:.4f} "
                  f"final={r['final_test_acc']:.4f} "
                  f"best={r['best_test_acc']:.4f} | {verdict} | "
                  f"{r['time']:.0f}s", flush=True)

            # Strip non-serializable
            sr = {k: v for k, v in r.items() if k != 'final_gate'}
            results['arms'][arm_key].append(sr)

            # Incremental save
            try:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            except Exception:
                pass

            # Progress file
            try:
                prog = {
                    'rate': rate, 'seed': seed,
                    'window_acc': r['window_avg_acc'],
                    'best_acc': r['best_test_acc'],
                    'grok_step': r['grok_step'],
                    'elapsed_s': time.time() - t_start,
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }
                with open(progress_path, 'w') as pf:
                    json.dump(prog, pf)
            except Exception:
                pass

            # Heartbeat-friendly: print arm progress
            arm_accs = [s['window_avg_acc'] for s in results['arms'][arm_key]]
            n_grok = sum(1 for a in arm_accs if a >= 0.90)
            print(f"    Arm {arm_key} progress: {n_grok}/{len(arm_accs)} grok "
                  f"(window median={np.median(arm_accs):.3f})", flush=True)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: FIRING-RATE ABLATION (L=6, N={HIDDEN}, mod-mult p={P_MOD})")
    print(f"{'='*70}")
    print(f"  Chance: {CHANCE:.4f}")
    print()
    print(f"{'Rate':>8} {'GrokRate':>10} {'WinMedian':>12} {'WinMean':>10} "
          f"{'WinStd':>8} {'BestMean':>10} {'BestStd':>8}")
    print("-" * 70)

    for rate in args.rates:
        arm_key = f"rate{rate:.2f}"
        arm = results['arms'].get(arm_key, [])
        if not arm:
            continue
        windows = [s['window_avg_acc'] for s in arm]
        bests = [s['best_test_acc'] for s in arm]
        n_grok = sum(1 for w in windows if w >= 0.90)
        med = float(np.median(windows))
        mean = float(np.mean(windows))
        std = float(np.std(windows))
        bmean = float(np.mean(bests))
        bstd = float(np.std(bests))
        print(f"{rate:>8.2f} {n_grok:>5}/{len(arm):<4} {med:>12.3f} "
              f"{mean:>10.3f} {std:>8.3f} {bmean:>10.3f} {bstd:>8.3f}")

    print(f"\n  Results: {output_path}")
    print(f"  Total elapsed: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
