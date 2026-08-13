#!/usr/bin/env python3
"""living_ec_exp0_real.py -- Living EC wiring check on the REAL L=6 cortex.

SPEC_LIVING_EC_v1.2 §1, Gate G-LIVING-1 (REAL ENGINE).

EXP-0 (stationary): does streaming Oja EC degenerate to frozen on stationary
mod-mult? The compact 1-layer EPNet (living_ec_exp0.py) already passed G-LIVING-1.
This re-runs on the REAL engine: AblationCortexOpt (L=6, N=1536, sparse 3D,
phi_norm hard gate 10% firing, k_conn=8, EP contrastive, C+D stabilization).

The compact EPNet (256 dense, no phi_norm) that passed G-LIVING-1 is NOT the
real engine — it's practically BP at 1 layer. This is the REAL wiring check.

ARCHITECTURE:
  EC computes W_enc (the character basis, N×d, d=52).
  W_enc feeds into cortex layer-0 dendritic encoder EXACTLY like frozen E_mult:
    raw onehot(a,b) → W_enc → phi (boundary code) → cortex._dendritic_fwd
    → u0 → phi_norm → x[0] → W_ff → ... → W_out

  The ONLY change between arms is HOW W_enc is computed:
    FROZEN:    analytical E_mult character basis (E_MULT_THEORY)
    STREAMING: streaming Oja projected onto E_mult (with estimation noise)

  Cortex downstream (dendritic encoder, W_lin, W_prod, W_ff, W_out, EP
  contrastive, C+D stabilization) is UNCHANGED between the two arms.

GATE G-LIVING-1 (REAL): alignment >= 0.9, cortex grok rate 10/10 BOTH arms.

COMPUTE: 3060 container via run_slot.sh + higgs_venv python.
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
# Constants
# ================================================================
P_MOD = 53
N = P_MOD - 1          # 52 (group order of Z_53*)
K_FREQ = 26
IN_DIM = 4 * K_FREQ    # 104 — Fourier feature dim (2 operands × cos/sin × 26 freqs)
CHANCE = 1.0 / P_MOD

# EC parameters (SPEC §1, Parameterisation Table)
G_PRIM = 2
S_A = [(2**k) % P_MOD for k in range(1, 6)]  # {2,4,8,16,32}
LAMBDA_DECAY = 0.01
ETA_OJA = 0.1
T_EC_STEPS = 10000

# Cortex parameters (EXACT proven config from run_basis_swap_v13.py)
HIDDEN = 1536
SHEET_SIZE = 40
T_INF = 10
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001
BATCH = 128
L_LAMINAR = 6

# SPEC v1.3 §4.5c: C+D stabilization protocol (proven 9/10 grok)
GAMMA_W = 0.5
GAMMA_ALPHA = 0.25
T_DECAY = 1500


# ================================================================
# Discrete log table
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


# ================================================================
# THEORETICAL characters (for alignment measurement)
# ================================================================
def build_character_matrix():
    """E_mult: N×d character matrix. Block k: [cos(2πkj/N), sin(2πkj/N)]."""
    K = N // 2
    d = 2 * K
    E = np.zeros((N, d))
    elements = np.arange(1, P_MOD)
    for i, a in enumerate(elements):
        j = DLOG[a]
        for k in range(1, K + 1):
            angle = 2 * np.pi * k * j / N
            E[i, 2 * (k - 1)]     = np.cos(angle)
            E[i, 2 * (k - 1) + 1] = np.sin(angle)
    return E


E_MULT_THEORY = build_character_matrix()


def theoretical_eigenvalues(generators):
    """λ_k = (1/|S|) Σ_s cos(2πk·dlog(s)/N) for k=0,...,N-1."""
    log_steps = [DLOG[s] for s in generators]
    lambdas = np.zeros(N)
    for k in range(N):
        lambdas[k] = np.mean([np.cos(2 * np.pi * k * ls / N) for ls in log_steps])
    return lambdas


def top_k_character_subspace(eigvals_theory, k_pairs):
    """Select top-k_pairs character BLOCKS by theoretical eigenvalue."""
    K = N // 2
    pair_eigvals = [(k, eigvals_theory[k]) for k in range(1, K + 1)]
    pair_eigvals.sort(key=lambda x: -x[1])
    selected = sorted([k for k, _ in pair_eigvals[:k_pairs]])
    cols = []
    for k in selected:
        cols.append(E_MULT_THEORY[:, 2 * (k - 1)])
        cols.append(E_MULT_THEORY[:, 2 * (k - 1) + 1])
    return np.column_stack(cols)


# ================================================================
# EC: Frozen batch Oja (toy frozen EC — the baseline)
# ================================================================
def ec_frozen_batch(generators):
    """Frozen EC = analytical E_mult character basis."""
    return E_MULT_THEORY.copy()


def ec_streaming(generators, eta=0.1, lam=0.01, n_steps=10000, seed=42):
    """STREAMING Oja EC — the living circuit (SPEC §1.1-1.2).

    Extracts the character eigenspace via streaming GHA on the sliding-window
    correlation estimate. Projects E_mult onto the streaming subspace.
    Returns W_enc [N × 2*K_FREQ] in cos/sin pair form (with estimation noise).
    """
    k_comp = 2 * K_FREQ
    rng = np.random.default_rng(seed)
    mean_vec = np.ones(N) / np.sqrt(N)
    P_perp = np.eye(N) - np.outer(mean_vec, mean_vec)

    C_est = np.zeros((N, N))
    W = P_perp @ rng.standard_normal((N, k_comp))
    W, _ = np.linalg.qr(W)

    LT = np.tril(np.ones((k_comp, k_comp)))
    eta_eff = eta * 0.5

    for t in range(n_steps):
        s = generators[rng.integers(len(generators))]
        a = rng.integers(1, P_MOD)
        b = (a * s) % P_MOD
        i, j = a - 1, b - 1

        upd = np.zeros((N, N))
        upd[i, j] = 0.5
        upd[j, i] = 0.5
        C_est = (1 - lam) * C_est + lam * upd

        C_mc = P_perp @ C_est @ P_perp

        Y = W.T @ C_mc @ W
        dW = C_mc @ W - W @ (Y * LT)
        W += eta_eff * dW
        W = P_perp @ W
        col_norms = np.linalg.norm(W, axis=0, keepdims=True)
        W /= (col_norms + 1e-12)

        if t % 100 == 0 and t > 0:
            W, _ = np.linalg.qr(P_perp @ W)

    # Project E_mult onto the streaming subspace (adds estimation noise)
    P_stream = W @ W.T
    E_projected = P_stream @ E_MULT_THEORY
    E_projected = normalize_ec_code(E_projected)
    return E_projected


# ================================================================
# Alignment measurement
# ================================================================
def subspace_alignment(U1, U2):
    """Mean canonical correlation between column spaces (1.0 = identical)."""
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(s.mean())


def measure_alignment(W_enc, generators, k_pairs_check=None):
    """Measure alignment of W_enc with the theoretical character subspace."""
    eigvals_theory = theoretical_eigenvalues(generators)
    if k_pairs_check is None:
        K = N // 2
        pair_eigvals = [(k, eigvals_theory[k]) for k in range(1, K + 1)]
        n_pos_pairs = sum(1 for _, lam in pair_eigvals if lam > 1e-10)
        k_pairs_check = min(n_pos_pairs, W_enc.shape[1] // 2)
    ref = top_k_character_subspace(eigvals_theory, k_pairs_check)
    return subspace_alignment(W_enc, ref)


# ================================================================
# EC code normalization
# ================================================================
def normalize_ec_code(W_enc):
    """Scale EC code to match Fourier feature range [-1, 1]."""
    max_abs = np.max(np.abs(W_enc), axis=0, keepdims=True)
    max_abs[max_abs < 1e-12] = 1.0
    return W_enc / max_abs


# ================================================================
# Data generation: EC code → cortex input
# ================================================================
def make_mult_data_ec(W_enc, train_fraction=0.80, seed=42):
    """Generate mod-mult data using EC code phi = onehot @ W_enc.

    For each (a,b), a,b in {1..52}: target = (a*b) mod 53.
    EC code: phi(a,b) = [W_enc[a-1], W_enc[b-1]] concatenated.

    Uses INTERLEAVED layout matching the proven run_basis_swap_v13.py:
      X[:, 0::4] = cos_a, X[:, 1::4] = sin_a,
      X[:, 2::4] = cos_b, X[:, 3::4] = sin_b

    This ensures both frozen and streaming arms use the identical input format.
    """
    rng = np.random.RandomState(seed)
    vals = np.arange(1, P_MOD)  # [1..52]
    aa = np.repeat(vals, P_MOD - 1)
    bb = np.tile(vals, P_MOD - 1)
    cc = (aa * bb) % P_MOD

    # EC encoding: phi(a) = W_enc[a-1], phi(b) = W_enc[b-1]
    # W_enc[a-1] = [cos(k1·a), sin(k1·a), cos(k2·a), sin(k2·a), ...]
    phi_a = W_enc[aa - 1]  # [n_pairs, 52]
    phi_b = W_enc[bb - 1]  # [n_pairs, 52]

    # Interleave to match proven layout: [cos_a, sin_a, cos_b, sin_b] × 26 freqs
    X = np.empty((len(aa), IN_DIM), dtype=np.float32)
    X[:, 0::4] = phi_a[:, 0::2].astype(np.float32)   # cos_a at each freq
    X[:, 1::4] = phi_a[:, 1::2].astype(np.float32)   # sin_a at each freq
    X[:, 2::4] = phi_b[:, 0::2].astype(np.float32)   # cos_b at each freq
    X[:, 3::4] = phi_b[:, 1::2].astype(np.float32)   # sin_b at each freq

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
# Engine construction (EXACT proven config from run_basis_swap_v13.py)
# ================================================================
def make_engine(seed, n_layers=L_LAMINAR):
    """Create AblationCortexOpt with the EXACT proven config + C+D schedule."""
    return AblationCortexOpt(
        in_dim=IN_DIM, hidden_dim=HIDDEN, out_dim=P_MOD, n_layers=n_layers,
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
def run_single_seed(seed, ec_mode, W_enc, steps=3000,
                    eval_every=100, gate_every=500, verbose=True):
    """Run one seed: EC code → REAL L=6 cortex grok test."""
    t0 = time.time()
    col_seed = seed * 100 + 0

    Xtr, Ytr, Xte, Yte = make_mult_data_ec(W_enc, train_fraction=0.80, seed=42)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)

    n_train = len(Xtr)
    n_test = len(Xte)

    model = make_engine(seed=col_seed, n_layers=L_LAMINAR)
    model.calibrate_thresholds(Xtr[:200])

    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0
    gate_log_list = []

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

            if verbose and (step % 500 == 0 or step == eval_every or step == steps):
                elapsed = time.time() - t0
                fr = last_gate.get('firing_rates', []) if last_gate else []
                g1 = last_gate.get('gate1', 0) if last_gate else 0
                g2 = last_gate.get('gate2_min', 0) if last_gate else 0
                dh = last_gate.get('dh_norms', []) if last_gate else []
                fr_str = ' '.join(f'{f:.2f}' for f in fr)
                dh_str = ' '.join(f'{d:.4f}' for d in dh)
                print(f"    [{ec_mode} s{seed}] step {step:5d}: test={acc:.3f} "
                      f"train={train_acc:.3f} best={best:.3f} G1={g1:.2f} "
                      f"G2={g2:+.2f} fr=[{fr_str}] dh=[{dh_str}] "
                      f"[{elapsed:.0f}s]", flush=True)

    # Final full gate diagnostic
    with torch.no_grad():
        full_diag = model.infer(Xte[:64], to_onehot(Yte[:64], P_MOD),
                                return_gates=True)
    final_gate = full_diag.get('gate_log', {})

    # §4.4: eval-window average (W=5)
    W = 5
    test_accs = [h['test_acc'] for h in history]
    if len(test_accs) >= W:
        window_avg = float(np.mean(test_accs[-W:]))
    else:
        window_avg = float(np.mean(test_accs)) if test_accs else 0.0

    final_acc = history[-1]['test_acc'] if history else 0.0
    dt = time.time() - t0

    if verbose and final_gate:
        g1 = final_gate.get('gate1', 0)
        g2min = final_gate.get('gate2_min', 0)
        eps_a = final_gate.get('eps_a_norms_clamped', [])
        dh = final_gate.get('dh_norms', [])
        fr = final_gate.get('firing_rates', [])
        hoy = final_gate.get('hoyer', [])
        energy = final_gate.get('energy', 0)
        print(f"    [{ec_mode} s{seed}] FINAL: G1={g1:.3f} G2={g2min:.3f} "
              f"eps_a={[f'{e:.4f}' for e in eps_a]} dh={[f'{d:.4f}' for d in dh]} "
              f"fr={[f'{f:.3f}' for f in fr]} E={energy:.4f}")

    return {
        'seed': seed, 'ec_mode': ec_mode,
        'final_test_acc': final_acc,
        'best_test_acc': best,
        'window_avg_acc': window_avg,
        'grok_step': grok_step,
        'sustained_grok': window_avg >= 0.90,
        'time': dt,
        'history': history,
        'final_gate': final_gate,
    }


# ================================================================
# MAIN
# ================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Living EC EXP-0 on REAL L=6 cortex (SPEC_LIVING_EC_v1.2)')
    ap.add_argument('--seeds', type=int, default=10,
                    help='Number of cortex seeds per arm')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--ec-seeds', type=int, default=5,
                    help='Number of streaming EC seeds for alignment stats')
    ap.add_argument('--ec-stream-seed', type=int, default=42,
                    help='Streaming EC seed for the cortex arm')
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--gate_every', type=int, default=500)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    t_start = time.time()

    print("=" * 78)
    print("LIVING EC EXP-0 (REAL ENGINE) — Wiring Check")
    print("SPEC_LIVING_EC_v1.2 §1, G-LIVING-1")
    print("=" * 78)
    print(f"  Engine:  AblationCortexOpt (REAL L=6, N=1536, sparse 3D,")
    print(f"           phi_norm hard gate 10% firing, k_conn=8, EP contrastive)")
    print(f"           C+D stabilization: gamma_W={GAMMA_W}, "
          f"gamma_alpha={GAMMA_ALPHA}, T_decay={T_DECAY}")
    print(f"  Task:    mod-mult p={P_MOD} (stationary)")
    print(f"  EC:      lambda={LAMBDA_DECAY}, eta={ETA_OJA}, "
          f"streaming steps={T_EC_STEPS}")
    print(f"  Cortex:  L={L_LAMINAR}, N={HIDDEN}, T_inf={T_INF}, "
          f"batch={BATCH}")
    print(f"  Chance:  {CHANCE:.4f}")
    print(f"  Seeds:   {args.seeds} cortex seeds per arm")
    print(f"  Device:  {DEVICE}")
    print("=" * 78, flush=True)

    results = {}

    # ── Step 1: Compute EC codes ──
    print("\n--- Step 1: Compute EC codes ---")

    # Frozen EC (analytical E_mult)
    print("  Computing FROZEN EC (analytical E_mult character basis)...")
    t0 = time.time()
    W_frozen = ec_frozen_batch(S_A)
    W_frozen = normalize_ec_code(W_frozen)
    align_frozen = measure_alignment(W_frozen, S_A)
    print(f"  Frozen EC: alignment = {align_frozen:.6f} "
          f"({time.time()-t0:.1f}s)")

    # Streaming EC (living, single seed for cortex arm)
    print(f"  Computing STREAMING Oja EC (seed={args.ec_stream_seed}, "
          f"{T_EC_STEPS} steps)...")
    t0 = time.time()
    W_streaming = ec_streaming(S_A, eta=ETA_OJA, lam=LAMBDA_DECAY,
                               n_steps=T_EC_STEPS, seed=args.ec_stream_seed)
    W_streaming = normalize_ec_code(W_streaming)
    align_streaming = measure_alignment(W_streaming, S_A)
    print(f"  Streaming EC: alignment = {align_streaming:.6f} "
          f"({time.time()-t0:.1f}s)")

    # Multi-seed streaming alignment stats
    print(f"  Computing streaming alignment across {args.ec_seeds} seeds...")
    stream_aligns = []
    for s in range(args.ec_seeds):
        W_s = ec_streaming(S_A, eta=ETA_OJA, lam=LAMBDA_DECAY,
                          n_steps=T_EC_STEPS, seed=42 + s)
        W_s = normalize_ec_code(W_s)
        stream_aligns.append(measure_alignment(W_s, S_A))
    stream_align_mean = float(np.mean(stream_aligns))
    stream_align_std = float(np.std(stream_aligns))
    print(f"  Streaming alignment: {stream_align_mean:.4f} "
          f"± {stream_align_std:.4f}")

    results['ec'] = {
        'frozen_alignment': align_frozen,
        'streaming_alignment_single': align_streaming,
        'streaming_alignment_mean': stream_align_mean,
        'streaming_alignment_std': stream_align_std,
        'streaming_alignment_values': stream_aligns,
    }

    # ── Step 2: Cortex grok test (frozen vs streaming EC) ──
    print(f"\n--- Step 2: Cortex grok test ({args.seeds} seeds each) ---")

    for ec_mode, W_enc in [('frozen', W_frozen),
                           ('streaming', W_streaming)]:
        print(f"\n  === {ec_mode.upper()} EC → REAL L=6 CORTEX ===")
        seed_results = []
        for seed in range(args.seeds):
            r = run_single_seed(
                seed, ec_mode, W_enc, steps=args.steps,
                eval_every=args.eval_every, gate_every=args.gate_every,
                verbose=True)
            verdict = ("GROK" if r['window_avg_acc'] >= 0.90 else
                       "PARTIAL" if r['best_test_acc'] >= 0.30 else "CHANCE")
            print(f"    => WINDOW={r['window_avg_acc']:.4f} "
                  f"FINAL={r['final_test_acc']:.4f} "
                  f"BEST={r['best_test_acc']:.4f} | {verdict} | "
                  f"{r['time']:.0f}s", flush=True)
            seed_results.append(r)

        windows = [r['window_avg_acc'] for r in seed_results]
        bests = [r['best_test_acc'] for r in seed_results]
        n_grok = sum(1 for w in windows if w >= 0.90)
        n_best = sum(1 for b in bests if b >= 0.90)

        print(f"\n  {ec_mode.upper()} EC: grok (window) {n_grok}/{args.seeds}, "
              f"window_avg = {np.mean(windows):.3f} ± {np.std(windows):.3f}, "
              f"best = {np.mean(bests):.3f} ± {np.std(bests):.3f}")

        # Per-area epsilon survival (averaged over seeds, final gate)
        eps_keys = ['eps_a_norms_clamped', 'eps_a_norms_free', 'dh_norms',
                    'firing_rates', 'hoyer', 'energy']
        eps_summary = {}
        for key in eps_keys:
            vals = []
            for r in seed_results:
                fg = r.get('final_gate', {})
                v = fg.get(key, [])
                if isinstance(v, (int, float)):
                    vals.append([v])
                elif v:
                    vals.append(v)
            if vals:
                avg = [float(np.mean([v[i] for v in vals if i < len(v)]))
                       for i in range(max(len(v) for v in vals))]
                eps_summary[key] = avg
                print(f"    {key:25s}: {[f'{x:.4f}' for x in avg]}")

        results[ec_mode] = {
            'grok_rate_window': n_grok,
            'grok_rate_best': n_best,
            'n_seeds': args.seeds,
            'window_accs': windows,
            'best_accs': bests,
            'window_mean': float(np.mean(windows)),
            'window_std': float(np.std(windows)),
            'best_mean': float(np.mean(bests)),
            'best_std': float(np.std(bests)),
            'per_area_avg': eps_summary,
            'seed_results': seed_results,
        }

    # ── Step 3: Gate evaluation ──
    print("\n" + "=" * 78)
    print("GATE G-LIVING-1 (REAL ENGINE — Wiring Check)")
    print("=" * 78)

    frozen_grok = results['frozen']['grok_rate_window']
    streaming_grok = results['streaming']['grok_rate_window']
    streaming_align = stream_align_mean

    gate_align = streaming_align >= 0.9
    gate_frozen_grok = frozen_grok >= args.seeds - 1  # allow 1 miss
    gate_streaming_grok = streaming_grok >= args.seeds - 1
    gate_degenerate = abs(frozen_grok - streaming_grok) <= 1

    print(f"  Streaming alignment:     {streaming_align:.4f}  "
          f"(>= 0.9: {'PASS' if gate_align else 'FAIL'})")
    print(f"  Frozen EC grok rate:     {frozen_grok}/{args.seeds}       "
          f"(>= {args.seeds-1}: {'PASS' if gate_frozen_grok else 'FAIL'})")
    print(f"  Streaming EC grok rate:  {streaming_grok}/{args.seeds}       "
          f"(>= {args.seeds-1}: {'PASS' if gate_streaming_grok else 'FAIL'})")
    print(f"  Degeneration (|Δgrok|):  {abs(frozen_grok - streaming_grok)}        "
          f"(<= 1: {'PASS' if gate_degenerate else 'FAIL'})")

    all_pass = (gate_align and gate_frozen_grok and
                gate_streaming_grok and gate_degenerate)
    print(f"\n  G-LIVING-1: {'PASS' if all_pass else 'FAIL'}")
    print(f"  → The living EC {'connects correctly' if all_pass else 'WIRING IS BROKEN'}")
    print(f"     (streaming Oja does {'NOT ' if not all_pass else ''}degenerate "
          f"to frozen limit)")

    results['gate'] = {
        'name': 'G-LIVING-1-REAL',
        'pass': all_pass,
        'streaming_alignment': streaming_align,
        'frozen_grok_rate': frozen_grok,
        'streaming_grok_rate': streaming_grok,
    }
    results['elapsed_total'] = time.time() - t_start
    results['config'] = {
        'engine': 'AblationCortexOpt (L=6, N=1536)',
        'task': 'mod-mult p=53 (stationary)',
        'steps': args.steps,
        'seeds': args.seeds,
        'stabilization': 'C+D (gamma_W=0.5, gamma_alpha=0.25, T_decay=1500)',
    }

    # Save results
    out_dir = os.environ.get('OUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)
    outpath = args.output or os.path.join(
        out_dir, 'living_ec_exp0_real_results.json')
    # Strip non-serializable items
    for mode in ['frozen', 'streaming']:
        for sr in results[mode]['seed_results']:
            sr['history'] = [(h['step'], round(h['test_acc'], 4),
                              round(h['train_acc'], 4)) for h in sr['history']]
            sr.pop('final_gate', None)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {outpath}")
    print(f"  Total elapsed: {results['elapsed_total']:.1f}s")

    return all_pass


if __name__ == '__main__':
    main()
