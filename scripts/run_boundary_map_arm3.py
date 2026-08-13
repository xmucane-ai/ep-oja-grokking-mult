#!/usr/bin/env python3
"""run_boundary_map_arm3.py -- SPEC_CD_BOUNDARY_MAP_ARM3_v1.1 (task t_ec220868).

The 3×3 P8-boundary-map grid: γ_W ∈ {0.3, 0.5, 0.7} × T_decay ∈ {750, 1500, 3000},
with γ_α = γ_W/2 (constant ratio 0.5), testing the axis-decoupling theorem
R(t) = 1.581·0.5^n.

Two-stage protocol (BINDING per §6):
  STAGE 1: 1-seed × 9-cell trajectory probe (~1.5h) with sign-off S1-S4.
  STAGE 2: full map (10 seeds or 8-seed fallback) gated on S1-S4.

Engine: FROZEN ablation_cortex_v14_1.py (via AblationCortexOpt). Schedule params only.
Engine hyperparameters EXACT proven config from run_basis_swap_v13.py.

COMPUTE LANE: 3060 container (training-container via run_slot.sh).
Local RTX2060 OFF-LIMITS.

Usage:
  # Stage 1: trajectory probe (1 seed × 9 cells)
  python -u run_boundary_map_arm3.py --stage 1 --seed 0

  # Stage 2: full map (8-seed fallback per §6.2 patch 6)
  python -u run_boundary_map_arm3.py --stage 2 --seeds 0-7

  # Stage 2: split run (seeds 0-4)
  python -u run_boundary_map_arm3.py --stage 2 --seeds 0-4
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
# Checkpoint utilities (task t_0db33cce — SAVE WEIGHTS invariant)
# ================================================================
def save_checkpoint(model, cell_idx, gamma_W, T_decay, seed, test_acc,
                    path, step, phase='terminal'):
    """Save weight checkpoint for schema VET (task t_0db33cce).

    Banks ALL learned weight tensors so a grokked schema can be re-loaded:
    W_ff (L-1 feedforward), W_out (readout), B_fb (feedback — P3 separate),
    B_hc (hippocampal broadcast — P5), thresholds (P8 homeostasis).

    NOTE: AblationCortexOpt has no W_enc (input is raw Fourier features).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'W_ff': [W.clone().cpu() for W in model.W_ff],
        'W_out': model.W_out.clone().cpu(),
        'B_fb': [B.clone().cpu() for B in model.B_fb],
        'B_hc': [B.clone().cpu() for B in model.B_hc],
        'thresholds': [t.clone().cpu() for t in model.thresholds],
        'cell_idx': cell_idx,
        'gamma_W': gamma_W,
        'T_decay': T_decay,
        'seed': seed,
        'test_acc': test_acc,
        'step': step,
        'phase': phase,
        'engine': 'AblationCortexOpt (ablation_cortex_v14_1_opt.py)',
    }
    torch.save(ckpt, path)
    return path

# ── Constants (EXACT proven config from run_basis_swap_v13.py) ──
P_MOD = 53
K_FREQ = 26
IN_DIM = 4 * K_FREQ  # 104
CHANCE = 1.0 / P_MOD
G_PRIM = 2

# Engine hyperparameters (byte-identical to proven config)
HIDDEN = 1536
SHEET_SIZE = 40
T_INF = 10
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001
BATCH = 128
L_LAMINAR = 6
ALPHA_THETA_0 = 0.05  # the hardcoded threshold EMA rate

# ── The 3×3 grid (SPEC §4.1 / Appendix B) ──
# Rows = γ_W, Cols = T_decay. γ_α = γ_W/2 always (P8 constraint KEPT).
GRID = [
    # (gamma_W, T_decay, gamma_alpha)  — 9 cells
    (0.3,  750, 0.15),   # [1,1]
    (0.3, 1500, 0.15),   # [1,2]
    (0.3, 3000, 0.15),   # [1,3] nostab
    (0.5,  750, 0.25),   # [2,1]
    (0.5, 1500, 0.25),   # [2,2] ★ C2 SANITY POINT
    (0.5, 3000, 0.25),   # [2,3] nostab
    (0.7,  750, 0.35),   # [3,1]
    (0.7, 1500, 0.35),   # [3,2]
    (0.7, 3000, 0.35),   # [3,3] nostab
]

# Indices for Stage-1 checks
CELL_22 = 4  # γ_W=0.5, T_decay=1500 (★ sanity point)
NOSTAB_ROW = [2, 5, 8]  # T_decay=3000 cells


# ================================================================
# Discrete log table for E_mult (identical to run_basis_swap_v13.py)
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
# Data generation (E_mult — identical to run_basis_swap_v13.py)
# ================================================================
def make_mult_data(n_train=2163, n_test=541, seed=42, train_fraction=0.80):
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
def make_engine(seed, gamma_W, gamma_alpha, T_decay):
    """Create AblationCortexOpt with the EXACT proven config + swept C+D schedule."""
    return AblationCortexOpt(
        in_dim=IN_DIM, hidden_dim=HIDDEN, out_dim=P_MOD, n_layers=L_LAMINAR,
        sheet_size=SHEET_SIZE,
        target_rate=0.10, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
        gamma_W=gamma_W, gamma_alpha=gamma_alpha, T_decay=T_decay,
        alpha_theta_0=ALPHA_THETA_0,
    )


# ================================================================
# Single cell × seed runner
# ================================================================
def run_cell_seed(cell_idx, gamma_W, T_decay, gamma_alpha,
                  seed, steps, eval_every=100, gate_every=500, ckpt_dir=None):
    """Run one cell (γ_W, T_decay, γ_α) for one seed. Returns trajectory + metrics."""
    t0 = time.time()
    col_seed = seed * 100 + 0  # matches proven l6eqcap seed mapping

    Xtr, Ytr, Xte, Yte = make_mult_data(seed=42, train_fraction=0.80)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)
    n_train = len(Xtr)

    model = make_engine(col_seed, gamma_W, gamma_alpha, T_decay)
    model.calibrate_thresholds(Xtr[:200])

    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0
    schedule_log = []

    cell_tag = f"[gW={gamma_W},Td={T_decay},s{seed}]"

    for step in range(1, steps + 1):
        idx = rng.randint(0, n_train, BATCH)
        do_gate = (step % gate_every == 0) or (step == 1) or (step == steps)
        model.train_step(Xtr[idx], Yoh[idx], return_gates=do_gate)

        if step % eval_every == 0 or step == 1:
            acc = model.evaluate(Xte, Yte)
            best = max(best, acc)
            if acc >= 0.9 and grok_step is None:
                grok_step = step
                # ── Save grok checkpoint (task t_0db33cce — SAVE WEIGHTS) ──
                if ckpt_dir is not None:
                    path = os.path.join(
                        ckpt_dir,
                        f'boundary_cell{cell_idx}_gW{gamma_W}_Td{T_decay}_seed{seed}_grok.pt')
                    save_checkpoint(model, cell_idx, gamma_W, T_decay, seed,
                                    acc, path, step, phase='grok')
                    print(f"  [CKPT] saved grok checkpoint: {path}", flush=True)
            history.append({'step': step, 'test_acc': acc})

            schedule_log.append({
                'step': step,
                'eta_W_eff': float(model._eta_W_eff),
                'alpha_theta_eff': float(model._alpha_theta_eff),
            })

            if step % 500 == 0 or step == eval_every or step == steps:
                elapsed = time.time() - t0
                print(f"  {cell_tag} step {step:5d}: test={acc:.3f} best={best:.3f} "
                      f"ηW={model._eta_W_eff:.5f} α={model._alpha_theta_eff:.5f} "
                      f"[{elapsed:.0f}s]", flush=True)

    # ── Final gate diagnostic (for ε_a norms — S4 check) ──
    with torch.no_grad():
        full_diag = model.infer(Xte[:64], to_onehot(Yte[:64], P_MOD),
                                return_gates=True)
    final_gate = full_diag.get('gate_log', {})
    eps_a_norms = final_gate.get('eps_a_norms_clamped', [])

    # ── Headline metrics (§4.4: window-avg W=5) ──
    W = 5
    test_accs = [h['test_acc'] for h in history]
    window_avg = float(np.mean(test_accs[-W:])) if len(test_accs) >= W else \
        (float(np.mean(test_accs)) if test_accs else 0.0)
    final_acc = test_accs[-1] if test_accs else 0.0
    sustained = window_avg >= 0.90

    dt = time.time() - t0
    result = {
        'cell_idx': cell_idx,
        'gamma_W': gamma_W,
        'T_decay': T_decay,
        'gamma_alpha': gamma_alpha,
        'seed': seed,
        'L': L_LAMINAR, 'N': HIDDEN, 'sheet_size': SHEET_SIZE, 't_inf': T_INF,
        'steps': steps,
        'best_acc': best,
        'final_acc': final_acc,
        'window_avg_acc': window_avg,
        'sustained_grok': sustained,
        'grok_step': grok_step,
        'eps_a_final': eps_a_norms,
        'trajectory': test_accs,
        'schedule_log': schedule_log,
        'time': dt,
        'chance': CHANCE,
    }
    print(f"  {cell_tag} => window={window_avg:.4f} final={final_acc:.4f} "
          f"best={best:.4f} eps_a={[f'{e:.3f}' for e in eps_a_norms]} "
          f"sustained={sustained} [{dt:.0f}s]", flush=True)

    # ── Save terminal checkpoint (task t_0db33cce — SAVE WEIGHTS) ──
    if ckpt_dir is not None:
        path = os.path.join(
            ckpt_dir,
            f'boundary_cell{cell_idx}_gW{gamma_W}_Td{T_decay}_seed{seed}_terminal.pt')
        save_checkpoint(model, cell_idx, gamma_W, T_decay, seed, window_avg,
                        path, steps, phase='terminal')
        print(f"  [CKPT] saved terminal checkpoint: {path}", flush=True)

    return result


# ================================================================
# Verdict computation (§6.2 taxonomy)
# ================================================================
def cell_verdict(results):
    """Compute boundary verdict for a cell across seeds.
    §6.2 MECE categories:
      grok_hold:    best_mean >= 0.90 AND sustained >= threshold
      grok_forget:  best_mean >= 0.90 AND sustained < threshold
      collapse:     0.50 <= best_mean < 0.90
      no_grok:      best_mean < 0.50
    """
    n = len(results)
    bests = [r['best_acc'] for r in results]
    windows = [r['window_avg_acc'] for r in results]
    best_mean = float(np.mean(bests))
    sustained_rate = sum(1 for r in results if r['sustained_grok'])

    # Threshold scales with seed count (§6.2 patch 6)
    thresh = max(1, int(np.ceil(0.8 * n)))  # 8/10 or 6/8 (rounds up)

    if best_mean >= 0.90 and sustained_rate >= thresh:
        verdict = 'grok_hold'
    elif best_mean >= 0.90 and sustained_rate < thresh:
        verdict = 'grok_forget'
    elif best_mean >= 0.50:
        verdict = 'collapse'
    else:
        verdict = 'no_grok'

    return {
        'best_mean': round(best_mean, 4),
        'window_avg_mean': round(float(np.mean(windows)), 4),
        'window_avg_median': round(float(np.median(windows)), 4),
        'sustained_rate': sustained_rate,
        'sustained_rate_str': f"{sustained_rate}/{n}",
        'grok_rate': sum(1 for b in bests if b >= 0.90),
        'threshold_needed': thresh,
        'verdict': verdict,
        'per_seed_window': [round(w, 4) for w in windows],
        'per_seed_best': [round(b, 4) for b in bests],
    }


# ================================================================
# Stage-1 sign-off evaluation (§6.1)
# ================================================================
def evaluate_stage1(results_by_cell):
    """Evaluate S1-S4 criteria for Stage 1.
    Returns dict with pass/fail per criterion + recommendation.
    """
    s1_pass = False
    s2_pass = False
    s3_pass = False
    s4_pass = False
    details = {}

    # S1: C2 sanity — cell [2,2] window-avg >= 0.90
    c22 = results_by_cell[CELL_22]
    s1_pass = c22['window_avg_acc'] >= 0.90
    details['S1'] = {
        'criterion': 'C2 [2,2] window-avg >= 0.90',
        'cell': '[2,2] γ_W=0.5, T_decay=1500',
        'window_avg': round(c22['window_avg_acc'], 4),
        'pass': s1_pass,
        'false_alarm_note': '10% chance seed-0 fails falsely (9/10 sustained)',
    }

    # S2: nostab row fails — cells [1,3],[2,3],[3,3] do NOT stabilize
    nostab_results = [results_by_cell[i] for i in NOSTAB_ROW]
    nostab_windows = [r['window_avg_acc'] for r in nostab_results]
    n_stabilized = sum(1 for w in nostab_windows if w >= 0.90)
    s2_pass = n_stabilized == 0  # ALL nostab cells must fail to stabilize
    details['S2'] = {
        'criterion': 'nostab cells [*,3] do NOT stabilize (window-avg < 0.90)',
        'nostab_windows': [round(w, 4) for w in nostab_windows],
        'n_stabilized': n_stabilized,
        'pass': s2_pass,
        'false_alarm_note': '40% chance seed-0 passes falsely (4/10 sustained)',
    }

    # S3: nostab-row identity — cells [1,3],[2,3],[3,3] identical SCHEDULE
    # (trajectories diverge on GPU due to FP chaos; schedule values are the
    # deterministic ground truth — n=0 for steps 1-2999 means γ_W is inert).
    nostab_schedules = [results_by_cell[i].get('schedule_log', []) for i in NOSTAB_ROW]
    max_sched_diff = 0.0
    if all(len(s) == len(nostab_schedules[0]) for s in nostab_schedules) and nostab_schedules[0]:
        for i_step in range(len(nostab_schedules[0])):
            for ci in range(1, len(nostab_schedules)):
                eta_diff = abs(nostab_schedules[0][i_step]['eta_W_eff'] -
                               nostab_schedules[ci][i_step]['eta_W_eff'])
                alpha_diff = abs(nostab_schedules[0][i_step]['alpha_theta_eff'] -
                                 nostab_schedules[ci][i_step]['alpha_theta_eff'])
                max_sched_diff = max(max_sched_diff, eta_diff, alpha_diff)
        # Steps 1-2999 must be identical; step 3000 (n=1) is allowed to differ
        # (it's the boundary step where decay kicks in, not part of nostab regime)
        s3_pass = max_sched_diff < 0.001  # allows for step-3000 boundary
    else:
        s3_pass = False
    details['S3'] = {
        'criterion': 'nostab-row [*,3] schedule IDENTICAL for steps 1-2999 '
                     '(FP-chaos makes trajectories diverge; schedule is deterministic)',
        'max_schedule_diff': round(max_sched_diff, 8),
        'note': 'step 3000 (n=1) has gamma_W^1 applied → different; steps 1-2999 identical',
        'pass': s3_pass,
    }

    # S4: P4 health — all cells ε_a norms in [0.10, 0.20]
    cells_out_of_range = []
    all_eps_ok = True
    for i, r in enumerate(results_by_cell):
        eps = r.get('eps_a_final', [])
        if not eps:
            cells_out_of_range.append({'cell': i, 'eps': 'MISSING'})
            all_eps_ok = False
            continue
        out_of_range = [e for e in eps if e < 0.10 or e > 0.20]
        if out_of_range:
            all_eps_ok = False
            cells_out_of_range.append({
                'cell': i,
                'gamma_W': r['gamma_W'],
                'T_decay': r['T_decay'],
                'eps': [round(e, 4) for e in eps],
                'out_of_range': [round(e, 4) for e in out_of_range],
            })
    s4_pass = all_eps_ok
    details['S4'] = {
        'criterion': 'all cells ε_a norms in [0.10, 0.20]',
        'pass': s4_pass,
        'cells_out_of_range': cells_out_of_range,
    }

    all_pass = s1_pass and s2_pass and s3_pass and s4_pass

    # Determine recommendation per false-alarm protocol (§6.1)
    if all_pass:
        recommendation = 'ALL PASS — proceed to Stage 2'
    elif s3_pass and s4_pass and not (s1_pass and s2_pass):
        recommendation = ('S3+S4 PASS, S1/S2 may be false alarm — '
                          'RE-RUN Stage 1 with seeds 1-2 (§6.1)')
    else:
        recommendation = ('S3 or S4 FAILED — grid confirmed MISCONFIGURED. '
                           'Debug before proceeding.')

    return {
        'S1': s1_pass, 'S2': s2_pass, 'S3': s3_pass, 'S4': s4_pass,
        'all_pass': all_pass,
        'recommendation': recommendation,
        'details': details,
    }


# ================================================================
# Seed parsing
# ================================================================
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
        description='SPEC_CD_BOUNDARY_MAP_ARM3_v1.1 — P8 boundary map')
    ap.add_argument('--stage', type=int, default=1, choices=[1, 2],
                    help='Stage 1: trajectory probe (1 seed × 9 cells). '
                         'Stage 2: full map (multi-seed × 9 cells)')
    ap.add_argument('--seed', type=int, default=0,
                    help='Stage 1: single seed to probe')
    ap.add_argument('--seeds', type=str, default='0-7',
                    help='Stage 2: seed spec (e.g. 0-7 for 8-seed fallback)')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--gate_every', type=int, default=500)
    ap.add_argument('--output', default=None)
    ap.add_argument('--cells', type=str, default=None,
                    help='Comma-sep cell indices (0-8) to run subset. Default: all 9')
    ap.add_argument('--no-checkpoint', action='store_true',
                    help='disable weight checkpointing (task t_0db33cce)')
    args = ap.parse_args()

    if args.stage == 1:
        seeds = [args.seed]
    else:
        seeds = parse_seeds(args.seeds)

    if args.cells:
        cell_indices = [int(c) for c in args.cells.split(',')]
    else:
        cell_indices = list(range(9))

    # Output paths
    out_dir = os.environ.get('OUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = None if args.no_checkpoint else os.path.join(out_dir, 'checkpoints')

    if args.output:
        base_name = args.output
    else:
        stage_tag = f"stage{args.stage}"
        seed_tag = f"seed{args.seed}" if args.stage == 1 else f"seeds{seeds[0]}-{seeds[-1]}"
        cell_tag = 'cells' + '-'.join(str(c) for c in cell_indices) if len(cell_indices) < 9 else 'all'
        base_name = os.path.join(
            out_dir, f'boundary_map_arm3_{stage_tag}_{seed_tag}_{cell_tag}.json')
    progress_path = os.path.join(out_dir, 'boundary_map_arm3_PROGRESS.json')

    config = {
        'spec': 'SPEC_CD_BOUNDARY_MAP_ARM3_v1.1',
        'stage': args.stage,
        'engine': 'ablation_cortex_v14_1.py (FROZEN) via AblationCortexOpt',
        'task': 'mult (E_mult discrete-log encoder)',
        'P': P_MOD, 'g': G_PRIM, 'K_freq': K_FREQ, 'in_dim': IN_DIM,
        'hidden_per_layer': HIDDEN, 'sheet_size': SHEET_SIZE,
        't_inf': T_INF, 'n_layers': L_LAMINAR,
        'steps': args.steps, 'batch': BATCH,
        'alpha_theta_0': ALPHA_THETA_0,
        'eta_W_0': ETA_W, 'eta_out': ETA_OUT,
        'chance': CHANCE,
        'eval_every': args.eval_every, 'gate_every': args.gate_every,
        'eval_window_W': 5,
        'grid': [{'cell': i, 'gamma_W': g[0], 'T_decay': g[1], 'gamma_alpha': g[2]}
                 for i, g in enumerate(GRID)],
        'seeds': seeds,
        'cells_run': cell_indices,
    }

    print(f"\n{'='*72}")
    print(f"SPEC_CD_BOUNDARY_MAP_ARM3_v1.1 — P8 Boundary Map")
    print(f"{'='*72}")
    print(f"  Stage:      {args.stage}")
    print(f"  Seeds:      {seeds}")
    print(f"  Cells:      {cell_indices} (of 9)")
    print(f"  Steps:      {args.steps}")
    print(f"  Grid:       γ_W ∈ {{0.3,0.5,0.7}} × T_decay ∈ {{750,1500,3000}}")
    print(f"  γ_α = γ_W/2 (P8 constraint KEPT)")
    print(f"  L={L_LAMINAR}, N={HIDDEN}/layer, T_inf={T_INF}, sheet={SHEET_SIZE}")
    print(f"  Chance:     {CHANCE:.4f}")
    print(f"  Device:     {DEVICE}")
    print(f"  Output:     {base_name}")
    print(f"{'='*72}\n", flush=True)

    all_results = []  # flat list of all cell×seed results

    for seed in seeds:
        for ci in cell_indices:
            gW, Td, gA = GRID[ci]
            print(f"\n--- Cell [{ci}] γ_W={gW}, T_decay={Td}, γ_α={gA}, seed={seed} ---")
            r = run_cell_seed(ci, gW, Td, gA, seed, args.steps,
                              args.eval_every, args.gate_every, ckpt_dir=ckpt_dir)
            all_results.append(r)

            # Write cumulative results after each cell×seed
            try:
                out_data = {'config': config, 'results': all_results}
                if args.stage == 1 and len(all_results) == 9:
                    # Stage 1 complete — evaluate S1-S4
                    results_by_cell = {r['cell_idx']: r for r in all_results}
                    signoff = evaluate_stage1(list(results_by_cell.values()))
                    out_data['stage1_signoff'] = signoff
                elif args.stage == 2 and len(all_results) >= 9 * len(seeds):
                    # Stage 2 complete — compute per-cell verdicts
                    verdicts = {}
                    for ci in cell_indices:
                        cell_results = [r for r in all_results if r['cell_idx'] == ci]
                        if cell_results:
                            verdicts[ci] = {
                                'gamma_W': GRID[ci][0],
                                'T_decay': GRID[ci][1],
                                'gamma_alpha': GRID[ci][2],
                                **cell_verdict(cell_results),
                            }
                    out_data['cell_verdicts'] = verdicts
                with open(base_name, 'w') as f:
                    json.dump(out_data, f, indent=1, default=str)
            except Exception as e:
                print(f"  WARNING: save failed: {e}")

            # Progress file
            try:
                prog = {
                    'stage': args.stage, 'seed': seed, 'cell': ci,
                    'total_done': len(all_results),
                    'total_expected': len(seeds) * len(cell_indices),
                    'last_acc': all_results[-1].get('window_avg_acc', 0),
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }
                with open(progress_path, 'w') as pf:
                    json.dump(prog, pf)
            except Exception:
                pass

    # ── Final summary ──
    print(f"\n{'='*72}")
    print(f"  STAGE {args.stage} COMPLETE — {len(all_results)} cell×seed runs")
    print(f"{'='*72}")

    if args.stage == 1:
        results_by_cell = {r['cell_idx']: r for r in all_results}
        signoff = evaluate_stage1(list(results_by_cell.values()))
        print(f"\n  STAGE 1 SIGN-OFF (§6.1):")
        print(f"    S1 (C2 sanity):       {'PASS' if signoff['S1'] else 'FAIL'}")
        print(f"    S2 (nostab fails):    {'PASS' if signoff['S2'] else 'FAIL'}")
        print(f"    S3 (nostab identity): {'PASS' if signoff['S3'] else 'FAIL'}")
        print(f"    S4 (P4 health):       {'PASS' if signoff['S4'] else 'FAIL'}")
        print(f"    ALL: {'PASS' if signoff['all_pass'] else 'FAIL'}")
        print(f"    → {signoff['recommendation']}")

        print(f"\n  PER-CELL TRAJECTORY SUMMARY (seed {seeds[0]}):")
        print(f"  {'Cell':>4s} {'γ_W':>5s} {'T_d':>5s} {'window':>7s} "
              f"{'final':>7s} {'best':>7s} {'grok?':>6s}")
        for ci in cell_indices:
            r = results_by_cell[ci]
            gr = 'YES' if r['grok_step'] else 'no'
            print(f"  {ci:4d} {r['gamma_W']:5.2f} {r['T_decay']:5d} "
                  f"{r['window_avg_acc']:7.4f} {r['final_acc']:7.4f} "
                  f"{r['best_acc']:7.4f} {gr:>6s}")
            eps = r.get('eps_a_final', [])
            print(f"       ε_a={[f'{e:.3f}' for e in eps]}")

    elif args.stage == 2:
        print(f"\n  PER-CELL BOUNDARY VERDICTS ({len(seeds)} seeds):")
        print(f"  {'Cell':>4s} {'γ_W':>5s} {'T_d':>5s} {'best_mean':>9s} "
              f"{'sustained':>10s} {'verdict':>14s}")
        for ci in cell_indices:
            cell_results = [r for r in all_results if r['cell_idx'] == ci]
            if not cell_results:
                continue
            v = cell_verdict(cell_results)
            gW = GRID[ci][0]
            Td = GRID[ci][1]
            marker = ' ★' if ci == CELL_22 else ''
            nmarker = ' (nostab)' if ci in NOSTAB_ROW else ''
            print(f"  {ci:4d} {gW:5.2f} {Td:5d} {v['best_mean']:9.4f} "
                  f"{v['sustained_rate_str']:>10s} {v['verdict']:>14s}{marker}{nmarker}")

        # Grok-then-forget row check
        nostab_verdicts = []
        for ci in NOSTAB_ROW:
            cell_results = [r for r in all_results if r['cell_idx'] == ci]
            if cell_results:
                nostab_verdicts.append(cell_verdict(cell_results)['verdict'])
        n_grok_forget = sum(1 for v in nostab_verdicts if v == 'grok_forget')
        print(f"\n  NOSTAB ROW (T_decay=3000): {nostab_verdicts}")
        print(f"  Grok-then-forget: {n_grok_forget}/{len(nostab_verdicts)} "
              f"(expected 3/3 per §5.1 ∀ condition)")

    print(f"\n  Results: {base_name}")
    print(f"  Progress: {progress_path}")


if __name__ == '__main__':
    main()
