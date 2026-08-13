#!/usr/bin/env python3
"""run_matched_mult_add.py -- MATCHED ADD-vs-MULT experiment at (L=6,N=1536,T=10).

Math spec: SPEC_MULT_VS_ADD_MATCHED_L6_v1.0 (task t_3b1530b9, commit 05ac583).
Parent: t_3b1530b9. Card: t_81a4f9ea.

The decisive cell that has NEVER been run: both add and mult on the SAME
driver (run_l6_falsifier), SAME single column, SAME N=1536, SAME T=10,
SAME steps=2000, 10 seeds. Isolates the task-structure variable.

Minimal modification of run_l6_falsifier.py:
- --task {add,mult} switches the label function (one line in make_data)
- Defaults to N=1536, sheet_size=40, T_inf=10 (the l6eqcap proven config)
- Per-product alignment instrumentation (spec section 2.3 / 8.2-8.3)

COMPUTE LANE: 3060 container (training-container via run_slot.sh).
The 2060 local GPU is OFF-LIMITS (user rule 2026-08-09).

Constitution: P1-P8 compliant (spec section 5). No engine internals changed
-- only the label function and instrumentation.

Usage:
  # ADD arm (sanity check -- must reproduce 10/10 grok)
  python run_matched_mult_add.py --task add --seeds 0-9 --steps 2000

  # MULT arm (the test)
  python run_matched_mult_add.py --task mult --seeds 0-9 --steps 2000
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
# Also allow finding the engine from the Windows lane tree
_win_lane = r"D:\PC-hermes\scripts"
if os.path.isdir(_win_lane) and _win_lane not in SCRIPTS_DIR:
    sys.path.insert(0, _win_lane)

from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE

# ── Data constants (IDENTICAL to run_l6_falsifier.py for direct comparability) ──
P_MOD = 53
K_FREQ = 26
IN_DIM = 4 * K_FREQ
CHANCE = 1.0 / P_MOD

L_LAMINAR = 6
HIDDEN_PER_LAYER = 1536       # l6eqcap proven config (coverage fix)
SHEET_SIZE = 40               # 40^2=1600 >= 1536 (P6 legality)
T_INF = 10                    # T_inference=10 (proven EPNet value)
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001
BATCH = 128


def make_data(n_train=2247, n_test=562, seed=42, task='add'):
    """Label function parameterized by task.

    IDENTICAL to run_l6_falsifier.py make_data except the label:
      add  -> cc = (aa + bb) % P_MOD   (existing)
      mult -> cc = (aa * bb) % P_MOD   (the matched experiment)
    Same RNG seed=42, same Fourier encoding, same train/test split.
    """
    rng = np.random.RandomState(seed)
    aa = np.repeat(np.arange(P_MOD), P_MOD)
    bb = np.tile(np.arange(P_MOD), P_MOD)
    if task == 'mult':
        cc = (aa * bb) % P_MOD
    else:
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
    perm = rng.permutation(P_MOD * P_MOD)
    Xtr, Ytr = X[perm[:n_train]], Y[perm[:n_train]]
    Xte, Yte = X[perm[n_train:n_train + n_test]], Y[perm[n_train:n_train + n_test]]
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def to_onehot(Y, n_classes):
    Yoh = torch.zeros(len(Y), n_classes, device=DEVICE)
    Yoh[torch.arange(len(Y)), Y] = 1.0
    return Yoh


def make_column(seed, n_layers=L_LAMINAR, hidden=None):
    """Same column construction as run_l6_falsifier.py make_column."""
    if hidden is None:
        hidden = HIDDEN_PER_LAYER
    net = AblationCortexOpt(
        in_dim=IN_DIM, hidden_dim=hidden, out_dim=P_MOD, n_layers=n_layers,
        sheet_size=SHEET_SIZE,
        target_rate=0.10, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
    )
    return net


def compute_product_alignment(model, X, Y_onehot, batch_size=128, n_permutations=200):
    """Per-product alignment instrumentation (spec section 2.3 / 8.2-8.3).

    For each hidden unit j at layer 0, for each of its 28 product pairs (p,q):
      align[j, p, q] = cos(d_eps_a_0[:, j], pv[:, j, p, q])
    where cos is over the batch dimension.

    Also classifies each product as cross-operand (a*b) vs same-operand
    (a*a or b*b) based on the Fourier feature structure.

    RED-TEAM FIX (T6, t_18d2e176): The 1/sqrt(B) CLT null is invalid for
    structured Fourier data — pv and d_eps_a_0 share spectral structure,
    producing systematic nonzero alignment. We use a PERMUTATION NULL:
    shuffle d_eps_a_0 along the batch dimension to break real input-alignment
    while preserving marginal statistics, then compare observed alignment to
    this empirical null (Bonferroni-corrected for N*n_pairs tests).

    Input features are structured as:
      idx % 4 == 0 -> cos(a), idx % 4 == 1 -> sin(a)  (operand a)
      idx % 4 == 2 -> cos(b), idx % 4 == 3 -> sin(b)  (operand b)
    """
    with torch.no_grad():
        X_dev = X.to(DEVICE)
        Y_dev = Y_onehot.to(DEVICE)
        B = X_dev.shape[0]

        # Run inference to get clamped/free eps_a
        result = model.infer(X_dev, Y_dev, return_gates=False)
        eps_a_free = result['eps_a_free']
        eps_a_clamped = result['eps_a_clamped']
        d_eps_a_0 = eps_a_clamped[0] - eps_a_free[0]  # [B, N]

        # Recompute product features (same as train_step L644-645)
        cv = X_dev[:, model.conn]                       # [B, N, k_conn]
        pv = cv[:, :, model.pi] * cv[:, :, model.pj]   # [B, N, n_pairs]

        # Cosine alignment over batch dim: cos(d_eps[:, j], pv[:, j, q])
        # Vectorized: numerator = einsum over batch
        num = torch.einsum('bn,bnp->np', d_eps_a_0, pv)  # [N, n_pairs]
        d_eps_norm = d_eps_a_0.norm(dim=0)                # [N]
        pv_norm = pv.norm(dim=0)                           # [N, n_pairs]
        align = num / (d_eps_norm.unsqueeze(1) * pv_norm + 1e-12)  # [N, n_pairs]

        # ── PERMUTATION NULL (red-team fix T6) ──
        # Shuffle d_eps_a_0 along batch dim, recompute alignment, build
        # empirical null distribution. This breaks real input-alignment
        # while preserving marginal (per-neuron) statistics.
        N_units = align.shape[0]
        n_pairs = align.shape[1]
        n_tests = N_units * n_pairs

        perm_max_aligns = torch.zeros(n_permutations, device=DEVICE)
        for perm_i in range(n_permutations):
            perm_idx = torch.randperm(B, device=DEVICE)
            d_eps_perm = d_eps_a_0[perm_idx]  # [B, N] shuffled along batch
            num_perm = torch.einsum('bn,bnp->np', d_eps_perm, pv)
            align_perm = num_perm / (d_eps_norm.unsqueeze(1) * pv_norm + 1e-12)
            perm_max_aligns[perm_i] = align_perm.abs().max()

        # Per-element null threshold: 95th percentile of permutation max
        # (Bonferroni-corrected: we're testing the max over all elements)
        perm_threshold_95 = float(torch.quantile(perm_max_aligns, 0.95).item())
        perm_threshold_99 = float(torch.quantile(perm_max_aligns, 0.99).item())
        # Also compute per-element null (mean + 2*std of perm alignments)
        # for FDR-style comparison
        all_perm_aligns = perm_max_aligns  # max over N*n_pairs per permutation

        # ── Classify each product as cross-operand vs same-operand ──
        conn_cpu = model.conn.cpu().numpy()
        operand = np.where((conn_cpu % 4) < 2, 'a', 'b')  # [N, k_conn]
        pi = model.pi.cpu().numpy()
        pj = model.pj.cpu().numpy()
        op_p = operand[:, pi]  # [N, n_pairs]
        op_q = operand[:, pj]  # [N, n_pairs]
        cross_mask = (op_p != op_q)  # [N, n_pairs]

        align_cpu = align.cpu().numpy()
        abs_align = np.abs(align_cpu)

        cross_aligns = abs_align[cross_mask]
        same_aligns = abs_align[~cross_mask]

        # OLD CLT floor (kept for reference / comparison)
        clt_noise_floor = 1.0 / np.sqrt(B)
        clt_thresh_2sigma = 2.0 * clt_noise_floor

        # NEW: permutation null thresholds
        frac_above_perm_95 = float((abs_align > perm_threshold_95).mean())
        frac_above_perm_99 = float((abs_align > perm_threshold_99).mean())
        frac_cross_above_perm_95 = float(
            (cross_aligns > perm_threshold_95).mean()) if len(cross_aligns) > 0 else 0.0
        frac_cross_above_perm_99 = float(
            (cross_aligns > perm_threshold_99).mean()) if len(cross_aligns) > 0 else 0.0

        # Also keep old CLT stats for comparison
        frac_above_clt = float((abs_align > clt_thresh_2sigma).mean())
        frac_cross_above_clt = float(
            (cross_aligns > clt_thresh_2sigma).mean()) if len(cross_aligns) > 0 else 0.0

        # Top-K aligned products per unit (sample a few units)
        n_top = 10
        top_per_unit = {}
        for j in range(min(5, align_cpu.shape[0])):
            top_idx = np.argsort(-abs_align[j])[:n_top]
            top_per_unit[f'unit_{j}'] = [
                {'pair_idx': int(idx),
                 'feat_i': int(conn_cpu[j, pi[idx]]),
                 'feat_j': int(conn_cpu[j, pj[idx]]),
                 'op': f"{op_p[j, idx]}*{op_q[j, idx]}",
                 'abs_align': float(abs_align[j, idx]),
                 'above_perm95': bool(abs_align[j, idx] > perm_threshold_95)}
                for idx in top_idx
            ]

        # Histogram bins
        hist_bins = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
        hist, _ = np.histogram(abs_align.flatten(), bins=hist_bins)
        hist_pct = (hist / len(abs_align.flatten()) * 100).tolist()

        return {
            'n_units': int(N_units),
            'n_pairs_per_unit': int(n_pairs),
            'total_products': int(n_tests),
            'batch_size': int(B),
            # Null models
            'clt_noise_floor': float(clt_noise_floor),
            'clt_thresh_2sigma': float(clt_thresh_2sigma),
            'perm_threshold_95': float(perm_threshold_95),
            'perm_threshold_99': float(perm_threshold_99),
            'n_permutations': int(n_permutations),
            # Alignment statistics
            'mean_abs_align': float(abs_align.mean()),
            'median_abs_align': float(np.median(abs_align)),
            'max_abs_align': float(abs_align.max()),
            'std_abs_align': float(abs_align.std()),
            # OLD CLT-based stats (kept for comparison, NOT for verdict)
            'frac_above_clt_2sigma': frac_above_clt,
            'frac_cross_above_clt_2sigma': frac_cross_above_clt,
            # NEW permutation-null-based stats (USE THESE for verdict)
            'frac_above_perm_95': frac_above_perm_95,
            'frac_above_perm_99': frac_above_perm_99,
            'frac_cross_above_perm_95': frac_cross_above_perm_95,
            'frac_cross_above_perm_99': frac_cross_above_perm_99,
            # Cross vs same operand
            'mean_abs_align_cross': float(cross_aligns.mean()) if len(cross_aligns) > 0 else 0.0,
            'mean_abs_align_same': float(same_aligns.mean()) if len(same_aligns) > 0 else 0.0,
            'n_cross': int(cross_mask.sum()),
            'n_same': int((~cross_mask).sum()),
            # Histogram of |align|
            'hist_bins': hist_bins.tolist(),
            'hist_pct': hist_pct,
            # Top aligned products per unit (first 5 units)
            'top_per_unit': top_per_unit,
            # Verdict using permutation null
            'verdict_branch_hint': _alignment_verdict_perm(
                abs_align, cross_aligns, perm_threshold_95,
                frac_above_perm_95, frac_cross_above_perm_95),
        }


def _alignment_verdict_perm(abs_align, cross_aligns, perm_threshold,
                            frac_above_perm, frac_cross_above_perm):
    """Classify into branch B/C using the PERMUTATION null (red-team fix T6).

    Branch B (expressivity ceiling): frac_above_perm < 0.01
      — almost no products exceed the permutation null = no real credit signal
    Branch C (credit dilution): frac_cross_above_perm > 0.02
      — some cross-operand products show real alignment above null = partial
      expressivity but credit is diluted
    Ambiguous: between B and C — run more permutations or larger batch
    """
    if frac_above_perm < 0.01:
        return 'B_expressivity_ceiling'
    elif frac_cross_above_perm > 0.02:
        return 'C_credit_dilution'
    else:
        return 'ambiguous_B_or_C'


def run_single_column(seed, task='add', steps=2000, eval_every=100,
                      progress_path=None, hidden=None):
    """Run ONE AblationCortex column at L=6 with task=add or mult."""
    t0 = time.time()
    col_seed = seed * 100 + 0  # matches CC3 column 0 seed

    Xtr, Ytr, Xte, Yte = make_data(seed=42, task=task)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Yoh = to_onehot(Ytr, P_MOD)

    model = make_column(seed=col_seed, hidden=hidden)
    model.calibrate_thresholds(Xtr[:200])

    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0
    diag_log = []

    for step in range(1, steps + 1):
        idx = rng.randint(0, len(Xtr), BATCH)
        model.train_step(Xtr[idx], Yoh[idx], return_gates=False)

        if step % eval_every == 0 or step == 1:
            acc = model.evaluate(Xte, Yte)
            best = max(best, acc)
            if acc >= 0.9 and grok_step is None:
                grok_step = step
            history.append((step, acc))

            diag_entry = {'step': step, 'test_acc': acc}
            diag_log.append(diag_entry)

            if step % 500 == 0 or step == eval_every:
                print(f"  [{task} s{seed}] step {step:5d}: test={acc:.3f} "
                      f"best={best:.3f} [{time.time()-t0:.0f}s]", flush=True)

            if progress_path:
                try:
                    prog = {
                        'task': task, 'seed': seed, 'step': step, 'acc': acc,
                        'best': best, 'grok_step': grok_step,
                        'elapsed_s': time.time() - t0,
                        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    }
                    with open(progress_path, 'w') as pf:
                        json.dump(prog, pf)
                except Exception:
                    pass

    # ── Final full credit propagation profile ──
    print(f"  [{task} s{seed}] computing final credit-propagation profile...", flush=True)
    with torch.no_grad():
        full_diag = model.infer(Xte[:64], to_onehot(Yte[:64], P_MOD),
                                return_gates=True)
    gate_log = full_diag.get('gate_log', {})
    diag_log.append({'step': steps, 'type': 'full_credit_diag', **gate_log})

    if gate_log:
        g1 = gate_log.get('gate1', 0)
        g2min = gate_log.get('gate2_min', 0)
        g2mean = gate_log.get('gate2_mean', 0)
        d_dep = gate_log.get('d_dependent_contrastive', [])
        cbias = gate_log.get('contrastive_bias_part', [])
        eps_a = gate_log.get('eps_a_norms_clamped', [])
        dh = gate_log.get('dh_norms', [])
        hoy = gate_log.get('hoyer', [])
        fr = gate_log.get('firing_rates', [])
        print(f"  [{task} s{seed}] CREDIT PROFILE:")
        print(f"    Gate1 = {g1:.3f}, Gate2(min/mean) = {g2min:.3f}/{g2mean:.3f}")
        print(f"    d_dep = {[round(d,6) for d in d_dep]}")
        print(f"    bias  = {[round(b,6) for b in cbias]}")
        print(f"    eps_a = {[round(e,4) for e in eps_a]}")
        print(f"    dh    = {[round(d,4) for d in dh]}")
        print(f"    fr    = {[round(f,3) for f in fr]}")

    dt = time.time() - t0
    return {
        'seed': seed, 'task': task, 'steps': steps,
        'final_test_acc': history[-1][1] if history else 0.0,
        'best_test_acc': best, 'grok_step': grok_step,
        'hidden_per_layer': hidden if hidden is not None else HIDDEN_PER_LAYER,
        'sheet_size': SHEET_SIZE,
        't_inf': T_INF,
        'effective_T': T_INF,
        'L': L_LAMINAR,
        'n_col': 1, 'chance': CHANCE,
        'time': dt,
        'diag_log': diag_log,
    }


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


def main():
    global HIDDEN_PER_LAYER, SHEET_SIZE, T_INF
    ap = argparse.ArgumentParser(
        description='MATCHED ADD-vs-MULT at (L=6,N=1536,T=10) -- spec t_3b1530b9')
    ap.add_argument('--task', choices=['add', 'mult'], default='add',
                    help='add: cc=(aa+bb)%%P; mult: cc=(aa*bb)%%P')
    ap.add_argument('--seeds', type=str, default='0-9',
                    help="seed spec: '0-9' or '0 1 2' or '0-4,7'")
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--output', default=None)
    ap.add_argument('--hidden-per-layer', type=int, default=None,
                    help='hidden units per layer (default: %(default)s)')
    ap.add_argument('--sheet-size', type=int, default=None,
                    help='sheet size (default: %(default)s)')
    ap.add_argument('--t-inf', type=int, default=None,
                    help='T_inference (default: %(default)s)')
    ap.add_argument('--alignment', action='store_true',
                    help='run per-product alignment instrumentation after '
                         'training (auto-enabled for mult if grok_rate < 8/10)')
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)

    if args.hidden_per_layer is not None:
        HIDDEN_PER_LAYER = args.hidden_per_layer
    if args.sheet_size is not None:
        SHEET_SIZE = args.sheet_size
    if args.t_inf is not None:
        T_INF = args.t_inf

    # P6 legality check
    assert HIDDEN_PER_LAYER <= SHEET_SIZE ** 2, (
        f"P6 VIOLATION: hidden_dim={HIDDEN_PER_LAYER} > sheet_size^2="
        f"{SHEET_SIZE**2}")

    if sys.platform == 'win32':
        out_dir = r"D:\PC-hermes\outputs"
    else:
        out_dir = os.environ.get('OUT_DIR', '/root/gate2/outputs')
    os.makedirs(out_dir, exist_ok=True)
    default_name = f'matched_{args.task}_L6_N{HIDDEN_PER_LAYER}_T{T_INF}.json'
    base_name = args.output or os.path.join(out_dir, default_name)
    progress_path = os.path.join(
        out_dir, f'matched_{args.task}_PROGRESS.json')

    print(f"MATCHED ADD-vs-MULT -- task={args.task}")
    print(f"  Spec: SPEC_MULT_VS_ADD_MATCHED_L6_v1.0 (t_3b1530b9)")
    print(f"  L={L_LAMINAR}, N={HIDDEN_PER_LAYER}/layer, T_inf={T_INF}")
    print(f"  sheet_size={SHEET_SIZE} ({SHEET_SIZE**2} positions, "
          f"{HIDDEN_PER_LAYER}/{SHEET_SIZE**2} used)")
    print(f"  task: {args.task} ({'cc=(aa+bb)%P' if args.task == 'add' else 'cc=(aa*bb)%P'})")
    print(f"  seeds: {seeds}")
    print(f"  steps: {args.steps}")
    print(f"  device: {DEVICE}")
    print(f"  output: {base_name}")
    print(f"  chance: {CHANCE:.4f}")

    results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  {args.task.upper()} SEED {seed}")
        print(f"{'='*60}")
        r = run_single_column(seed, task=args.task, steps=args.steps,
                              eval_every=args.eval_every,
                              progress_path=progress_path,
                              hidden=HIDDEN_PER_LAYER)
        verdict = "GROK" if r['final_test_acc'] >= 0.9 else \
                  ("LEARN" if r['best_test_acc'] >= 0.3 else "CHANCE")
        print(f"  [DONE] {args.task} s{seed}: final={r['final_test_acc']:.3f} "
              f"best={r['best_test_acc']:.3f} grok={r['grok_step']} "
              f"[{verdict}] ({r['time']:.0f}s)", flush=True)
        results.append(r)

        # Write cumulative results after EACH seed
        try:
            with open(base_name, 'w') as f:
                config = {
                    'task': args.task,
                    'label_fn': '(aa+bb)%P' if args.task == 'add' else '(aa*bb)%P',
                    'hidden_per_layer': HIDDEN_PER_LAYER,
                    'sheet_size': SHEET_SIZE,
                    't_inf': T_INF,
                    'effective_T': T_INF,
                    'n_layers': L_LAMINAR,
                    'steps': args.steps,
                    'seeds': seeds,
                    'chance': CHANCE,
                    'spec': 'SPEC_MULT_VS_ADD_MATCHED_L6_v1.0_t_3b1530b9',
                }
                out = {'config': config, 'results': results}
                json.dump(out, f, indent=1, default=str)
        except Exception:
            pass

    # ── Summary ──
    finals = [r['final_test_acc'] for r in results]
    bests = [r['best_test_acc'] for r in results]
    n_grok = sum(1 for r in results if r['grok_step'] is not None)
    grok_steps = [r['grok_step'] for r in results if r['grok_step'] is not None]

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {args.task.upper()} arm (L=6, N={HIDDEN_PER_LAYER}, T={T_INF})")
    print(f"{'='*70}")
    print(f"  Seeds: {len(results)}")
    print(f"  Final acc: median={np.median(finals):.3f} "
          f"mean={np.mean(finals):.3f} ±{np.std(finals):.3f}")
    print(f"  Best acc:  median={np.median(bests):.3f} "
          f"mean={np.mean(bests):.3f} ±{np.std(bests):.3f}")
    print(f"  Grok rate (>=0.9 final): {n_grok}/{len(results)}")
    if grok_steps:
        print(f"  Grok steps: {grok_steps}, median={int(np.median(grok_steps))}")
    print(f"  Chance: {CHANCE:.4f}")

    # ── Per-product alignment instrumentation (if mult fails or --alignment) ──
    alignment_report = None
    do_alignment = args.alignment or (args.task == 'mult' and n_grok < 8)
    if do_alignment:
        print(f"\n{'='*70}")
        print(f"  PER-PRODUCT ALIGNMENT INSTRUMENTATION")
        print(f"{'='*70}")
        print(f"  (spec section 2.3 / 8.2-8.3)")

        # Rebuild the last-seed model for instrumentation
        last_seed = seeds[-1]
        col_seed = last_seed * 100 + 0
        Xtr, Ytr, Xte, Yte = make_data(seed=42, task=args.task)
        Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
        model = make_column(seed=col_seed, hidden=HIDDEN_PER_LAYER)
        model.calibrate_thresholds(Xtr[:200])

        # Re-train to last step (model state is not persisted between seeds)
        rng = np.random.RandomState(last_seed)
        Yoh_full = to_onehot(Ytr, P_MOD)
        for step in range(1, args.steps + 1):
            idx = rng.randint(0, len(Xtr), BATCH)
            model.train_step(Xtr[idx], Yoh_full[idx], return_gates=False)

        # Run alignment on a test batch
        align_data = compute_product_alignment(
            model, Xte[:128], to_onehot(Yte[:128], P_MOD))

        print(f"  N={align_data['n_units']}, pairs/unit={align_data['n_pairs_per_unit']}")
        print(f"  Total products: {align_data['total_products']}")
        print(f"  CLT noise floor (1/sqrt(B)): {align_data['clt_noise_floor']:.4f} [REFERENCE ONLY]")
        print(f"  Permutation null thresholds (n={align_data['n_permutations']}):")
        print(f"    95th pct: {align_data['perm_threshold_95']:.4f}")
        print(f"    99th pct: {align_data['perm_threshold_99']:.4f}")
        print(f"  Mean |align|: {align_data['mean_abs_align']:.4f}")
        print(f"  Max |align|:  {align_data['max_abs_align']:.4f}")
        print(f"  Frac above perm-95: {align_data['frac_above_perm_95']:.4f}")
        print(f"  Frac above perm-99: {align_data['frac_above_perm_99']:.4f}")
        print(f"  [OLD CLT frac above 2sigma: {align_data['frac_above_clt_2sigma']:.4f} — REFERENCE ONLY]")
        print(f"  Cross-operand (n={align_data['n_cross']}):")
        print(f"    mean |align| = {align_data['mean_abs_align_cross']:.4f}")
        print(f"    frac above perm-95 = {align_data['frac_cross_above_perm_95']:.4f}")
        print(f"  Same-operand (n={align_data['n_same']}):")
        print(f"    mean |align| = {align_data['mean_abs_align_same']:.4f}")
        print(f"  Histogram |align| (bins {align_data['hist_bins']}):")
        for i, pct in enumerate(align_data['hist_pct']):
            print(f"    [{align_data['hist_bins'][i]:.2f}-{align_data['hist_bins'][i+1]:.2f}): "
                  f"{pct:.1f}%")
        print(f"  Top-10 aligned products (unit 0):")
        for item in align_data['top_per_unit']['unit_0']:
            flag = " ***PERM95***" if item['above_perm95'] else ""
            print(f"    pair {item['pair_idx']:2d}: feat({item['feat_i']:3d},{item['feat_j']:3d}) "
                  f"{item['op']:4s} |align|={item['abs_align']:.4f}{flag}")
        print(f"\n  VERDICT BRANCH HINT (permutation null): {align_data['verdict_branch_hint']}")

        alignment_report = align_data

        # Save alignment report
        align_path = base_name.replace('.json', '_alignment.json')
        try:
            with open(align_path, 'w') as f:
                json.dump(align_data, f, indent=1, default=str)
            print(f"  Saved alignment report: {align_path}")
        except Exception as e:
            print(f"  WARNING: could not save alignment report: {e}")

    # Write final summary JSON with alignment
    try:
        with open(base_name, 'r') as f:
            full_out = json.load(f)
        full_out['summary'] = {
            'task': args.task,
            'n_seeds': len(results),
            'final_acc_median': float(np.median(finals)),
            'final_acc_mean': float(np.mean(finals)),
            'final_acc_std': float(np.std(finals)),
            'best_acc_median': float(np.median(bests)),
            'best_acc_mean': float(np.mean(bests)),
            'grok_rate': f"{n_grok}/{len(results)}",
            'grok_steps': grok_steps,
            'chance': CHANCE,
        }
        if alignment_report:
            full_out['alignment'] = alignment_report
        with open(base_name, 'w') as f:
            json.dump(full_out, f, indent=1, default=str)
    except Exception:
        pass

    print(f"\n  Results: {base_name}")


if __name__ == '__main__':
    main()
