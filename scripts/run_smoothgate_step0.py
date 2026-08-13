#!/usr/bin/env python3
"""run_smoothgate_step0.py -- Step 0: Provenance re-verification of Fix 1'.

SPEC v1.5 §9 Step 0 (MANDATORY before any L=6 test):
  Re-run L=2 CC3 add-sanity gate with Fix 1' applied (β=12 + HARD gate kept)
  and NOTHING else changed (compound_cap/preconditioner/clip_w_out all OFF,
  T=40). The v14.1 code path is preserved EXCEPT beta is raised 4→12.

  Config: L=2, N=4096, T=40 (10*L), beta=12, HARD gate KEPT.
  10 seeds, 3000 steps.

  PASS: >= 8/10 seeds grok add (held-out accuracy > 0.9).
  Proven baseline: 10/10 with hard gate at beta=4.
  If FAIL (<8/10): Fix 1' FULLY WITHDRAWN, revert to Fix 2+3 at OLD beta=4.

Usage:
  python -u run_smoothgate_step0.py
"""
import sys, os, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_cortex_v14_2 import AblationCortex, DEVICE

P = 53
K_FREQ = 26
IN_DIM = 4 * K_FREQ
CHANCE = 1.0 / P


def make_data(n_train=2247, n_test=562, seed=42):
    rng = np.random.RandomState(seed)
    aa = np.repeat(np.arange(P), P)
    bb = np.tile(np.arange(P), P)
    cc = (aa + bb) % P
    freqs = np.arange(1, K_FREQ + 1, dtype=np.float32)
    ta = 2.0 * np.pi * np.outer(aa, freqs) / P
    tb = 2.0 * np.pi * np.outer(bb, freqs) / P
    X = np.empty((P * P, IN_DIM), dtype=np.float32)
    X[:, 0::4] = np.cos(ta); X[:, 1::4] = np.sin(ta)
    X[:, 2::4] = np.cos(tb); X[:, 3::4] = np.sin(tb)
    Y = cc.astype(np.int64)
    perm = rng.permutation(P * P)
    Xtr, Ytr = X[perm[:n_train]], Y[perm[:n_train]]
    Xte, Yte = X[perm[n_train:n_train + n_test]], Y[perm[n_train:n_train + n_test]]
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def to_onehot(Y, n_classes):
    Yoh = torch.zeros(len(Y), n_classes, device=DEVICE)
    Yoh[torch.arange(len(Y)), Y] = 1.0
    return Yoh


def run_single_seed(seed, L=2, N=4096, steps=3000, batch_size=128,
                    eval_every=200, gate_every=200):
    Xtr, Ytr, Xte, Yte = make_data()
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    Ytr_oh = to_onehot(Ytr, P)

    # Step 0 config: Fix 1' ONLY (β=12 + HARD gate). Everything else = OLD v14.1.
    # smooth_gate=False (hard gate KEPT), compound_cap=False, preconditioner=False,
    # clip_w_out=False. T=40 (= 10*L, the OLD default), beta_softplus=12.
    net = AblationCortex(
        in_dim=IN_DIM, hidden_dim=N, out_dim=P, n_layers=L,
        sheet_size=int(np.ceil(np.sqrt(N))),
        target_rate=0.10, sigma_norm=1.0, beta_softplus=12.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=40,  # OLD: 10*L, NOT the Fix 2a 20*L
        eta_h=0.5, eta_W=0.01, eta_out=0.01, eta_theta=0.001,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
        smooth_gate=False,    # Fix 1': HARD gate KEPT, β=12
        compound_cap=False,   # Fix 2d OFF (keep old 1/sqrt(L))
        preconditioner=False, # Fix 2b OFF
        clip_w_out=False,     # Fix 3 OFF
    )
    net.calibrate_thresholds(Xtr[:200])

    results = {
        'seed': seed, 'L': L, 'N': N,
        'steps': [], 'train_acc': [], 'test_acc': [],
    }

    t0 = time.time()
    rng_batch = np.random.RandomState(seed * 1000)

    for step in range(1, steps + 1):
        idx = rng_batch.randint(0, len(Xtr), batch_size)
        Xb = Xtr[idx]
        Yb_oh = Ytr_oh[idx]
        do_gates = (step % gate_every == 0) or (step == 1) or (step == steps)
        gate_log = net.train_step(Xb, Yb_oh, return_gates=do_gates)

        if step % eval_every == 0 or step == 1 or step == steps:
            train_acc = net.evaluate(Xtr[:500], Ytr[:500])
            test_acc = net.evaluate(Xte, Yte)
            results['steps'].append(step)
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            fr = gate_log.get('firing_rates', []) if gate_log else []
            fr_str = ' '.join(f'{f:.2f}' for f in fr)
            g2m = gate_log.get('gate2_mean', 0) if gate_log else 0
            print(f"  seed={seed} step={step:4d} train={train_acc:.3f} "
                  f"test={test_acc:.3f} G2={g2m:+.2f} fr=[{fr_str}] "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    results['final_test_acc'] = results['test_acc'][-1]
    results['best_test_acc'] = max(results['test_acc'])
    results['grokked'] = results['final_test_acc'] > 0.9
    results['elapsed'] = time.time() - t0
    return results


def main():
    seeds = 10
    L = 2
    N = 4096
    steps = 3000

    print(f"\n{'='*70}")
    print(f"Step 0: Provenance re-verification of Fix 1' (β=12 + HARD gate)")
    print(f"  L={L}, N={N}, T=40 (10*L), β=12, HARD gate KEPT")
    print(f"  compound_cap=OFF, preconditioner=OFF, clip_w_out=OFF")
    print(f"  {seeds} seeds, {steps} steps, chance={CHANCE:.4f}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}\n", flush=True)

    all_results = []
    for seed in range(seeds):
        print(f"--- Seed {seed}/{seeds} ---")
        result = run_single_seed(seed, L=L, N=N, steps=steps)
        all_results.append(result)
        verdict = "GROK" if result['grokked'] else "CHANCE"
        print(f"  => FINAL={result['final_test_acc']:.4f} "
              f"BEST={result['best_test_acc']:.4f} | {verdict} | "
              f"{result['elapsed']:.0f}s\n", flush=True)

    final_accs = [r['final_test_acc'] for r in all_results]
    grokked = sum(r['grokked'] for r in all_results)

    print(f"\n{'='*70}")
    print(f"STEP 0 SUMMARY")
    print(f"  Grokked: {grokked}/{seeds}")
    print(f"  Final:   {np.mean(final_accs):.4f} ± {np.std(final_accs):.4f}")
    print(f"  Chance:  {CHANCE:.4f}")
    print(f"  PASS criterion: >= 8/10 grok")
    if grokked >= 8:
        print(f"  VERDICT: PASS — Fix 1' (β=12+hard gate) is safe on proven L=2 path")
    else:
        print(f"  VERDICT: FAIL — Fix 1' WITHDRAWN, revert to Fix 2+3 at OLD β=4")
    print(f"{'='*70}\n", flush=True)

    save_data = {
        'step': 'step0',
        'config': {
            'L': L, 'N': N, 'T': 40, 'beta': 12,
            'smooth_gate': False, 'compound_cap': False,
            'preconditioner': False, 'clip_w_out': False,
            'seeds': seeds, 'steps': steps,
        },
        'results': all_results,
        'summary': {
            'grokked': grokked, 'seeds': seeds,
            'final_mean': float(np.mean(final_accs)),
            'final_std': float(np.std(final_accs)),
            'chance': CHANCE,
            'pass': grokked >= 8,
        },
    }
    outpath = '/root/gate2/outputs/smoothgate_step0_L2_beta12.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Results saved to {outpath}")


if __name__ == '__main__':
    main()
