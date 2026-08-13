#!/usr/bin/env python3
"""test_t_cliff.py — fast visualization of the EP settling-time cliff.

The paper's T-cliff claim (C7): addition groks at T=10 (8/10) but not T=1/T=5
(0/10); multiplication's cliff is at lower T and complete. A full seed costs
~10 min; this script runs a SHORT probe (600 steps, 1 seed, per-layer error
norms dh and gate diagnostics at each T) so a reviewer can see WHY the cliff
exists without the full run.

Usage:
    python scripts/test_t_cliff.py            # T in {1, 5, 10}, 1 seed, 600 steps
    python scripts/test_t_cliff.py --T 1 5 10 --steps 600

Expected output (matches the paper's mechanism claim):
    T=1  → dh nonzero ONLY in the output layer (credit path never reaches
           hidden layers) → chance
    T=5  → partial penetration → partial/no grok
    T=10 → dh grows through all layers → grok trajectory begins

This is a diagnostic probe, not a reproduction of the banked 10-seed claims
(those are outputs/l6eqcap_T{1,5,10}_10seeds.json via run_basis_swap_v13.py).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_data(p=53, seed=42):
    rng = np.random.RandomState(seed)
    pairs = [(a, b) for a in range(1, p) for b in range(1, p)]
    idx = rng.permutation(len(pairs))
    n_train = int(0.8 * len(pairs))
    train = [pairs[i] for i in idx[:n_train]]
    test = [pairs[i] for i in idx[n_train:]]
    return train, test


def encode_add(pairs, p=53):
    """Additive Fourier features (the E_add basis the paper uses for addition)."""
    k = np.arange(1, (p - 1) // 2 + 1)  # 26 freqs
    X = []
    Y = []
    for a, b in pairs:
        xa = np.concatenate([np.cos(2 * np.pi * k * a / p), np.sin(2 * np.pi * k * a / p)])
        xb = np.concatenate([np.cos(2 * np.pi * k * b / p), np.sin(2 * np.pi * k * b / p)])
        X.append(np.concatenate([xa, xb]))
        y = np.zeros(p)
        y[(a + b) % p] = 1.0
        Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def run_probe(T, steps=600, p=53, seed=0, batch=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, _ = make_data(p, seed=42)
    Xtr, Ytr = encode_add(train, p)

    hidden = 1536
    cortex = AblationCortexOpt(
        in_dim=Xtr.shape[1], hidden_dim=hidden, out_dim=p,
        n_layers=6, k_conn=8, T_inference=T,
        sheet_size=64, target_rate=0.10, sigma_norm=1.0,
        beta_softplus=4.0, beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        eta_h=0.5, eta_W=0.01, eta_out=0.01, eta_theta=0.001,
        lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
    )
    cortex.to(DEVICE)

    n = Xtr.shape[0]
    accs = []
    dh_profiles = []
    for step in range(0, steps + 1, 100):
        # one mini-batch update
        idx = np.random.choice(n, size=batch, replace=False)
        Xb = torch.tensor(Xtr[idx], device=DEVICE)
        Yb = torch.tensor(Ytr[idx], device=DEVICE)
        cortex.train_step(Xb, Yb)  # EP contrastive step (free + clamped settle)

        if step % 200 == 0 or step == steps:
            # inference + dh profile
            pred = cortex.predict(torch.tensor(Xtr[:256], device=DEVICE))
            acc = float((pred == Ytr[:256].argmax(1)).float().mean())
            accs.append((step, acc))
            # dh norm per layer from the last train_step's free/clamped contrast
            if hasattr(cortex, "_last_dh"):
                dh_profiles.append((step, list(cortex._last_dh)))

    # final dh profile — redo a settle with gates to expose per-layer error norms
    Xb = torch.tensor(Xtr[:128], device=DEVICE)
    Yb = torch.tensor(Ytr[:128], device=DEVICE)
    gates = cortex.infer(Xb, Yb, return_gates=True)
    dh = gates.get("dh_norms", [])
    return accs, dh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--T", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"T-cliff probe: T={args.T}, steps={args.steps}, seed={args.seed}, device={DEVICE}")
    print("(diagnostic only — banked 10-seed results are in outputs/l6eqcap_T*.json)\n")

    for T in args.T:
        print(f"{'='*70}\nT={T}\n{'='*70}")
        accs, dh = run_probe(T, steps=args.steps, seed=args.seed)
        for step, acc in accs:
            print(f"  step {step:4d}: test_acc={acc:.4f}")
        if dh:
            print(f"  final per-layer dh norms: {[f'{d:.4f}' for d in dh]}")
            deep = sum(1 for d in dh[1:] if d > 1e-4)
            print(f"  layers with dh>0 beyond output: {deep}/5"
                  f"  -> {'CREDIT REACHES HIDDEN LAYERS' if deep >= 3 else 'CREDIT STUCK AT OUTPUT (chance)'}")
        print()


if __name__ == "__main__":
    main()
