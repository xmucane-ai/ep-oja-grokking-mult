#!/usr/bin/env python3
"""run_temporal_v14.py -- SPEC v14.4/v14.5 Experiment Harness.

Three arms (§9.1):
  1. Treatment (temporal): TemporalColumn / MultiColumnCortex (Algorithm 2)
  2. Baseline (spatial): PC-ALM activity-contrast (v13.12)
  3. Ceiling (dense-BP): standard backprop MLP

Task: modular addition (P=53, Fourier-encoded inputs).
Falsification gate (§9.3): temporal FAILS TO GROK on >=8/10 seeds -> FALSIFIED.

Usage:
  python -u run_temporal_v14.py --arm temporal --seeds 10 --steps 20000
  python -u run_temporal_v14.py --arm dense_bp --seeds 10 --steps 20000
  python -u run_temporal_v14.py --arm spatial --seeds 10 --steps 20000
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_v14 import (TemporalColumn, MultiColumnCortex, DenseBPControl,
                          DEVICE)


# ================================================================
# Task: Modular addition with Fourier encoding
# ================================================================
def fourier_encode(a, b, P):
    """Fourier input encoding: [cos/sin(2*pi*k*a/P), cos/sin(2*pi*k*b/P)]."""
    k = np.arange(1, P // 2 + 1)
    a_feat = np.concatenate([
        np.cos(2 * np.pi * k[None, :] * a[:, None] / P),
        np.sin(2 * np.pi * k[None, :] * a[:, None] / P)], axis=1)
    b_feat = np.concatenate([
        np.cos(2 * np.pi * k[None, :] * b[:, None] / P),
        np.sin(2 * np.pi * k[None, :] * b[:, None] / P)], axis=1)
    X = np.concatenate([a_feat, b_feat], axis=1).astype(np.float32)
    return torch.from_numpy(X)


def generate_modular_addition(P=53, train_frac=0.30, seed=0):
    rng = np.random.RandomState(seed)
    a = np.arange(P)
    b = np.arange(P)
    aa, bb = np.meshgrid(a, b, indexing='ij')
    a_flat = aa.flatten()
    b_flat = bb.flatten()
    y_flat = (a_flat + b_flat) % P
    n = P * P
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    X_train = fourier_encode(a_flat[train_idx], b_flat[train_idx], P)
    X_test = fourier_encode(a_flat[test_idx], b_flat[test_idx], P)
    Y_train = torch.from_numpy(y_flat[train_idx].astype(np.int64))
    Y_test = torch.from_numpy(y_flat[test_idx].astype(np.int64))
    return (X_train, Y_train, X_test, Y_test, P)


# ================================================================
# Temporal arm (one seed)
# ================================================================
def run_temporal_seed(seed, P=53, steps=20000, batch_size=512,
                      N=4096, L_c=2, n_columns=1,
                      lambda_trace=0.95, T_behavior=5, T_replay=5,
                      T_theta=20, eta_W=0.01, eta_out=0.01, eta_theta=0.01,
                      beta_hc=0.1, target_rate=0.10,
                      p_reactivate=0.3, swr_noise=0.1, T_recal=200,
                      eval_interval=500, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, Y_train, X_test, Y_test, _ = generate_modular_addition(
        P, train_frac=0.30, seed=seed)
    in_dim = X_train.shape[1]
    out_dim = P

    Y_train_oh = torch.nn.functional.one_hot(Y_train, P).float()
    X_train = X_train.to(DEVICE); Y_train_oh = Y_train_oh.to(DEVICE)
    Y_train_y = Y_train.to(DEVICE)
    X_test = X_test.to(DEVICE); Y_test_y = Y_test.to(DEVICE)

    n_train = X_train.shape[0]
    rng = np.random.RandomState(seed)

    cortex_kwargs = dict(
        N=N, out_dim=out_dim, L_c=L_c,
        sigma_0=1.0, beta=4.0, target_rate=target_rate,
        lambda_trace=lambda_trace, lambda_hier=0.9,
        T_behavior=T_behavior, T_replay=T_replay, T_theta=T_theta,
        eta_W=eta_W, eta_out=eta_out, eta_theta=eta_theta,
        beta_hc=beta_hc, alpha_ema=0.10,
        lambda_wd=0.001, gamma_rms=0.9, w_clip=5.0,
        p_reactivate=p_reactivate, swr_noise=swr_noise, T_recal=T_recal)

    if n_columns > 1:
        cortex = MultiColumnCortex(in_dim, n_columns=n_columns,
                                   seed=seed, **cortex_kwargs)
    else:
        cortex = TemporalColumn(in_dim=in_dim, seed=seed, **cortex_kwargs)

    # Calibrate thresholds from data (pre-activation quantiles)
    if isinstance(cortex, TemporalColumn):
        with torch.no_grad():
            x, u_list = cortex.forward(X_train[:256])
            for l in range(cortex.L_c):
                cortex.thresholds[l] = torch.quantile(
                    u_list[l].float(), 1.0 - target_rate, dim=0).clamp(0, 5.0)
    elif isinstance(cortex, MultiColumnCortex):
        for col in cortex.columns:
            with torch.no_grad():
                x, u_list = col.forward(X_train[:256])
                for l in range(col.L_c):
                    col.thresholds[l] = torch.quantile(
                        u_list[l].float(), 1.0 - target_rate, dim=0).clamp(0, 5.0)

    results = {
        'seed': seed, 'arm': 'temporal',
        'config': {'N': N, 'L_c': L_c, 'n_columns': n_columns,
                   'lambda_trace': lambda_trace,
                   'T_behavior': T_behavior, 'T_replay': T_replay,
                   'T_theta': T_theta, 'eta_W': eta_W, 'eta_theta': eta_theta,
                   'beta_hc': beta_hc, 'target_rate': target_rate,
                   'p_reactivate': p_reactivate, 'swr_noise': swr_noise,
                   'T_recal': T_recal,
                   'P': P, 'steps': steps, 'batch_size': batch_size},
        'train_acc_curve': [], 'test_acc_curve': [],
        'gate_T1': [], 'r_norms': [], 'trace_norms': [],
        'firing_rates': [], 'gate_T2_final': None}

    t0 = time.time()

    for step in range(steps):
        batch_idx = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
        Xb = X_train[batch_idx]; Yb = Y_train_oh[batch_idx]
        return_gates = (step % eval_interval == 0) or (step == steps - 1)
        gates = cortex.train_step(Xb, Yb, return_gates=return_gates)

        if return_gates:
            train_acc = cortex.evaluate(Xb, Y_train_y[batch_idx])
            test_acc = cortex.evaluate(X_test, Y_test_y)
            results['train_acc_curve'].append(train_acc)
            results['test_acc_curve'].append(test_acc)
            if 'gate_T1' in gates:
                results['gate_T1'].append(gates['gate_T1'])
            if 'r_norms' in gates:
                results['r_norms'].append(gates['r_norms'])
            if 'trace_norms' in gates:
                results['trace_norms'].append(gates['trace_norms'])
            if 'firing_rates' in gates:
                results['firing_rates'].append(gates['firing_rates'])
            if verbose:
                fr = gates.get('firing_rates', [])
                fr_str = ','.join(f'{v:.3f}' for v in fr) if fr else ''
                print(f"  seed={seed} step={step:5d}/{steps} "
                      f"train={train_acc:.4f} test={test_acc:.4f} "
                      f"T1={gates.get('gate_T1', 0):.3f} fr=[{fr_str}] "
                      f"({time.time()-t0:.0f}s)", flush=True)

    if isinstance(cortex, TemporalColumn):
        try:
            Y_test_oh = torch.nn.functional.one_hot(Y_test, P).float().to(DEVICE)
            t2 = cortex.compute_gate_T2(X_test[:256], Y_test_oh[:256])
            results['gate_T2_final'] = t2
        except Exception as e:
            results['gate_T2_final'] = {'error': str(e)}

    results['final_train_acc'] = cortex.evaluate(X_train, Y_train_y)
    results['final_test_acc'] = cortex.evaluate(X_test, Y_test_y)
    results['grok'] = results['final_test_acc'] > 0.5
    results['chance'] = 1.0 / P
    results['elapsed_s'] = time.time() - t0
    if verbose:
        print(f"\n  seed={seed} DONE: train={results['final_train_acc']:.4f} "
              f"test={results['final_test_acc']:.4f} grok={results['grok']}", flush=True)
    return results


# ================================================================
# Dense-BP ceiling arm (one seed)
# ================================================================
def run_dense_bp_seed(seed, P=53, steps=20000, batch_size=512,
                      hidden_dim=4096, n_hidden=2, lr=0.001,
                      weight_decay=0.1,
                      eval_interval=500, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    X_train, Y_train, X_test, Y_test, _ = generate_modular_addition(
        P, train_frac=0.30, seed=seed)
    in_dim = X_train.shape[1]; out_dim = P
    Y_train_oh = torch.nn.functional.one_hot(Y_train, P).float()
    X_train = X_train.to(DEVICE); Y_train_oh = Y_train_oh.to(DEVICE)
    Y_train_y = Y_train.to(DEVICE)
    X_test = X_test.to(DEVICE); Y_test_y = Y_test.to(DEVICE)

    net = DenseBPControl(in_dim, hidden_dim, out_dim, n_hidden=n_hidden,
                         lr=lr, weight_decay=weight_decay, seed=seed)
    results = {'seed': seed, 'arm': 'dense_bp',
               'config': {'hidden_dim': hidden_dim, 'n_hidden': n_hidden,
                          'lr': lr, 'weight_decay': weight_decay,
                          'P': P, 'steps': steps},
               'train_acc_curve': [], 'test_acc_curve': []}
    n_train = X_train.shape[0]; rng = np.random.RandomState(seed); t0 = time.time()

    for step in range(steps):
        batch_idx = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
        loss = net.train_step(X_train[batch_idx], Y_train_oh[batch_idx])
        if step % eval_interval == 0 or step == steps - 1:
            train_acc = net.evaluate(X_train[batch_idx], Y_train_y[batch_idx])
            test_acc = net.evaluate(X_test, Y_test_y)
            results['train_acc_curve'].append(train_acc)
            results['test_acc_curve'].append(test_acc)
            if verbose:
                print(f"  seed={seed} step={step:5d}/{steps} "
                      f"train={train_acc:.4f} test={test_acc:.4f} "
                      f"loss={loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

    results['final_train_acc'] = net.evaluate(X_train, Y_train_y)
    results['final_test_acc'] = net.evaluate(X_test, Y_test_y)
    results['grok'] = results['final_test_acc'] > 0.5
    results['chance'] = 1.0 / P
    results['elapsed_s'] = time.time() - t0
    if verbose:
        print(f"\n  seed={seed} DONE: train={results['final_train_acc']:.4f} "
              f"test={results['final_test_acc']:.4f} grok={results['grok']}", flush=True)
    return results


# ================================================================
# Spatial (PC-ALM) baseline arm (one seed)
# ================================================================
def run_spatial_seed(seed, P=53, steps=20000, batch_size=512,
                     N=4096, L_c=2, eta_W=0.01,
                     eval_interval=500, verbose=True):
    try:
        from pc_alm_cortex_v13 import PCALMCortex
    except ImportError:
        return {'seed': seed, 'arm': 'spatial',
                'error': 'pc_alm_cortex_v13.py not available'}

    torch.manual_seed(seed); np.random.seed(seed)
    X_train, Y_train, X_test, Y_test, _ = generate_modular_addition(
        P, train_frac=0.30, seed=seed)
    in_dim = X_train.shape[1]; out_dim = P
    Y_train_oh = torch.nn.functional.one_hot(Y_train, P).float()
    X_train = X_train.to(DEVICE); Y_train_oh = Y_train_oh.to(DEVICE)
    Y_train_y = Y_train.to(DEVICE)
    X_test = X_test.to(DEVICE); Y_test_y = Y_test.to(DEVICE)

    cortex = PCALMCortex(in_dim=in_dim, hidden_dim=N, out_dim=out_dim,
                         n_layers=L_c, eta_W=eta_W, seed=seed)
    cortex.calibrate_thresholds(X_train[:256])

    results = {'seed': seed, 'arm': 'spatial',
               'config': {'N': N, 'L_c': L_c, 'eta_W': eta_W, 'P': P, 'steps': steps},
               'train_acc_curve': [], 'test_acc_curve': [], 'gate2': []}
    n_train = X_train.shape[0]; rng = np.random.RandomState(seed); t0 = time.time()

    for step in range(steps):
        batch_idx = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
        return_gates = (step % eval_interval == 0) or (step == steps - 1)
        gate_log = cortex.train_step(X_train[batch_idx], Y_train_oh[batch_idx],
                                     return_gates=return_gates)
        if return_gates:
            train_acc = cortex.evaluate(X_train[batch_idx], Y_train_y[batch_idx])
            test_acc = cortex.evaluate(X_test, Y_test_y)
            results['train_acc_curve'].append(train_acc)
            results['test_acc_curve'].append(test_acc)
            if gate_log and 'gate2_min' in gate_log:
                results['gate2'].append(gate_log['gate2_min'])
            if verbose:
                g2 = gate_log['gate2_min'] if gate_log and 'gate2_min' in gate_log else 0
                print(f"  seed={seed} step={step:5d}/{steps} "
                      f"train={train_acc:.4f} test={test_acc:.4f} "
                      f"gate2={g2:.3f} ({time.time()-t0:.0f}s)", flush=True)

    results['final_train_acc'] = cortex.evaluate(X_train, Y_train_y)
    results['final_test_acc'] = cortex.evaluate(X_test, Y_test_y)
    results['grok'] = results['final_test_acc'] > 0.5
    results['chance'] = 1.0 / P
    results['elapsed_s'] = time.time() - t0
    if verbose:
        print(f"\n  seed={seed} DONE: train={results['final_train_acc']:.4f} "
              f"test={results['final_test_acc']:.4f} grok={results['grok']}", flush=True)
    return results


# ================================================================
# Main: multi-seed experiment
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='SPEC v14.4/v14.5 temporal credit experiment')
    parser.add_argument('--arm', choices=['temporal', 'dense_bp', 'spatial'],
                        default='temporal')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=4096)
    parser.add_argument('--L_c', type=int, default=2)
    parser.add_argument('--n_columns', type=int, default=1)
    parser.add_argument('--lambda_trace', type=float, default=0.95)
    parser.add_argument('--T_behavior', type=int, default=5)
    parser.add_argument('--T_replay', type=int, default=5)
    parser.add_argument('--T_theta', type=int, default=20)
    parser.add_argument('--eta_W', type=float, default=0.01)
    parser.add_argument('--eta_theta', type=float, default=0.01)
    parser.add_argument('--beta_hc', type=float, default=0.1)
    parser.add_argument('--target_rate', type=float, default=0.10)
    parser.add_argument('--p_reactivate', type=float, default=0.3,
                        help='SWR reactivation probability (ADV-BG-V140-3)')
    parser.add_argument('--swr_noise', type=float, default=0.1,
                        help='SWR reactivation Gaussian noise std')
    parser.add_argument('--T_recal', type=int, default=50,
                        help='Threshold hard-recalibration period (P8)')
    parser.add_argument('--lr', type=float, default=0.001, help='dense_bp lr')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='dense_bp weight decay for grokking')
    parser.add_argument('--hidden_dim', type=int, default=4096, help='dense_bp width')
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    all_results = []
    t_total = time.time()

    for seed in range(args.start_seed, args.start_seed + args.seeds):
        print(f"\n{'='*60}", flush=True)
        print(f"ARM={args.arm} SEED={seed}", flush=True)
        print(f"{'='*60}", flush=True)

        if args.arm == 'temporal':
            r = run_temporal_seed(
                seed, P=53, steps=args.steps, batch_size=args.batch_size,
                N=args.N, L_c=args.L_c, n_columns=args.n_columns,
                lambda_trace=args.lambda_trace,
                T_behavior=args.T_behavior, T_replay=args.T_replay,
                T_theta=args.T_theta, eta_W=args.eta_W, eta_theta=args.eta_theta,
                beta_hc=args.beta_hc, target_rate=args.target_rate,
                p_reactivate=args.p_reactivate, swr_noise=args.swr_noise,
                T_recal=args.T_recal,
                eval_interval=args.eval_interval)
        elif args.arm == 'dense_bp':
            r = run_dense_bp_seed(
                seed, P=53, steps=args.steps, batch_size=args.batch_size,
                hidden_dim=args.hidden_dim, n_hidden=args.L_c,
                lr=args.lr, weight_decay=args.weight_decay,
                eval_interval=args.eval_interval)
        elif args.arm == 'spatial':
            r = run_spatial_seed(
                seed, P=53, steps=args.steps, batch_size=args.batch_size,
                N=args.N, L_c=args.L_c, eta_W=args.eta_W,
                eval_interval=args.eval_interval)
        else:
            print(f"Unknown arm: {args.arm}"); continue

        all_results.append(r)

        # Save incrementally
        if args.output:
            outpath = args.output
        else:
            tag = f"_{args.tag}" if args.tag else ""
            outpath = f"results_v14_{args.arm}{tag}_N{args.N}_L{args.L_c}.json"
        with open(outpath, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved {len(all_results)} results -> {outpath}", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY: {args.arm} arm, {len(all_results)} seeds", flush=True)
    print(f"{'='*60}", flush=True)
    valid = [r for r in all_results if 'final_test_acc' in r]
    if valid:
        test_accs = [r['final_test_acc'] for r in valid]
        grok_count = sum(r['grok'] for r in valid)
        chance = valid[0].get('chance', 1.0/53)
        print(f"  Test accuracy: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}", flush=True)
        print(f"  Grok rate: {grok_count}/{len(valid)} seeds", flush=True)
        print(f"  Chance: {chance:.4f}", flush=True)
        print(f"  Total elapsed: {time.time()-t_total:.0f}s", flush=True)

    print(f"\nResults saved to: {outpath}", flush=True)


if __name__ == '__main__':
    main()
