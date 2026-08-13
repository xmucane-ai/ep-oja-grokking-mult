#!/usr/bin/env python3
"""living_ec_exp5.py -- SPEC_LIVING_EC_v1.2 §5, Gate G-LIVING-6.

NON-STATIONARY TRACKING: can the living EC track a drifting eigenspace?

THE test that proves the EC is actually alive, not just matching frozen.

KEY MATHEMATICAL FINDING (implemented below):
  The character basis of Z_p* is the FIXED eigenbasis for ALL transition
  operators Omega_S, for any generator set S. Blending generators S_A -> S_B
  does NOT rotate the eigenspace — it only reshuffles eigenvalue rankings.
  This means streaming Oja CANNOT track generator changes (once W spans a
  set of character pairs, C@W stays within that span — the update is trapped).

  To test actual tracking, we rotate the PHYSICAL basis: Omega(t) = R(t) @
  Omega_A @ R(t)^T, where R(t) is a continuous rotation (geodesic on O(N)).
  This rotates the eigenvectors in the standard basis while preserving the
  eigenvalue spectrum. The streaming Oja must track this rotation.

EXPERIMENT:
  Phase 1 (T_CONV): converge W to top eigenspace of Omega_A at eta_conv.
  Phase 2 (T_DRIFT): continuously rotate Omega(t) via R(alpha).
    alpha ramps 0->1 linearly. R(alpha) = expm(alpha * B) for fixed
    antisymmetric B. Eigenspace rotates continuously.
  Phase 3 (T_SETTLED): rotation stops at R(1). W should re-converge.

3 ARMS (share convergence, diverge at drift):
  (1) Frozen: W locked, should FAIL to track
  (2) eta=0.1: normal, should TRACK (alignment >= 0.9)
  (3) eta=0.001: too slow, should FAIL (staged failure)

PREDICT (Theorem ST(b)): alignment >= 0.9 while rho_drift < eta * Delta.
FALSIFIER: alignment < 0.9 when eta * Delta < rho_drift.

COMPUTE: numpy CPU (52x52 matrices). GPU only for optional cortex test.
"""
import numpy as np
import time
import json
import sys

# ================================================================
# Constants
# ================================================================
P = 53
N = P - 1           # 52
K_EXTRACT = 6       # top 3 character pairs (6 components) — well-separated eigenvalues

# Discrete log
G_PRIM = 2
def build_dlog_table(p=53, g=2):
    dlog = {}
    val = 1
    for exp in range(p - 1):
        dlog[val] = exp
        val = (val * g) % p
    return dlog

DLOG = build_dlog_table(P, G_PRIM)
DLOG_INV = np.array([0]*N, dtype=np.int64)
for a_val, j_val in DLOG.items():
    DLOG_INV[j_val] = a_val

DLOG_ARR = np.array([DLOG[a] for a in range(1, P)])

# Domain A: first 5 powers of g=2
S_A = [(2**k) % P for k in range(1, 6)]
S_A_DLOG = np.array([DLOG[s] for s in S_A])

# Streaming params
LAMBDA_DECAY = 0.02    # sliding-window forgetting (W_eff ~ 50)
BATCH_SIZE = 200
ETA_CONV = 0.5         # shared convergence rate
T_CONV = 3000
T_DRIFT = 4000
T_SETTLED = 1000
T_TOTAL = T_CONV + T_DRIFT + T_SETTLED  # 8000
CHECK_EVERY = 50

# Arm etas (during drift/settled phases)
ETA_NORMAL = 0.1
ETA_SLOW = 0.001

# Rotation strength (how far the eigenspace rotates during drift)
# Full rotation angle = ||B||_F * alpha. We calibrate B so that at alpha=1,
# the eigenspace has rotated enough that frozen W alignment drops to ~0.3.
ROTATION_STRENGTH = 10.0  # calibrated: gives ~0.33 alignment R(0) vs R(1)

# ================================================================
# Build Omega_A (the base transition operator, symmetric)
# ================================================================
def build_omega_A():
    """Build symmetric transition operator for S_A on Z_p*."""
    Omega = np.zeros((N, N))
    for s_step in S_A_DLOG:
        for a in range(1, P):
            j_a = DLOG[a]
            j_b = (j_a + s_step) % N
            b = int(DLOG_INV[j_b])
            i, j = a - 1, b - 1
            Omega[i, j] += 0.5
            Omega[j, i] += 0.5
    Omega /= len(S_A_DLOG)
    return Omega

OMEGA_A = build_omega_A()

# Mean-center
MEAN_VEC = np.ones(N) / np.sqrt(N)
P_PERP = np.eye(N) - np.outer(MEAN_VEC, MEAN_VEC)

# ================================================================
# Rotation: R(alpha) = expm(alpha * B) for fixed antisymmetric B
# ================================================================
def build_rotation_generator(seed=12345):
    """Build a fixed random antisymmetric matrix B.

    R(alpha) = expm(alpha * B) is a proper rotation.
    The eigenspace of R @ Omega_A @ R^T rotates continuously with alpha.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((N, N))
    B = ROTATION_STRENGTH * (M - M.T) / N  # antisymmetric, scaled
    return B

B_ROT = build_rotation_generator()

# Precompute the per-step rotation increment during drift
# During drift, alpha increases by 1/T_DRIFT per step.
# R(t+1) = expm(B/T_DRIFT) @ R(t) = dR @ R(t)
# Compute dR once using scipy or numpy eigendecomposition
def matrix_expm(A):
    """Matrix exponential via eigendecomposition (for antisymmetric A, this is exact).

    For antisymmetric A: A = V @ diag(i*lambda) @ V^H
    expm(A) = V @ diag(exp(i*lambda)) @ V^H = V @ diag(cos(lambda) + i*sin(lambda)) @ V^H
    Result is real.
    """
    eigvals, eigvecs = np.linalg.eig(A)
    exp_eigvals = np.exp(eigvals)
    result = eigvecs @ np.diag(exp_eigvals) @ np.linalg.inv(eigvecs)
    return np.real(result)

# Per-step rotation during drift
DR_PER_STEP = matrix_expm(B_ROT / T_DRIFT)
# Full rotation at alpha=1
R_FULL = matrix_expm(B_ROT)

# ================================================================
# Alignment measurement
# ================================================================
def subspace_alignment(U1, U2):
    """Mean canonical correlation between column spaces (1.0 = identical)."""
    Q1, _ = np.linalg.qr(U1)
    Q2, _ = np.linalg.qr(U2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(s.mean())

def instantaneous_top_eigenspace(R, k=K_EXTRACT):
    """Top-k eigenvectors of R @ Omega_A @ R^T (mean-centered)."""
    Omega_t = R @ OMEGA_A @ R.T
    Omega_t_mc = P_PERP @ Omega_t @ P_PERP
    eigvals, eigvecs = np.linalg.eigh(Omega_t_mc)
    top_idx = np.argsort(-eigvals)[:k]
    return eigvecs[:, top_idx]

# ================================================================
# Transition sampling (rotated)
# ================================================================
def sample_rotated_batch(R, alpha, batch_size, rng):
    """Sample a batch of transitions from Omega_A, then rotate by R.

    Returns the symmetric C_update matrix [N x N].
    For each transition (a, b):
      contribution = R[:,a] @ R[:,b]^T / 2 + R[:,b] @ R[:,a]^T / 2

    Vectorized: if Ra = R[:, a_idx], Rb = R[:, b_idx] (both N x batch):
      C_update = (Ra @ Rb.T + Rb @ Ra.T) / (2 * batch)
    """
    # Sample transitions from Omega_A
    s_steps = S_A_DLOG[rng.integers(len(S_A_DLOG), size=batch_size)]
    a_idx = rng.integers(0, N, size=batch_size)
    j_a = DLOG_ARR[a_idx]
    j_b = (j_a + s_steps) % N
    b_idx = DLOG_INV[j_b] - 1  # 0-indexed

    # Extract rotated columns
    Ra = R[:, a_idx]  # N x batch
    Rb = R[:, b_idx]  # N x batch

    # Symmetric rank-2 updates averaged over batch
    C_update = (Ra @ Rb.T + Rb @ Ra.T) / (2.0 * batch_size)
    return C_update

# ================================================================
# Per-arm runner
# ================================================================
def run_arm(arm_name, drift_eta, seed=42, verbose=True):
    """Run one arm through convergence -> drift -> settled."""
    rng = np.random.default_rng(seed)

    # Initialize W
    W = P_PERP @ rng.standard_normal((N, K_EXTRACT))
    W, _ = np.linalg.qr(W)
    C_est = np.zeros((N, N))

    trajectory = []
    frozen = False
    frozen_W = None
    R = np.eye(N)  # current rotation (starts at identity)

    for t in range(T_TOTAL):
        # Determine phase, alpha, eta
        if t < T_CONV:
            alpha = 0.0
            eta = ETA_CONV
        elif t < T_CONV + T_DRIFT:
            alpha = (t - T_CONV) / T_DRIFT
            eta = drift_eta
            # Update rotation
            R = DR_PER_STEP @ R
        else:
            alpha = 1.0
            eta = drift_eta
            # R stays at R_FULL (settled)

        # Freeze frozen arm at start of drift
        if arm_name == 'frozen' and t >= T_CONV:
            if not frozen:
                frozen = True
                frozen_W = W.copy()

        # Sample transitions and update C_est
        C_update = sample_rotated_batch(R, alpha, BATCH_SIZE, rng)
        C_est = (1 - LAMBDA_DECAY) * C_est + LAMBDA_DECAY * C_update

        # Update W (unless frozen)
        if not frozen:
            C_mc = P_PERP @ C_est @ P_PERP
            # Oja subspace rule (no LT deflation — rotation invariant)
            Y = W.T @ C_mc @ W
            dW = C_mc @ W - W @ Y
            W += eta * dW
            W = P_PERP @ W
            col_norms = np.linalg.norm(W, axis=0, keepdims=True)
            W /= (col_norms + 1e-12)
            if t % 50 == 0:
                W, _ = np.linalg.qr(P_PERp_update(W))

        # Measure alignment
        if t % CHECK_EVERY == 0 or t == T_TOTAL - 1:
            ref = instantaneous_top_eigenspace(R)
            W_cur = frozen_W if frozen else W
            align = subspace_alignment(W_cur, ref)
            trajectory.append((t, alpha, align))

            phase = ('conv' if t < T_CONV else
                     'drift' if t < T_CONV + T_DRIFT else 'settled')
            if verbose and (t % 500 == 0 or t == T_TOTAL - 1):
                print(f"    [{arm_name:20s}] t={t:5d} alpha={alpha:.3f} "
                      f"phase={phase:7s} align={align:.4f}", flush=True)

    # Stats
    drift_traj = [(t,al,a) for t,al,a in trajectory
                  if T_CONV <= t < T_CONV + T_DRIFT]
    min_drift = min(a for _,_,a in drift_traj) if drift_traj else 0.0
    mean_drift = float(np.mean([a for _,_,a in drift_traj])) if drift_traj else 0.0
    settled_traj = [(t,al,a) for t,al,a in trajectory if t >= T_CONV + T_DRIFT]
    final_align = settled_traj[-1][2] if settled_traj else trajectory[-1][2]
    conv_traj = [(t,al,a) for t,al,a in trajectory if t < T_CONV]
    final_conv = conv_traj[-1][2] if conv_traj else 0.0

    return {
        'arm': arm_name,
        'drift_eta': drift_eta,
        'trajectory': trajectory,
        'final_alignment': final_align,
        'min_drift_alignment': min_drift,
        'mean_drift_alignment': mean_drift,
        'conv_alignment': final_conv,
    }


def P_PERp_update(W):
    """Helper: project to perpendicular space."""
    return P_PERP @ W


# ================================================================
# MAIN
# ================================================================
def main():
    t_start = time.time()
    print("=" * 78)
    print("LIVING EC EXP-5 -- Non-stationary Tracking (G-LIVING-6)")
    print("SPEC_LIVING_EC_v1.2 §5")
    print("=" * 78)
    print(f"  p={P}, N={N}")
    print(f"  S_A (g=2): {S_A}")
    print(f"  Drift: continuous physical-basis rotation R(alpha)=expm(alpha*B)")
    print(f"  Omega(t) = R(alpha) @ Omega_A @ R(alpha)^T")
    print(f"  lambda={LAMBDA_DECAY} (W_eff~{1/LAMBDA_DECAY:.0f}), batch={BATCH_SIZE}")
    print(f"  T_conv={T_CONV} (eta={ETA_CONV}), T_drift={T_DRIFT}, T_settled={T_SETTLED}")
    print(f"  T_total={T_TOTAL}, k_extract={K_EXTRACT} ({K_EXTRACT//2} pairs)")
    print(f"  Rotation strength: {ROTATION_STRENGTH}")
    print("=" * 78)

    # Verify rotation magnitude
    print("\n--- Rotation verification ---")
    ref_0 = instantaneous_top_eigenspace(np.eye(N))
    ref_1 = instantaneous_top_eigenspace(R_FULL)
    base_align = subspace_alignment(ref_0, ref_1)
    print(f"  Alignment R(0) vs R(1) top eigenspaces: {base_align:.4f}")
    print(f"  (Should be < 0.9 for frozen arm to fail)")
    # Check intermediate
    R_half = matrix_expm(B_ROT * 0.5)
    ref_half = instantaneous_top_eigenspace(R_half)
    align_half = subspace_alignment(ref_0, ref_half)
    print(f"  Alignment R(0) vs R(0.5): {align_half:.4f}")

    # Eigenvalue spectrum of Omega_A
    Omega_A_mc = P_PERP @ OMEGA_A @ P_PERP
    eigvals_A = np.sort(np.linalg.eigvalsh(Omega_A_mc))[::-1]
    print(f"  Omega_A top-{K_EXTRACT} eigenvalues: {eigvals_A[:K_EXTRACT]}")
    print(f"  Eigenvalue gap ({K_EXTRACT} vs {K_EXTRACT+1}): "
          f"{eigvals_A[K_EXTRACT-1]-eigvals_A[K_EXTRACT]:.4f}")

    # Tracking bandwidth analysis
    print("\n--- Tracking bandwidth (Theorem ST(b)) ---")
    Delta = 1.0 / LAMBDA_DECAY
    # rho_drift: approximate per-step rotation rate of eigenspace
    # At R(0) vs R(1): alignment drops from 1.0 to base_align
    # Total "distance" ~ arccos(base_align). Over T_DRIFT steps.
    rho_drift = np.arccos(np.clip(base_align, -1, 1)) / T_DRIFT
    print(f"  rho_drift ~ {rho_drift:.6f}/step (eigenspace angular velocity)")
    print(f"  Delta = 1/lambda = {Delta:.0f}")
    for name, eta in [('frozen', 0.0), ('normal(0.1)', ETA_NORMAL), ('slow(0.001)', ETA_SLOW)]:
        bw = eta * Delta
        verdict = 'TRACK' if bw > rho_drift else 'FAIL'
        print(f"  {name:14s}: eta*Delta={bw:8.3f} vs rho_drift={rho_drift:.6f} -> {verdict}")
    print("=" * 78)

    # ── Run 3 arms ──
    results = {}
    arms = [
        ('frozen', 0.0),
        ('streaming_eta0.1', ETA_NORMAL),
        ('streaming_eta0.001', ETA_SLOW),
    ]

    for arm_name, eta in arms:
        print(f"\n--- Arm: {arm_name} (drift eta={eta}) ---")
        t0 = time.time()
        r = run_arm(arm_name, eta, seed=42, verbose=True)
        r['elapsed'] = time.time() - t0
        results[arm_name] = r
        print(f"  Done in {r['elapsed']:.1f}s")
        print(f"  Conv alignment:   {r['conv_alignment']:.4f}")
        print(f"  Min drift align:  {r['min_drift_alignment']:.4f}")
        print(f"  Mean drift align: {r['mean_drift_alignment']:.4f}")
        print(f"  Final alignment:  {r['final_alignment']:.4f}")

    # ── Gate evaluation ──
    print("\n" + "=" * 78)
    print("GATE G-LIVING-6 (Non-stationary Tracking)")
    print("=" * 78)

    frozen_min = results['frozen']['min_drift_alignment']
    normal_min = results['streaming_eta0.1']['min_drift_alignment']
    slow_min = results['streaming_eta0.001']['min_drift_alignment']

    frozen_fail = frozen_min < 0.9
    normal_track = normal_min >= 0.9
    slow_fail = slow_min < 0.9

    print(f"  Arm 1 (frozen):    min_drift={frozen_min:.4f}  "
          f"(expect <0.9 FAIL):  {'CONFIRMED' if frozen_fail else 'UNEXPECTED'}")
    print(f"  Arm 2 (eta=0.1):   min_drift={normal_min:.4f}  "
          f"(expect >=0.9 TRACK): {'CONFIRMED' if normal_track else 'FAILED'}")
    print(f"  Arm 3 (eta=0.001): min_drift={slow_min:.4f}  "
          f"(expect <0.9 FAIL):  {'CONFIRMED' if slow_fail else 'UNEXPECTED'}")

    all_pass = frozen_fail and normal_track and slow_fail
    print(f"\n  G-LIVING-6: {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print("  -> Living EC tracks non-stationary statistics (Theorem ST(b))")
        print("  -> Frozen fails, streaming adapts (eta=0.1), fails (eta=0.001)")
        print("  -> The EC IS alive: it tracks, not just matches frozen.")
    else:
        fails = []
        if not frozen_fail: fails.append("frozen did NOT fail")
        if not normal_track: fails.append("normal did NOT track")
        if not slow_fail: fails.append("slow did NOT fail")
        print(f"  -> Unexpected: {'; '.join(fails)}")

    # ── Trajectory table ──
    print("\n--- Alignment trajectory ---")
    print(f"  {'Step':>6s} {'alpha':>6s} {'phase':>7s} {'frozen':>8s} {'eta=0.1':>8s} {'eta=0.001':>9s}")
    f_traj = results['frozen']['trajectory']
    n_traj = results['streaming_eta0.1']['trajectory']
    s_traj = results['streaming_eta0.001']['trajectory']
    stride = max(1, len(f_traj) // 30)
    for i in range(0, len(f_traj), stride):
        t, al, af = f_traj[i]
        _, _, an = n_traj[i]
        _, _, asl = s_traj[i]
        phase = 'conv' if t < T_CONV else 'drift' if t < T_CONV+T_DRIFT else 'set'
        mark = ' ***' if (af < 0.9 or an < 0.9 or asl < 0.9) else ''
        print(f"  {t:6d} {al:6.3f} {phase:7s} {af:8.4f} {an:8.4f} {asl:9.4f}{mark}")

    # Save JSON
    serializable = {}
    for arm_name, r in results.items():
        sr = dict(r)
        sr['trajectory'] = [(int(t), round(float(al),4), round(float(a),4))
                            for t, al, a in r['trajectory']]
        serializable[arm_name] = sr
    serializable['gate'] = {
        'name': 'G-LIVING-6',
        'pass': all_pass,
        'frozen_min_drift': round(frozen_min, 4),
        'normal_min_drift': round(normal_min, 4),
        'slow_min_drift': round(slow_min, 4),
    }
    serializable['config'] = {
        'P': P, 'N': N, 'S_A': S_A,
        'lambda_decay': LAMBDA_DECAY, 'batch_size': BATCH_SIZE,
        'eta_conv': ETA_CONV,
        'T_conv': T_CONV, 'T_drift': T_DRIFT, 'T_settled': T_SETTLED,
        'k_extract': K_EXTRACT,
        'eta_normal': ETA_NORMAL, 'eta_slow': ETA_SLOW,
        'rotation_strength': ROTATION_STRENGTH,
        'base_align_R0_vs_R1': round(base_align, 4),
        'rho_drift': rho_drift,
    }
    serializable['elapsed_total'] = time.time() - t_start

    with open('living_ec_exp5_results.json', 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to living_ec_exp5_results.json")
    print(f"  Total elapsed: {serializable['elapsed_total']:.1f}s")

    return all_pass


if __name__ == '__main__':
    main()
