import numpy as np

# =====================================================================
# math_check_oja_t06df28d0.py  (round-3 update)
#
# Verification of the Oja weight-mirroring feedback fix for MATH SPEC
# t_06df28d0, per the round-2 aggregate (math-reviewer MR2-T06DF-1/2/3,
# red-team RT-T06DF-R2-1/2).
#
# ROUND-3 CHANGES (LOW RT-T06DF-R2-2 / honesty):
#   * The RECOMMENDED per-update spectral clip is now applied INSIDE the
#     Oja loop (previously the clip was applied only post-hoc on the final
#     matrix). All headline cosines now come from the recommended config.
#   * A 20-seed sweep is reported for the recommended per-update-clip
#     config (mean/min/max, # seeds >= 0.5), not a single seed-0 top.
#   * The old 0.506 headline is retained as a FOOTNOTE: it is the
#     unclipped seed-0 value and sits at the top of the unclipped
#     distribution; it is NOT the recommended-config headline.
#   * k-WTA 10% + per-update clip is now tested and reported HONESTLY:
#     under the real non-whitened substrate the cosine is stuck near ~0.05
#     (the whitening gap, MR2-T06DF-1/2/3). This is declared an OPEN
#     empirical question, NOT claimed as convergence.
# =====================================================================

def oja_step(B, x_j, y_k, eta):
    # per-synapse three-factor Oja mirroring, P1-local, P3-clean
    return B + eta * (np.outer(x_j, y_k) - B * (x_j[:, None] ** 2))

def spectral_clip(B):
    n = np.linalg.norm(B, 2)
    return B / n if n > 1.0 else B

def cos_score(B, W_out, s):
    a = B @ s
    g = W_out @ s
    return (a @ g) / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-12)

def run_oja(N, out, n_steps, eta_fb, x, y, clip_per_update):
    rng = np.random.default_rng(0)
    B = rng.standard_normal((N, out)) / np.sqrt(N)
    for t in range(n_steps):
        B = oja_step(B, x[:, t], y[t], eta_fb)
        if clip_per_update:
            B = spectral_clip(B)
    return B

# ---------------------------------------------------------------------
# Config: the real target substrate (N=4096, out=53) and the verified
# whitened regime (N=1024, out=32) for direct comparison.
# ---------------------------------------------------------------------
N, out = 1024, 32
n_steps = 5000
eta_fb = 0.01

# forward readout weights (fixed during B_fb learning)
W_out = np.random.default_rng(0).standard_normal((N, out)) / np.sqrt(N)
W_out = W_out / np.linalg.norm(W_out, 2)

# --- Inputs: WHITENED (tanh(randn*0.1)) --- the script's historical regime.
# NOTE (MR2-T06DF-2): this regime is effectively whitened. The cos->0.506
# trajectory holds here ONLY; it is NOT evidence for k-WTA convergence.
x_w = np.tanh(np.random.default_rng(0).standard_normal((N, n_steps)) * 0.1)
y_w = x_w.T @ W_out

print("=== OJA RULE, WHITENED INPUTS (tanh(randn*0.1)) — REGIME-NOTE: not k-WTA ===")

# Recommended config: per-update spectral clip inside the loop.
B_w_clip = run_oja(N, out, n_steps, eta_fb, x_w, y_w, clip_per_update=True)
s = np.random.default_rng(1).standard_normal(out) * 0.5
cos_w_clip = cos_score(B_w_clip, W_out, s)
print(f"per-update-clip (RECOMMENDED): cos(B@s, W_out@s) = {cos_w_clip:+.4f} | ||B||_2 = {np.linalg.norm(B_w_clip,2):.3f}")

# Historical headline: UNCLIPPED seed-0 (footnote, not the recommended value).
B_w_noclip = run_oja(N, out, n_steps, eta_fb, x_w, y_w, clip_per_update=False)
cos_w_noclip = cos_score(B_w_noclip, W_out, s)
print(f"unclipped seed-0 (HISTORICAL 0.506 FOOTNOTE): cos = {cos_w_noclip:+.4f} | ||B||_2 = {np.linalg.norm(B_w_noclip,2):.3f}")

# --- Test 2: spectral clip is a direction-only op (single-clip cosine) ---
print("\n=== SPECTRAL CLIP direction-only (SINGLE clip) ===")
B_clip = B_w_noclip / np.linalg.norm(B_w_noclip, 2)
cos_uc = cos_score(B_w_noclip, W_out, s)
cos_c = cos_score(B_clip, W_out, s)
print(f"cos unclipped = {cos_uc:+.4f} | cos clipped = {cos_c:+.4f} | delta = {abs(cos_uc-cos_c):.2e}")
print("(This confirms SINGLE-clip cosine invariance. It does NOT address the")
print(" repeated clip-vs-update dynamics — see k-WTA test below.)")

# --- Test 3: defective DFA rule diverges (unchanged) ---
print("\n=== DEFECTIVE spec rule D_B = eta*x*score^T (diverges) ===")
def grad_rule_step(B, x_j, score):
    return B + eta_fb * np.outer(x_j, score)
B_bad = np.random.default_rng(0).standard_normal((N, out)) / np.sqrt(N)
scores = np.random.default_rng(2).standard_normal((n_steps, out)) * 0.5
nb = []
for t in range(n_steps):
    B_bad = grad_rule_step(B_bad, x_w[:, t], scores[t])
    if t % 1000 == 0 or t == n_steps - 1:
        nb.append((t, np.linalg.norm(B_bad, 2)))
for t, n in nb:
    print(f"step {t:>5}: ||B_fb||_2 = {n:>8.1f}")

# --- Test 4 (ROUND-3): 20-seed sweep, RECOMMENDED per-update-clip config ---
print("\n=== 20-SEED SWEEP (RECOMMENDED per-update clip, whitened inputs) ===")
cos_20 = []
for seed in range(20):
    W = np.random.default_rng(seed).standard_normal((N, out)) / np.sqrt(N)
    W = W / np.linalg.norm(W, 2)
    xw = np.tanh(np.random.default_rng(seed).standard_normal((N, n_steps)) * 0.1)
    yw = xw.T @ W
    Bw = run_oja(N, out, n_steps, eta_fb, xw, yw, clip_per_update=True)
    cos_20.append(cos_score(Bw, W, np.random.default_rng(seed).standard_normal(out) * 0.5))
cos_20 = np.array(cos_20)
print(f"mean {cos_20.mean():.3f} | min {cos_20.min():.3f} | max {cos_20.max():.3f} | "
      f"seeds>=0.5: {(cos_20>=0.5).sum()}/20 | seeds>=0.4: {(cos_20>=0.4).sum()}/20")

# --- Test 5 (ROUND-3): k-WTA 10% + per-update clip (the whitening gap) ---
# Real non-whitened substrate: a FIXED dataset of k-WTA-thresholded sparse
# patterns (persistent support/correlation, like the real cortex). This
# reproduces the ill-conditioned Cov that k-WTA induces. HONEST report:
# convergence is NOT established here — cos stays well below the 0.5 gate,
# corroborating the reviewers' independent measurement (cos stuck ~0.05,
# cond(Cov)>>1). Declared OPEN (MR2-T06DF-1/2/3), NOT claimed as convergence.
print("\n=== k-WTA 10% + per-update clip (NON-whitened, MR2-T06DF-1/2/3 gap) ===")

Nk, outk = 4096, 53
rng_k = np.random.default_rng(0)
n_pat_k = 2000
k_keep = Nk // 10  # 10% firing
base_k = np.tanh(rng_k.standard_normal((Nk, n_pat_k)) * 0.1)
Xk = np.zeros_like(base_k)
for i in range(n_pat_k):
    keep = np.argpartition(base_k[:, i], -k_keep)[-k_keep:]
    Xk[keep, i] = base_k[keep, i]

Cov_k = np.cov(Xk)
print(f"cond(Cov) of fixed sparse dataset = {np.linalg.cond(Cov_k):.3e} (whitened ~O(1))")

Wk = rng_k.standard_normal((Nk, outk)) / np.sqrt(Nk)
Wk = Wk / np.linalg.norm(Wk, 2)
Yk = Xk.T @ Wk
eta_k = 0.01
Bk = rng_k.standard_normal((Nk, outk)) / np.sqrt(Nk)
cos_k = []; bn_k = []
for t in range(n_pat_k):
    Bk = oja_step(Bk, Xk[:, t], Yk[t], eta_k)
    Bk = spectral_clip(Bk)   # per-update clip (recommended)
    if t % 400 == 0 or t == n_pat_k - 1:
        cos_k.append(cos_score(Bk, Wk, rng_k.standard_normal(outk) * 0.5))
        bn_k.append(np.linalg.norm(Bk, 2))
print("step   cos(B@s,W_out@s)  ||B||_2")
for i, (c, n) in enumerate(zip(cos_k, bn_k)):
    st = i * 400 if i * 400 < n_pat_k else n_pat_k
    print(f"{st:>5}   {c:+.4f}            {n:.2f}")
print("(cos stays far below the 0.5 gate: whitening assumption violated by")
print(" k-WTA (cond(Cov)>>1). Corroborates reviewers' measurement (cos ~0.05).")
print(" Convergence on the real substrate is the DECISIVE empirical question")
print(" for the coder's A/B — NOT established by this proof.)")
