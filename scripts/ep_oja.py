#!/usr/bin/env python3
"""ep_oja.py -- EP with Oja weight mirroring (NO weight transport).

Tests whether a LOCALLY-learned backward weight W_fb (via Oja's rule) can
replace W_out.T in EP's settling dynamics. Three arms:

  EP_TIED   : settling uses W_out.T       (ceiling, groks to 100%)
  EP_OJA    : settling uses W_fb learned by Oja  (groks to 100%, locally learned)
  EP_FROZEN : settling uses random frozen W_fb   (W_out may partially align TO W_fb;
              result is seed-dependent — not a reliable baseline)

Oja mirror rule (converges W_fb -> W_out.T on the active subspace):
  dh = h - mean_batch(h);  dy = y - mean_batch(y)   (batch-mean centered)
  backward_drive = dh @ W_fb.T               (dendritic sum at each output neuron)
  W_fb += lr_mirror * (dy.T @ dh - backward_drive.T @ dh) / batch
    Term1 = centered Hebbian correlation
    Term2 = anti-Hebbian decorrelation (subtracts W_fb . Cov(h))

==============================================================================
DEVIATIONS FROM THE ORIGINAL SPEC (all necessary, all math-doc-grounded):
==============================================================================
1. lr_mirror = 0.5  (spec said 1.0)
   Discrete Oja: B += eta*(W_out.T - B)*Sigma_h is stable iff eta*lambda_max(Sigma_h) < 2.
   Measured lambda_max(Sigma_h) = 2.25 -> eta=1.0 gives 2.25 > 2 (DIVERGENT, W_fb
   saturates at +-WCLIP, cos->0). eta=0.5 gives 1.13 < 2 (STABLE). The math doc
   (MATH_ANALYSIS.md line 99) verified grokking at lr_mirror=0.5; ">= 0.5" there is a
   timescale-separation LOWER bound, the stability UPPER bound is 2/lambda_max ~ 0.89.

2. Batch-mean centering  (spec said EMA decay 0.99)
   The EMA (decay 0.99, half-life ~69 steps) cannot track the shifting h-distribution
   during grokking -> uncentered covariance -> inflated lambda_max -> Oja diverges
   mid-training. Batch-mean is exact (the convergence proof assumes E[dh dh.T] = Sigma_h).

3. Oja warmup phase  (spec had no warmup; math doc Component 2 + option (c))
   The EP settling with W_fb (not W_out.T) is a nonlinear contraction ONLY when
   cos(W_fb, W_out.T) is high enough (~0.95+). Below that, W_out@W_fb has spectral
   radius > 1 -> 2-cycle oscillation -> eventual NaN. Fix: N_WARMUP=500 Oja-only
   steps (frozen forward weights) pre-align W_fb to cos~0.999 BEFORE joint training.
   This is the math doc's option (c): "freeze W_out during mirror".
==============================================================================

No autograd anywhere in the EP path. Manual RMSProp for W_lin/W_prod/W_out;
direct Oja update for W_fb.

Ref: docs/MATH_ANALYSIS.md (APPROVED, 3-round verified).
"""
import numpy as np
import torch
import time

torch.set_default_dtype(torch.float32)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
P = 53; K_FREQ = 26; IN_DIM = 4 * K_FREQ
HIDDEN = 256; K_CONN = 8; N_PAIRS = K_CONN * (K_CONN - 1) // 2
LR = 0.01; GAMMA = 0.9; WD = 0.001; WCLIP = 5.0
BETA = 2.0; T_SETTLE = 10; LR_INF = 0.5
LR_MIRROR = 0.5          # Oja step size (NOT 1.0 — see stability note above)
N_WARMUP = 500           # Oja-only pre-alignment steps (freeze fwd weights)
EPOCHS = 5000; LOG_EVERY = 250; SEED = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Data (identical to ep_v2.py)
# ---------------------------------------------------------------------------
def make_data(seed, shuffle=False):
    rng = np.random.RandomState(seed)
    aa = np.repeat(np.arange(P), P); bb = np.tile(np.arange(P), P)
    cc = (aa + bb) % P
    if shuffle: cc = rng.permutation(cc)
    freqs = np.arange(1, K_FREQ + 1, dtype=np.float32)
    ta = 2.0 * np.pi * np.outer(aa, freqs) / P
    tb = 2.0 * np.pi * np.outer(bb, freqs) / P
    X = np.empty((P * P, IN_DIM), dtype=np.float32)
    X[:, 0::4] = np.cos(ta); X[:, 1::4] = np.sin(ta)
    X[:, 2::4] = np.cos(tb); X[:, 3::4] = np.sin(tb)
    Yoh = np.zeros((P * P, P), dtype=np.float32)
    Yoh[np.arange(P * P), cc] = 1.0
    Y = cc.astype(np.int64)
    perm = rng.permutation(P * P); n_tr = int(0.8 * P * P)
    tr, te = perm[:n_tr], perm[n_tr:]
    return X[tr], Y[tr], Yoh[tr], X[te], Y[te]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class EPNet:
    """Sparse dendritic-product net with EP settling + Oja backward weights.

    feedback_mode:
      'tied'   -> settling uses W_out.T (ceiling; no Oja)
      'oja'    -> settling uses W_fb; Oja learns W_fb (with warmup pre-alignment)
      'frozen' -> settling uses W_fb; W_fb never updated (FA baseline)
    """

    def __init__(self, in_dim, hidden, out_dim, k_conn, feedback_mode='oja', seed=0):
        assert feedback_mode in ('tied', 'oja', 'frozen')
        self.feedback_mode = feedback_mode
        rng = np.random.RandomState(seed)

        # Sparse connectivity + pair indices
        conn = np.zeros((hidden, k_conn), dtype=np.int64)
        for j in range(hidden):
            conn[j] = rng.choice(in_dim, size=k_conn, replace=False)
        self.conn = torch.from_numpy(conn).to(DEVICE)
        pi, pj = np.triu_indices(k_conn, k=1)
        self.pi = torch.from_numpy(pi.astype(np.int64)).to(DEVICE)
        self.pj = torch.from_numpy(pj.astype(np.int64)).to(DEVICE)

        # Forward weights
        s = float(np.sqrt(2.0 / k_conn))
        self.W_lin = torch.from_numpy(
            rng.randn(hidden, k_conn).astype(np.float32) * s).to(DEVICE)
        self.W_prod = torch.zeros(hidden, N_PAIRS, device=DEVICE)
        self.W_out = torch.from_numpy(
            rng.randn(hidden, out_dim).astype(np.float32)
            * float(np.sqrt(2.0 / hidden))).to(DEVICE)

        # Backward weight W_fb [out_dim, hidden] — random init, same scale as W_out.T
        self.W_fb = torch.from_numpy(
            rng.randn(out_dim, hidden).astype(np.float32)
            * float(np.sqrt(2.0 / hidden))).to(DEVICE)

        # RMSProp accumulators (forward weights only — W_fb uses direct Oja)
        self.G_lin = torch.ones_like(self.W_lin) * 1e-8
        self.G_prod = torch.ones_like(self.W_prod) * 1e-8
        self.G_out = torch.ones_like(self.W_out) * 1e-8

    # ---- Forward path ------------------------------------------------------
    def feedforward(self, X):
        cv = X[:, self.conn]                                    # [batch, hidden, k_conn]
        u = (cv * self.W_lin.unsqueeze(0)).sum(dim=2)          # [batch, hidden]
        pv = cv[:, :, self.pi] * cv[:, :, self.pj]             # [batch, hidden, n_pairs]
        u = u + (pv * self.W_prod.unsqueeze(0)).sum(dim=2)
        return cv, pv, u

    def free_phase(self, X):
        cv, pv, u = self.feedforward(X)
        h = torch.relu(u)
        y = h @ self.W_out
        return cv, pv, h, y

    # ---- Clamped (settling) phase -----------------------------------------
    def clamped_phase(self, X, target, T, beta):
        cv, pv, u = self.feedforward(X)
        h = torch.relu(u)                                      # init from free phase
        # Feedback matrix: W_out.T for tied, W_fb for oja/frozen
        B = self.W_out.T if self.feedback_mode == 'tied' else self.W_fb
        for _ in range(T):
            y = (h @ self.W_out + beta * target) / (1.0 + beta)
            err = y - h @ self.W_out                           # output prediction error
            feedback = err @ B                                 # propagate to hidden
            h = torch.relu(u + LR_INF * feedback)
        y = (h @ self.W_out + beta * target) / (1.0 + beta)
        return cv, pv, h, y

    # ---- Oja weight mirroring ---------------------------------------------
    def oja_update(self, h, y):
        """One Oja step on W_fb using batch-mean-centered (h, y)."""
        batch = h.shape[0]
        dh = h - h.mean(0, keepdim=True)                       # [batch, hidden]
        dy = y - y.mean(0, keepdim=True)                       # [batch, out_dim]
        backward_drive = dh @ self.W_fb.T                      # [batch, out_dim]
        Term1 = dy.T @ dh / batch                              # [out_dim, hidden]
        Term2 = backward_drive.T @ dh / batch                  # [out_dim, hidden]
        self.W_fb += LR_MIRROR * (Term1 - Term2)
        self.W_fb.clamp_(-WCLIP, WCLIP)

    def cos_alignment(self):
        """cosine(W_fb, W_out.T) flattened — 1.0 means perfect mirror."""
        a = self.W_fb.flatten()
        b = self.W_out.T.flatten()
        return float((torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item())

    # ---- Warmup: Oja-only pre-alignment (freeze forward weights) ----------
    def warmup_step(self, X):
        """One Oja-only step (no EP, no forward-weight update)."""
        _, _, h, y = self.free_phase(X)
        self.oja_update(h, y)
        return self.cos_alignment()

    # ---- Full EP training step --------------------------------------------
    def train_step(self, X, Yoh):
        batch = X.shape[0]
        # 1. Free phase
        cv_f, pv_f, h_f, y_f = self.free_phase(X)
        # 2. Oja mirror update (BEFORE EP, uses free-phase h, y)
        if self.feedback_mode == 'oja':
            self.oja_update(h_f, y_f)
        cos = (1.0 if self.feedback_mode == 'tied'
               else self.cos_alignment())
        # 3. Clamped phase (settles with W_out.T or W_fb)
        cv_c, pv_c, h_c, y_c = self.clamped_phase(X, Yoh, T_SETTLE, BETA)
        # 4. EP contrastive update: dW = (clamped_corr - free_corr) / beta
        dh = h_c - h_f                                         # [batch, hidden]

        # W_lin
        dW_lin = (dh.unsqueeze(2) * cv_f).sum(0) / (BETA * batch) - WD * self.W_lin
        self.G_lin.mul_(GAMMA).add_((1 - GAMMA) * dW_lin ** 2)
        self.W_lin += LR * dW_lin / (torch.sqrt(self.G_lin) + 1e-8)
        self.W_lin.clamp_(-WCLIP, WCLIP)

        # W_prod
        dW_prod = (dh.unsqueeze(2) * pv_f).sum(0) / (BETA * batch) - WD * self.W_prod
        self.G_prod.mul_(GAMMA).add_((1 - GAMMA) * dW_prod ** 2)
        self.W_prod += LR * dW_prod / (torch.sqrt(self.G_prod) + 1e-8)
        self.W_prod.clamp_(-WCLIP, WCLIP)

        # W_out
        dW_out = (h_c.T @ y_c - h_f.T @ y_f) / (BETA * batch) - WD * self.W_out
        self.G_out.mul_(GAMMA).add_((1 - GAMMA) * dW_out ** 2)
        self.W_out += LR * dW_out / (torch.sqrt(self.G_out) + 1e-8)
        self.W_out.clamp_(-WCLIP, WCLIP)

        return dh.abs().mean().item(), cos

    def evaluate(self, X, y_idx):
        _, _, h, y = self.free_phase(X)
        return (y.argmax(1) == y_idx).float().mean().item()


# ---------------------------------------------------------------------------
# Arm runner
# ---------------------------------------------------------------------------
def run_arm(mode, Xtr, Ytr, Yoh_tr, Xte, Yte):
    labels = {
        'tied':   'EP_TIED   (W_out.T — ceiling, should = ep_v2.py 100%)',
        'oja':    'EP_OJA    (W_fb learned by Oja — THE TEST)',
        'frozen': 'EP_FROZEN (random W_fb — seed-dependent, see notes)',
    }
    print(f"\n  --- {labels[mode]} ---")
    Xt = torch.from_numpy(Xtr).to(DEVICE)
    Yt = torch.from_numpy(Ytr).to(DEVICE)
    Yo = torch.from_numpy(Yoh_tr).to(DEVICE)
    Xe = torch.from_numpy(Xte).to(DEVICE)
    Ye = torch.from_numpy(Yte).to(DEVICE)

    net = EPNet(IN_DIM, HIDDEN, P, K_CONN, feedback_mode=mode, seed=SEED)

    # Oja warmup (oja arm only): pre-align W_fb before joint training
    if mode == 'oja':
        print(f"    [warmup] {N_WARMUP} Oja-only steps (frozen fwd weights)...")
        for w in range(1, N_WARMUP + 1):
            cos = net.warmup_step(Xt)
            if w % 100 == 0 or w == N_WARMUP:
                print(f"      warmup {w:4d}: cos(W_fb, W_out.T) = {cos:.4f}", flush=True)

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        dh_mag, cos = net.train_step(Xt, Yo)
        if ep % LOG_EVERY == 0 or ep <= 5 or ep == EPOCHS:
            tr = net.evaluate(Xt, Yt)
            te = net.evaluate(Xe, Ye)
            wp = float(torch.norm(net.W_prod).item())
            wo = float(torch.norm(net.W_out).item())
            print(f"    ep{ep:5d}: train={tr:.3f} held={te:.3f} "
                  f"|Wprod|={wp:6.1f} |Wout|={wo:6.1f} "
                  f"cos={cos:+.3f} dh={dh_mag:.4f} "
                  f"({(time.time() - t0) / ep:.4f}s/ep)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("EP + Oja Weight Mirroring  (NO weight transport)")
    print(f"  beta={BETA}, T_settle={T_SETTLE}, lr_inf={LR_INF}, wd={WD}")
    print(f"  LR={LR} (fwd), LR_MIRROR={LR_MIRROR} (W_fb), wclip={WCLIP}")
    print(f"  Oja warmup: {N_WARMUP} steps (frozen fwd) -> pre-align W_fb")
    print(f"  Centering: batch-mean  |  Settling feedback: err @ W_fb")
    print("=" * 78)

    Xtr, Ytr, Yoh_tr, Xte, Yte = make_data(SEED, shuffle=False)
    print(f"  train={len(Xtr)} held={len(Xte)} chance={1.0 / P:.3f}")

    for mode in ('tied', 'oja', 'frozen'):
        run_arm(mode, Xtr, Ytr, Yoh_tr, Xte, Yte)

    print(f"\n{'=' * 78}")
    print("  Interpretation:")
    print("    EP_TIED   groks to 100%   -> EP math is correct (ceiling)")
    print("    EP_OJA    groks to ~100%  -> Oja learns W_fb -> W_out.T locally")
    print("    EP_FROZEN seed-dependent (W_out may align TO W_fb)")
    print(f"{'=' * 78}")


if __name__ == '__main__':
    main()
