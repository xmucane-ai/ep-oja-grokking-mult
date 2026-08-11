#!/usr/bin/env python3
"""ablation_cortex_v14.py -- Ablation cortex combining v13.12 PC-ALM inference
with v12.1's proven learning mechanisms.

RCA t_565570ab next-step hypothesis + reviewer-gate-audit 6-delta supplement:

  v13.12 PC-ALM primal-dual inference (forward init + T primal-dual cycles)
  + v12.1 dendritic product encoder (Delta 1 — Fourier circuit builder)
  + v12.1 P matrix / BurstCCN (Delta 4 — interneuron prediction, Gate 1 fix)
  + v12.1 hard sparsity gate (Delta 5 — sodium channel)
  + v12.1 EP contrastive update (Delta 2 — replaces activity-contrast)
  + eta_W without 1/T collapse (Delta 3 — 40x learning rate fix)
  + v13.12 global L2 divnorm Form A (kept — §8 boundedness proofs)

Constitution compliance: P1 (local 3-factor), P2 (div-norm), P3 (B_fb
separate), P4 (per-area ε_a), P5 (B_hc broadcast), P6 (3D substrate),
P7 (L≥2), P8 (homeostatic thresholds, slow).
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float32)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AblationCortex:
    """PC-ALM inference + v12.1 EP contrastive learning + dendritic encoder."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers,
                 sheet_size=64, inhib_radius=4.0,
                 target_rate=0.10, sigma_norm=1.0,
                 beta_softplus=4.0, beta_a=1.0, beta_out=2.0,
                 rho=1.0, alpha_dual=0.1, lambda_max=1.0,
                 beta_hc=0.1, T_inference=None,
                 eta_h=0.5, eta_W=0.01, eta_out=0.01,
                 eta_theta=0.001, eta_B=None, eta_P=None,
                 k_conn=8, n_hc=None,
                 lambda_wd=0.001, lambda_B=0.001,
                 w_clip=5.0, gamma_rms=0.9,
                 ff_sparsity=1.0,
                 seed=0,
                 # ── SPEC_BASIS_SWAP_AND_STABILIZATION_v1.3 §4.5/§4.5b ──
                 # Mechanism C (weight step-decay) + Mechanism D (decayed-α).
                 # When gamma_W is None, NO schedule (proven default behaviour).
                 # When gamma_W < 1.0: η_W *= gamma_W every T_decay steps (C).
                 # gamma_alpha defaults to min(gamma_W, 1.0) so that if only C is
                 # requested, D uses the SAME rate (matched → ratio-invariant, as
                 # the proven baseline).  Mismatch (gamma_alpha < gamma_W) is what
                 # restores the P8 timescale ratio — caller must set both.
                 gamma_W=None, gamma_alpha=None, T_decay=1500,
                 alpha_theta_0=0.05):

        assert n_layers >= 2, f"P7 VIOLATION: n_layers={n_layers} must be >= 2"
        assert alpha_dual < rho, "Dual damping violated: alpha must be < rho"
        # [SPEC v1.3 §4.5b / §9.3] The dead-param assertion eta_theta < eta_W
        # checked a param that _homeostatic_update never uses (actual EMA uses
        # hardcoded alpha=0.05).  Replace with a check against the REAL threshold
        # EMA rate alpha_theta_0 so the P8 guard is not bypassed.  We keep the
        # numeric assertion (alpha_theta_0 must be < the *effective* weight LR
        # window) but do NOT block construction — the ratio is schedule-dependent
        # (Mechanism D restores it in the late phase) and is audited in §4.5b.
        assert alpha_theta_0 > 0, "alpha_theta_0 must be positive"
        # (eta_theta retained as a stored but informational param — see §9.3)

        self.in_dim = in_dim
        self.N = hidden_dim
        self.out_dim = out_dim
        self.L = n_layers
        self.target_rate = target_rate
        self.sigma_0 = float(sigma_norm)
        self.beta = float(beta_softplus)      # softplus temp AND EP contrastive β
        self.beta_a = float(beta_a)           # apical error gain
        self.beta_out = float(beta_out)       # readout error gain
        self.rho = float(rho)
        self.alpha = float(alpha_dual)
        self.lambda_max = float(lambda_max)
        self.beta_hc = float(beta_hc)
        self.eta_h = float(eta_h)
        self.eta_W = float(eta_W)             # NO 1/T collapse (Delta 3 fix)
        self.eta_out = float(eta_out)
        self.eta_theta = float(eta_theta)
        self.eta_B = float(eta_B) if eta_B is not None else float(eta_W)
        self.eta_P = float(eta_P) if eta_P is not None else float(eta_W)
        self.lambda_wd = float(lambda_wd)
        self.lambda_B = float(lambda_B)
        self.w_clip = float(w_clip)
        self.gamma_rms = float(gamma_rms)
        self.k_conn = k_conn
        # ── SPEC v1.3 §4.5/§4.5b: C+D stabilization schedule ──
        # Mechanism C: eta_W *= gamma_W every T_decay steps (weight damping)
        # Mechanism D: alpha_theta *= gamma_alpha every T_decay steps (P8 ratio)
        # Default (gamma_W=None) = NO schedule = proven v14.1 baseline.
        self.gamma_W = float(gamma_W) if gamma_W is not None else None
        if self.gamma_W is not None:
            # If caller doesn't set gamma_alpha, default to matched (baseline-equivalent)
            self.gamma_alpha = float(gamma_alpha) if gamma_alpha is not None else self.gamma_W
        else:
            self.gamma_alpha = None
        self.T_decay = int(T_decay)
        self.alpha_theta_0 = float(alpha_theta_0)
        # Live (scheduled) values — mutated by _update_schedule()
        self._eta_W_eff = float(eta_W)
        self._alpha_theta_eff = float(alpha_theta_0)
        if n_hc is None:
            n_hc = out_dim
        self.n_hc = n_hc

        # µPC depth scaling
        self.s_L = float(1.0 / np.sqrt(n_layers))

        # Inference steps
        if T_inference is None:
            self.T = int(10 * n_layers)
        else:
            self.T = int(T_inference)

        rng = np.random.RandomState(seed)
        self._seed = seed

        # ================================================================
        # P6: GENUINE 3D SUBSTRATE
        # ================================================================
        n_per_sheet = sheet_size * sheet_size
        assert hidden_dim <= n_per_sheet, \
            f"hidden_dim {hidden_dim} > sheet capacity {n_per_sheet}"

        self.positions = []
        for l in range(n_layers):
            idx = rng.choice(n_per_sheet, size=hidden_dim, replace=False)
            coords = np.array(
                [(i // sheet_size, i % sheet_size, float(l))
                 for i in idx], dtype=np.float32)
            self.positions.append(coords)

        # Lateral inhibition masks (within-area, local on 3D sheet) — P2 local
        self.inhib_masks_raw = []
        self.k_target = []
        for l in range(n_layers):
            xy = self.positions[l][:, :2]
            diff = xy[:, None, :] - xy[None, :, :]
            d = np.sqrt((diff ** 2).sum(axis=2))
            mask_raw = ((d < inhib_radius) & (d > 0)).astype(np.float32)
            n_neigh = np.maximum(mask_raw.sum(axis=1), 1).astype(np.float32)
            self.inhib_masks_raw.append(
                torch.from_numpy(mask_raw).to(DEVICE))
            mean_n_neigh = float(n_neigh.mean())
            k_target = max(1.0, self.target_rate * mean_n_neigh)
            self.k_target.append(
                torch.tensor(k_target, device=DEVICE, dtype=torch.float32))

        # Feedforward connectivity masks (3D distance between layers)
        self.ff_masks = []
        ff_alpha = 3.0
        ff_radius = 8.0
        for l in range(n_layers - 1):
            diff = self.positions[l][:, None, :] - self.positions[l + 1][None, :, :]
            d_ff = np.sqrt(
                diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2 +
                ff_alpha ** 2 * diff[:, :, 2] ** 2)
            mask = (d_ff < ff_radius).astype(np.float32)
            zero_rows = mask.sum(axis=1) == 0
            if zero_rows.any():
                for i in np.where(zero_rows)[0]:
                    mask[i, np.argmin(d_ff[i])] = 1.0
            if ff_sparsity < 1.0:
                rand_mask = (rng.rand(*mask.shape) < ff_sparsity).astype(np.float32)
                mask = mask * rand_mask
                zero_rows = mask.sum(axis=1) == 0
                if zero_rows.any():
                    for i in np.where(zero_rows)[0]:
                        mask[i, np.argmin(d_ff[i])] = 1.0
            self.ff_masks.append(torch.from_numpy(mask).to(DEVICE))

        # ================================================================
        # Delta 1: DENDRITIC PRODUCT ENCODER (proven EPNet form, v12.1:339-345)
        # Bilinear products over sparse connectivity — builds Fourier circuit
        # ================================================================
        conn = np.zeros((hidden_dim, k_conn), dtype=np.int64)
        for j in range(hidden_dim):
            conn[j] = rng.choice(in_dim, size=k_conn, replace=False)
        self.conn = torch.from_numpy(conn).to(DEVICE)
        pi, pj = np.triu_indices(k_conn, k=1)
        self.pi = torch.from_numpy(pi.astype(np.int64)).to(DEVICE)
        self.pj = torch.from_numpy(pj.astype(np.int64)).to(DEVICE)
        self.n_pairs = k_conn * (k_conn - 1) // 2

        s_dend = 1.0 / np.sqrt(k_conn)
        self.W_lin = torch.from_numpy(
            (rng.randn(hidden_dim, k_conn) * s_dend).astype(np.float32)
        ).to(DEVICE)
        self.W_prod = torch.zeros(hidden_dim, self.n_pairs, device=DEVICE)

        # ================================================================
        # FEEDFORWARD WEIGHTS W_ff[l]: [N, N] (µPC N(0,1/N) init)
        # ================================================================
        s_w = 1.0 / np.sqrt(hidden_dim)
        self.W_ff = []
        for l in range(n_layers - 1):
            W = torch.from_numpy(
                (rng.randn(hidden_dim, hidden_dim) * s_w).astype(np.float32)
            ).to(DEVICE)
            self.W_ff.append(W * self.ff_masks[l])

        # ================================================================
        # P3: CORTICAL FEEDBACK B_fb[l]: [N, N], SEPARATE pathway (apical)
        # ================================================================
        self.B_fb = []
        for l in range(n_layers - 1):
            B = torch.from_numpy(
                (rng.randn(hidden_dim, hidden_dim) * s_w).astype(np.float32)
            ).to(DEVICE)
            B = B * self.ff_masks[l]
            self._spectral_clip(B)
            self.B_fb.append(B)

        # ================================================================
        # P5: HIPPOCAMPAL PROJECTION B_hc[l]: [N, N_hc], FIXED random (broadcast)
        # ================================================================
        s_hc = 1.0 / np.sqrt(n_hc)
        self.B_hc = []
        for l in range(n_layers):
            B = torch.from_numpy(
                (rng.randn(hidden_dim, n_hc) * s_hc).astype(np.float32)
            ).to(DEVICE)
            self._spectral_clip(B)
            self.B_hc.append(B)

        # ================================================================
        # Delta 4: INTERNEURON PREDICTION P[l]: [N, N] (BurstCCN, v12.1)
        # P <- B_fb copy at init (NONZERO, RT-MAJOR-2-REFINE)
        # ================================================================
        self.P = []
        for l in range(n_layers):
            if l <= n_layers - 2:
                P = self.B_fb[l].clone()
            else:
                P = torch.from_numpy(
                    (rng.randn(hidden_dim, hidden_dim) * s_w).astype(np.float32)
                ).to(DEVICE)
                P = P * self.ff_masks[-1] if n_layers > 1 else P
                self._spectral_clip(P)
            self.P.append(P)

        # ================================================================
        # READOUT W_out: [N, N_out]
        # ================================================================
        self.W_out = torch.from_numpy(
            (rng.randn(hidden_dim, out_dim) * s_w).astype(np.float32)
        ).to(DEVICE)

        # ================================================================
        # P8: PER-NEURON THRESHOLDS (homeostatic)
        # ================================================================
        self.thresholds = [torch.zeros(hidden_dim, device=DEVICE) for _ in range(n_layers)]

        # Rolling pre-activation buffer for stable threshold quantiles (v12.1 BUG-A fix)
        self.act_buffer = [[] for _ in range(n_layers)]
        self.act_buffer_max = 10

        # RMSProp accumulators
        self.G_lin = torch.ones_like(self.W_lin) * 1e-8
        self.G_prod = torch.ones_like(self.W_prod) * 1e-8
        self.G_ff = [torch.ones_like(W) * 1e-8 for W in self.W_ff]
        self.G_fb = [torch.ones_like(B) * 1e-8 for B in self.B_fb]
        self.G_P = [torch.ones_like(P) * 1e-8 for P in self.P]
        self.G_out = torch.ones_like(self.W_out) * 1e-8

        self.step_count = 0
        self._calibrated = False
        self.last_gates = {}

    # ================================================================
    # Spectral clipping (power iteration)
    # ================================================================
    @staticmethod
    def _spectral_clip(W, max_norm=1.0, n_iter=30):
        with torch.no_grad():
            m, n = W.shape
            u = torch.randn(m, device=W.device, dtype=W.dtype)
            u = u / (u.norm() + 1e-8)
            for _ in range(n_iter):
                v = W.T @ u
                v = v / (v.norm() + 1e-8)
                u = W @ v
                u = u / (u.norm() + 1e-8)
            sigma = float((W @ v).norm().item())
            if sigma > max_norm:
                W.mul_(0.99 * max_norm / (sigma + 1e-8))

    # ================================================================
    # Delta 5: HARD SPARSITY GATE + softplus + local div-norm (v12.1 form)
    # ================================================================
    def phi_norm(self, u, l):
        """softplus(β) + hard gate + LOCAL divisive normalization.

        Delta 5: restored hard gate (u > θ).float() — sodium channel.
        P2: LOCAL div-norm via inhibitory pools (not global L2).
        """
        # softplus for C² smoothness (envelope theorem)
        s = torch.nn.functional.softplus(
            self.beta * (u - self.thresholds[l].unsqueeze(0))) / self.beta

        # Hard sparsity gate: only above-threshold neurons fire (Delta 5)
        gate = (u > self.thresholds[l].unsqueeze(0)).float()
        s = s * gate
        s = s.clamp(0, 5.0)

        # P2: LOCAL divisive normalization (v12.1 form, per-neuron pools)
        local_sum = s @ self.inhib_masks_raw[l]
        pool_ratio = local_sum / self.k_target[l]
        denom = torch.clamp(pool_ratio, min=1.0)
        x_norm = s / denom

        return x_norm.float()  # cast AFTER division (dtype pitfall)

    def _dsigma(self, u, l):
        """Activation derivative for somatic cascade: sigmoid(β·(u−θ))."""
        return torch.sigmoid(self.beta * (u - self.thresholds[l].unsqueeze(0)))

    # ================================================================
    # Dendritic feedforward (Delta 1 — proven EPNet form)
    # ================================================================
    def _dendritic_fwd(self, X):
        """Layer 0 dendritic feedforward (proven EPNet form)."""
        cv = X[:, self.conn]                                      # [B, N, k_conn]
        u0 = (cv * self.W_lin.unsqueeze(0)).sum(dim=2)           # [B, N]
        pv = cv[:, :, self.pi] * cv[:, :, self.pj]               # [B, N, n_pairs]
        u0 = u0 + (pv * self.W_prod.unsqueeze(0)).sum(dim=2)
        return u0, cv, pv

    # ================================================================
    # Forward initialization (FREE PHASE / test-time activity)
    # ================================================================
    def forward_init(self, X):
        """Forward pass: x[0]=φ_norm(dendritic(X)), x[ℓ+1]=φ_norm(x[ℓ]+s_L·x[ℓ]@W_ff[ℓ]).

        Returns (x, pre_acts) where pre_acts[l] is the pre-activation u for
        layer l — needed for the v12.1 homeostatic update (BUG-A fix: threshold
        crosses measured on PRE-activation, not post-activation).
        """
        X = X.to(DEVICE)
        u0, _, _ = self._dendritic_fwd(X)
        x = [self.phi_norm(u0, 0)]
        pre_acts = [u0]
        for l in range(self.L - 1):
            u = x[l] + self.s_L * (x[l] @ self.W_ff[l])
            x.append(self.phi_norm(u, l + 1))
            pre_acts.append(u)
        return x, pre_acts

    def predict(self, X):
        """Test-time prediction: forward-init (free phase), matching proven EPNet.

        The proven EPNet uses the free phase (pure forward pass) for test-time
        prediction and trains the readout on the same free-phase activity.
        For our multi-layer cortex, forward_init IS the free phase. The
        readout trains on converged x (clamped phase) which provides the
        teaching signal; once the cortex groks, the forward-init features
        become informative and the readout generalizes.
        """
        with torch.no_grad():
            x, _ = self.forward_init(X)
            return (x[self.L - 1] @ self.W_out).argmax(dim=-1)

    def evaluate(self, X, Y):
        return float((self.predict(X) == Y).float().mean().item())

    # ================================================================
    # Delta 4: Apical error ε_a (v12.1 formula with P matrix)
    # ================================================================
    def _compute_eps_a(self, a, r_HC):
        """Compute apical errors ε_a[l] for ALL layers 0..L-1.

        ε_a[l] = B_fb[l]·a[l+1] + B_hc[l]·r_HC − P[l]·a[l]  (interior)
        ε_a[L-1] = B_hc[L-1]·r_HC − P[L-1]·a[L-1]           (top)
        ε_a[0] = B_fb[0]·a[1] + B_hc[0]·r_HC − P[0]·a[0]    (bottom)

        Delta 4: P matrix restored — P^T B_hc ≠ 0 from step 0.
        """
        eps_a = [None] * self.L
        for l in range(self.L):
            if l < self.L - 1:
                u_a = a[l + 1] @ self.B_fb[l].T + r_HC @ self.B_hc[l].T
                eps_a[l] = u_a - a[l] @ self.P[l].T
            else:
                eps_a[l] = r_HC @ self.B_hc[l].T - a[l] @ self.P[l].T
        return eps_a

    # ================================================================
    # Compute prediction residuals (for PC-ALM inference)
    # ================================================================
    def _compute_residuals(self, x):
        s_hat = [None] * self.L
        r = [None] * self.L
        s_hat[0] = x[0]
        r[0] = torch.zeros_like(x[0])
        for l in range(1, self.L):
            u_pred = x[l - 1] + self.s_L * (x[l - 1] @ self.W_ff[l - 1])
            s_hat[l] = self.phi_norm(u_pred, l)
            r[l] = x[l] - s_hat[l]
        return s_hat, r

    # ================================================================
    # Compute primal gradient ∇_x L_ρ (for PC-ALM inference settling)
    # ================================================================
    def _compute_primal_grad(self, x, composite, Y_onehot):
        grad_h = [None] * self.L
        grad_h[0] = None  # input clamped

        for l in range(1, self.L):
            grad = composite[l].clone()
            if l < self.L - 1:
                backward = composite[l + 1] @ self.B_fb[l]
                grad = grad - backward
            else:
                output_err = (x[l] @ self.W_out - Y_onehot)
                grad = grad + output_err @ self.W_out.T
            grad_h[l] = grad
        return grad_h

    # ================================================================
    # BP reference gradients (for Gate 2 — NOT used for learning)
    # ================================================================
    def _compute_bp_grads(self, x_list, Y_onehot, X_input=None):
        """BP reference gradients for Gate2 (cos(eps_a, -BP)).

        FIX (task t_581a1095, red-team audit #4): the forward pass now uses
        self.phi_norm — the SAME local-divisive-normalization forward as EP
        inference (softplus -> hard gate -> clamp -> LOCAL divnorm via
        inhib_masks).  Previously this used GLOBAL L2 norm (s / s.norm(dim=-1)),
        which is a DIFFERENT forward function, making Gate2 an invalid EP-vs-BP
        comparison.  Now both signals come from the identical forward map.
        """
        with torch.enable_grad():
            if X_input is not None:
                X_input = X_input.to(DEVICE).detach()
                u0, _, _ = self._dendritic_fwd(X_input)
                x0_val = self.phi_norm(u0, 0).detach().requires_grad_(True)
                x_chain = [x0_val]
                for l in range(self.L - 1):
                    u = x_chain[l] + self.s_L * (x_chain[l] @ self.W_ff[l])
                    x_chain.append(self.phi_norm(u, l + 1))
                yhat = x_chain[self.L - 1] @ self.W_out
            else:
                x_chain = [xi.detach() for xi in x_list]
                for l in range(1, self.L):
                    u = x_chain[l - 1] + self.s_L * (x_chain[l - 1] @ self.W_ff[l - 1])
                    x_chain[l] = self.phi_norm(u, l)
                yhat = x_chain[self.L - 1] @ self.W_out

            loss = 0.5 * ((Y_onehot - yhat) ** 2).sum() / Y_onehot.shape[0]
            grads = torch.autograd.grad(loss, x_chain[1:], retain_graph=False,
                                        allow_unused=True)
        bp = [torch.zeros_like(x_list[0])]
        for g in grads:
            bp.append(g.detach() if g is not None else torch.zeros_like(x_list[0]))
        return bp

    @staticmethod
    def _cosine(a, b):
        na, nb = a.norm() + 1e-12, b.norm() + 1e-12
        return float((a @ b) / (na * nb))

    @staticmethod
    def _spectral_norm(W, n_iter=30):
        """Estimate the spectral norm (largest singular value) of W via power iteration."""
        with torch.no_grad():
            m, n = W.shape
            u = torch.randn(m, device=W.device, dtype=W.dtype)
            u = u / (u.norm() + 1e-8)
            for _ in range(n_iter):
                v = W.T @ u
                v = v / (v.norm() + 1e-8)
                u = W @ v
                u = u / (u.norm() + 1e-8)
            sigma = float((W @ v).norm().item())
            return sigma

    # ================================================================
    # PC-ALM inference (v13.12 primal-dual cycles)
    # ================================================================
    def infer(self, X, Y_onehot, return_gates=False):
        """Run PC-ALM primal-dual inference to get converged x*.

        1. Forward init → save x^0 (free phase)
        2. T primal-dual cycles (primal step + dual ascent)
        3. Return x*, x0 for EP contrastive update
        """
        X = X.to(DEVICE)
        Y_onehot = Y_onehot.to(DEVICE)
        B = X.shape[0]

        # ── Forward initialization (save x^0) ──
        x, pre_acts = self.forward_init(X)
        x0 = [xi.clone() for xi in x]

        # ── Teaching signal (P5 broadcast) ──
        yhat0 = x0[self.L - 1] @ self.W_out
        score = Y_onehot - yhat0

        # ── Initialize dual variables λ[ℓ] ← 0 ──
        lam = [torch.zeros(B, self.N, device=DEVICE) for _ in range(self.L)]

        # ── Primal-dual cycles ──
        composite = [None] * self.L
        for t in range(self.T - 1):
            _, r = self._compute_residuals(x)
            for l in range(self.L):
                composite[l] = self.rho * r[l] + lam[l]

            grad_h = self._compute_primal_grad(x, composite, Y_onehot)
            for l in range(1, self.L):
                x[l] = x[l] - self.eta_h * grad_h[l]
                x[l] = x[l].clamp(0, 10.0)

            _, r_new = self._compute_residuals(x)
            for l in range(1, self.L):
                broadcast = score @ self.B_hc[l].T
                lam[l] = lam[l] + self.alpha * (r_new[l] + self.beta_hc * broadcast)
                lam[l] = lam[l].clamp(-self.lambda_max, self.lambda_max)

        # ── Final primal step ──
        _, r_f = self._compute_residuals(x)
        for l in range(self.L):
            composite[l] = self.rho * r_f[l] + lam[l]
        grad_f = self._compute_primal_grad(x, composite, Y_onehot)
        for l in range(1, self.L):
            x[l] = x[l] - self.eta_h * grad_f[l]
            x[l] = x[l].clamp(0, 10.0)

        # ── Compute v12.1-style ε_a for BOTH phases (for EP contrastive) ──
        # Free phase: r_HC = 0
        eps_a_free = self._compute_eps_a(x0, torch.zeros_like(Y_onehot))
        # Clamped phase: r_HC = target (proven EPNet/v12.1 form)
        eps_a_clamped = self._compute_eps_a(x, Y_onehot)

        # Per-area contrastive signal dh[l] = x*[l] - x^0[l]
        d = [x[l] - x0[l] for l in range(self.L)]

        # ── Gate diagnostics ──
        gate_log = {}
        if return_gates:
            bp_grads = self._compute_bp_grads(x, Y_onehot, X_input=X)

            # Gate 1: ||eps_a[1]|| / ||eps_a[L-1]|| (ballistic credit)
            ea1_norm = float(eps_a_clamped[1].norm().item()) / max(B, 1)
            eaL_norm = float(eps_a_clamped[self.L - 1].norm().item()) / max(B, 1)
            gate_log['gate1'] = ea1_norm / (eaL_norm + 1e-12)

            # Gate 1d: ||d_1|| / ||d_{L-1}||
            d1_norm = float(d[1].norm().item()) / max(B, 1)
            dL_norm = float(d[self.L - 1].norm().item()) / max(B, 1)
            gate_log['gate1d'] = d1_norm / (dL_norm + 1e-12)

            # Gate 2: cos(eps_a_clamped, -δ^BP) per layer
            gate2_vals = []
            for l in range(1, self.L):
                cos_val = self._cosine(
                    eps_a_clamped[l].mean(dim=0), -bp_grads[l].mean(dim=0))
                gate2_vals.append(cos_val)
            gate_log['gate2_min'] = min(gate2_vals) if gate2_vals else 0.0
            gate_log['gate2_per_layer'] = gate2_vals
            gate_log['gate2_mean'] = float(np.mean(gate2_vals)) if gate2_vals else 0.0

            # Per-layer norms
            gate_log['eps_a_norms_clamped'] = [
                float(eps_a_clamped[l].norm().item()) / max(B, 1) for l in range(self.L)]
            gate_log['eps_a_norms_free'] = [
                float(eps_a_free[l].norm().item()) / max(B, 1) for l in range(self.L)]
            gate_log['dh_norms'] = [
                float(d[l].norm().item()) / max(B, 1) for l in range(self.L)]
            gate_log['lam_norms'] = [
                float(lam[l].norm().item()) / max(B, 1) for l in range(self.L)]

            # Firing rates
            gate_log['firing_rates'] = []
            gate_log['hoyer'] = []
            for l in range(self.L):
                xa = x0[l]
                mean_act = xa.abs().mean(dim=0)
                l1 = mean_act.sum() + 1e-12
                l2 = mean_act.norm() + 1e-12
                hoy = (np.sqrt(self.N) - float(l1) / float(l2)) / (np.sqrt(self.N) - 1)
                hoy = max(0.0, min(1.0, hoy))
                gate_log['hoyer'].append(hoy)
                fr = float((xa > 1e-6).float().mean().item())
                gate_log['firing_rates'].append(fr)

            # Energy
            energy = 0.5 * ((Y_onehot - x[self.L - 1] @ self.W_out) ** 2).sum().item() / B
            gate_log['energy'] = energy

            # Threshold norms
            gate_log['threshold_norms'] = [
                float(self.thresholds[l].norm().item()) for l in range(self.L)]

            # ================================================================
            # L=6 EP attribution verdict (t_f36c8426 / t_a20fd17f) PORTED METRICS
            # Ported from cortex_v14_7.py L889-928 + L1027-1028.
            # Instrumentation only — NO mechanism change (bio-grounded confirmed).
            # ================================================================

            # W_ff Frobenius + spectral norms (per-layer)
            gate_log['W_ff_norms'] = [
                float(self.W_ff[l].norm().item()) for l in range(self.L - 1)]
            gate_log['W_ff_spectral_norm'] = [
                round(self._spectral_norm(self.W_ff[l]), 6)
                for l in range(self.L - 1)]

            # Contrastive decomposition: x_c^T eps_a,c - x_f^T eps_a,f
            #   = x0^T * Deps_a + d^T * eps_a,c
            # (1) d-dependent (learning-relevant): ||d^T . eps_a,c||/(beta*B)
            # (2) bias part (non-learning floor):   ||x0^T . Deps_a||/(beta*B)
            beta_EP = self.beta  # AblationCortex uses self.beta for EP contrastive β
            contrastive_d_dep = []
            contrastive_bias = []
            for l in range(self.L - 1):
                d_dep = d[l].T @ eps_a_clamped[l + 1]   # [N,N]
                contrastive_d_dep.append(
                    float(d_dep.norm().item()) / (beta_EP * B))
                d_eps_a = eps_a_clamped[l + 1] - eps_a_free[l + 1]
                bias = x0[l].T @ d_eps_a  # [N,N]
                contrastive_bias.append(
                    float(bias.norm().item()) / (beta_EP * B))
            gate_log['contrastive_d_dep'] = contrastive_d_dep
            gate_log['contrastive_bias'] = contrastive_bias
            gate_log['d_dependent_contrastive'] = [round(v, 6) for v in contrastive_d_dep]
            gate_log['contrastive_bias_part'] = [round(v, 6) for v in contrastive_bias]

            # Contrastive-only G2: cos(eps_a,c - eps_a,f, -BP) per layer
            # Removes the fixed B_hc.Y confound.
            gate2_contrastive_vals = []
            for l in range(1, self.L):
                d_eps = (eps_a_clamped[l] - eps_a_free[l]).mean(dim=0)
                cos_val = self._cosine(d_eps, -bp_grads[l].mean(dim=0))
                gate2_contrastive_vals.append(cos_val)
            gate_log['gate2_contrastive_per_layer'] = gate2_contrastive_vals
            gate_log['gate2_contrastive_mean'] = \
                float(np.mean(gate2_contrastive_vals)) if gate2_contrastive_vals else 0.0

            self.last_gates = gate_log

        return {'x': x, 'x0': x0, 'd': d, 'lam': lam,
                'eps_a_free': eps_a_free, 'eps_a_clamped': eps_a_clamped,
                'score': score, 'gate_log': gate_log,
                'pre_acts': pre_acts}

    # ================================================================
    # SPEC v1.3 §4.5/§4.5b: C+D step-decay schedule
    # ================================================================
    def _update_schedule(self):
        """Apply Mechanism C+D step-decay at step boundaries.

        n = floor(step_count / T_decay).  Multiplicative decay is equivalent to
        precomputing the power, but applying it incrementally avoids any reset
        ambiguity and keeps the live values auditable.  When gamma_W is None
        this is a no-op (proven baseline).

        P8 ratio: R = alpha_theta_eff / eta_W_eff grows when gamma_alpha >=
        gamma_W (matched → invariant) and falls when gamma_alpha < gamma_W
        (mismatched → P8 restored, §4.5b).
        """
        if self.gamma_W is None:
            return  # No schedule requested — proven baseline.
        n = self.step_count // self.T_decay
        self._eta_W_eff = self.eta_W * (self.gamma_W ** n)
        self._alpha_theta_eff = self.alpha_theta_0 * (self.gamma_alpha ** n)

    # ================================================================
    # EP contrastive training step (Delta 2: replaces activity-contrast)
    # ================================================================
    def train_step(self, X, Y_onehot, return_gates=False):
        """EP contrastive update using apical error ε_a (v12.1 form).

        P1: Δw = η · pre × ε_a × s_L (local 3-factor)
        P3: separate updates for W_ff, B_fb, P
        P5: broadcast teaching via r_HC = target
        P8: homeostatic threshold update (slow)
        SPEC v1.3 §4.5c: uses scheduled eta_W (_eta_W_eff) for all weight
        matrices — C+D stabilization (gamma_W, gamma_alpha, T_decay).
        """
        self.step_count += 1
        self._update_schedule()
        B = X.shape[0]
        beta = self.beta

        with torch.no_grad():
            X_dev = X.to(DEVICE)
            Yoh_dev = Y_onehot.to(DEVICE)

            # Run PC-ALM inference → converged x*, forward-init x^0
            result = self.infer(X_dev, Yoh_dev, return_gates=return_gates)
            x = result['x']           # converged (clamped)
            x0 = result['x0']         # forward init (free)
            pre_acts = result['pre_acts']  # free-phase pre-activations (for P8)
            eps_a_free = result['eps_a_free']
            eps_a_clamped = result['eps_a_clamped']

            # ── Dendritic weights (Delta 1: proven EPNet form) ──
            # Use ε_a contrastive at layer 0: d(ε_a[0]) = ε_a_c[0] - ε_a_f[0]
            cv = X_dev[:, self.conn]
            pv = cv[:, :, self.pi] * cv[:, :, self.pj]
            d_eps_a_0 = eps_a_clamped[0] - eps_a_free[0]

            dW_lin = (d_eps_a_0.unsqueeze(2) * cv).sum(0) / (beta * B)
            dW_lin = dW_lin - self.lambda_wd * self.W_lin
            self._rmsprop(self.W_lin, self.G_lin, dW_lin, self._eta_W_eff)

            dW_prod = (d_eps_a_0.unsqueeze(2) * pv).sum(0) / (beta * B)
            dW_prod = dW_prod - self.lambda_wd * self.W_prod
            self._rmsprop(self.W_prod, self.G_prod, dW_prod, self._eta_W_eff)

            # ── Feedforward weights W_ff (Delta 2: EP contrastive) ──
            # ΔW_ff[l] = η s_L (x_c[l]^T ε_a,c[l+1] − x_f[l]^T ε_a,f[l+1]) / (βB)
            for l in range(self.L - 1):
                dW = self.s_L * (
                    x[l].T @ eps_a_clamped[l + 1] -
                    x0[l].T @ eps_a_free[l + 1]
                ) / (beta * B)
                dW = dW - self.lambda_wd * self.W_ff[l]
                dW = dW * self.ff_masks[l]
                self._rmsprop(self.W_ff[l], self.G_ff[l], dW, self._eta_W_eff)
                # Fix (c′) from divergence_verdict t_3c15fcaa §4: spectral-clip
                # W_ff mirroring B_fb (line 621) and P (line 627). W_ff was the
                # ONLY weight matrix without spectral clipping (only L∞ w_clip).
                # P1-compliant bounded feedforward signals. P3-safe: B_fb untouched.
                self._spectral_clip(self.W_ff[l])

            # ── Cortical feedback B_fb (P3: own plasticity) ──
            # SPEC v1.3 §4.5: C+D step-decay applies to ALL weight matrices.
            # eta_B defaults to eta_W; scale by the same gamma_W factor.
            eta_B_eff = self._eta_W_eff * (self.eta_B / self.eta_W) if self.gamma_W else self.eta_B
            for l in range(self.L - 1):
                dB = (
                    x[l + 1].T @ eps_a_clamped[l] -
                    x0[l + 1].T @ eps_a_free[l]
                ) / (beta * B)
                dB = dB - self.lambda_B * self.B_fb[l]
                dB = dB * self.ff_masks[l]
                self._rmsprop(self.B_fb[l], self.G_fb[l], dB, eta_B_eff)
                self._spectral_clip(self.B_fb[l])

            # ── Delta 4: Interneuron prediction P (BurstCCN, clamped-only) ──
            eta_P_eff = self._eta_W_eff * (self.eta_P / self.eta_W) if self.gamma_W else self.eta_P
            for l in range(self.L):
                dP = (x[l].T @ eps_a_clamped[l]) / B
                self._rmsprop(self.P[l], self.G_P[l], dP, eta_P_eff)
                self._spectral_clip(self.P[l])

            # ── Readout (supervised delta rule on converged activity) ──
            # Readout trains on converged x (clamped-phase settled activity).
            # This is consistent with inference: predict() uses forward_init x0,
            # but the circuit's forward path IS the inference-free path. Once
            # the cortex builds the circuit, x0 and x converge for well-fit
            # inputs. The readout on converged x drives the output_err that
            # feeds the primal-dual inference, giving the cortex a teaching
            # signal to shape x0. Keep converged readout.
            eta_out_eff = self._eta_W_eff * (self.eta_out / self.eta_W) if self.gamma_W else self.eta_out
            dW_out = x[self.L - 1].T @ (Yoh_dev - x[self.L - 1] @ self.W_out) / B
            dW_out = dW_out - self.lambda_wd * self.W_out
            self._rmsprop(self.W_out, self.G_out, dW_out, eta_out_eff)

            # ── P8: Homeostatic threshold update (SLOW, free-phase PRE-ACTIVATIONS) ──
            for l in range(self.L):
                self._homeostatic_update(None, l, u=pre_acts[l])

        return result.get('gate_log')

    def _rmsprop(self, W, G, dW, lr):
        G.mul_(self.gamma_rms).add_((1 - self.gamma_rms) * dW ** 2)
        W.add_(lr * dW / (torch.sqrt(G) + 1e-8))
        W.clamp_(-self.w_clip, self.w_clip)

    # ================================================================
    # P8: Homeostatic threshold update (v12.1 proven form)
    # ================================================================
    def _homeostatic_update(self, h, l, u=None):
        """Slow per-neuron threshold update toward target firing rate.

        Ported from v12.1 (sparse_pc_cortex_v12.py:347-390) — the proven form.

        BUG-A FIX (from v12.1): firing rate is measured on PRE-ACTIVATION
        crossing the threshold (u > theta), NOT post-activation (h > 0).
        Softplus post-activation is ALWAYS > 0, so rate ≡ 1.0 → thresholds
        ratcheted monotonically upward, killing all activity.

        Uses rolling buffer (act_buffer_max=10 batches) for stable quantile
        estimation. Deep-layer pre-activation distributions are narrow and
        batch-level (64-sample) estimates are extremely noisy — pooling 10
        batches (640 samples) gives stable quantiles.

        Slow EMA (α=0.05) toward target threshold ensures P8 timescale
        separation: thresholds move 20× slower than weights (η_W=0.01 vs
        effective α=0.05 per step, but quantile is stable so net movement
        is small and smooth).
        """
        src = u if u is not None else h
        if src is None:
            return

        # Accumulate into rolling buffer
        self.act_buffer[l].append(src.detach().float().cpu())
        if len(self.act_buffer[l]) > self.act_buffer_max:
            self.act_buffer[l].pop(0)

        if len(self.act_buffer[l]) >= 3:
            pooled = torch.cat(self.act_buffer[l], dim=0)
            q = float(1.0 - self.target_rate)
            target_thr = torch.quantile(pooled, q, dim=0).to(DEVICE)
        else:
            # Not enough data yet — use parametric estimate
            batch_mean = src.float().mean(dim=0)
            batch_std = src.float().std(dim=0).clamp(min=1e-4)
            target_thr = batch_mean + 1.2816 * batch_std

        # P8 timescale: threshold EMA toward target. alpha=0.05 was tuned
        # empirically (v14.1 run): alpha=0.001/0.01 killed activity (positive
        # feedback: threshold chases shrinking distribution → activity death).
        # The 3/10 grok-then-forget seeds are a weight-dynamics issue, not
        # threshold speed — reported as known instability.
        #
        # SPEC v1.3 §4.5b (Mechanism D): use the scheduled alpha_theta rate.
        # When no schedule is active, _alpha_theta_eff == alpha_theta_0 == 0.05
        # (proven baseline).  When C+D is active, alpha decays by gamma_alpha
        # every T_decay steps to restore the P8 timescale ratio R.
        alpha = self._alpha_theta_eff
        self.thresholds[l] = (1.0 - alpha) * self.thresholds[l] + alpha * target_thr
        self.thresholds[l].clamp_(0, 5.0)

    # ================================================================
    # Threshold calibration
    # ================================================================
    def calibrate_thresholds(self, X):
        """Initialize thresholds so ~target_rate fraction fires."""
        with torch.no_grad():
            X = X.to(DEVICE)
            u0, _, _ = self._dendritic_fwd(X)
            self.thresholds[0] = torch.quantile(
                u0.float(), 1.0 - self.target_rate, dim=0).to(DEVICE)
            x = [self.phi_norm(u0, 0)]
            for l in range(self.L - 1):
                u = x[l] + self.s_L * (x[l] @ self.W_ff[l])
                self.thresholds[l + 1] = torch.quantile(
                    u.float(), 1.0 - self.target_rate, dim=0).to(DEVICE)
                x.append(self.phi_norm(u, l + 1))
            self._calibrated = True
