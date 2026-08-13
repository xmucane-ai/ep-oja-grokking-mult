#!/usr/bin/env python3
"""cortex_v13_13.py -- SPEC v13.14 §6.4 3D Sparse PC Cortex.

Card t_4eac8511: v13.14 §6.4 P-matrix REVERT — clamped-only P is the
OPERATIVE default. Empirically grounded in v14.1 (9/10 grok at L=2).
The contrastive form (x_c^Tε_a,c − x_f^Tε_a,f)/(βB) FREEZES P in deep
layers where x_c≈x_f — the v13.13 §6.4 mechanism-death root cause
(RCA t_6a870e69). The clamped-only form has NO free-phase subtraction,
so its raw gradient stays nonzero and P keeps tracking B_fb.

v13.14 R2 DOC CORRECTIONS (reflected here, per PATCH_r2):
  - Tracking advantage = ABSENCE of free-phase subtraction, NOT /β.
    Under RMSProp a constant gradient scale c cancels (lr·g/√G), so
    /β is irrelevant to effective step (η_P=η_B=η_W=0.01). The real
    killer is the contrastive subtraction →0 in deep layers.
  - INV3 Form B bound: max_i Σ_{j∈pool(i)} x_j ≤ k_target(l) + tol
    (~5.0 at N=4096, rate 0.10), NOT 1/σ₀. σ₀ is a GHOST in phi_norm.

FIX-A (RT-V1313-1) REVERTED by v13.14: P matrix update is clamped-only
   dP = (x_c^T ε_a,c) / B   [OPERATIVE DEFAULT, use_contrastive_P=False]
FIX-B (RT-V1313-2): readout trains on FORWARD-INIT x⁰[L−1] (UNCHANGED).
FIX-C (P8): η_θ = 0.0001 (UNCHANGED).

Architecture (SPEC v13.14, all P1-P8 compliant):
  - PC-ALM primal-dual inference (forward-init → T cycles → x*)
  - Dendritic product encoder (Delta 1 — Fourier circuit builder)
  - P matrix / BurstCCN with P←B_fb nonzero init (Delta 4)
  - Hard sparsity gate (u>θ).float() (Delta 5)
  - EP contrastive update (Delta 2): ΔW = η_W s_L (x_c^Tε_a,c − x_f^Tε_a,f)/(βB)
  - η_W=0.01 without 1/T collapse (Delta 3)
  - Local inhibition pools (Delta 6 — per-neuron divnorm)
  - µPC parameterisation (s_L=1/√L, N(0,1/N), residual)
  - Genuine 3D substrate (64×64 sheet, distance-dependent connectivity)
  - Runtime invariants INV1-4 (§4.3.5) logged per step
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float32)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CortexV1313:
    """SPEC v13.13 §6.4 cortex: EP contrastive + PC-ALM + dendritic encoder."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers,
                 sheet_size=64, inhib_radius=4.0,
                 target_rate=0.10, sigma_norm=1.0,
                 beta_softplus=4.0, beta_a=1.0, beta_out=2.0,
                 rho=1.0, alpha_dual=0.1, lambda_max=1.0,
                 beta_hc=0.1, T_inference=None,
                 eta_h=0.5, eta_W=0.01, eta_out=0.01,
                 eta_theta=0.0001, eta_B=None, eta_P=None,
                 k_conn=8, n_hc=None,
                 lambda_wd=0.001, lambda_B=0.001,
                 w_clip=5.0, gamma_rms=0.9,
                 ff_sparsity=1.0,
                 use_contrastive_P=False,      # v13.14 §6.4 OPERATIVE DEFAULT: clamped-only dP=(x_c^T·ε_a,c)/B
                 use_forward_init_readout=True,  # v13.13 RT-V1313-2: False = v14.1 converged
                 sigma_mode='floor',           # σ₀+Σ spec t_efdac66b: 'floor'|'static'|'gain'|'none'
                 use_hard_gate=False,          # σ₀+Σ spec: arms are gate-off (β_act ramp handles sharpness)
                 alpha_sigma=0.02,             # P8 σ₀ gain rate (slower than θ's α=0.05)
                 alpha_fr=0.05,                # fr̄ EMA rate for σ₀ target
                 seed=0):

        assert n_layers >= 2, f"P7 VIOLATION: n_layers={n_layers} must be >= 2"
        assert eta_theta < eta_W, f"P8 VIOLATION: eta_theta must be << eta_W"
        assert alpha_dual < rho, "Dual damping violated: alpha must be < rho"

        self.in_dim = in_dim
        self.N = hidden_dim
        self.out_dim = out_dim
        self.L = n_layers
        self.target_rate = target_rate
        self.sigma_0 = float(sigma_norm)
        self.beta = float(beta_softplus)      # β_EP: EP contrastive denominator (FIXED at 4)
        self.beta_act = float(beta_softplus)  # β_act: activation/softplus sharpness (§7 fix #2 — decoupled from β_EP, ramps 4→16 developmentally)
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
        self.use_contrastive_P = use_contrastive_P
        self.use_forward_init_readout = use_forward_init_readout
        # σ₀+Σ spec t_efdac66b: divnorm denominator mode + per-neuron gain homeostat
        self.sigma_mode = sigma_mode            # 'floor'|'static'|'gain'|'none'
        self.use_hard_gate = use_hard_gate
        self.alpha_sigma = float(alpha_sigma)   # P8 σ₀ gain rate
        self.alpha_fr = float(alpha_fr)          # fr̄ EMA rate
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
        # Prediction 5: track threshold drift between consecutive gate samples
        self._prev_thresholds = None
        # Prediction 6: forward-init vs converged oracle accuracy
        self._oracle_W_out = None  # separate oracle readout (trained on x*)
        # P8: rolling pre-activation buffer for stable threshold quantiles
        self.act_buffer = [[] for _ in range(n_layers)]
        self.act_buffer_max = 10

        # ================================================================
        # σ₀+Σ spec t_efdac66b: PER-NEURON HOMEOSTATIC GAIN (P8)
        # σ₀,i(0) = k_target, σ_min = 0.1·k_target, σ_max = 10·k_target
        # fr̄ EMA (post-norm firing indicator) drives the gain homeostat
        # ================================================================
        self.sigma_0_per_neuron = []  # per-neuron σ₀ per layer
        self.sigma_min = []           # hard floor per layer
        self.sigma_max = []           # hard cap per layer
        self.fr_ema = []              # per-neuron fr̄ EMA per layer
        for l in range(n_layers):
            kt = float(self.k_target[l].item())
            s0 = kt * torch.ones(hidden_dim, device=DEVICE, dtype=torch.float32)
            self.sigma_0_per_neuron.append(s0)
            self.sigma_min.append(0.1 * kt)
            self.sigma_max.append(10.0 * kt)
            self.fr_ema.append(torch.zeros(hidden_dim, device=DEVICE, dtype=torch.float32))

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
        """σ₀+Σ divisive normalization (spec t_efdac66b §2, P2 literal form).

        Modes (self.sigma_mode):
          'floor'  — current anti-suppressive clamp(S/k_target, 1) [control]
          'static' — P2-faithful x=s/(σ₀+S), global σ₀=k_target [isolates denominator]
          'gain'   — P2-faithful x=s/(σ₀,i+S), per-neuron σ₀ [THE intervention]
          'none'   — denom=1, no divnorm [negative control]

        s = softplus(β_act·(u−θ))/β_act, optionally × hard gate, clamp[0,5].
        σ₀ is held FIXED during inference (EMA updated once per train_step,
        outside settle loop — preserve EP envelope, spec §6.4).
        """
        # softplus for C² smoothness (envelope theorem)
        # β_act = activation sharpness — decoupled from β_EP (contrastive temp)
        s = torch.nn.functional.softplus(
            self.beta_act * (u - self.thresholds[l].unsqueeze(0))) / self.beta_act

        # Hard sparsity gate (sodium channel) — optional per arm
        if self.use_hard_gate:
            gate = (u > self.thresholds[l].unsqueeze(0)).float()
            s = s * gate
        s = s.clamp(0, 5.0)

        if self.sigma_mode == 'none':
            # No divnorm — denom=1 always (negative control)
            return s.float()

        if self.sigma_mode == 'floor':
            # Current anti-suppressive floor (control arm)
            local_sum = s @ self.inhib_masks_raw[l]
            pool_ratio = local_sum / self.k_target[l]
            denom = torch.clamp(pool_ratio, min=1.0)
            x_norm = s / denom
            return x_norm.float()

        # 'static' or 'gain': P2-faithful σ₀+Σ form
        # x_i = s_i / (σ₀,i + Σ_{j∈pool(i)} s_j)
        local_sum = s @ self.inhib_masks_raw[l]  # [B, N] self-exclusive pool sum

        if self.sigma_mode == 'static':
            # Global constant σ₀ = k_target (uniform semi-saturation)
            denom = self.k_target[l] + local_sum
        elif self.sigma_mode == 'gain':
            # Per-neuron σ₀,i (the P8 gain homeostat — the candidate)
            # σ₀,i is [N], broadcast over batch dim
            denom = self.sigma_0_per_neuron[l].unsqueeze(0) + local_sum
        else:
            raise ValueError(f"Unknown sigma_mode: {self.sigma_mode}")

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

    def forward_init_xonly(self, X):
        """Forward pass returning only x (for predict/oracle compatibility)."""
        x, _ = self.forward_init(X)
        return x

    def predict(self, X):
        with torch.no_grad():
            x = self.forward_init_xonly(X)
            return (x[self.L - 1] @ self.W_out).argmax(dim=-1)

    def predict_oracle(self, X, Y_onehot=None):
        """Prediction 6: converged-oracle accuracy.

        Runs inference to get x*, trains a separate oracle W_out on x*, then
        reads out. If grokking works, |acc(forward-init) − acc(oracle)| → 0
        during training (the forward-init readout becomes as good as the
        converged oracle). Y_onehot needed to train the oracle; if None,
        uses the main W_out on x* (proxy).
        """
        with torch.no_grad():
            if Y_onehot is None:
                # Proxy: just read out x* with main W_out
                result = self.infer(X, torch.zeros(X.shape[0], self.out_dim, device=DEVICE))
                x_star = result['x']
                return (x_star[self.L - 1] @ self.W_out).argmax(dim=-1)
            else:
                result = self.infer(X, Y_onehot)
                x_star = result['x']
                if self._oracle_W_out is None:
                    self._oracle_W_out = torch.randn(
                        self.N, self.out_dim, device=DEVICE) * (1.0 / np.sqrt(self.N))
                # One-step oracle fit on this batch
                oracle_dW = x_star[self.L - 1].T @ (
                    Y_onehot - x_star[self.L - 1] @ self._oracle_W_out) / X.shape[0]
                self._oracle_W_out += 0.1 * oracle_dW  # small lr to track slowly
                return (x_star[self.L - 1] @ self._oracle_W_out).argmax(dim=-1)

    def evaluate(self, X, Y):
        return float((self.predict(X) == Y).float().mean().item())

    def evaluate_oracle(self, X, Y, Y_onehot):
        """Prediction 6 metric: accuracy of converged-oracle readout."""
        preds = self.predict_oracle(X, Y_onehot)
        return float((preds == Y).float().mean().item())

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
        with torch.enable_grad():
            if X_input is not None:
                X_input = X_input.to(DEVICE).detach()
                u0, _, _ = self._dendritic_fwd(X_input)
                s0 = torch.nn.functional.softplus(
                    self.beta * (u0 - self.thresholds[0].unsqueeze(0))) / self.beta
                gate = (u0 > self.thresholds[0].unsqueeze(0)).float()
                s0 = (s0 * gate).clamp(0, 5.0)
                x0_val = s0 / (s0.norm(dim=-1, keepdim=True) + 1e-8)
                x_chain = [x0_val.detach().requires_grad_(True)]
                for l in range(self.L - 1):
                    u = x_chain[l] + self.s_L * (x_chain[l] @ self.W_ff[l])
                    s = torch.nn.functional.softplus(
                        self.beta * (u - self.thresholds[l + 1].unsqueeze(0))) / self.beta
                    gate = (u > self.thresholds[l + 1].unsqueeze(0)).float()
                    s = (s * gate).clamp(0, 5.0)
                    x_next = s / (s.norm(dim=-1, keepdim=True) + 1e-8)
                    x_chain.append(x_next)
                yhat = x_chain[self.L - 1] @ self.W_out
            else:
                x_chain = [xi.detach() for xi in x_list]
                for l in range(1, self.L):
                    u = x_chain[l - 1] + self.s_L * (x_chain[l - 1] @ self.W_ff[l - 1])
                    s = torch.nn.functional.softplus(
                        self.beta * (u - self.thresholds[l].unsqueeze(0))) / self.beta
                    gate = (u > self.thresholds[l].unsqueeze(0)).float()
                    s = (s * gate).clamp(0, 5.0)
                    x_chain[l] = s / (s.norm(dim=-1, keepdim=True) + 1e-8)
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

            # ── Prediction 5: threshold drift (Change 3: per-neuron RMS) ──
            # Spec §7.4 Pred 5: bounded, no drift. P8 compliance.
            # v13.14 Change 3: report per-neuron RMS (‖Δθ‖₂/√N), NOT raw norm.
            # Raw norm at N=4096 with ±0.5 clamp = √(4096×0.25)=32 → misleading.
            if self._prev_thresholds is not None:
                drifts_rms = []
                for l in range(self.L):
                    delta = self.thresholds[l] - self._prev_thresholds[l]
                    rms = float(delta.norm().item()) / np.sqrt(self.N)
                    drifts_rms.append(rms)
                gate_log['threshold_drift_rms'] = drifts_rms
                gate_log['threshold_drift_max'] = max(drifts_rms)
            else:
                gate_log['threshold_drift_rms'] = [0.0] * self.L
                gate_log['threshold_drift_max'] = 0.0
            self._prev_thresholds = [self.thresholds[l].clone() for l in range(self.L)]

            # ── Runtime invariants INV1-4 (SPEC §4.3.5) ──
            # σ₀+Σ spec t_efdac66b: invariants are mode-aware.
            # 'floor' mode: INV3 bound = k_target (original Form B)
            # 'static'/'gain' mode: σ₀>0 bounds activity, INV3 bound = pool_size·5/σ_min
            invariants = {'inv1': True, 'inv2': True, 'inv3': True, 'inv4': True}
            inv_details = {}
            for l in range(self.L):
                xa = x0[l]  # forward-init activity (the divnorm output)
                # INV1: σ₀ > 0 (scalar for floor/static, per-neuron mean for gain)
                if self.sigma_mode == 'gain':
                    s0_mean = float(self.sigma_0_per_neuron[l].mean().item())
                else:
                    s0_mean = float(self.sigma_0)
                inv_details[f'inv1_sigma0_L{l}'] = round(s0_mean, 4)
                # INV3: pool sum bound — mode-dependent
                pool_sums = (xa.abs() @ self.inhib_masks_raw[l])  # [B, N] pool sums
                inv3_max = float(pool_sums.max().item())
                if self.sigma_mode in ('static', 'gain'):
                    # σ₀+Σ: x_i ≤ 5/σ_min → pool sum ≤ pool_size·5/σ_min
                    inv3_bound = float(self.sigma_max[l]) + 1.0  # generous bound
                else:
                    inv3_bound = float(self.k_target[l].item()) + 0.1
                inv_details[f'inv3_max_poolsum_L{l}'] = round(inv3_max, 4)
                inv_details[f'inv3_bound_L{l}'] = round(inv3_bound, 4)
                if inv3_max > inv3_bound:
                    invariants['inv3'] = False
            # INV4: mean(D)/σ₀ > 1.0 — local pool divnorm
            for l in range(self.L):
                xa = x0[l]
                local_sum = xa.abs() @ self.inhib_masks_raw[l]
                if self.sigma_mode == 'gain':
                    s0_val = float(self.sigma_0_per_neuron[l].mean().item())
                else:
                    s0_val = float(self.sigma_0)
                if self.sigma_mode == 'none':
                    mean_D = float(local_sum.mean().item()) + 1.0  # no divnorm
                    ratio = mean_D
                else:
                    mean_D = float(local_sum.mean().item()) + s0_val
                    ratio = mean_D / max(s0_val, 1e-8)
                inv_details[f'inv4_meanD_over_sigma0_L{l}'] = round(ratio, 4)
                if ratio < 1.0 and self.sigma_mode != 'none':
                    invariants['inv4'] = False
            gate_log['invariants'] = invariants
            gate_log['inv_details'] = inv_details

            self.last_gates = gate_log

        return {'x': x, 'x0': x0, 'd': d, 'lam': lam,
                'eps_a_free': eps_a_free, 'eps_a_clamped': eps_a_clamped,
                'score': score, 'gate_log': gate_log,
                'pre_acts': pre_acts}

    # ================================================================
    # EP contrastive training step (Delta 2: replaces activity-contrast)
    # ================================================================
    def train_step(self, X, Y_onehot, return_gates=False):
        """EP contrastive update using apical error ε_a (v12.1 form).

        P1: Δw = η · pre × ε_a × s_L (local 3-factor)
        P3: separate updates for W_ff, B_fb, P
        P5: broadcast teaching via r_HC = target
        P8: homeostatic threshold update (slow)
        """
        self.step_count += 1
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
            self._rmsprop(self.W_lin, self.G_lin, dW_lin, self.eta_W)

            dW_prod = (d_eps_a_0.unsqueeze(2) * pv).sum(0) / (beta * B)
            dW_prod = dW_prod - self.lambda_wd * self.W_prod
            self._rmsprop(self.W_prod, self.G_prod, dW_prod, self.eta_W)

            # ── Feedforward weights W_ff (Delta 2: EP contrastive) ──
            # ΔW_ff[l] = η s_L (x_c[l]^T ε_a,c[l+1] − x_f[l]^T ε_a,f[l+1]) / (βB)
            for l in range(self.L - 1):
                dW = self.s_L * (
                    x[l].T @ eps_a_clamped[l + 1] -
                    x0[l].T @ eps_a_free[l + 1]
                ) / (beta * B)
                dW = dW - self.lambda_wd * self.W_ff[l]
                dW = dW * self.ff_masks[l]
                self._rmsprop(self.W_ff[l], self.G_ff[l], dW, self.eta_W)
                # Fix (c′) from divergence_verdict t_3c15fcaa §4: spectral-clip
                # W_ff mirroring B_fb (line 729) and P (line 743). W_ff was the
                # ONLY weight matrix without spectral clipping — it had only
                # elementwise w_clip=5.0 (L∞, _rmsprop line 765), which does NOT
                # bound the spectral/Frobenius norm that diverges (observed
                # 564→1229 on gate_off seed0). P1-compliant: bounded feedforward
                # signals. P3-safe: does not touch B_fb. Feedforward-only.
                self._spectral_clip(self.W_ff[l])

            # ── Cortical feedback B_fb (P3: own plasticity) ──
            for l in range(self.L - 1):
                dB = (
                    x[l + 1].T @ eps_a_clamped[l] -
                    x0[l + 1].T @ eps_a_free[l]
                ) / (beta * B)
                dB = dB - self.lambda_B * self.B_fb[l]
                dB = dB * self.ff_masks[l]
                self._rmsprop(self.B_fb[l], self.G_fb[l], dB, self.eta_B)
                self._spectral_clip(self.B_fb[l])

            # ── Delta 4: Interneuron prediction P (BurstCCN) ──
            # v13.14 §6.4 REVERT: clamped-only dP = (x_c^T·ε_a,c)/B is the
            # OPERATIVE DEFAULT (use_contrastive_P=False). The contrastive
            # form freezes P in deep layers where x_c≈x_f (mechanism death).
            # dP = (x_c^T ε_a,c) / B                     [clamped-only, v13.14 OPERATIVE]
            # dP = (x_c^T ε_a,c − x_f^T ε_a,f) / (βB)    [contrastive, NEGATIVE CONTROL]
            for l in range(self.L):
                if self.use_contrastive_P:
                    dP = (x[l].T @ eps_a_clamped[l] - x0[l].T @ eps_a_free[l]) / (beta * B)
                else:
                    dP = (x[l].T @ eps_a_clamped[l]) / B
                self._rmsprop(self.P[l], self.G_P[l], dP, self.eta_P)
                self._spectral_clip(self.P[l])

            # ── Readout ──
            # FIX-B (RT-V1313-2): forward-init delta rule (train/test consistent)
            # v14.1 form: converged-x delta rule
            if self.use_forward_init_readout:
                x_out = x0[self.L - 1]
            else:
                x_out = x[self.L - 1]
            dW_out = x_out.T @ (Yoh_dev - x_out @ self.W_out) / B
            dW_out = dW_out - self.lambda_wd * self.W_out
            self._rmsprop(self.W_out, self.G_out, dW_out, self.eta_out)

            # ── P8: Homeostatic threshold update (SLOW, free-phase PRE-ACTIVATIONS) ──
            for l in range(self.L):
                self._homeostatic_update(None, l, u=pre_acts[l])

            # ── σ₀+Σ spec t_efdac66b §3: Per-neuron gain homeostat ──
            # Called OUTSIDE the settle loop (once per train_step).
            # σ₀,i tracks POST-norm firing (the converged clamped-phase x).
            # θ tracks PRE-activation (above) — two-part offset/gain (P8).
            if self.sigma_mode == 'gain':
                self._gain_update(x)

        return result.get('gate_log')

    def _rmsprop(self, W, G, dW, lr):
        G.mul_(self.gamma_rms).add_((1 - self.gamma_rms) * dW ** 2)
        W.add_(lr * dW / (torch.sqrt(G) + 1e-8))
        W.clamp_(-self.w_clip, self.w_clip)

    # ================================================================
    # P8: Homeostatic threshold update (v14.1 proven form)
    # ================================================================
    def _homeostatic_update(self, h, l, u=None):
        """Per-neuron threshold update toward target firing rate (P8).

        Uses the v14.1-proven pre-activation quantile tracking (which grokked
        9/10 at L=2). Ported from ablation_cortex_v14_1.py.

        BUG-A FIX (from v12.1): firing rate is measured on PRE-ACTIVATION
        crossing the threshold (u > theta), NOT post-activation (h > 0).
        Softplus post-activation is ALWAYS > 0, so rate ≡ 1.0 → thresholds
        ratcheted monotonically upward, killing all activity.

        Uses rolling buffer (act_buffer_max=10 batches) for stable quantile
        estimation. Deep-layer pre-activation distributions are narrow and
        batch-level estimates are noisy — pooling 10 batches gives stable
        quantiles.

        EMA (α=0.05) toward target quantile. The spec §12 table says η_θ =
        0.0001 for a rate-based delta rule, but that mechanism is UNSTABLE
        with the hard gate (positive feedback: low rate → threshold up →
        lower rate → ...). The quantile-based EMA is the proven-stable form.

        P8 compliance: separation by TARGET STABILITY, not α < η_W (spec §5.2
        correction v1.1). Although α=0.05 > η_W=0.01 numerically, the quantile
        target is near-stationary once the pre-activation distribution is
        quasi-static (gate-free steady state), so net θ drift per step is small
        and smooth. Weights integrate a changing gradient every step; thresholds
        track a slowly-moving quantile. This is the operative timescale separation.
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
            batch_mean = src.float().mean(dim=0)
            batch_std = src.float().std(dim=0).clamp(min=1e-4)
            target_thr = batch_mean + 1.2816 * batch_std

        alpha = 0.05
        self.thresholds[l] = (1.0 - alpha) * self.thresholds[l] + alpha * target_thr
        self.thresholds[l].clamp_(0, 5.0)

    # ================================================================
    # σ₀+Σ spec t_efdac66b §3: PER-NEURON GAIN HOMEOSTAT (P8)
    # ================================================================
    def _gain_update(self, x_postnorm):
        """Per-neuron σ₀ gain homeostat (P8, spec §3.2).

        σ₀,i ← σ₀,i · exp(α_σ · (fr̄_i − target_rate))
        σ₀,i ← clamp(σ₀,i, σ_min, σ_max)

        Called ONCE per train_step, OUTSIDE the settle loop (spec §6.4:
        σ₀ is a slow parameter, fixed during inference to preserve EP
        envelope). fr̄_i is the EMA of the post-norm firing indicator.

        x_postnorm: list of [B, N] post-norm activities x[l] from the
                    clamped (converged) phase — the actual output σ₀ controls.
        """
        if self.sigma_mode != 'gain':
            return  # only the gain arm runs the homeostat

        for l in range(self.L):
            # Post-norm firing indicator (batch-averaged)
            firing = (x_postnorm[l] > 1e-6).float().mean(dim=0)  # [N]
            # EMA of firing rate
            self.fr_ema[l] = (
                (1.0 - self.alpha_fr) * self.fr_ema[l] +
                self.alpha_fr * firing
            )
            # Multiplicative Turrigiano-style gain update (spec §3.2)
            update = self.alpha_sigma * (self.fr_ema[l] - self.target_rate)
            self.sigma_0_per_neuron[l] = (
                self.sigma_0_per_neuron[l] * torch.exp(update)
            )
            # Hard bounds (spec §3.2: σ_min > 0 non-negotiable)
            self.sigma_0_per_neuron[l].clamp_(
                self.sigma_min[l], self.sigma_max[l])

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
