#!/usr/bin/env python3
"""quantized_ablation_cortex.py -- Low-bit training wrapper for the L=6 main engine.

SPEC_LOWBIT_TRAINING_FALSIFIER_v1.1 (commit 0fadc72). Ports the proven CiM
quantization recipe (BFP per-neuron scale + STE-by-construction + latent fp32
master + two-stage LR/wd) onto the L=6 stabilized main engine
(ablation_cortex_v14_1.py AblationCortex / AblationCortexOpt, FROZEN commit
87c7250, local RMSProp, C+D stabilization).

The EP contrastive learning rule is UNTOUCHED. Quantization is a
weight-representation wrapper (G4): the engine's forward/infer reads
QUANTIZED weights; the EP contrastive update (computed from activities that
settled through quantized weights) is applied to the fp32 LATENT master. The
quantizer is bypassed by construction (STE-by-construction, spec §2.2).

Mechanics implemented here (all per spec §7):
- BFP per-neuron scale (gamma_{l,j} = max_k|W_jk|/q_max) or per-tensor (R4t).
- Stochastic rounding (SR) or nearest.
- ECO (Kahan residual) on the quantization step — harness-implemented (spec
  §2.3 patch 2: ECO is NOT in the engine).
- Latent fp32 master accumulates sub-LSB updates; write-mode quantize once per
  train_step (the main engine's train_step == one mini-batch == the CiM's
  "epoch" = one optimizer step).
- Two-stage wd schedule: wd(t) = wd_0 if t < T_decay_wd else 0 (spec §7.4),
  applied to the latent master. Constant wd = wd_0 always (R1/R4c control).
- Thresholds (P8) + normalization (P2) NOT quantized (fp32, per-neuron analog).

The engine's internal lambda_wd is set to 0 for all arms — the spec's wd
schedule REPLACES the constant wd (spec §4.3 "Replace constant wd with BitNet
two-stage wd"). The spec's wd is the sole wd mechanism.

P1-P8 compliance: P1 (local 3-factor, no Sigma), P2 (div-norm fp32), P3
(separate B_fb, own plasticity), P4 (per-area eps_a), P5 (B_hc broadcast),
P6 (3D substrate), P7 (L>=2), P8 (thresholds fp32, slow). All preserved.
"""
import numpy as np
import torch

from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE


def _sr_round(u, gen=None):
    """Unbiased stochastic rounding: floor(u) + Bernoulli(u - floor(u)). E[round]=u."""
    fl = torch.floor(u)
    frac = u - fl
    if gen is not None:
        r = torch.rand(u.shape, generator=gen, device=u.device)
    else:
        r = torch.rand(u.shape, device=u.device)
    return fl + (r < frac)


class QuantizedAblationCortex(AblationCortexOpt):
    """AblationCortexOpt + low-bit weight-representation wrapper.

    precision: 'fp32' (no quant), 'int4' (qmax=7), 'int3' (qmax=3).
    bfp: 'neuron' (per-neuron BFP scale) or 'tensor' (per-tensor scale).
    wd_schedule: 'two_stage' (wd_0 phase 1, 0 phase 2) or 'constant' (wd_0 always).
    eco: bool — Kahan residual on the quantization step.
    rounding: 'stochastic' (SR) or 'nearest'.
    """

    def __init__(self, *args, precision='fp32', bfp='neuron',
                 wd_schedule='two_stage', wd_0=1.0, eco=True,
                 rounding='stochastic', T_decay_wd=None, **kwargs):
        # The spec's wd schedule REPLACES the engine's constant lambda_wd.
        kwargs['lambda_wd'] = 0.0
        super().__init__(*args, **kwargs)

        self.precision = precision
        self.bfp = bfp
        self.wd_schedule = wd_schedule
        self.wd_0 = float(wd_0)
        self.eco = eco
        self.rounding = rounding
        self.T_decay_wd = int(T_decay_wd) if T_decay_wd is not None else self.T_decay

        if precision == 'fp32':
            self.bits = None
            self.qmax = None
        elif precision == 'int4':
            self.bits = 4
            self.qmax = 2 ** (4 - 1) - 1  # 7
        elif precision == 'int3':
            self.bits = 3
            self.qmax = 2 ** (3 - 1) - 1  # 3
        else:
            raise ValueError(f"unknown precision {precision!r}")

        # ── Latent fp32 masters (copy of engine init weights) ──
        self._lat_W_lin = self.W_lin.clone()
        self._lat_W_prod = self.W_prod.clone()
        self._lat_W_ff = [W.clone() for W in self.W_ff]
        self._lat_B_fb = [B.clone() for B in self.B_fb]
        self._lat_P = [P.clone() for P in self.P]
        self._lat_W_out = self.W_out.clone()

        # ── ECO (Kahan) residuals ──
        self._eco_W_lin = torch.zeros_like(self.W_lin)
        self._eco_W_prod = torch.zeros_like(self.W_prod)
        self._eco_W_ff = [torch.zeros_like(W) for W in self.W_ff]
        self._eco_B_fb = [torch.zeros_like(B) for B in self.B_fb]
        self._eco_P = [torch.zeros_like(P) for P in self.P]
        self._eco_W_out = torch.zeros_like(self.W_out)

        # Quantize the engine weights to the latent masters (write-mode init).
        self._quantize_all()
        self._last_vanish = (0.0, 0.0)

    # ================================================================
    # Quantization primitives
    # ================================================================
    def _quantize_tensor(self, lat, eco_residual):
        """BFP fake-quant (quantize->dequantize) of a latent master.

        Returns the quantized tensor (used by the engine forward/infer).
        Updates the ECO residual in-place (Kahan): r = lat + r_prev - Q(lat + r_prev).
        """
        if self.precision == 'fp32':
            return lat
        qmax = self.qmax
        # ECO: add the carried residual before quantizing.
        w = lat + eco_residual if self.eco else lat
        # BFP scale: per-neuron (reduce over last dim) or per-tensor.
        if self.bfp == 'neuron':
            amax = w.abs().amax(dim=-1, keepdim=True)
        else:
            amax = w.abs().amax()
        q = (amax / qmax).clamp(min=torch.finfo(torch.float32).tiny)
        u = w / q
        if self.rounding == 'stochastic':
            k = _sr_round(u)
        else:
            k = torch.round(u)
        k = k.clamp(-qmax, qmax)
        quantized = k * q
        if self.eco:
            eco_residual.copy_(w - quantized)
        return quantized

    def _quantize_all(self):
        """Write-mode: quantize all latent masters -> engine weight attributes."""
        if self.precision == 'fp32':
            # No quantization — engine weights ARE the latent masters.
            self.W_lin = self._lat_W_lin
            self.W_prod = self._lat_W_prod
            self.W_ff = self._lat_W_ff
            self.B_fb = self._lat_B_fb
            self.P = self._lat_P
            self.W_out = self._lat_W_out
            return
        self.W_lin = self._quantize_tensor(self._lat_W_lin, self._eco_W_lin)
        self.W_prod = self._quantize_tensor(self._lat_W_prod, self._eco_W_prod)
        self.W_ff = [self._quantize_tensor(self._lat_W_ff[l], self._eco_W_ff[l])
                     for l in range(self.L - 1)]
        self.B_fb = [self._quantize_tensor(self._lat_B_fb[l], self._eco_B_fb[l])
                     for l in range(self.L - 1)]
        self.P = [self._quantize_tensor(self._lat_P[l], self._eco_P[l])
                  for l in range(self.L)]
        self.W_out = self._quantize_tensor(self._lat_W_out, self._eco_W_out)

    def _snapshot_engine_weights(self):
        return {
            'W_lin': self.W_lin.clone(),
            'W_prod': self.W_prod.clone(),
            'W_ff': [W.clone() for W in self.W_ff],
            'B_fb': [B.clone() for B in self.B_fb],
            'P': [P.clone() for P in self.P],
            'W_out': self.W_out.clone(),
        }

    def _transfer_to_latent(self, q_before):
        """Transfer the engine's applied deltas (on quantized weights) to latent masters.

        The engine's train_step updated the quantized weights in-place via
        _rmsprop. The applied delta (quantized_after - quantized_before) is the
        STE-by-construction update — add it to the latent master. Clamp the
        latent master to the engine's w_clip range for stability (the quantized
        weights are clamped there by _rmsprop; the latent master must stay in
        the same bounded range so sub-LSB accumulation is meaningful).
        """
        wc = self.w_clip
        self._lat_W_lin = (self._lat_W_lin + (self.W_lin - q_before['W_lin'])).clamp(-wc, wc)
        self._lat_W_prod = (self._lat_W_prod + (self.W_prod - q_before['W_prod'])).clamp(-wc, wc)
        for l in range(self.L - 1):
            self._lat_W_ff[l] = (self._lat_W_ff[l] + (self.W_ff[l] - q_before['W_ff'][l])).clamp(-wc, wc)
            self._lat_B_fb[l] = (self._lat_B_fb[l] + (self.B_fb[l] - q_before['B_fb'][l])).clamp(-wc, wc)
        for l in range(self.L):
            self._lat_P[l] = (self._lat_P[l] + (self.P[l] - q_before['P'][l])).clamp(-wc, wc)
        self._lat_W_out = (self._lat_W_out + (self.W_out - q_before['W_out'])).clamp(-wc, wc)

    def _apply_wd(self):
        """Two-stage wd schedule (spec §7.4): W_lat -= wd(t)*eta*W_lat.

        wd(t) = wd_0 if t < T_decay_wd else 0 (two_stage), or wd_0 always
        (constant). eta = the scheduled weight LR (_eta_W_eff, C+D).
        """
        if self.wd_schedule == 'constant':
            wd = self.wd_0
        else:  # two_stage
            wd = self.wd_0 if self.step_count < self.T_decay_wd else 0.0
        if wd == 0.0:
            return
        eta = self._eta_W_eff
        shrink = wd * eta
        self._lat_W_lin = self._lat_W_lin - shrink * self._lat_W_lin
        self._lat_W_prod = self._lat_W_prod - shrink * self._lat_W_prod
        for l in range(self.L - 1):
            self._lat_W_ff[l] = self._lat_W_ff[l] - shrink * self._lat_W_ff[l]
            self._lat_B_fb[l] = self._lat_B_fb[l] - shrink * self._lat_B_fb[l]
        for l in range(self.L):
            self._lat_P[l] = self._lat_P[l] - shrink * self._lat_P[l]
        self._lat_W_out = self._lat_W_out - shrink * self._lat_W_out

    # ================================================================
    # Override train_step
    # ================================================================
    def train_step(self, X, Y_onehot, return_gates=False):
        """Run the proven engine train_step on quantized weights, then transfer
        the applied update to the latent master + apply the wd schedule."""
        if self.precision == 'fp32':
            # Engine weights ARE latent masters — no quant, no transfer.
            lat_before = self._snapshot_latent()
            gate_log = super().train_step(X, Y_onehot, return_gates=return_gates)
            self._apply_wd()
            self._last_vanish = self.vanish_terminal(lat_before)
            return gate_log

        q_before = self._snapshot_engine_weights()
        lat_before = self._snapshot_latent()
        gate_log = super().train_step(X, Y_onehot, return_gates=return_gates)
        self._transfer_to_latent(q_before)
        self._apply_wd()
        self._quantize_all()
        self._last_vanish = self.vanish_terminal(lat_before)
        return gate_log

    # ================================================================
    # Metrics
    # ================================================================
    def _snapshot_latent(self):
        return {
            'W_lin': self._lat_W_lin.clone(),
            'W_prod': self._lat_W_prod.clone(),
            'W_ff': [W.clone() for W in self._lat_W_ff],
            'B_fb': [B.clone() for B in self._lat_B_fb],
            'P': [P.clone() for P in self._lat_P],
            'W_out': self._lat_W_out.clone(),
        }

    def vanish_terminal(self, lat_before):
        """Fraction of latent-master updates below the quantum (spec §11.3).

        q = per-neuron BFP quantum (max|W|/qmax). For fp32 arms, report against
        a REFERENCE int4 quantum for cross-arm comparability (like the CiM
        _edge_quantum). Returns (fraction, quantum).
        """
        if self.precision == 'fp32':
            ref_qmax = 7  # reference int4 quantum
        else:
            ref_qmax = self.qmax
        fracs = []
        for lat, prev in [(self._lat_W_lin, lat_before['W_lin']),
                          (self._lat_W_prod, lat_before['W_prod'])]:
            d = (lat - prev).abs()
            q = (lat.abs().amax() / ref_qmax).clamp(min=1e-12)
            fracs.append(float((d < q / 2).float().mean().item()))
        for l in range(self.L - 1):
            d = (self._lat_W_ff[l] - lat_before['W_ff'][l]).abs()
            q = (self._lat_W_ff[l].abs().amax() / ref_qmax).clamp(min=1e-12)
            fracs.append(float((d < q / 2).float().mean().item()))
        for l in range(self.L - 1):
            d = (self._lat_B_fb[l] - lat_before['B_fb'][l]).abs()
            q = (self._lat_B_fb[l].abs().amax() / ref_qmax).clamp(min=1e-12)
            fracs.append(float((d < q / 2).float().mean().item()))
        for l in range(self.L):
            d = (self._lat_P[l] - lat_before['P'][l]).abs()
            q = (self._lat_P[l].abs().amax() / ref_qmax).clamp(min=1e-12)
            fracs.append(float((d < q / 2).float().mean().item()))
        d = (self._lat_W_out - lat_before['W_out']).abs()
        q = (self._lat_W_out.abs().amax() / ref_qmax).clamp(min=1e-12)
        fracs.append(float((d < q / 2).float().mean().item()))
        return float(np.mean(fracs)), float(q)

    def fourier_conc(self):
        """Fourier concentration of the layer-0 dendritic encoder (W_lin).

        Each hidden unit j selects k_conn=8 input features (conn[j, c]). Each
        input feature index i has frequency k = (i // 4) + 1 (1..26) and
        component i % 4 (0=cos_a, 1=sin_a, 2=cos_b, 3=sin_b). For each unit,
        build a per-frequency energy vector over K=26 frequencies from W_lin,
        drop DC (none here), and measure the top-frequency fraction.
        conc = importance-weighted mean over units, weighted by output drive
        ||W_out[:, j]||. Mirrors the CiM fourier_code (per-neuron spectral
        concentration, importance-weighted). Returns (conc, dc_frac).
        """
        W_lin = self._lat_W_lin.detach().cpu().numpy()  # (N, k_conn)
        W_out = self._lat_W_out.detach().cpu().numpy()  # (N, P)
        conn = self.conn.detach().cpu().numpy()          # (N, k_conn)
        N, kc = W_lin.shape
        K = 26
        # Frequency of each selected feature.
        freq = (conn // 4) + 1  # (N, kc) in 1..26
        # Per-unit frequency energy: E[j, k] = sum of W_lin[j,c]^2 over c with freq k.
        E = np.zeros((N, K), dtype=np.float64)
        for k in range(1, K + 1):
            mask = (freq == k)
            E[:, k - 1] = (W_lin ** 2 * mask).sum(axis=1)
        total = E.sum(axis=1) + 1e-12
        conc_j = E.max(axis=1) / total  # top-frequency fraction per unit
        imp = np.linalg.norm(W_out, axis=1)  # output drive per unit
        w = imp / (imp.sum() + 1e-12)
        conc = float(np.sum(conc_j * w))
        # dc_frac: fraction of W_out spectral energy in the DC (constant) component.
        dc_energy = np.linalg.norm(W_out.mean(axis=1, keepdims=True), axis=1) ** 2
        tot_energy = np.linalg.norm(W_out, axis=1) ** 2 + 1e-12
        dc_frac = float(np.sum(dc_energy * w) / np.sum(tot_energy * w))
        return conc, dc_frac

    def dh_per_area(self, X, Y_onehot):
        """dh_l = x_clamped[l] - x_free[l] per area (spec §11.3)."""
        with torch.no_grad():
            res = self.infer(X.to(DEVICE), Y_onehot.to(DEVICE), return_gates=False)
            d = res['d']
            B = X.shape[0]
            return [float(d[l].norm().item()) / max(B, 1) for l in range(self.L)]
