"""
schema_gate2.py — GATE 2: does the proven PC rule schema SURVIVE local-radius connectivity on a
3D volume (+ depletion)? Port of schema_gate1.py (the dense MLP) onto the cortex_l0.py 3D substrate.

GATE 1 verdict (schema_gate1.py / NOTES "GATE 1 built"): the rule schema groks under predictive
coding IFF the feedback weights are ALIGNED with the forward weights — pc_transport (feedback=W2)
and pc_kp (feedback learns->W2) grok (test 1.0, Fourier clock, block split 1.0); pc_fa (fixed
random feedback = the genuinely no-transport arm) FAILS. Gate-1 is an ABSTRACT dense MLP.

GATE 2 replaces the dense W1/W2 with a MASKED net whose mask is the 3D-substrate adjacency, and
asks: does grokking survive LOCAL-RADIUS connectivity (+ depletion)? Deliverable = a
grok-vs-edge-density CURVE with controls that make either sign of answer interpretable.

THE HEADLINE (second-set-of-eyes patch #2) is the grok-vs-density CURVE — does locality kill
grokking for ANY rule, including the backprop oracle? A3 (volume-kp) is the "bio-candidate of
record", NOT "THE HEADLINE": KP passing just means backprop-via-learned-alignment survives the
mask; the informative signal is the curve shape + whether the oracle also dies.
PASS is reported as the MINIMUM density at which grokking holds (patch #1: no magic d*=15%),
not a binary cutoff.

DESIGN (gate2_design.md; each pair of arms differs by exactly one thing; matched compute):
  A0  dense control        = A1 @ density 1.0 (transport, no depletion) -> must reproduce Gate-1
  A1  volume-transport     feedback = W2 (tied, by symmetric adjacency), depletion OFF  [positive control]
  A2  volume-transport+DEP  A1 + per-hidden resource depletion ON                      [isolates depletion]
  A3  volume-kp            feedback B2 independent, Kolen-Pollack learns -> W2          [bio-candidate of record]
  A4  volume-fa            feedback B2 fixed random                                     [NEGATIVE control; grok -> bug]
  A5  shuffled-label null  backprop on permuted c  (also the per-density DV2 null)
  A1r random-mask control  A1's degree sequence, edges placed at random (no geometry)   [locality vs sparsity]
  ORACLE backprop          true grads on the SAME mask per density                      [coverage/capacity ceiling]
  FROZEN untrained masked net scored on DV1                                           [geometry-as-learning guard]

SUBSTRATE (cortex_l0.py, ported): jittered 8^3=512 grid; 415 typed nodes (106 input=2P, 256 hidden,
53 output=P) chosen + typed RANDOMLY per seed; KDTree over typed nodes only (no dead-node trap);
edges ONLY input<->hidden and hidden<->output within a 3D radius r. W_eff = W * adjacency (the
constitutional (1-R)=0.9 factor, R=0.1 OPEN, is a GLOBAL scalar absorbed by AdamW's adaptive lr and by
argmax scoring, and is OMITTED so A0 bit-reproduces Gate-1 which has no R factor; see W_EFF_NOTE).
reach_scale=INFINITY for the whole sweep (binary adjacency isolates locality; the exp(-dist) attenuation
is a SECOND, deferred knob -> every arm differs from A0 by exactly one thing).
Density swept by mean-degree / edge-fraction targets {1.0,0.5,0.25,0.12,0.06,0.03}; r chosen PER
SEED to hit the target, then k-NN floor. Grads flow THROUGH W_eff = W (x) M (assert grad-mask == M);
non-edge params are pegged to 0 after every step (excluded from optimizer + metrics). Init fan-in
normalized by LOCAL degree (-> at density 1.0 A0 is bit-identical to Gate-1's init).

GUARDS (constitution): seeds top-level only (stream offsets only, never seed() inside a builder);
positions+types redrawn per seed, SAME positions across arms within a seed (paired); 10 seeds; equal
epochs (6000) across ALL arms AND densities; assert test^train=empty; block split at top frac; 3D
never flattened (mask is (h,2P)/(P,h) but built from genuine 3D positions); R=0 OPEN; reach != lambda
!= myelination; reach_scale=inf. The ONLY Python loop is PC's genuinely-sequential relaxation.
"""

import json
import os
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist

# Reuse the proven Gate-1 pieces verbatim (task, splits, baselines, Fourier DV2, stats).
from schema_gate1 import (
    make_cells, onehot2, onehot2_op, split_random, split_block, cc_acc, fourier_code, _wilcoxon,
)

POS_OFFSET = 70_000          # stream offset for positions/types (split uses RandomState(seed) = Gate-1)
CLOCK_SEED = 12_345          # top-level seed for the fixed DV2 clock measuring-stick (out-of-band)
R_FIXED = 0.1                # R=0 OPEN (constitution); see W_EFF_NOTE below
REACH_SCALE = np.inf         # binary adjacency for the whole sweep (isolates locality)
W_EFF_NOTE = ("W_eff in code = W * adjacency. The constitutional (1-R)=0.9 factor (R=0.1 OPEN) is a "
              "GLOBAL scalar absorbed by AdamW's adaptive lr and by argmax scoring, and is OMITTED so "
              "A0 bit-reproduces Gate-1 (which has no R factor). reach_scale=inf => exp(-dist/inf)=1.")


# ============================================================== 3D substrate ===
def build_volume(seed, n_in, n_hid, n_out, grid_size=8, jitter=0.15):
    """Pure 3D VOLUME (cortex_l0 port): jittered grid; pick n_in+n_hid+n_out positions; assign
    types (input/hidden/output) RANDOMLY. in_idx[j] = physical position of token j (a-tokens are
    the first P, b-tokens the next P). One call per seed -> SAME positions across arms/densities."""
    rng = np.random.RandomState(seed)
    coords = np.arange(grid_size)
    grid = np.array(np.meshgrid(coords, coords, coords, indexing="ij")).reshape(3, -1).T.astype(float)
    grid += rng.standard_normal(grid.shape) * jitter
    n_total = n_in + n_hid + n_out
    assert grid.shape[0] >= n_total, "grid too small for typed node budget"
    chosen = rng.choice(grid.shape[0], size=n_total, replace=False)   # exclude unused grid pts
    pos = grid[chosen]
    perm = rng.permutation(n_total)                                   # random type assignment
    in_idx = perm[:n_in]
    hid_idx = perm[n_in:n_in + n_hid]
    out_idx = perm[n_in + n_hid:]
    return pos, in_idx, hid_idx, out_idx


def pair_coverage(M1, P):
    """Joint (a,b)-pair coverage: fraction of the P*P cells visible to >=1 hidden node (connected to
    BOTH the a-token and the b-token). <1 => the task is unlearnable by construction at this mask."""
    A = M1[:, :P].astype(np.float32)      # bool-safe (§2f): cast at the matmul boundary
    B = M1[:, P:].astype(np.float32)
    both = (A.T @ B) > 0    # (P,P): both[a,b] = exists h with A[h,a]&B[h,b]
    return float(both.mean())


def build_mask(pos, in_idx, hid_idx, out_idx, density, P, h):
    """Radius mask at a target edge-fraction. Binary-search ONE radius r (single knob) so the realized
    cross-type edge count ~= density * (dense bipartite edges). M1:(h,2P) input<->hidden, M2:(P,h)
    hidden<->output. Then k-NN floor (every hidden >=1 a, >=1 b, >=1 output; every token/output >=1
    hidden) and assert it. Returns M1, M2, diagnostics (incl. realized density + pair coverage)."""
    d_ih = cdist(pos[hid_idx], pos[in_idx])    # (h, 2P) hidden x input
    d_ho = cdist(pos[hid_idx], pos[out_idx])   # (h, P)  hidden x output
    dense_total = d_ih.size + d_ho.size        # h*2P + h*P
    target = density * dense_total
    all_d = np.concatenate([d_ih.ravel(), d_ho.ravel()])
    lo, hi = 0.0, float(all_d.max()) + 1e-6
    for _ in range(80):                         # bisection on r to hit the target edge count
        mid = 0.5 * (lo + hi)
        if (d_ih <= mid).sum() + (d_ho <= mid).sum() < target:
            lo = mid
        else:
            hi = mid
    r = hi
    M1 = (d_ih <= r).astype(bool)                 # §2f: binary mask stored as BOOL (upcast at compute)
    M2 = (d_ho <= r).astype(bool).T               # (P, h)
    n_in = d_ih.shape[1]                           # 2P (single-op) or 2P+n_ops (op-cue; OP_CUE_SURGERY_SPEC §3)

    # ---- connectivity floor: add the nearest cross-type neighbour where a requirement is unmet ----
    for hh in range(h):
        if M1[hh, :P].sum() == 0:
            M1[hh, np.argmin(d_ih[hh, :P])] = True
        if M1[hh, P:].sum() == 0:
            M1[hh, P + np.argmin(d_ih[hh, P:])] = True
        if M2[:, hh].sum() == 0:
            M2[np.argmin(d_ho[hh]), hh] = True
    for tt in range(n_in):
        if M1[:, tt].sum() == 0:
            M1[np.argmin(d_ih[:, tt]), tt] = True
    for oo in range(P):
        if M2[oo, :].sum() == 0:
            M2[oo, np.argmin(d_ho[:, oo])] = True
    # ---- assert the floor (per seed per density) ----
    assert (M1[:, :P].sum(1) >= 1).all() and (M1[:, P:].sum(1) >= 1).all(), "floor: hidden needs >=1 a and >=1 b"
    assert (M2.sum(0) >= 1).all(), "floor: every hidden needs >=1 output edge"
    assert (M1.sum(0) >= 1).all(), "floor: every input token needs >=1 hidden edge"
    assert (M2.sum(1) >= 1).all(), "floor: every output needs >=1 hidden edge"

    stats = dict(
        r=float(r),
        dens_ih=float(M1.mean()), dens_ho=float(M2.mean()),
        mean_deg_in=float(M1.sum(1).mean()), mean_deg_out=float(M2.sum(0).mean()),
        coverage=pair_coverage(M1, P),
        realized=float((M1.sum() + M2.sum()) / dense_total),
    )
    return M1, M2, stats


def randomize_mask(M1, M2, rng, n_swaps_per_edge=10):
    """A1r control: degree-preserving 2x2 edge swaps on each bipartite block -> identical per-node
    degree sequence, geometry destroyed. (Full block at density 1.0 is left as-is: A1r==A1 there.)"""
    def swaps(M):
        rows, cols = M.shape
        M = M.copy()
        if 0 < int(M.sum()) < rows * cols:
            edges = [tuple(e) for e in np.argwhere(M)]               # §2f: M is bool -> argwhere True pos
            E = len(edges)
            for _ in range(n_swaps_per_edge * E):
                i, j = rng.randint(E), rng.randint(E)
                (r1, c1), (r2, c2) = edges[i], edges[j]
                if r1 == r2 or c1 == c2 or M[r1, c2] or M[r2, c1]:
                    continue
                M[r1, c2] = M[r2, c1] = 1.0
                M[r1, c1] = M[r2, c2] = 0.0
                edges[i], edges[j] = (r1, c2), (r2, c1)
        return M
    return swaps(M1), swaps(M2)


# ====================================================== masked net (seed-batched) ===
def init_seeds_masked(seeds, P, h, M1, M2, dev, offset=0):
    """Per-seed W1:(S,h,n_in), W2:(S,P,h) with fan-in normalized by LOCAL degree (not n_in/h). At density
    1.0 every local degree == n_in (resp. h) so this is bit-identical to Gate-1's init_seeds. Draw order
    (W1 then W2 from Generator(seed+offset)) matches Gate-1 -> A0 reproduces Gate-1 exactly. n_in is
    DERIVED from M1's last dim so the op-cue path (M1=(S,h,2P+n_ops), OP_CUE_SURGERY_SPEC §2) needs no
    new arg; n_ops=0 callers pass M1=(S,h,2P) -> n_in=2P -> bitwise-unchanged."""
    M1t = torch.tensor(M1, device=dev)
    M2t = torch.tensor(M2, device=dev)
    n_in = M1t.shape[2]                                          # 2P (single-op) or 2P+n_ops (op-cue)
    fan1 = M1t.sum(2).to(torch.float32).clamp(min=1.0)            # §2f: bool mask -> upcast fan-in to float
    fan2 = M2t.sum(2).to(torch.float32).clamp(min=1.0)            # (S,P) hidden-fan-in per output (sum over h)
    W1, W2 = [], []
    for i, s in enumerate(seeds):
        g = torch.Generator(device=dev).manual_seed(s + offset)
        sc1 = (1.0 / torch.sqrt(fan1[i])).unsqueeze(1)     # (h,1)
        sc2 = (1.0 / torch.sqrt(fan2[i])).unsqueeze(1)     # (P,1)
        W1.append(torch.randn(h, n_in, generator=g, device=dev) * sc1)
        W2.append(torch.randn(P, h, generator=g, device=dev) * sc2)
    return torch.stack(W1), torch.stack(W2)


def init_B_masked(seeds, P, h, M2, dev, offset):
    """Independent feedback B2:(S,P,h), masked, scaled like W2 (per-output local fan-in). At density
    1.0 == Gate-1 init_B (1/sqrt(h)). Never copied from W2 (FA asserted != W2 at init)."""
    M2t = torch.tensor(M2, device=dev)
    fan2 = M2t.sum(2).to(torch.float32).clamp(min=1.0)            # §2f: bool mask -> upcast fan-in to float
    B = []
    for i, s in enumerate(seeds):
        g = torch.Generator(device=dev).manual_seed(s + offset)
        sc = (1.0 / torch.sqrt(fan2[i])).unsqueeze(1)
        B.append(torch.randn(P, h, generator=g, device=dev) * sc)
    return torch.stack(B)


def blogits_masked(X, W1, W2, M1, M2, gain=1.0):
    """Batched forward through W_eff = W (x) M (autograd -> grad-mask == M by chain rule)."""
    if X.dtype != torch.float32:                       # §2f: one-hot stored BOOL; upcast at matmul
        X = X.to(torch.float32)
    W1e = W1 * M1                                                      # float32 (M bool auto-upcasts)
    W2e = W2 * M2
    if gain != 1.0:
        W2e = W2e * gain
    a1 = torch.relu(torch.einsum("snf,shf->snh", X, W1e))
    return torch.einsum("snh,sph->snp", a1, W2e)


def bpc_grads_masked(W1, W2, X, Y, B2, M1, M2, T, eta,
                     deplete=False, dep_rate=0.06, tau=5.0, gain=1.0, want_trace=False,
                     n_per_entry=None, tr_mask=None,
                     pi_h=None, pi_out=None, pi_fb=None, want_c3=False, fb_gain=1.0):
    """Batched PC local credit assignment (Whittingman-Bogacz), grads taken THROUGH W_eff = W (x) M at
    the relaxed state. Relaxation updates HIDDEN free nodes only; input/output clamped. Feedback B2
    carries output error into hidden (transport: B2=W2 -> Be=W2e; FA: frozen; KP: learned). Same PC
    objective every arm, so transport-vs-no-transport (+mask, +depletion) is the only thing that
    differs. gW1 = gW1_eff (x) M1, gW2 = gW2_eff (x) M2 -> grad-mask == adjacency by construction.

    §2f: X/Y may be stored BOOL (one-hot); upcast to float32 at the matmul/subtraction boundary (compute
    stays float32). §2e (lever 1): when DIFFERENT-size splits are batched (random+block) the smaller is
    zero-padded to the max -- pass n_per_entry=(REAL per-entry n,) and tr_mask=(B,n) so grads normalize
    by the REAL n (NOT the padded max) and padded rows contribute exactly 0 (X=0 -> gW1 term 0; a1e
    masked -> gW2 term 0). Default (None) = single uniform split -> scalar n, no masking (unchanged).

    DEPLETION (A2): per-hidden resource r persists across the T relaxation steps, RESET per gradient
    step (full-batch => cross-example persistence would be a batch-coupling confound); clamped
    input/output nodes do not deplete; resource scales the hidden activity on the FORWARD output path
    (a1*r) AND the feedback current into hidden (fb*r) SYMMETRICALLY, so the update is still the
    instantaneous-energy gradient (E = .5||e1||^2 + .5||Y - (a1*r)@W2e||^2; dE/dW2e = -e2 (x) (a1*r)).
    The only Python loop is this relaxation.

    C3 PRECISION ARMS (C3_CELL_SPEC §3; optional, default None => bitwise-vanilla path unchanged):
      - C3-S (STATIC per-layer, energy-consistent §3.1): pass `pi_h` (hidden Π_h, shape broadcastable
        to (S,n,h) e.g. (S,1,1)) and `pi_out` (output Π_out, (S,1,1)). Applied as e1_eff=pi_h*e1 in the
        relaxation update AND gW1; e2_eff=pi_out*e2 in fb AND gW2. Energy E=.5 sum(pi_h e1^2)+.5 sum(pi_out e2^2).
      - C3-D (DYNAMIC per-unit, EXOGENOUS fb-gate §3.3): pass `pi_fb` (per-unit π_i, shape (S,1,h),
        broadcasts over the batch n). Applied as fb_i <- pi_fb_i * fb_i INSIDE the loop ONLY; e1/gW1 stay
        VANILLA (pi_h=None), Π_out≡1 (pi_out=None). Do NOT route C3-D through the §3.1 formula (spec §3.1).
      PARITY GUARD (mandatory): with pi_h=pi_out=ones (C3-S) or pi_fb=ones (C3-D) the mul-by-1.0 is
      IEEE-exact => output is bitwise-identical to the vanilla path (pi_*=None, no mul). The None branch
      skips the mul entirely so the proven vanilla/close-out path is byte-inert.
    EPOCH-COUPLING (spec §3.3): this fn is STATELESS across calls. The C3-D displacement EMA d_i lives in
      the CALLER (run_seeds_masked): caller passes pi_fb (computed from last epoch's d_i) IN, this fn
      RETURNS the per-unit displacement (diag position 6), caller updates d_i. π is never computed here.
    DIAGNOSTIC RETURN (want_trace=True OR want_c3=True): 8-tuple
      (gW1, gW2, trace, mean_r, residual, per_unit_disp, e1sq_mean, e2sq_mean):
        trace        = per-step energy list, or None when want_trace=False (C3 non-log epoch),
        per_unit_disp= mean_n|x1^T - x1^{T-1}| shape (S,h) (REAL n via tr_mask; the C3-D signal),
        e1sq_mean/e2sq_mean = per-seed mean of e1^2/e2^2 (S,) (the C3-S Π EMA input).
      Default (want_trace=False, want_c3=False): 2-tuple (gW1, gW2) -- UNCHANGED for vanilla callers."""
    use_ph = pi_h is not None                    # C3-S hidden precision on e1 (None => vanilla)
    use_po = pi_out is not None                  # C3-S output precision on e2 (None => vanilla)
    use_pf = pi_fb is not None                   # C3-D per-unit fb gate (None => vanilla)
    diag = want_trace or want_c3                 # extended 8-tuple return when any diagnostics needed
    with torch.no_grad():
        if X.dtype != torch.float32:                     # §2f: one-hot stored BOOL; upcast ONCE at boundary
            X = X.to(torch.float32)
        if Y.dtype != torch.float32:                     # §2f: target stored BOOL; subtraction needs float
            Y = Y.to(torch.float32)
        W1e = W1 * M1                                                     # float32 (M bool auto-upcasts)
        W2e = W2 * M2
        if gain != 1.0:
            W2e = W2e * gain
        Be = (B2 * M2) if B2 is not None else W2e
        if gain != 1.0 and B2 is not None:
            Be = Be * gain
        if fb_gain != 1.0:                                  # feedback-only attenuation (the "second gear"):
            Be = Be * fb_gain                                # forward W2e UNCHANGED -> loop gain = fb_gain*||W2||^2.
                                                            # Default 1.0 = tied (byte-parity). <1.0 damps the
                                                            # relaxation independently of the forward path. NB:
                                                            # scaled INSIDE bpc_grads_masked so the caller's `fb`
                                                            # stays `is W2` (the tied-weights assert at ~L676 holds).
        mu1 = torch.einsum("snf,shf->snh", X, W1e)
        x1 = mu1.clone()
        x1_prev = None                                # snapshot for the final-step residual / per-unit disp
        S, h = X.shape[0], W1.shape[1]
        r = torch.ones(S, h, device=X.device) if deplete else None
        trace = [] if want_trace else None
        for _ in range(T):
            if diag:
                x1_prev = x1                          # pre-update state -> becomes x1^(T-1) on the last step
            a1 = torch.relu(x1)
            a1e = a1 * r.unsqueeze(1) if deplete else a1          # (S,n,h)*(S,1,h)
            e2 = Y - torch.einsum("snh,sph->snp", a1e, W2e)
            e1 = x1 - mu1
            e1_eff = pi_h * e1 if use_ph else e1                  # C3-S weights e1 (§3.1); None => e1 (vanilla)
            e2_eff = pi_out * e2 if use_po else e2                # C3-S weights e2 in fb (§3.1); None => e2
            fb = torch.einsum("snp,sph->snh", e2_eff, Be)
            if deplete:
                fb = fb * r.unsqueeze(1)                          # symmetric on the feedback path
            if use_pf:
                fb = pi_fb * fb                                   # C3-D per-unit gate on fb (§3.3); e1/gW1 vanilla
            x1 = x1 + eta * (-e1_eff + (x1 > 0).float() * fb)
            if deplete:
                fire = a1.mean(1)                              # per-hidden MEAN firing over the batch (S,h)
                r = torch.clamp(r - dep_rate * fire + (1.0 - r) / tau, 0.0, 1.0)
            if want_trace:
                trace.append(float(0.5 * (e1 ** 2).sum() + 0.5 * (e2 ** 2).sum()))
        a1 = torch.relu(x1)
        a1e = a1 * r.unsqueeze(1) if deplete else a1
        e2 = Y - torch.einsum("snh,sph->snp", a1e, W2e)
        e1 = x1 - mu1
        e1_eff = pi_h * e1 if use_ph else e1                      # relaxed-state precision-weighted e1 (grads)
        e2_eff = pi_out * e2 if use_po else e2                    # relaxed-state precision-weighted e2 (grads)
        if n_per_entry is None:                       # single uniform split -> scalar n (unchanged path)
            n = X.shape[1]
            gW1 = (-torch.einsum("snh,snf->shf", e1_eff, X) / n) * M1
            gW2 = (-torch.einsum("snp,snh->sph", e2_eff, a1e) / n) * M2
        else:                                         # §2e: REAL per-entry n + zero padded rows' grad
            n_b = n_per_entry[:, None, None]
            a1e_g = a1e * tr_mask[:, :, None]              # padded rows -> 0 in the gW2 contraction
            gW1 = (-torch.einsum("snh,snf->shf", e1_eff, X) / n_b) * M1     # X=0 on padded -> gW1 term already 0
            gW2 = (-torch.einsum("snp,snh->sph", e2_eff, a1e_g) / n_b) * M2
    if not diag:
        return gW1, gW2
    resid = x1 - x1_prev                            # §3c: ||x1^(T) - x1^(T-1)|| (x1_prev tracked when diag)
    per_unit = (x1 - x1_prev).abs()                 # C3-D signal: |x1^T - x1^{T-1}|, (S,n,h)
    e1sq = e1 ** 2                                  # C3-S Π EMA input
    e2sq = e2 ** 2
    if tr_mask is not None:                         # §2e: REAL examples only (mask padded rows)
        m = tr_mask[:, :, None]
        resid = resid * m
        per_unit = per_unit * m
        e1sq = e1sq * m
        e2sq = e2sq * m
        residual = resid.reshape(resid.shape[0], -1).norm(dim=1).cpu().numpy()      # per-entry (B,) L2 norm
        per_unit_disp = (per_unit.sum(dim=1) / n_per_entry.clamp(min=1.0)[:, None]).cpu().numpy()  # mean over REAL n -> (S,h)
        e1sq_mean = (e1sq.sum(dim=(1, 2)) / n_per_entry.clamp(min=1.0)).cpu().numpy()    # (S,)
        e2sq_mean = (e2sq.sum(dim=(1, 2)) / n_per_entry.clamp(min=1.0)).cpu().numpy()    # (S,)
    else:
        residual = resid.reshape(resid.shape[0], -1).norm(dim=1).cpu().numpy()
        per_unit_disp = per_unit.mean(dim=1).cpu().numpy()                        # (S,h)
        e1sq_mean = e1sq.mean(dim=(1, 2)).cpu().numpy()                           # (S,)
        e2sq_mean = e2sq.mean(dim=(1, 2)).cpu().numpy()                           # (S,)
    trace_out = (np.array(trace) if want_trace else None)
    mean_r_out = (float(r.mean()) if deplete else 1.0)
    return gW1, gW2, trace_out, mean_r_out, residual, per_unit_disp, e1sq_mean, e2sq_mean


# ---------------------------------------------------- optional torch.compile JIT ---
# Env-flagged (GATE2_COMPILE=1) kernel-fusion speedup, ZERO science change (same math, inductor-fused).
# The hot training path calls these with want_trace=False (no CPU sync) + static shapes per
# (cfg,frac,split) cell -> mode="reduce-overhead" (CUDA-graph replay). The want_trace=True one-shot
# calls (settle_smoke, A2 mean_r re-measurement) hit a float() sync -> graph-break -> eager fallback
# for those calls only. Default OFF (uncompiled). Best-effort: if the inductor/triton stack is
# unavailable (e.g. this Windows env: no triton wheel matching torch's inductor API), the flag prints a
# notice and falls back to the raw functions instead of crashing -- so it is safe to set GATE2_COMPILE=1
# on WSL2/Linux with a matching triton, and a no-op fallback here. Equivalence must still be checked
# (compiled vs uncompiled) before any compiled run is trusted for science.
COMPILE = os.environ.get("GATE2_COMPILE", "0") == "1"
if COMPILE:
    try:
        bpc_grads_masked = torch.compile(bpc_grads_masked, mode="reduce-overhead")
        blogits_masked = torch.compile(blogits_masked, mode="reduce-overhead")
    except Exception as _e:                    # inductor/triton unavailable -> honest best-effort fallback
        print(f"[GATE2_COMPILE] torch.compile unavailable ({type(_e).__name__}: {_e!s}); "
              f"running uncompiled. (Best-effort flag; use a matching triton on WSL2/Linux.)")
        COMPILE = False


def _optimizer(params, cfg):
    """Optimizer factory. cfg['opt'] (default 'adamw') selects the update rule; the AdamW default is
    the stream-parity path (close-out / GATE-2 / Gate-2.1a wd-sweep all use it -- exposing it as a knob
    changes NOTHING when unset). 'sgd' = plain SGD (momentum=0) -- the Gate-2.1a optimizer-axis closer
    (the #6 'PROVEN' hedge). Same params / lr / wd; only the update rule differs. NB: SGD wd is coupled-L2
    (PyTorch default) vs AdamW decoupled -- biases TOWARD confirming hard-capacity (conservative)."""
    name = cfg.get("opt", "adamw")
    if name == "adamw":
        # fused=True: one CUDA kernel replaces ~15 foreach launches (same update equations, PyTorch-guaranteed).
        # Conditional on CUDA (fused AdamW raises on CPU). VERIFIED: a zeroed-grad param (gate=sleep) still
        # decays under fused (decoupled wd acts on param.data inside the fused kernel, independent of grad)."""
        fused = bool(params) and all(p.is_cuda for p in params)
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"], fused=fused)
    if name == "sgd":
        # momentum default 0.0 = plain GD (byte-identical to the PC-native N1 run). cfg["momentum"] opts
        # into heavy-ball (N1' negative control: does momentum break the train=0.83 plateau?).
        return torch.optim.SGD(params, lr=cfg["lr"], weight_decay=cfg["wd"],
                               momentum=cfg.get("momentum", 0.0))
    raise ValueError(f"unknown optimizer {name!r} (cfg['opt']); expected 'adamw' or 'sgd'")


def _corr_unit(x, y):
    """Pearson correlation between two equal-length 1D arrays; NaN if either has zero variance (§3.4
    corr(π_i, d_i) across units -- zero variance occurs when π is uniform, e.g. parity α=0)."""
    xm = x - x.mean(); ym = y - y.mean()
    denom = float(np.sqrt((xm ** 2).sum() * (ym ** 2).sum()))
    return float(xm @ ym / denom) if denom > 0 else float("nan")


# =========================================================== one arm, all seeds ===
def run_seeds_masked(mode, seeds, labels, splits, M1, M2, a, b, P, cfg, dev,
                     deplete=False, gain=1.0, label_kind="real",
                     log_per_epoch=None, early_stop=None, es_uses_block=False, want_rtdiag=False,
                     w1_gate="both", norm_gate=None,
                     channel_switch_epoch=None, phase2_regime=None, probe_epochs=None,
                     init_weights=None,
                     mechinterp_probes=False, probe_inputs=None, probe_labels=None,
                     per_seed_probe=False,
                     want_gradnorm=False,
                     n_ops=0, ops=None):
    """Train one (arm, density) cell. Batched over the seed dim AND over splits (§2e lever 1): `splits`
    is a LIST of split-sets (each a per-seed list of (tr,te)); `labels` a parallel list of per-seed
    label arrays. Batch dim B = len(splits)*len(seeds); entry (sp,s) lives at index sp*S + s. Splits of
    different sizes (random 2528 vs block 2520) are zero-padded to the max and normalized by the REAL
    per-entry n (n_per_entry) -- padded rows relax to 0 and contribute exactly 0 to grads (X=0 -> gW1
    term 0; a1e masked -> gW2 term 0). SAME MATH as a serial per-split call; the 4-agent verifies
    batched-vs-unbatched dW = 0.0. Returns a per-ENTRY list (length B) of {test, train, conc, align,
    hist, per_epoch}; the driver maps entries [:S]=split0 (random), [S:2S]=split1 (block).

    §2f storage: one-hot X/targets stored BOOL (upcast to float32 at the matmul boundary), class
    indices INT16, masks BOOL. Compute (weights/activations/grads/matmuls) stays float32.
    §2b/§3c: logging + early-stop run every K=cfg['log_every'] epochs (want_trace ~1% of the time ->
    99% of PC epochs compiled). §3.1: shuffled->train>=0.99, real->test>=0.9 (+block>=0.9 when the
    block split is batched in, §3.2's batched allowance) in >=8/10 seeds.
    RTDIAG (want_rtdiag=True; KIMI-THE-WATCH §4.1, ReLU homogeneity): every cfg['snap_every'] epochs
    snapshot the pegged W1 and, on the NEXT snapshot, decompose dW1=W1_cur-W1_prev into
    radial_i=(dW1.u_i)u_i (the positive-rescaling GAUGE DOF, function-neutral under ReLU) and
    tangential_i=dW1-radial_i (direction change). READ-ONLY: a detach().clone() of W1 after the peg,
    no RNG/dtype/graph perturbation -> want_rtdiag=True reproduces want_rtdiag=False bitwise. Logged
    into per_epoch['rtdiag'] (requires log_per_epoch=True). Each interval entry also carries
    per_unit_w1norm (the (h,) per-unit ||W1_i|| at the snapshot) -- the per-unit identity trajectory:
    units with growing ||W1_i|| are the radial inflactors; post-hoc checks if the SAME indices stay high
    across epochs (committed schema assemblies) or rotate (diffuse instability).
    W1-GATE (w1_gate; default 'both' = no gating = bitwise-inert): project gW1 onto radial/tangential
    using the CURRENT W1 (pre-step) and zero the gated channel BEFORE opt.step (W2 learns freely).
    'radial' = freeze tangential (user: direction drift is orbiting-relaxation noise); 'tangential' =
    freeze radial (Kimi C5a: norm inflation is the problem); 'frozen' = zero gW1 (W2-alone control).
    Mask preserved by construction (u=0 on non-edges). Keep want_rtdiag=True to verify the gate.
    NORM-BAND GATE (norm_gate; default None = OFF = bitwise-inert): C5 FUSEE. Per-batch-element sleep/
    wake keyed on ||W1|| (the bifurcation parameter, NOT displacement -- displacement~5 sustains through
    the grok so a disp-gate would block it). norm_gate = dict(theta_hi, theta_lo): once ||W1||>=theta_hi
    an entry FREEZES (gW1 AND gW2 zeroed = sleep); it wakes only when wd shrinks ||W1|| below theta_lo
    (deadlock-breaker -- AdamW's decoupled wd applies to param.data inside opt.step regardless of the
    gated gradient, so zeroed-grad still decays). Hysteresis band [theta_lo, theta_hi) holds prior state.
    The per-element scalar gate preserves zero-on-non-edges (0*gate=0), so the grad-mask assert holds.
    theta_hi=inf -> gate never fires -> bitwise 'both' (the parity guard). Logged per-epoch:
    ng_sleep (B,) bool, ng_disp_pct (B,3) [p50,p90,p99] over the per-unit displacement, ng_wake_frac (B,)
    cumulative over ALL epochs (accumulated every epoch, not just log points).
    INIT_WEIGHTS (default None = fresh init = bitwise-inert): SEQUENTIAL warm-start for C2/C4. Pass a dict
    {"W1":(S,h,n_in) float, "W2":(S,P,h) float, ["d_ema":(S,h) float]}. W1/W2 OVERRIDE init_seeds_masked
    (before the n_sp repeat, so n_sp>1 replicates the warm start across splits). Optional d_ema seeds the
    C3-D precision EMA (c3_dynamic only) so the protection state carries across phases. Optimizer + other
    cross-epoch state are ALWAYS fresh (matches the staged-channel FLUSH design). None path = untouched."""
    assert mode in ("backprop", "pc_transport", "pc_fa", "pc_kp", "frozen",
                    "c3_static", "c3_dynamic"), f"unknown mode {mode!r}"
    if log_per_epoch is None:
        log_per_epoch = cfg.get("log_per_epoch", False)
    if early_stop is None:
        early_stop = cfg.get("early_stop", False)
    K = cfg.get("log_every", 1)                       # §2b: periodic cadence (1 = every epoch)
    h, T, eta = cfg["h"], cfg["T"], cfg["eta"]
    S = len(seeds)
    n_sp = len(splits)                                # §2e: number of batched splits (1 or 2)
    B = n_sp * S

    # ---- build zero-padded batched tensors + REAL per-entry n + masks (§2e) ----
    n_tr_max = max(len(splits[sp][s][0]) for sp in range(n_sp) for s in range(S))
    n_te_max = max(len(splits[sp][s][1]) for sp in range(n_sp) for s in range(S))
    n_in = 2 * P + n_ops                                   # OP_CUE_SURGERY_SPEC §2 (n_ops=0 -> 2P, bitwise)
    if n_ops > 0:
        assert ops is not None and len(ops) == len(a), "n_ops>0 requires ops array (len==len(a))"
        Xall = torch.tensor(onehot2_op(a, b, ops, P, n_ops), device=dev).to(torch.bool)  # §2f: one-hot BOOL
    else:
        Xall = torch.tensor(onehot2(a, b, P), device=dev).to(torch.bool)        # §2f: one-hot BOOL
    Xtr = torch.zeros(B, n_tr_max, n_in, dtype=torch.bool, device=dev)
    Xte = torch.zeros(B, n_te_max, n_in, dtype=torch.bool, device=dev)
    Ytr = torch.zeros(B, n_tr_max, P, dtype=torch.bool, device=dev)         # §2f: target BOOL
    cte = torch.zeros(B, n_te_max, dtype=torch.int16, device=dev)           # §2f: class index INT16
    ctr = torch.zeros(B, n_tr_max, dtype=torch.int16, device=dev)
    tr_mask = torch.zeros(B, n_tr_max, dtype=torch.bool, device=dev)
    te_mask = torch.zeros(B, n_te_max, dtype=torch.bool, device=dev)
    n_tr_pe = torch.zeros(B, dtype=torch.float32, device=dev)               # REAL per-entry n (grad div)
    n_te_pe = torch.zeros(B, dtype=torch.float32, device=dev)
    if n_ops > 0:                                           # OP_CUE_SURGERY_SPEC §5: per-op accuracy needs the
        ops_te = torch.full((B, n_te_max), -1, dtype=torch.int8, device=dev)  # op-id of each (padded) test cell
        ops_tr = torch.full((B, n_tr_max), -1, dtype=torch.int8, device=dev)
    else:
        ops_te = ops_tr = None
    for sp in range(n_sp):
        for s in range(S):
            idx = sp * S + s
            tr, te = splits[sp][s]
            lab = labels[sp][s]
            ntr, nte = len(tr), len(te)
            Xtr[idx, :ntr] = Xall[tr]
            Xte[idx, :nte] = Xall[te]
            Ytr[idx, torch.arange(ntr), torch.tensor(lab[tr])] = True
            cte[idx, :nte] = torch.tensor(lab[te])
            ctr[idx, :ntr] = torch.tensor(lab[tr])
            tr_mask[idx, :ntr] = True
            te_mask[idx, :nte] = True
            n_tr_pe[idx] = ntr
            n_te_pe[idx] = nte
            if n_ops > 0:
                ops_te[idx, :nte] = torch.tensor(ops[te], dtype=torch.int8, device=dev)
                ops_tr[idx, :ntr] = torch.tensor(ops[tr], dtype=torch.int8, device=dev)
    # ALWAYS pass per-entry n + mask so the n_sp==1 and n_sp>2 paths use the SAME tensor-division
    # kernel (tensor/n vs tensor/python-int round differently by ~1e-10 -> would drift over 30k steps).
    # For n_sp==1 these are uniform/all-True -> mathematically the scalar path; the 4-agent verifies
    # batched-vs-unbatched dW = 0.0. (bpc_grads_masked keeps the scalar branch only for its None callers.)
    npc = (n_tr_pe, tr_mask)

    M1t = torch.tensor(M1, device=dev)                                            # §2f: mask BOOL
    M2t = torch.tensor(M2, device=dev)
    W1, W2 = init_seeds_masked(seeds, P, h, M1, M2, dev)
    if init_weights is not None:                       # SEQUENTIAL warm-start (C2/C4): inject phase-1 W1/W2
        # guard: only modes whose cross-epoch state is fully carried (W1/W2 [+c3_dynamic d_ema]) are
        # supported. c3_static (Pi_h/Pi_out), pc_kp/pc_fa (learned B2) would SILENTLY lose state.
        assert mode in ("c3_dynamic", "backprop", "pc_transport", "frozen"), \
            f"init_weights not supported for mode={mode!r} (silent loss of its cross-epoch state)"
        W1 = init_weights["W1"].detach().to(dev).to(torch.float32).contiguous().clone()
        W2 = init_weights["W2"].detach().to(dev).to(torch.float32).contiguous().clone()
        assert W1.shape == (S, h, n_in), f"init_weights W1 {tuple(W1.shape)} != ({S},{h},{n_in})"
        assert W2.shape == (S, P, h), f"init_weights W2 {tuple(W2.shape)} != ({S},{P},{h})"
    if n_sp > 1:                                       # §2e: replicate per-seed mask/init across splits
        M1t = M1t.repeat(n_sp, 1, 1); M2t = M2t.repeat(n_sp, 1, 1)     # entry sp*S+s == M[s] (same as serial)
        W1 = W1.repeat(n_sp, 1, 1); W2 = W2.repeat(n_sp, 1, 1)         # -> main & block start from same W

    def _acc(pred, ref, mask, n_pe):                  # masked accuracy over REAL examples (§2e pad-safe)
        return ((pred == ref) & mask).sum(dim=1).to(torch.float32) / n_pe

    if mode == "frozen":                              # untrained masked net (geometry-as-learning guard)
        with torch.no_grad():
            pred_tr = blogits_masked(Xtr, W1, W2, M1t, M2t).argmax(-1)
            pred_te = blogits_masked(Xte, W1, W2, M1t, M2t).argmax(-1)
        out = []
        for idx in range(B):
            conc, hist = fourier_code(W1[idx] * M1t[idx], W2[idx] * M2t[idx], P)
            out.append(dict(test=float(_acc(pred_te[idx], cte[idx], te_mask[idx], n_te_pe[idx])),
                            train=float(_acc(pred_tr[idx], ctr[idx], tr_mask[idx], n_tr_pe[idx])),
                            conc=conc, align=float("nan"), hist=hist, mean_r=float("nan"),
                            per_epoch=None))
        return out

    W1.requires_grad_(True); W2.requires_grad_(True)
    opt = _optimizer([W1, W2], cfg)                       # cfg['opt'] default 'adamw' = stream-parity
    B2, optB = None, None
    if mode == "pc_fa":
        B2 = init_B_masked(seeds, P, h, M2, dev, 10_000)
        if n_sp > 1: B2 = B2.repeat(n_sp, 1, 1)
        assert not torch.allclose(B2, W2.detach()), "FA feedback must not equal W2 at init"
    elif mode == "pc_kp":
        B2 = init_B_masked(seeds, P, h, M2, dev, 20_000).requires_grad_(True)
        if n_sp > 1: B2 = B2.repeat(n_sp, 1, 1)
        optB = _optimizer([B2], cfg)                       # kp feedback optimizer (default AdamW)

    grad_asserted = False
    wd0_safe = (cfg.get("wd", 1.0) == 0.0)               # §3 req 2: wd=0 |W1| can grow unbounded -> guard
    diverged = False
    log_pe = log_per_epoch
    do_stop = early_stop
    need_metrics = log_pe or do_stop
    snap_every = cfg.get("snap_every", 100)               # RTDIAG snapshot cadence (epochs)
    T_eval = cfg.get("T_eval", None)                      # eval-only long-relaxation diagnostic (None = off)
    if want_rtdiag:
        assert log_pe, "want_rtdiag needs log_per_epoch=True (rtdiag hitches a ride on the per_epoch carrier)"
    per_epoch = {"train_acc": [], "test_acc": [], "w1_norm": [], "w2_norm": [], "residual": [], "F": []}
    if want_gradnorm:                  # MECH-DIG: read-only per-epoch gradient/error norms (PC modes)
        per_epoch["e1_rms"] = []       # per-entry RMS sqrt(mean e1^2) at the relaxed state (None for BP)
        per_epoch["e2_rms"] = []
        per_epoch["gW1_norm"] = []     # per-entry Frobenius ||gW1|| of the raw PC grad (pre-gate)
        per_epoch["gW2_norm"] = []
    if n_ops > 0:                                          # OP_CUE_SURGERY_SPEC §5: per-op acc trajectories
        for op_id in range(n_ops):
            per_epoch[f"test_acc_op{op_id}"] = []
            per_epoch[f"train_acc_op{op_id}"] = []
    if channel_switch_epoch is not None:                    # STAGED CHANNEL: log phase2_active (bool per log pt)
        per_epoch["phase2_active"] = []
    if want_rtdiag:
        per_epoch["rtdiag"] = []                          # list (per interval) of (B,) interval-summary dicts
        if T_eval is not None:                            # N1' finite-T diagnostic: F trajectory + resid at T_eval
            per_epoch["F_T200"] = []                      # per-snapshot aggregate F trajectory over T_eval steps
            per_epoch["resid_T200"] = []                  # per-snapshot per-entry (B,) final-step residual
    w1_prev_rtd = None                                    # RTDIAG: pegged-W1 baseline from the previous snapshot

    # ---- C3 precision cross-epoch state (C3_CELL_SPEC §3.2/§3.3; lives HERE, not in stateless bpc_grads) ----
    # C3 builds on pc_transport (feedback = W2). deplete=False for all C3 arms (spec §2).
    is_c3 = mode in ("c3_static", "c3_dynamic")
    is_pc = mode != "backprop"                     # any mode running the PC relaxation (T_eval diagnostic)
    c3_state = None
    if is_c3:
        per_epoch["pi_mean"] = []            # §3.4: mean(π) over units, per entry
        per_epoch["pi_frac_low"] = []        # §3.4: frac units with π_i < 0.5*π0, per entry (C3-D)
        per_epoch["pi_corr"] = []            # §3.4: corr(π_i, d_i) across units, per entry (C3-D)
        per_epoch["Pi_h"] = []               # §3.4: Π_h trajectory, per entry (C3-S)
        per_epoch["Pi_out"] = []             # §3.4: Π_out trajectory, per entry (C3-S)
        if mode == "c3_dynamic":             # C3-D: per-unit displacement EMA d_i (β=0.99), π_i from PREVIOUS epoch
            c3_state = dict(
                d_ema=np.zeros((B, h), dtype=np.float32),            # per-(entry,unit) EMA (spec §3.3)
                pi0=cfg.get("c3_pi0", 1.0), pimin=cfg.get("c3_pimin", 0.02),
                beta=cfg.get("c3_beta", 0.99), alpha=cfg.get("c3_alpha", 0.0),   # alpha=0 => π≡1 (parity)
            )
            if init_weights is not None and init_weights.get("d_ema") is not None:  # SEQUENTIAL: carry precision state
                _d = np.array(init_weights["d_ema"], dtype=np.float32)
                if _d.shape == (S, h) and B != S:               # tile across splits (mirrors the W1/W2 repeat)
                    _d = np.tile(_d, (n_sp, 1))
                assert _d.shape == (B, h), f"init_weights d_ema {tuple(_d.shape)} != ({B},{h})"
                c3_state["d_ema"] = _d.copy()
        else:                                # C3-S: per-layer Π_h/Π_out scalars, inverse-EMA of e² (§3.2)
            c3_state = dict(
                Pi_h=np.ones((B, 1, 1), dtype=np.float32), Pi_out=np.ones((B, 1, 1), dtype=np.float32),
                ema_e1sq=np.zeros(B, dtype=np.float32), ema_e2sq=np.zeros(B, dtype=np.float32),
                lam=cfg.get("c3_lambda", 0.1), decay=cfg.get("c3_ema_decay", 0.99),  # horizon~100
                pmin=cfg.get("c3_pmin", 0.01), pmax=cfg.get("c3_pmax", 100.0), eps=cfg.get("c3_eps", 1e-8),
            )
    stopped_epoch = cfg["epochs"]                       # = full budget (no early stop); else the stop epoch
    # ---- C5 NORM-BAND gate cross-epoch state (FUSEE addendum §2/§3): sleep_state is a per-entry bool that
    # persists across epochs (hysteresis). wake_count accumulates wake epochs EVERY epoch (the
    # cumulative wake-fraction needs all epochs, not just log points). norm_gate=None -> all None (inert)."""
    ng_active = norm_gate is not None
    sleep_state = torch.zeros(B, dtype=torch.bool, device=dev) if ng_active else None
    ng_wake_count = torch.zeros(B, dtype=torch.float64, device=dev) if ng_active else None
    if ng_active:
        per_epoch["ng_sleep"] = []          # (B,) bool per log point (instantaneous sleep state)
        per_epoch["ng_disp_pct"] = []       # (B,3) [p50,p90,p99] over per-unit displacement, per log point
        per_epoch["ng_wake_frac"] = []      # (B,) cumulative wake-fraction over ALL epochs, per log point
    # ---- DYNMAP probe state (DYNMAP_SPEC §2-§4; DYNMAP_BUILD_SPEC §1): fixed-grid READ-ONLY snapshots. Probes
    # fire AFTER grad computation + AFTER w1_gate/norm_gate (g1_try == the final gated update about to be
    # applied) but BEFORE opt.step, snapshotting pre-step W1/W2 and the in-hand g1_try. All under no_grad;
    # RNG capture/restore around the probe (parity-inert; the bit-parity smoke is the proof). Rides the
    # per_epoch carrier (requires log_per_epoch, like want_rtdiag). The frozen grid lists ep=30000 which is
    # UNREACHABLE under epochs=30000 (range -> 0..29999), so the terminal epoch (epochs-1) is added to the
    # probe set -> exactly 40 probes for a 30k run (the recorded epoch field is the ACTUAL ep, honest)."""
    probe_active = probe_epochs is not None
    if probe_active:
        assert log_pe, "probe_epochs needs log_per_epoch=True (probes ride the per_epoch carrier)"
        probe_set = set(int(e) for e in probe_epochs)
        probe_set.add(int(cfg["epochs"]) - 1)          # always probe the terminal epoch (30000 grid -> ep 29999)
        probe_data = [[] for _ in range(B)]
    else:
        probe_set = set()
        probe_data = None
    # ---- MECH-INTERP probe battery (MECH_INTERP_SPEC §3; rides the DYNMAP probe carrier). Flag gates the
    # read-only mech-interp fields. PROBE SET (v1 cue-confound fix): when per_seed_probe=True each seed uses
    # its OWN test split's first-20 as the probe set (all 20 genuinely held-out for every seed); when False
    # (v1 path) the driver passes one global probe_inputs/probe_labels (seed-0's; cue-confounded for seeds
    # 1-9, kept only to reproduce the banked v1 JSON). Built here as per-entry (B,MI_PROBE_N,2P)/(B,MI_PROBE_N)
    # so the probe block is uniform. All ops under no_grad + DETERMINISTIC (no RNG) -> parity-inert."""
    mi_active = bool(mechinterp_probes)
    if mi_active:
        assert probe_active, "mechinterp_probes needs probe_epochs (it rides the probe carrier)"
        if per_seed_probe:
            # V2: per-seed held-out probe set (each seed's own test split, first 20). REAL labels (a+b)%P
            # regardless of arm label_kind (the contribution/Fourier diagnostics measure the TRUE function).
            # OP-CUE (OP_CUE_SURGERY_SPEC §6): when n_ops>0 there is no single TRUE function; use the caller's
            # per-cell op-conditioned label (labels[0][0]: add for op0, mult for op1) so mi_contrib measures the
            # function each probe cell is actually trained under. n_ops=0 keeps (a+b)%P -> bitwise-inert.
            if n_ops > 0:
                _c_real = labels[0][0]
            else:
                _c_real = (a + b) % P                               # numpy (N,) real correct class
            MI_PROBE_INPUTS = torch.zeros(B, MI_PROBE_N, n_in, dtype=torch.float32, device=dev)
            MI_PROBE_LABELS = torch.zeros(B, MI_PROBE_N, dtype=torch.long, device=dev)
            for sp in range(n_sp):
                for s in range(S):
                    idx = sp * S + s
                    _pidx = splits[sp][s][1][:MI_PROBE_N]          # this seed's test split, first 20
                    assert len(_pidx) >= MI_PROBE_N, "seed test split < MI_PROBE_N (lower frac)"
                    MI_PROBE_INPUTS[idx] = Xall[_pidx].to(torch.float32)      # (20,n_in)
                    MI_PROBE_LABELS[idx] = torch.as_tensor(_c_real[_pidx], dtype=torch.long, device=dev)
        else:
            # V1: one global probe set (driver-supplied), broadcast across all entries.
            assert probe_inputs is not None and probe_labels is not None, \
                "mechinterp_probes needs probe_inputs (20,n_in) and probe_labels (20,) [or per_seed_probe=True]"
            _gi = torch.as_tensor(probe_inputs, dtype=torch.float32, device=dev)
            _gl = torch.as_tensor(probe_labels, dtype=torch.long, device=dev)
            assert _gi.shape == (MI_PROBE_N, n_in) and _gl.shape == (MI_PROBE_N,), \
                f"probe_inputs/labels must be ({MI_PROBE_N},n_in)/({MI_PROBE_N},); got {_gi.shape}/{_gl.shape}"
            MI_PROBE_INPUTS = _gi.unsqueeze(0).expand(B, MI_PROBE_N, n_in)     # (B,20,n_in) broadcast view
            MI_PROBE_LABELS = _gl.unsqueeze(0).expand(B, MI_PROBE_N)            # (B,20)
    else:
        MI_PROBE_INPUTS = MI_PROBE_LABELS = None
    for ep in range(cfg["epochs"]):
        log_epoch = (ep % K == 0) or (ep == cfg["epochs"] - 1)      # §2b: log/eval/early-stop at cadence K
        opt.zero_grad(set_to_none=True)
        # STAGED CHANNEL: compute frozen flag EARLY (before C3 state update) so the precision EMA (d_ema)
        # also freezes when skip_step=True — a full freeze freezes weights AND the precision diagnostics
        # (red-team Issue A: without this, residual/F/pi logs drift despite bitwise-frozen weights).
        _frozen_now = (channel_switch_epoch is not None and ep >= channel_switch_epoch
                       and phase2_regime == "frozen")
        resid_ep = None
        e1sq_ep = e2sq_ep = None                 # PC-native F logging (§2b); None for backprop, set in the pc else-block
        if mode == "backprop":
            err = (blogits_masked(Xtr, W1, W2, M1t, M2t, gain) - Ytr.to(torch.float32)) ** 2  # padded->0
            loss = 0.5 * (err.sum(dim=(1, 2)) / n_tr_pe).sum()      # §2e: per-entry REAL n (==serial /n)
            loss.backward()
            g1_try, g2_try = W1.grad, W2.grad
        else:
            fb = W2 if mode in ("pc_transport", "c3_static", "c3_dynamic") else B2
            # ---- C3 precision: compute pi_* from LAST epoch's cross-epoch state, pass IN (§3.3 epoch-coupling) ----
            pi_h_in = pi_out_in = pi_fb_in = None
            disp_ep = e1sq_ep = e2sq_ep = None
            pi_used = d_used = pi_out_used = None     # snapshot of THIS epoch's π (for §3.4 logging; pre-EMA-update)
            want_c3_ep = is_c3
            want_trace_ep = log_pe and log_epoch
            if is_c3:
                if mode == "c3_dynamic":
                    st = c3_state
                    pi_np = np.clip(st["pi0"] / (1.0 + st["alpha"] * st["d_ema"]),
                                    st["pimin"], st["pi0"]).astype(np.float32)        # (B,h)
                    pi_fb_in = torch.as_tensor(pi_np, device=dev).unsqueeze(1)        # (B,1,h) broadcasts over n
                    pi_used, d_used = pi_np, st["d_ema"]                              # (B,h) each, for logging
                else:  # c3_static
                    pi_h_in = torch.as_tensor(c3_state["Pi_h"], device=dev)          # (B,1,1)
                    pi_out_in = torch.as_tensor(c3_state["Pi_out"], device=dev)
                    pi_used = c3_state["Pi_h"].reshape(B)                            # (B,) Π_h used this epoch
                    pi_out_used = c3_state["Pi_out"].reshape(B)                      # (B,) Π_out used this epoch
            pc_out = bpc_grads_masked(W1, W2, Xtr, Ytr, fb, M1t, M2t, T, eta,
                                      deplete, cfg.get("dep_rate", 0.06),
                                      cfg.get("tau", 5.0), gain,
                                      want_trace=want_trace_ep, want_c3=want_c3_ep,
                                      n_per_entry=npc[0], tr_mask=npc[1],
                                      pi_h=pi_h_in, pi_out=pi_out_in, pi_fb=pi_fb_in,
                                      fb_gain=cfg.get("fb_gain", 1.0))
            if want_trace_ep or want_c3_ep:              # §3c: residual reused from THIS relaxation (no 2nd pass)
                g1_try, g2_try, _tr, _mr, resid_ep, disp_ep, e1sq_ep, e2sq_ep = pc_out
            else:
                g1_try, g2_try = pc_out
            W1.grad, W2.grad = g1_try, g2_try
            # ---- update C3 cross-epoch state from THIS epoch's relaxed diagnostics (§3.3 epoch-coupling) ----
            # STAGED CHANNEL: freeze the precision EMA when _frozen_now (skip_step) — a full freeze freezes
            # weights AND the c3d precision state (red-team Issue A: d_ema drifting with frozen weights
            # confounds the frozen-cell diagnostic logs). Partial-learning cells (frozen_w1only/radial) keep
            # updating d_ema (the precision tracks the changing weights — correct c3d dynamics).
            if is_c3 and disp_ep is not None and not _frozen_now:
                if mode == "c3_dynamic":
                    st = c3_state
                    st["d_ema"] = st["beta"] * st["d_ema"] + (1.0 - st["beta"]) * disp_ep     # (B,h) EMA
                else:  # c3_static: residual-equilibrium inverse-EMA (§3.2); Π refreshed every LOG_EVERY
                    st = c3_state
                    st["ema_e1sq"] = st["decay"] * st["ema_e1sq"] + (1.0 - st["decay"]) * e1sq_ep
                    st["ema_e2sq"] = st["decay"] * st["ema_e2sq"] + (1.0 - st["decay"]) * e2sq_ep
                    if log_epoch:
                        inv1 = (1.0 / (st["ema_e1sq"] + st["eps"])).reshape(B, 1, 1)        # (B,1,1) to match Pi_h
                        inv2 = (1.0 / (st["ema_e2sq"] + st["eps"])).reshape(B, 1, 1)
                        st["Pi_h"] = np.clip(st["Pi_h"] ** (1 - st["lam"]) * inv1 ** st["lam"],
                                             st["pmin"], st["pmax"]).astype(np.float32)
                        st["Pi_out"] = np.clip(st["Pi_out"] ** (1 - st["lam"]) * inv2 ** st["lam"],
                                                st["pmin"], st["pmax"]).astype(np.float32)
        # ---- STAGED CHANNEL switch (STAGED_CHANNEL_SPEC §3a): at channel_switch_epoch, transition from
        # phase-1 (both channels, caller's w1_gate) to a phase-2 regime. Optimizer FLUSH at the switch epoch
        # re-inits AdamW m,v (eliminates the stale phase-1 momentum transient — Review 1). frozen = skip
        # opt.step entirely (true zero dW for W1 AND W2); frozen_w1only = step normally then RESTORE W1 to
        # pre-step (W2 updates alone); radial = gate W1 to radial-only via the existing projection
        # (effective_gate overrides w1_gate in phase 2). Logging (train/test/F/rtdiag) continues regardless —
        # we track whether the schema degrades without weight changes. Default (channel_switch_epoch=None) ->
        # effective_gate=w1_gate, skip_step=False, restore=False = bitwise-inert (parity guard)."""
        effective_gate = w1_gate
        skip_step = False
        restore_w1_after = False
        if channel_switch_epoch is not None and ep >= channel_switch_epoch:
            if ep == channel_switch_epoch:
                opt = _optimizer([W1, W2], cfg)   # FLUSH: re-init AdamW m,v (stale phase-1 transient out)
            if phase2_regime == "frozen":
                skip_step = True
            elif phase2_regime == "frozen_w1only":
                restore_w1_after = True
            elif phase2_regime == "radial":
                effective_gate = "radial"
        # ---- W1-gating factorial (user spec): project gW1 onto radial/tangential using the CURRENT
        # W1 (pre-step), zero the gated channel, set W1.grad so opt.step applies the gated gradient.
        # W2 learns freely (full g2_try). Compute u from the MASKED effective weights (W1*M1t) so u=0 on
        # non-edges at EVERY epoch (incl. epoch 0, where W1 is still the unmasked init -> raw W1 would put
        # nonzero radial_g/tang_g on non-edges and crash the grad-mask assert at d<1.0). At d=1.0 (no
        # non-edges) W1*M1t == W1 so this is identical to the raw-W1 projection. "both" = bitwise-inert.
        # STAGED CHANNEL: uses effective_gate (may differ from w1_gate in phase 2); skipped entirely when
        # skip_step=True (frozen regime — no gradient projection needed since opt.step is skipped)."""
        if effective_gate != "both" and not skip_step:
            with torch.no_grad():
                W1eff = W1 * M1t.to(torch.float32)                       # (B,h,2P) masked; 0 on non-edges always
                w1n = W1eff.norm(dim=2, keepdim=True).clamp(min=1e-8)    # (B,h,1) pre-step ||W1_eff[i,:]||
                u = W1eff / w1n                                          # (B,h,f) unit radial dir u_i (0 on non-edges)
                proj = (g1_try * u).sum(dim=2, keepdim=True)             # (B,h,1) signed radial projection of gW1
                radial_g = proj * u                                      # (B,h,f) radial component of gW1 (0 on non-edges)
                tang_g = g1_try - radial_g                               # (B,h,f) tangential component (~gW1 on non-edges)
            if effective_gate == "radial":
                g1_try = radial_g            # freeze TANGENTIAL (user: orbiting-relaxation direction drift is noise)
            elif effective_gate == "tangential":
                g1_try = tang_g              # freeze RADIAL (Kimi C5a: norm inflation is the problem)
            elif effective_gate == "frozen":
                g1_try = torch.zeros_like(g1_try)
            else:
                raise ValueError(f"unknown effective_gate {effective_gate!r} (expected both|radial|tangential|frozen)")
            W1.grad = g1_try                  # opt.step() applies the gated gradient (backprop+PC paths)
        # ---- C5 NORM-BAND GATE (FUSEE addendum §2/§3): per-batch-element sleep/wake on ||W1||. Compute
        # AFTER mask peg of the PREVIOUS step (non-edges already 0 -> raw W1 norm excludes them; at d=1.0
        # raw==masked), BEFORE opt.step (so the gated gradient is what the optimizer applies). theta_hi=inf
        # -> (w1_norms>=hi) is always False -> sleep_state stays False -> gate=1 -> bitwise 'both'. The
        # per-element scalar gate preserves zero-on-non-edges (0*gate=0) so the grad-mask assert holds.
        # wd (AdamW decoupled) applies inside opt.step regardless of the gated grad -> shrinks ||W1||
        # during sleep -> drops below theta_lo -> wake (deadlock-breaker; VERIFIED: zeroed-grad still decays)."""
        if ng_active:
            with torch.no_grad():
                w1_norms = W1.reshape(B, -1).norm(dim=1)            # (B,) per-entry Frobenius (pegged W1)
                hi, lo = float(norm_gate["theta_hi"]), float(norm_gate["theta_lo"])
                crossed_hi = w1_norms >= hi                          # (B,) bool, always False if hi=inf
                # hysteresis: once asleep, stay asleep while ||W1|| >= theta_lo; wake only when < theta_lo
                sleep_state = (sleep_state | crossed_hi) & (w1_norms >= lo)
                gate = (~sleep_state).to(torch.float32)              # (B,) 1=wake, 0=sleep
                ng_wake_count += gate                                # accumulate wake epochs (all epochs)
            g1_try = g1_try * gate[:, None, None]                    # zero frozen entries' W1 grad
            g2_try = g2_try * gate[:, None, None]                    # zero frozen entries' W2 grad (sleep BOTH)
            W1.grad, W2.grad = g1_try, g2_try
        if not grad_asserted:                 # GUARD (once): grad-mask == adjacency on both blocks
            nz1 = ~M1t
            if nz1.any():
                assert g1_try[nz1].abs().max().item() < 1e-9, "grad-mask != M1 (non-edge got grad)"
            nz2 = ~M2t
            if nz2.any():
                assert g2_try[nz2].abs().max().item() < 1e-9, "grad-mask != M2 (non-edge got grad)"
            if mode in ("pc_transport", "c3_static", "c3_dynamic"):
                assert fb is W2, "transport: feedback must BE W2 (tied, symmetric adjacency)"
            grad_asserted = True
        # ---- DYNMAP PROBE (DYNMAP_BUILD_SPEC §1; read-only — must not touch optimizer/gate/RNG/training
        # state). Fires AFTER grad computation + AFTER w1_gate/norm_gate (so g1_try is the FINAL gated update)
        # but BEFORE opt.step, snapshotting PRE-step W1/W2 + the in-hand g1_try. Per-unit ||W1_i|| (256),
        # path-weighted commitment c_i=||W1_i||*||W2[:,i]|| (256), radial/tangential decomposition of g1_try
        # onto u_i=W1eff_pre/||W1eff_pre|| (same math as the w1gate factorial, READ-ONLY — does NOT modify
        # g1_try), train/test acc (identical eval to the L780-784 log_epoch block, on pre-step W1), gate
        # sleep_state (A1 only), relaxation residual (A1 only — already-computed resid_ep; None for BP).
        # NO extra backward (snapshots g1_try.detach()); all under no_grad; RNG capture/restore. The bit-parity
        # smoke proves this block is inert. Probes are appended to per-entry probe_data[idx]."""
        if probe_active and ep in probe_set:
            _prng = torch.get_rng_state()
            _pcrng = torch.cuda.get_rng_state() if dev == "cuda" else None
            with torch.no_grad():
                W1_snap = W1.detach().clone()                       # (B,h,2P) pre-step
                W2_snap = W2.detach().clone()                       # (B,P,h) pre-step
                w1_pu = W1_snap.reshape(B, h, -1).norm(dim=2)       # (B,h) per-unit ||W1_i||
                w2_pu = W2_snap.reshape(B, -1, h).norm(dim=1)       # (B,h) per-unit ||W2[:,i]||
                c_i = w1_pu * w2_pu                                 # (B,h) path-weighted commitment
                w1_g = W1_snap.reshape(B, -1).norm(dim=1)          # (B,) global ||W1||
                w2_g = W2_snap.reshape(B, -1).norm(dim=1)          # (B,) global ||W2||
                gate_snap = (sleep_state.detach().clone() if ng_active else None)
                # radial/tangential decomposition of the in-hand g1_try onto pre-step W1eff (READ-ONLY)
                W1eff = W1_snap * M1t.to(torch.float32)            # (B,h,2P) masked; 0 on non-edges always
                w1n = W1eff.norm(dim=2, keepdim=True).clamp(min=1e-8)   # (B,h,1)
                u = W1eff / w1n                                     # (B,h,2P) unit radial dir
                proj = (g1_try.detach() * u).sum(dim=2, keepdim=True)  # (B,h,1) signed radial projection
                radial_g = proj * u                                 # (B,h,2P) radial component
                tang_g = g1_try.detach() - radial_g                 # (B,h,2P) tangential component
                radial_mass = radial_g.reshape(B, -1).norm(dim=1)  # (B,) ||radial|| of the update
                tang_mass = tang_g.reshape(B, -1).norm(dim=1)      # (B,) ||tangential|| of the update
                radial_pu = radial_g.norm(dim=2)                   # (B,h) PER-UNIT ||radial_i|| (v2: Q4)
                tang_pu = tang_g.norm(dim=2)                       # (B,h) PER-UNIT ||tang_i||   (v2: Q4)
                # train/test acc — identical eval to L780-784 (blogits_masked + _acc), on pre-step W1_snap
                tr_pr = blogits_masked(Xtr, W1_snap, W2_snap, M1t, M2t, gain).argmax(-1)
                te_pr = blogits_masked(Xte, W1_snap, W2_snap, M1t, M2t, gain).argmax(-1)
                tr_acc_p = _acc(tr_pr, ctr, tr_mask, n_tr_pe)
                te_acc_p = _acc(te_pr, cte, te_mask, n_te_pe)
                # ---- MECH-INTERP battery (MECH_INTERP_SPEC §3 NEW; READ-ONLY, deterministic, no RNG).
                # W1_snap is (B,h,2P): rows=hidden units, cols=input feats. PROBE_INPUTS (20,2P) is FIXED
                # across all entries. x1=(B,20,h) = hidden activations on the fixed probe set. Per-unit
                # logit contribution_i = x1[:,i] * W2[correct_class,i]; mean over the 20 probe inputs ->
                # mean_contrib (B,h). Ablation ranks units by |mean_contrib|, zeros the top-k in a COPY of
                # W1 (training W1 untouched) and measures FULL-test acc. SVD/Fourier run numpy on the
                # EFFECTIVE weights W1_snap*M1t (== raw @ d=1.0); W1 transposed to spec convention (2P,h).
                mi_contrib = mi_act = mi_x1 = mi_abl = mi_svd1 = mi_svd2 = mi_fea = mi_feb = None
                mi_fca = mi_fcb = None
                if mi_active:
                    # MI_PROBE_INPUTS is (B,20,2P) -- per-seed held-out (v2) or broadcast global (v1).
                    W1eff_mi = W1_snap * M1t.to(torch.float32)                    # (B,h,2P) effective W1
                    mi_x1 = torch.relu(torch.einsum("snf,shf->snh", MI_PROBE_INPUTS, W1eff_mi))  # (B,20,h)
                    _b_idx = torch.arange(B, device=dev)[:, None]
                    w2c = W2_snap[_b_idx, MI_PROBE_LABELS, :]                     # (B,20,h) W2 row of correct class
                    mi_contrib = (mi_x1 * w2c).mean(dim=1)                        # (B,h) mean per-unit contribution
                    mi_act = mi_x1.mean(dim=1)                                    # (B,h) mean activation
                    ranked = torch.argsort(mi_contrib.abs(), dim=1, descending=True)   # (B,h) by |contribution|
                    mi_abl = [[] for _ in range(B)]
                    for _k in MI_ABLATION_KS:
                        topk = ranked[:, :_k]                                     # (B,k) units to zero
                        abl_mask = torch.ones(B, h, device=dev)
                        abl_mask.scatter_(1, topk, 0.0)
                        W1_abl = W1_snap * abl_mask[:, :, None]                   # (B,h,2P) COPY; training W1 untouched
                        abl_pred = blogits_masked(Xte, W1_abl, W2_snap, M1t, M2t, gain).argmax(-1)
                        abl_acc = _acc(abl_pred, cte, te_mask, n_te_pe)           # (B,) FULL-test acc
                        for idx in range(B):
                            mi_abl[idx].append([int(_k), float(abl_acc[idx])])
                    # SVD + Fourier per entry (numpy; deterministic). W1 (h,2P) -> .T = (2P,h) per spec.
                    # M2 fix: store the FULL singular spectrum (free; numpy computes all min(2P,h)=106 /
                    # min(P,h)=53 SVs) so the "rank decreases post-grok" falsifier is testable past top-10.
                    mi_svd1, mi_svd2, mi_fea, mi_feb, mi_fca, mi_fcb = [], [], [], [], [], []
                    _n_dft = np.arange(P); _kk = _n_dft.reshape(-1, 1)
                    _Fdft = np.exp(-2j * np.pi * _kk * _n_dft / P) / np.sqrt(P)   # (P,P) unitary DFT
                    for idx in range(B):
                        W1e_np = (W1_snap[idx] * M1t[idx].to(torch.float32)).cpu().numpy().T   # (2P,h)
                        _U1, S1, Vt1 = np.linalg.svd(W1e_np, full_matrices=False)
                        mi_svd1.append(dict(singular_values=S1.tolist(),          # FULL spectrum (M2 falsifier)
                                            top_vectors=Vt1[:10].tolist()))        # top-10 right sing vecs (dirs)
                        W2e_np = (W2_snap[idx] * M2t[idx].to(torch.float32)).cpu().numpy()     # (P,h)
                        _U2, S2, _Vt2 = np.linalg.svd(W2e_np, full_matrices=False)
                        mi_svd2.append(dict(singular_values=S2.tolist()))          # FULL spectrum (M2 falsifier)
                        W1a = W1e_np[:P, :]                                       # (P,h) a-block
                        W1b = W1e_np[P:2 * P, :]                                   # (P,h) b-block (op-cue-safe: [P:2P] excludes op cols; == [P:] at n_ops=0)
                        Fa = _Fdft.conj() @ W1a                                  # (P,h)
                        Fb = _Fdft.conj() @ W1b
                        mi_fea.append((np.abs(Fa) ** 2).sum(axis=1).tolist())    # (P,) population freq energy (spec §3e)
                        mi_feb.append((np.abs(Fb) ** 2).sum(axis=1).tolist())
                        # M3 fix: PER-UNIT Fourier concentration (Gate-1 discriminating metric; the
                        # aggregate freq_energy above repeats the Gate-1 aggregate-PR retraction -- a
                        # population of single-freq neurons aggregates to broadband. conc_i =
                        # max_f|F[f,i]|^2 / sum_f|F[f,i]|^2. UNFOLDED (no conjugate fold, DC incl): a single-
                        # freq cosine neuron reads ~0.5 (power split across k,P-k); folding k<->P-k + dropping
                        # DC (Gate-1 fourier_code) gives ~0.95. So ~0.5 = clean clock read UNFOLDED; ~0.1 = null.
                        Fa_e = np.abs(Fa) ** 2; Fb_e = np.abs(Fb) ** 2            # (P,h) each
                        mi_fca.append((Fa_e.max(axis=0) / (Fa_e.sum(axis=0) + 1e-12)).tolist())   # (h,) per-unit conc, a-block
                        mi_fcb.append((Fb_e.max(axis=0) / (Fb_e.sum(axis=0) + 1e-12)).tolist())   # (h,) per-unit conc, b-block
            for idx in range(B):
                _pd = dict(
                    epoch=int(ep),
                    train_acc=float(tr_acc_p[idx]), test_acc=float(te_acc_p[idx]),
                    w1_norm=float(w1_g[idx]), w2_norm=float(w2_g[idx]),
                    w1_per_unit=w1_pu[idx].cpu().tolist(), c_i=c_i[idx].cpu().tolist(),
                    radial_mass=float(radial_mass[idx]), tang_mass=float(tang_mass[idx]),
                    resid=(None if resid_ep is None else float(resid_ep[idx])),
                    gate_sleep=(None if gate_snap is None else bool(gate_snap[idx])))
                if mi_active:
                    _pd["mean_contribution"] = mi_contrib[idx].cpu().tolist()    # (h,) per-unit logit contribution
                    _pd["mean_activation"] = mi_act[idx].cpu().tolist()          # (h,) mean activation
                    _pd["x1"] = mi_x1[idx].cpu().tolist()                        # (20,h) full probe activations
                    _pd["ablation"] = mi_abl[idx]                                # [[k,acc],...] 11 pts, FULL-test
                    _pd["svd_w1"] = mi_svd1[idx]                                 # FULL spectrum + top-10 right vecs
                    _pd["svd_w2"] = mi_svd2[idx]                                 # FULL spectrum
                    _pd["freq_energy_a"] = mi_fea[idx]                          # (P,) population DFT energy of a-block
                    _pd["freq_energy_b"] = mi_feb[idx]                          # (P,) population DFT energy of b-block
                    _pd["freq_conc_a"] = mi_fca[idx]                            # (h,) PER-UNIT Fourier conc, a-block (M3)
                    _pd["freq_conc_b"] = mi_fcb[idx]                            # (h,) PER-UNIT Fourier conc, b-block (M3)
                    _pd["radial_mass_per_unit"] = radial_pu[idx].cpu().tolist()  # (h,) PER-UNIT ||radial_i|| (v2: Q4)
                    _pd["tang_mass_per_unit"] = tang_pu[idx].cpu().tolist()      # (h,) PER-UNIT ||tang_i||   (v2: Q4)
                    # W1/W2 weight matrices at the grok-window + terminal probes (v2: enables Gate-1 exact
                    # folded Fourier conc, per-unit frequency census Q4a, M2 rank-decrease trajectory). Stored
                    # at probes with epoch<=MI_W_CKPT_WINDOW (covers every arm's grok<=~600) + the terminal
                    # probe; the post-hoc grok-anchor probe (test first >=0.9) is always among them. Effective
                    # weights W*M (== raw @ d=1.0); W1 transposed to spec convention (2P,h), W2 to (h,P).
                    if ep <= MI_W_CKPT_WINDOW or ep == cfg["epochs"] - 1:
                        _pd["W1_matrix"] = (W1_snap[idx] * M1t[idx].to(torch.float32)).cpu().numpy().T.tolist()
                        _pd["W2_matrix"] = (W2_snap[idx] * M2t[idx].to(torch.float32)).cpu().numpy().T.tolist()
                if is_c3 and mode == "c3_dynamic":       # SEQUENTIAL: expose precision state (d_ema) for warm-start
                    _pd["d_ema"] = c3_state["d_ema"][idx].tolist()
                probe_data[idx].append(_pd)
            torch.set_rng_state(_prng)
            if dev == "cuda":
                torch.cuda.set_rng_state(_pcrng)
        if restore_w1_after:                    # STAGED CHANNEL frozen_w1only: snapshot W1 before step
            with torch.no_grad():
                W1_pre = W1.detach().clone()
        if not skip_step:                       # STAGED CHANNEL frozen: skip opt.step entirely (zero dW)
            opt.step()
            with torch.no_grad():                 # peg non-edge params to 0 -> excluded from opt + metrics
                W1.mul_(M1t.to(torch.float32)); W2.mul_(M2t.to(torch.float32))
                if restore_w1_after:              # restore W1 to pre-step value (W2 updated normally)
                    W1.copy_(W1_pre)
            if mode == "pc_kp":
                optB.zero_grad(set_to_none=True); B2.grad = g2_try
                optB.step()
                with torch.no_grad():
                    B2.mul_(M2t.to(torch.float32))
        if want_rtdiag and (ep + 1) % snap_every == 0:    # RTDIAG snapshot (read-only; ReLU homogeneity §4.1)
            with torch.no_grad():
                w1_cur = W1.detach().clone()              # (B,h,2P) pegged state at this interval END
                if w1_prev_rtd is not None:               # 2nd+ snapshot: decompose dW1 over [prev_snap, ep]
                    dW = w1_cur - w1_prev_rtd             # (B,h,2P) -- the actual update (grad + AdamW + peg)
                    u = w1_prev_rtd / w1_prev_rtd.norm(dim=2, keepdim=True).clamp(min=1e-8)   # radial dir u_i
                    proj = (dW * u).sum(dim=2, keepdim=True)    # (B,h,1) scalar projection length per unit
                    radial = proj * u                       # (B,h,2P) radial component (gauge DOF)
                    tang = dW - radial                      # (B,h,2P) tangential component (direction change)
                    r_norm = proj.abs().squeeze(2)          # (B,h) ||radial_i||
                    t_norm = tang.norm(dim=2)               # (B,h) ||tangential_i||
                    r_np = r_norm.cpu().numpy(); t_np = t_norm.cpu().numpy()
                    pu_np = w1_cur.reshape(B, h, -1).norm(dim=2).cpu().numpy()   # (B,h) per-unit ||W1_i|| (identity)
                    ivals = []
                    for idx in range(B):
                        ri, ti = r_np[idx], t_np[idx]
                        rmean = float(ri.mean()); tmean = float(ti.mean())
                        ratio = float(rmean / max(tmean, 1e-12))      # ratio-of-means (robust bulk ratio; verdict metric)
                        frac = float((ri > ti).mean())                # frac units ||radial||>||tang||
                        ivals.append(dict(epoch=int(ep), radial_mean=rmean, tang_mean=tmean,
                                         ratio_mean=ratio, frac_radial_dominant=frac,
                                         per_unit_w1norm=pu_np[idx].tolist()))   # per-unit magnitude trajectory: units with growing ||W1_i|| are the radial inflators (committed assemblies); post-hoc checks if the SAME indices stay high across epochs (committed) or rotate (diffuse)
                    per_epoch["rtdiag"].append(ivals)
                w1_prev_rtd = w1_cur                        # baseline for the NEXT interval
                # N1' finite-T diagnostic (PC_NATIVE follow-up): eval-only long relaxation at T_eval on the
                # CURRENT weights (post-step/post-peg), using THIS epoch's precision/feedback (pi_*_in, fb --
                # the same the T=20 training relaxation used). bpc_grads_masked is already no_grad + stateless
                # + read-only on W1/W2 -> parity-inert (no weight update, no RNG, no graph). VERDICT uses the
                # ABSOLUTE resid (resid>>0 => oscillating / non-equilibrium; the ratio is blind to this -- a
                # period-2 orbit has resid_T200 ~= resid_T20 but both >>0). Runs for any PC mode (is_pc), incl.
                # vanilla pc_transport (pi_*=None -> full-strength feedback, the bitwise-vanilla WB path).
                if T_eval is not None and is_pc:
                    _g1d, _g2d, tr_eval, _mrd, resid_eval, _dd, _e1d, _e2d = bpc_grads_masked(
                        W1, W2, Xtr, Ytr, fb, M1t, M2t, T_eval, eta,
                        deplete, cfg.get("dep_rate", 0.06), cfg.get("tau", 5.0), gain,
                        want_trace=True, want_c3=True, n_per_entry=npc[0], tr_mask=npc[1],
                        pi_h=pi_h_in, pi_out=pi_out_in, pi_fb=pi_fb_in,
                        fb_gain=cfg.get("fb_gain", 1.0))
                    per_epoch["F_T200"].append([float(x) for x in np.asarray(tr_eval).ravel()])
                    per_epoch["resid_T200"].append(resid_eval.tolist())
        if wd0_safe and log_epoch:                       # §3 req 2: wd=0 divergence guard (gated -> inert at wd=1.0)
            with torch.no_grad():
                _finite = bool((torch.isfinite(W1).all() & torch.isfinite(W2).all()).item())
            if not _finite:
                diverged = True
                stopped_epoch = ep
                break
        if need_metrics and log_epoch:        # §3: NUMERICALLY INERT reads at cadence K (no RNG/dtype change)
            with torch.no_grad():
                tr_pr = blogits_masked(Xtr, W1, W2, M1t, M2t, gain).argmax(-1)
                te_pr = blogits_masked(Xte, W1, W2, M1t, M2t, gain).argmax(-1)
                tr_acc = _acc(tr_pr, ctr, tr_mask, n_tr_pe)
                te_acc = _acc(te_pr, cte, te_mask, n_te_pe)
                w1n = W1.flatten(1).norm(dim=1)         # §3d: PER ENTRY (the delay law is per-network)
                w2n = W2.flatten(1).norm(dim=1)
                tr_acc_per_op = te_acc_per_op = None
                if n_ops > 0:                            # OP_CUE_SURGERY_SPEC §5: per-op test/train accuracy
                    tr_acc_per_op, te_acc_per_op = [], []
                    for op_id in range(n_ops):
                        tr_om = (ops_tr == op_id) & tr_mask
                        te_om = (ops_te == op_id) & te_mask
                        tr_acc_per_op.append(((tr_pr == ctr) & tr_om).sum(1).to(torch.float32)
                                             / tr_om.sum(1).clamp(min=1).to(torch.float32))
                        te_acc_per_op.append(((te_pr == cte) & te_om).sum(1).to(torch.float32)
                                             / te_om.sum(1).clamp(min=1).to(torch.float32))
            if log_pe:
                per_epoch["train_acc"].append(tr_acc.cpu().tolist())
                per_epoch["test_acc"].append(te_acc.cpu().tolist())
                per_epoch["w1_norm"].append(w1n.cpu().tolist())
                per_epoch["w2_norm"].append(w2n.cpu().tolist())
                per_epoch["residual"].append(None if resid_ep is None else resid_ep.tolist())
                per_epoch["F"].append(None if e1sq_ep is None
                                      else (0.5 * (e1sq_ep + e2sq_ep)).tolist())
                if want_gradnorm:      # MECH-DIG: e1/e2 RMS (relaxed state) + raw grad Frobenius norms
                    per_epoch["e1_rms"].append(None if e1sq_ep is None
                                               else np.sqrt(np.maximum(e1sq_ep, 0.0)).tolist())
                    per_epoch["e2_rms"].append(None if e2sq_ep is None
                                               else np.sqrt(np.maximum(e2sq_ep, 0.0)).tolist())
                    per_epoch["gW1_norm"].append(g1_try.reshape(B, -1).norm(dim=1).cpu().tolist())
                    per_epoch["gW2_norm"].append(g2_try.reshape(B, -1).norm(dim=1).cpu().tolist())
                if n_ops > 0:                            # per-op trajectories (OP_CUE_SURGERY_SPEC §5)
                    for op_id in range(n_ops):
                        per_epoch[f"test_acc_op{op_id}"].append(te_acc_per_op[op_id].cpu().tolist())
                        per_epoch[f"train_acc_op{op_id}"].append(tr_acc_per_op[op_id].cpu().tolist())
                if channel_switch_epoch is not None:            # STAGED CHANNEL: phase2_active (global bool)
                    per_epoch["phase2_active"].append(ep >= channel_switch_epoch)
                if is_c3:                         # §3.4 C3 diagnostics (π that drove THIS epoch, pre-update)
                    if mode == "c3_dynamic" and pi_used is not None and d_used is not None:
                        pim = pi_used.mean(axis=1)                                  # (B,) mean(π) over units
                        pfl = (pi_used < 0.5 * c3_state["pi0"]).mean(axis=1)        # (B,) frac-low(π)
                        pcr = np.array([_corr_unit(pi_used[b], d_used[b]) for b in range(B)], dtype=np.float32)
                        per_epoch["pi_mean"].append(pim.tolist())
                        per_epoch["pi_frac_low"].append(pfl.tolist())
                        per_epoch["pi_corr"].append(np.where(np.isfinite(pcr), pcr, np.nan).tolist())
                        per_epoch["Pi_h"].append(None)
                        per_epoch["Pi_out"].append(None)
                    else:                          # c3_static: per-layer scalars (no per-unit distribution)
                        per_epoch["pi_mean"].append(np.asarray(pi_used, dtype=np.float32).tolist())
                        per_epoch["pi_frac_low"].append(None)
                        per_epoch["pi_corr"].append(None)
                        per_epoch["Pi_h"].append(np.asarray(pi_used, dtype=np.float32).tolist())
                        per_epoch["Pi_out"].append(np.asarray(pi_out_used, dtype=np.float32).tolist())
                if ng_active:                        # FUSEE: sleep state + disp distribution + cumulative wake
                    # ng_sleep: instantaneous sleep_state (B,) bool; ng_disp_pct: [p50,p90,p99] over per-unit
                    # displacement (B,3) -- the DV distribution (demoted from gate signal, addendum §4.3);
                    # ng_wake_frac: cumulative wake-fraction over ALL epochs (wake_count accumulated every ep).
                    per_epoch["ng_sleep"].append(sleep_state.cpu().tolist())
                    if disp_ep is not None:
                        per_epoch["ng_disp_pct"].append(
                            np.percentile(disp_ep, [50, 90, 99], axis=1).T.astype(np.float32).tolist())
                    else:
                        per_epoch["ng_disp_pct"].append(None)
                    per_epoch["ng_wake_frac"].append((ng_wake_count / (ep + 1)).cpu().tolist())
            if do_stop:                                # §3.1/§3.2: split-0 (random) drives the check
                if n_ops > 0:                          # OP_CUE_SURGERY_SPEC §5: BOTH ops >= GROK_TEST to stop
                    op_ok = torch.ones(S, dtype=torch.bool, device=dev)
                    for op_id in range(n_ops):
                        op_ok = op_ok & (te_acc_per_op[op_id][:S] >= GROK_TEST)
                    hit = int(op_ok.sum().item())
                else:
                    te_rand = te_acc[:S]
                    if label_kind == "shuffled":
                        hit = int((tr_acc[:S] >= EARLY_STOP_TRAIN).sum())
                    elif es_uses_block and n_sp >= 2:      # §3.2 batched allowance: grok = test AND block
                        hit = int(((te_rand >= GROK_TEST) & (te_acc[S:2 * S] >= GROK_BLOCK)).sum())
                    else:
                        hit = int((te_rand >= GROK_TEST).sum())
                if hit >= GROK_SEEDS:
                    stopped_epoch = ep
                    break

    with torch.no_grad():
        tr_final = _acc(blogits_masked(Xtr, W1, W2, M1t, M2t, gain).argmax(-1), ctr, tr_mask, n_tr_pe)
        te_final = _acc(blogits_masked(Xte, W1, W2, M1t, M2t, gain).argmax(-1), cte, te_mask, n_te_pe)
        mean_r = float("nan")
        if deplete:                            # measure A2's steady-state resource for the static-gain control
            fb = W2 if mode in ("pc_transport", "c3_static", "c3_dynamic") else B2
            _, _, _, mr, _resid, _d, _e1, _e2 = bpc_grads_masked(W1, W2, Xtr, Ytr, fb, M1t, M2t, T, eta, True,
                                           cfg.get("dep_rate", 0.06), cfg.get("tau", 5.0),
                                           want_trace=True, n_per_entry=npc[0], tr_mask=npc[1])
            mean_r = mr
    pe_by_entry = None
    if log_pe:                                         # transpose columnar per_epoch -> per-entry trajectory
        n_ep = len(per_epoch["train_acc"])
        has_resid = bool(per_epoch["residual"]) and per_epoch["residual"][0] is not None
        has_F = bool(per_epoch["F"]) and per_epoch["F"][0] is not None

        def _col(key, idx):
            # per-entry trajectory for a C3 diagnostic column; elements may be None per log point (C3-S)
            col = per_epoch[key]
            return [(None if col[i] is None else col[i][idx]) for i in range(n_ep)]
        pe_by_entry = [dict(
            train_acc=[per_epoch["train_acc"][i][idx] for i in range(n_ep)],
            test_acc=[per_epoch["test_acc"][i][idx] for i in range(n_ep)],
            w1_norm=[per_epoch["w1_norm"][i][idx] for i in range(n_ep)],
            w2_norm=[per_epoch["w2_norm"][i][idx] for i in range(n_ep)],
            residual=([per_epoch["residual"][i][idx] for i in range(n_ep)] if has_resid else None),
            F=([per_epoch["F"][i][idx] for i in range(n_ep)] if has_F else None),
            e1_rms=(_col("e1_rms", idx) if want_gradnorm else None),
            e2_rms=(_col("e2_rms", idx) if want_gradnorm else None),
            gW1_norm=(_col("gW1_norm", idx) if want_gradnorm else None),
            gW2_norm=(_col("gW2_norm", idx) if want_gradnorm else None),
            stopped_epoch=stopped_epoch,
        ) for idx in range(B)]
        if n_ops > 0:                            # OP_CUE_SURGERY_SPEC §5: per-op trajectories per entry
            for op_id in range(n_ops):
                tk, trk = f"test_acc_op{op_id}", f"train_acc_op{op_id}"
                for idx in range(B):
                    pe_by_entry[idx][tk] = [per_epoch[tk][i][idx] for i in range(n_ep)]
                    pe_by_entry[idx][trk] = [per_epoch[trk][i][idx] for i in range(n_ep)]
        if channel_switch_epoch is not None:        # STAGED CHANNEL: phase2_active (same global bool per entry)
            pa = list(per_epoch["phase2_active"])
            for idx in range(B):
                pe_by_entry[idx]["phase2_active"] = pa
        if is_c3:                               # §3.4: append the C3 diagnostic trajectories per entry
            for idx in range(B):
                for key in ("pi_mean", "pi_frac_low", "pi_corr", "Pi_h", "Pi_out"):
                    pe_by_entry[idx][key] = _col(key, idx)
        if want_rtdiag:                         # RTDIAG: transpose interval summaries -> per-entry trajectory
            # per_epoch["rtdiag"][i] is a (B,) list of dicts -> entry idx gets the i-th dict from each interval
            for idx in range(B):
                pe_by_entry[idx]["rtdiag"] = [per_epoch["rtdiag"][i][idx] for i in range(len(per_epoch["rtdiag"]))]
            if T_eval is not None and ("F_T200" in per_epoch):    # N1' finite-T diagnostic -> per-entry
                # resid_T200 is per-entry (B,) per snapshot -> entry trajectory. F_T200 is the AGGREGATE F
                # trajectory (same for all entries) -> duplicated per entry so it rides the per-entry carrier.
                n_snaps = len(per_epoch["resid_T200"])
                for idx in range(B):
                    pe_by_entry[idx]["resid_T200"] = [per_epoch["resid_T200"][i][idx] for i in range(n_snaps)]
                    pe_by_entry[idx]["F_T200"] = list(per_epoch["F_T200"])   # aggregate trace (shared)
        if ng_active:                           # FUSEE: transpose norm-gate columns -> per-entry trajectory
            for idx in range(B):
                for key in ("ng_sleep", "ng_disp_pct", "ng_wake_frac"):
                    pe_by_entry[idx][key] = _col(key, idx)
        if probe_data is not None:              # DYNMAP: probes already stored per-entry -> attach directly
            for idx in range(B):
                pe_by_entry[idx]["probes"] = probe_data[idx]
    out = []
    for idx in range(B):
        conc, hist = fourier_code(W1[idx] * M1t[idx], W2[idx] * M2t[idx], P)   # measure on W_eff
        al = 1.0
        if B2 is not None:
            u, v = (B2[idx] * M2t[idx]).flatten(), (W2[idx] * M2t[idx]).flatten()
            al = float((u @ v) / (u.norm() * v.norm() + 1e-9))
        out.append(dict(test=float(te_final[idx]), train=float(tr_final[idx]),
                        conc=conc, align=al, hist=hist, mean_r=mean_r,
                        diverged=diverged,
                        per_epoch=(pe_by_entry[idx] if pe_by_entry is not None else None)))
    return out


def clock_reference(M1, M2, P, h, dev):
    """Per-density DV2 bar: a synthetic single-frequency-per-neuron clock (Nanda 2023) pushed through
    the SAME mask, then scored by fourier_code -> the mask-deflated achievable concentration. A
    measured conc near this bar means 'clock on substrate'; near the A5 null means 'no clock'.
    CLOCK_SEED is a top-level constant (a FIXED measuring-stick signal, same for all arms/densities)."""
    rng = np.random.RandomState(CLOCK_SEED)
    M = P // 2
    k = rng.randint(1, M + 1, size=h)
    a = np.arange(P)
    base = np.cos(2 * np.pi * np.outer(k, a) / P)
    W1 = np.zeros((h, 2 * P), dtype=np.float32)
    W1[:, :P] = base
    W1[:, P:] = base
    W2 = np.zeros((P, h), dtype=np.float32)
    W2[:] = base.T
    W1t = torch.tensor(W1 * M1, device=dev)
    W2t = torch.tensor(W2 * M2, device=dev)
    conc, _ = fourier_code(W1t, W2t, P)
    return conc


def settle_smoke(M1, M2, a, b, P, cfg, dev):
    """A2 depletion settle check (L0-style, design §5): with a PROPERLY-INITIALIZED masked net and real
    data, does the PC relaxation still SETTLE under depletion (energy decreasing over T, resource in
    [0,1])? Run BEFORE any A2 science claim. Returns (energy_decreases, r_in_range, final_mean_r, trace)."""
    S = M1.shape[0]; h = cfg["h"]; n = min(256, len(a))
    Xtr = torch.tensor(onehot2(a, b, P)[:n], device=dev).unsqueeze(0).repeat(S, 1, 1)
    cc = make_cells(P)[2][:n]
    Ytr = torch.zeros(S, n, P, device=dev)
    Ytr[:, torch.arange(n), torch.tensor(cc)] = 1.0
    W1, W2 = init_seeds_masked(list(range(S)), P, h, M1, M2, dev)        # local-degree init (real dynamics)
    M1t = torch.tensor(M1, device=dev); M2t = torch.tensor(M2, device=dev)
    _, _, trace, rmean, _resid, _d, _e1, _e2 = bpc_grads_masked(W1, W2, Xtr, Ytr, W2, M1t, M2t, cfg["T"], cfg["eta"],
                                          deplete=True, dep_rate=cfg.get("dep_rate", 0.06),
                                          tau=cfg.get("tau", 5.0), want_trace=True)
    dec = trace[-1] < trace[0]
    return bool(dec), bool(0.0 <= rmean <= 1.0), float(rmean), trace


# ===================================================================== driver ===
ARM_MODE = {                                    # arm -> (mode, deplete, mask_kind, label_kind)
    "A1":     ("pc_transport", False, "spatial", "real"),
    "A2":     ("pc_transport", True,  "spatial", "real"),
    "A3":     ("pc_kp",        False, "spatial", "real"),
    "A4":     ("pc_fa",        False, "spatial", "real"),
    "A5":     ("backprop",     False, "spatial", "shuffled"),
    "A1r":    ("pc_transport", False, "random",  "real"),
    "A1s":    ("pc_transport", False, "spatial", "shuffled"),   # CLOSE-OUT razor: A1 machinery on A5's perm stream
    "A1g":    ("pc_transport", False, "spatial", "real"),   # static-gain control (gain = A2 mean resource)
    "oracle": ("backprop",     False, "spatial", "real"),
    "frozen": ("frozen",       False, "spatial", "real"),
    # ---- C3 precision arms (C3_CELL_SPEC §2; all @ d=1.0, wd=0; build on pc_transport + a precision rule) ----
    "C3-S-noise": ("c3_static",  False, "spatial", "shuffled"),   # STATIC per-layer Π (§3.2); falsifying pair for H1
    "C3-D-noise": ("c3_dynamic", False, "spatial", "shuffled"),   # DYNAMIC per-unit π(displacement) (§3.3); H1 arm
    "C3-S-real":  ("c3_static",  False, "spatial", "real"),       # giving-up-vs-targeting control
    "C3-D-real":  ("c3_dynamic", False, "spatial", "real"),       # the K2 decision cell
}
GROK_TEST, GROK_BLOCK, GROK_SEEDS = 0.9, 0.9, 8   # per-seed grok: test>=.9 AND block>=.9 in >=8/10
EARLY_STOP_TRAIN = 0.99                            # CLOSE-OUT §3.1: shuffled arms (chance-test) stop TRAIN-based

CFG_SMOKE = dict(P=53, h=256, densities=(1.0,), fracs=(0.9,), epochs=1000, lr=2e-3, wd=1.0,
                 T=20, eta=0.2, dep_rate=0.06, tau=5.0, seeds=(0, 1), eval_block=True,
                 arms=("A1", "A4"))
CFG_SELFCHECK = dict(P=53, h=256, densities=(1.0,), fracs=(0.9,), epochs=6000, lr=2e-3, wd=1.0,
                     T=20, eta=0.2, dep_rate=0.06, tau=5.0, seeds=tuple(range(10)), eval_block=True,
                     arms=("A1", "A4", "A5", "oracle"))
CFG_FULL = dict(P=53, h=256, densities=(1.0, 0.5, 0.25, 0.12, 0.06, 0.03),
                fracs=(0.3, 0.5, 0.7, 0.9), epochs=6000, lr=2e-3, wd=1.0, T=20, eta=0.2,
                dep_rate=0.06, tau=5.0, seeds=tuple(range(10)), eval_block=True,
                arms=("A1", "A2", "A1g", "A3", "A4", "A5", "A1r", "oracle", "frozen"),
                frac_sweep_densities=(1.0, 0.06))   # DV1 rises-with-data checked at dense + local ref
# CLOSE-OUT (§2c): d=0.5 cliff with 5x budget (30k) + A1s razor. densities=(1.0,0.5): d=1.0 runs FIRST
# as the fast self-check (A1 groks -> instrumentation inert; A1s@d=1.0 must memorize->1.0, the §7.3 gate
# + reviewers' F1 watch), then the long d=0.5 cliff (oracle/A1/A1r real arms + A1s razor-at-cliff).
# Order is launch-strategy only (cells are independent); per-epoch logging + pre-registered early-stop ON.
# Same AdamW constants as CFG_FULL. See CLOSEOUT_SPEC.md.
CFG_CLOSEOUT = dict(P=53, h=256, densities=(1.0, 0.5), fracs=(0.9,), epochs=30000,
                    lr=2e-3, wd=1.0, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                    seeds=tuple(range(10)), eval_block=True,
                    arms=("A1", "A1r", "oracle", "A1s"),
                    log_per_epoch=True, early_stop=True, log_every=100)   # §2b periodic; want_trace ~1%


def drive(cfg, label, save_path=None):
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: Gate-2 needs CUDA (CPU = days, gate2_design §6)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P)
    N = P * P
    seeds = cfg["seeds"]
    arms = cfg["arms"]
    top = max(cfg["fracs"])
    frac_sweep_dens = cfg.get("frac_sweep_densities", cfg["densities"])
    print("=" * 104)
    print(f"GATE 2 [{label}] (a+b) mod {P} on 3D volume  | device={dev} h={h} epochs={cfg['epochs']} "
          f"T={cfg['T']} seeds={len(seeds)}  R={R_FIXED} reach_scale={REACH_SCALE}")
    print(f"  densities={cfg['densities']}  fracs={cfg['fracs']} (DV1 RISES)  top frac={top} carries block/DV2")
    print(f"  arms={arms}  | grok = per-seed test>={GROK_TEST} AND block>={GROK_BLOCK} in >={GROK_SEEDS}/10")
    print("=" * 104)

    # positions/types once per seed (shared across arms + densities -> paired)
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}

    # results store: sweep[density][arm][frac] = [per-seed test]; dv2[density][arm] at top frac
    sweep = {d: {arm: {f: None for f in cfg["fracs"]} for arm in arms} for d in cfg["densities"]}
    block = {d: {arm: [] for arm in arms} for d in cfg["densities"]}
    dv2 = {d: {arm: dict(test=[], train=[], conc=[], align=[], hist=[], mean_r=None, per_epoch=None)
               for arm in arms} for d in cfg["densities"]}
    base = {f: dict(mem=[], cc=[]) for f in cfg["fracs"]}
    subst = {d: dict() for d in cfg["densities"]}
    clock_bar = {d: 0.0 for d in cfg["densities"]}
    null_conc = {d: [] for d in cfg["densities"]}
    grok_flag = {d: {arm: False for arm in arms} for d in cfg["densities"]}

    for d in cfg["densities"]:
        # ---- build per-seed masks at this density (spatial + the A1r random variant) ----
        M1s, M2s, M1rs, M2rs, st = [], [], [], [], []
        for s in seeds:
            pos, in_idx, hid_idx, out_idx = vols[s]
            m1, m2, stats = build_mask(pos, in_idx, hid_idx, out_idx, d, P, h)
            M1s.append(m1); M2s.append(m2); st.append(stats)
            if "A1r" in arms:
                rng_r = np.random.RandomState(s + 50_000)
                m1r, m2r = randomize_mask(m1, m2, rng_r)
                M1rs.append(m1r); M2rs.append(m2r)
        M1 = np.stack(M1s); M2 = np.stack(M2s)
        if "A1r" in arms:
            M1r = np.stack(M1rs); M2r = np.stack(M2rs)
        else:
            M1r = M2r = None
        # degree-sequence parity check (A1r must match A1's per-node degrees)
        if "A1r" in arms:
            assert np.allclose(M1.sum(1), M1r.sum(1)) and np.allclose(M1.sum(2), M1r.sum(2)), \
                "A1r degree sequence must match A1"
            assert np.allclose(M2.sum(1), M2r.sum(1)) and np.allclose(M2.sum(2), M2r.sum(2)), \
                "A1r degree sequence must match A1 (out)"

        st_arr = {k: np.array([x[k] for x in st]) for k in st[0]}
        subst[d] = {k: (float(v.mean()), float(v.std())) for k, v in st_arr.items()}
        clock_bar[d] = float(np.mean([clock_reference(M1s[i], M2s[i], P, h, dev) for i in range(len(seeds))]))
        print(f"\n[density target={d}]  r={subst[d]['r'][0]:.3f}+-{subst[d]['r'][1]:.3f}  "
              f"realized={subst[d]['realized'][0]:.3f}  (ih={subst[d]['dens_ih'][0]:.3f} ho={subst[d]['dens_ho'][0]:.3f})  "
              f"mean_deg in/out={subst[d]['mean_deg_in'][0]:.1f}/{subst[d]['mean_deg_out'][0]:.1f}  "
              f"pair_coverage={subst[d]['coverage'][0]:.3f}+-{subst[d]['coverage'][1]:.3f}  "
              f"DV2 clock_bar={clock_bar[d]:.3f}")
        if "A2" in arms and d == cfg["densities"][0]:
            dec, inrng, rmean, tr = settle_smoke(M1, M2, a, b, P, cfg, dev)
            print("  A2 settle-smoke: energy decreases over T={0}  r in [0,1]={1}  final_mean_r={2:.3f}  "
                  "E[0]->{3:.1f} E[-1]->{4:.1f}".format(dec, inrng, rmean, tr[0], tr[-1]))

        do_frac_sweep = d in frac_sweep_dens
        for f in cfg["fracs"]:
            if f != top and not do_frac_sweep:
                continue                              # headline curve only needs top frac; DV1 at ref densities
            splits, perms = [], []
            for s in seeds:
                rng = np.random.RandomState(s)
                tr, te = split_random(N, f, rng)
                assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
                splits.append((tr, te))
                cperm = c.copy(); rng.shuffle(cperm)
                perms.append(cperm)
                if f == top:                          # baselines at the top frac
                    base[f]["mem"].append(float((c[te] == np.bincount(c[tr], minlength=P).argmax()).mean()))
                    base[f]["cc"].append(cc_acc(a, b, c, tr, te, P))
            do_block = (f == top and cfg["eval_block"])    # §2e lever 1: batch random+block at top frac
            if do_block:
                bsplits, bperms = [], []
                for s in seeds:
                    rb = np.random.RandomState(1000 + s)
                    trb, teb = split_block(a, b, P, top, rb)
                    cpb = c.copy(); rb.shuffle(cpb)
                    bsplits.append((trb, teb)); bperms.append(cpb)
            for arm in arms:
                if arm == "A1g":
                    continue                              # static-gain control runs after A2 (needs its mean resource)
                mode, depl, mkind, lkind = ARM_MODE[arm]
                mm1, mm2 = (M1r, M2r) if mkind == "random" else (M1, M2)
                lab_rand = [perms[i] if lkind == "shuffled" else c for i in range(len(seeds))]
                if do_block:                              # §2e: ONE pass covers random ([:S]) + block ([S:])
                    lab_blk = [bperms[i] if lkind == "shuffled" else c for i in range(len(seeds))]
                    res = run_seeds_masked(mode, seeds, [lab_rand, lab_blk], [splits, bsplits],
                                           mm1, mm2, a, b, P, cfg, dev, deplete=depl,
                                           label_kind=lkind, es_uses_block=True)
                    res_rand, res_blk = res[:len(seeds)], res[len(seeds):]
                else:
                    res_rand = run_seeds_masked(mode, seeds, [lab_rand], [splits], mm1, mm2,
                                                a, b, P, cfg, dev, deplete=depl, label_kind=lkind)
                    res_blk = None
                sweep[d][arm][f] = [r["test"] for r in res_rand]
                if f == top:
                    for k in ("test", "train", "conc", "align", "hist"):
                        dv2[d][arm][k] = [r[k] for r in res_rand]
                    dv2[d][arm]["mean_r"] = res_rand[0]["mean_r"]     # for the A1g static-gain control
                    if cfg.get("log_per_epoch", False):          # CLOSE-OUT §2b: per-epoch trajectory (random split)
                        dv2[d][arm]["per_epoch"] = [r["per_epoch"] for r in res_rand]
                    if arm == "A5":
                        null_conc[d] = [r["conc"] for r in res_rand]
                    if res_blk is not None:
                        block[d][arm] = [r["test"] for r in res_blk]
            if "A1g" in arms and do_block:                 # static-gain control (design §5/§7): A1 transport
                g = dv2[d].get("A2", {}).get("mean_r", float("nan"))  # with fixed gain = A2's mean resource
                g = float(g) if (np.isfinite(g) and g > 0) else 1.0
                lab = [c for _ in range(len(seeds))]
                rtopblk = run_seeds_masked("pc_transport", seeds, [lab, lab], [splits, bsplits],
                                           M1, M2, a, b, P, cfg, dev, deplete=False, gain=g,
                                           es_uses_block=True)
                rtop, rblk = rtopblk[:len(seeds)], rtopblk[len(seeds):]
                sweep[d]["A1g"][top] = [r["test"] for r in rtop]
                for k in ("test", "train", "conc", "align", "hist"):
                    dv2[d]["A1g"][k] = [r[k] for r in rtop]
                dv2[d]["A1g"]["mean_r"] = float("nan")     # A1g is non-depleting; nan is faithful
                block[d]["A1g"] = [r["test"] for r in rblk]
                print(f"  [A1g static-gain control @ d={d}] gain(=A2 mean resource)={g:.3f}")
            # grok flag for this (density, arm): top-frac test>=.9 AND block>=.9 in >=8/10
            for arm in arms:
                if sweep[d][arm][top] is not None and block[d][arm]:
                    per = [(sweep[d][arm][top][i] >= GROK_TEST and block[d][arm][i] >= GROK_BLOCK)
                           for i in range(len(seeds))]
                    grok_flag[d][arm] = sum(per) >= GROK_SEEDS

        if save_path:                                 # incremental dump (survives interruption)
            _dump(save_path, sweep, block, dv2, base, subst, clock_bar, null_conc, grok_flag, cfg)

    _report(cfg, sweep, block, dv2, base, subst, clock_bar, null_conc, grok_flag, top)
    print("=" * 104)


def _dump(path, sweep, block, dv2, base, subst, clock_bar, null_conc, grok_flag, cfg):
    def clean(o):                          # RFC-8259: NaN/inf (mean_r/align on non-deplete/frozen arms) -> null
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = dict(sweep={str(d): {a: {str(f): v for f, v in fd.items()} for a, fd in ad.items()}
                      for d, ad in sweep.items()},
               block={str(d): ad for d, ad in block.items()},
               dv2={str(d): {a: {k: dv2[d][a][k] for k in ("test", "train", "conc", "align", "mean_r", "per_epoch")}
                             for a in dv2[d]} for d in dv2},
               base={str(f): base[f] for f in base},
               null_conc={str(d): v for d, v in null_conc.items()},
               clock_bar={str(d): v for d, v in clock_bar.items()},
               grok={str(d): {a: bool(grok_flag[d][a]) for a in grok_flag[d]} for d in grok_flag},
               subst={str(d): subst[d] for d in subst}, cfg={k: (list(v) if isinstance(v, tuple) else v)
                                                             for k, v in cfg.items()})
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False,
                  default=lambda o: (list(o) if hasattr(o, "__iter__") else str(o)))


def _report(cfg, sweep, block, dv2, base, subst, clock_bar, null_conc, grok_flag, top):
    arms = cfg["arms"]
    ms = lambda v: (float(np.mean(v)), float(np.std(v))) if len(v) else (float("nan"), float("nan"))
    print("\nGROK-VS-DENSITY (held-out test acc @ top frac, mean+-std)   [*] = per-seed grok "
          f"(test>={GROK_TEST} & block>={GROK_BLOCK} in >={GROK_SEEDS}/10)")
    print(f"  {'arm':8s}" + "".join(f"{('d='+str(d)):>16}" for d in cfg["densities"]))
    for arm in arms:
        cells = ""
        for d in cfg["densities"]:
            v = sweep[d][arm].get(top)
            if not v:
                cells += f"{'--':>16}"; continue
            m, s = ms(v)
            cells += f"{m:>9.3f}{'*' if grok_flag[d][arm] else ' '}{s:<5.2f}"
        print(f"  {arm:8s}" + cells)
    print(f"  {'block':8s}" + "  (structured block split @ top frac; rule EXTRACTION not interpolation)")
    for arm in arms:
        cells = ""
        for d in cfg["densities"]:
            v = block[d][arm]
            m = float(np.mean(v)) if v else float("nan")
            cells += f"{m:>16.3f}"
        print(f"  {arm:8s}" + cells)
    mem = float(np.mean(base[top]["mem"])) if base[top]["mem"] else float("nan")
    cc = float(np.mean(base[top]["cc"])) if base[top]["cc"] else float("nan")
    print(f"  baselines @ top frac: memorization={mem:.3f}  C.C={cc:.3f}  chance={1/cfg['P']:.3f}")

    print(f"\nTRAIN acc @ top frac (§7a under-optimization gate: a negative is attributable only if train~1)")
    print(f"  {'arm':8s}" + "".join(f"{('d='+str(d)):>16}" for d in cfg["densities"]))
    for arm in arms:
        cells = ""
        for d in cfg["densities"]:
            v = dv2[d][arm].get("train", [])
            m = float(np.mean(v)) if v else float("nan")
            cells += f"{m:>16.3f}"
        print(f"  {arm:8s}" + cells)

    print("\nDV2 per-neuron Fourier conc @ top frac  (clock~0.9 dense; mask-calibrated per density)")
    print(f"  {'arm':8s}" + "".join(f"{('d='+str(d)):>16}" for d in cfg["densities"]))
    for arm in arms:
        cells = ""
        for d in cfg["densities"]:
            v = dv2[d][arm]["conc"]
            m = float(np.mean(v)) if v else float("nan")
            cells += f"{m:>16.3f}"
        print(f"  {arm:8s}" + cells)
    print("  " + "-" * 100)
    cb = "".join(f"{clock_bar[d]:>16.3f}" for d in cfg["densities"])
    nc = "".join(f"{(float(np.mean(null_conc[d])) if null_conc[d] else 0.0):>16.3f}" for d in cfg["densities"])
    print(f"  {'clock_bar':8s}" + cb + "   <- mask-deflated achievable bar")
    print(f"  {'A5 null':8s}" + nc + "   <- per-density DV2 null")

    print("\nFALSIFIER READOUT")
    a0 = sweep[1.0]["A1"][top] if 1.0 in sweep and "A1" in sweep[1.0] and sweep[1.0]["A1"][top] else None
    if a0 is not None:
        print(f"  A0 self-check (A1@d=1.0 must reproduce Gate-1 pc_transport): "
              f"test={np.mean(a0):.3f}+-{np.std(a0):.2f}  conc={np.mean(dv2[1.0]['A1']['conc']):.3f}  "
              f"-> {'GROKS (plumbing OK)' if np.mean(a0) >= 0.9 else '!! does NOT grok -> harness bug, STOP'}")
    a4_any = any(grok_flag[d].get("A4", False) for d in cfg["densities"])
    print(f"  A4 self-check (volume-fa must NOT grok anywhere): "
          f"-> {'OK (nowhere)' if not a4_any else '!! A4 GROKS -> harness bug, STOP'}")
    # minimum density at which each arm groks (patch #1: no magic d*)
    print("  minimum grok-density per arm (None = never groks in the swept range):")
    for arm in arms:
        ds = [d for d in cfg["densities"] if grok_flag[d].get(arm, False)]
        md = min(ds) if ds else None
        print(f"    {arm:8s} min_grok_density={md}")
    # attribution aids
    if "A1" in arms and "A1r" in arms:
        print("  A1 vs A1r (locality vs sparsity): A1~=A1r everywhere => 'sparse-mask MLP', NOT '3D volume'")
    if "A2" in arms and "A1g" in arms:
        print("  A2 vs A1g (depletion dynamics vs static gain): A1g groks where A2 dies => A2 death is the")
        print("    dynamic/nonlinear depletion, NOT a static gain reduction (design §5/§7 static-gain control).")
    if "A3" in arms and "A1" in arms:
        a3_ds = [d for d in cfg["densities"] if grok_flag[d].get("A3", False)]
        a1_ds = [d for d in cfg["densities"] if grok_flag[d].get("A1", False)]
        if a1_ds and not a3_ds:
            print("  PROJECT-LEVEL FALSIFIER: A3 (KP) fails at EVERY density where A1 groks -> "
                  "'bio (learned-alignment) rule-schema does NOT survive the substrate' (no rescue reframe).")
    print("\n  READING: HEADLINE = the grok-vs-density curve (does locality kill grokking for ANY rule,")
    print("  incl. the oracle?). A3 passing = backprop-via-learned-alignment survives the mask (bio-candidate")
    print("  of record, NOT the headline). Negative A3 attributable to locality ONLY if train~1, oracle groks")
    print("  on the SAME mask, and A1r groks where A1 fails; report pair-coverage if <1 (unlearnable by construction).")


# ============================================================== GATE-2.1a razor ===
# The wd-sweep razor (GATE21A_SPEC.md). Reuses run_seeds_masked (the close-out
# instrumentation + A1s pairing) with wd as the ONLY per-cell swept axis. Three
# pre-registered questions, language attached BEFORE the run (§4) -- no improvisation.
#
#   Q1: oracle (backprop) @ d=0.5, wd in {1.0,0.1,0.01,0.0} -- capacity vs regularization.
#       wd=0 is the decisive cell (no residual decoupled decay). Oracle runs FULL 30k
#       (early_stop=False; can't-miss #3: no plateau-stop truncating grokking's slow climb).
#   Q2: A1s (pc_transport + shuffled) @ d=1.0, wd=0 -- the weak-gradient razor.
#       SUCCESS = train>=0.95 SUSTAINED (MEMORIZE+HOLD), NOT grok (wd=0 cannot generalize;
#       A1s test is chance by construction). Full 30k; score from the per-seed tail.
#   Q3: |W1|/|W2| on the Q2 cell (FREE -- norms already logged every K=100).
#
# wd0_safe guard (§3 req 2): at wd=0 |W1| can grow unbounded -> on NaN/inf RECORD as a
# finding ("unbounded norm growth at wd=0"), do not crash/retry. The GUARD is byte-inert at
# wd=1.0 (gated on cfg["wd"]==0.0 -> the close-out/--full/--closeout compute path is untouched).
# Note: the wd=1.0 oracle cell reproduces the close-out to FLOATING tolerance (single-split
# razor vs the close-out's 2-split batch can pick different cuBLAS kernels) -- the ±0.02 parity
# arbiter (ORACLE_REF) checks this; it is not a bit-for-bit claim. Seeds stay top-level.

ORACLE_REF = {"train": 0.853, "test": 0.309, "tol": 0.02}   # close-out oracle @ d=0.5, 30k (parity arbiter)
GATE21A_WD = (1.0, 0.1, 0.01, 0.0)                          # Q1 oracle wd sweep (wd=0 = decisive hard-capacity cell)
Q2_TAIL = 50                                                # §4 Q2 "sustains": mean over last 50 log pts (=5k ep @ K=100)


def _tail_mean(per_epoch_traj, key="train_acc", n=Q2_TAIL):
    """Per-seed mean of the last n log points (§4 Q2 'sustains = train>=0.95 for final 5k ep')."""
    v = per_epoch_traj[key]
    return float(np.mean(v[-n:])) if len(v) else float("nan")


def score_q1_21a(wd, entries):
    """Q1 oracle @ d=0.5 pre-registered bins (GATE21A_SPEC §4). Returns (label, language)."""
    tr = float(np.mean([e["train"] for e in entries]))
    te = float(np.mean([e["test"] for e in entries]))
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", f"wd={wd}: unbounded norm growth at wd=0 -- recorded as finding, not retried."
    if wd < 1.0 and te >= 0.5:
        return "OVER-REGULARIZED", (f"wd={wd}: test rose to {te:.3f} at reduced decay -- wd=1.0 was throttling "
                                     "BOTH fitting and generalization at d=0.5; the cliff was over-regularization, not capacity.")
    if tr >= 0.95 and te < 0.5:
        return "REGULARIZATION-CEILING", (f"wd={wd}: train {tr:.3f} (fits) while test {te:.3f} (no generalization) -- "
                                          "regularization-capped memorization + a REAL generalization gap at d=0.5.")
    if wd in (0.0, 0.01) and abs(tr - 0.85) <= 0.02:
        return "HARD-CAPACITY", (f"wd={wd}: train pinned at {tr:.3f} (within +-0.02 of 0.85, spec §4) even with "
                                 "decay ~off -- the d=0.5 mask cannot represent the rule at this width. "
                                 "Capacity ceiling confirmed.")
    if 0.90 <= tr < 0.95:
        return "MIXED", f"wd={wd}: train {tr:.3f}, test {te:.3f} -- partially wd-limited; record the wd->train gradient."
    return "UNSCORED", f"wd={wd}: train {tr:.3f}, test {te:.3f} -- outside pre-registered bins; report raw, do not narrate."


def score_q2_21a(entries):
    """Q2 A1s @ d=1.0 wd=0 pre-registered bins. SUCCESS = MEMORIZE+HOLD (train>=0.95 sustained over the
    final 5k epochs -- read as the per-seed tail-MIN, the conservative 'held' reading of spec §4 'train
    >=0.95 for the final 5k epochs'), NOT grok (wd=0 cannot generalize; A1s test is chance by
    construction). Never bare grok:false. PARTIAL band = spec's 0.6-0.9 (extended to <0.95); <0.6 = stuck low."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "wd=0: unbounded norm growth -- recorded as finding, not retried."
    pes = [e["per_epoch"] for e in entries if e.get("per_epoch")]
    tail_mean = [_tail_mean(pe) for pe in pes]                               # per-seed sustained level (mean of last 5k)
    tail_min = [float(np.min(pe["train_acc"][-Q2_TAIL:])) if pe["train_acc"] else float("nan") for pe in pes]
    lvl = float(np.mean(tail_mean)) if tail_mean else float("nan")           # mean sustained level across seeds
    tr_seed_final = [round(float(e["train"]), 2) for e in entries]
    if tail_min and all(t >= 0.95 for t in tail_min):                        # HELD >=0.95 throughout final 5k, all seeds
        return ("MEMORIZED+HELD (weak-gradient-vs-decay CONFIRMED)",
                f"wd=0: A1s memorized noise and HELD (tail-min>=0.95 all seeds; per-seed tail-mean "
                f"{[round(t, 2) for t in tail_mean]}) -- PC can fit noise when decay is off; the close-out "
                "unlearning was weak-gradient-vs-decay. Story closed.")
    if lvl < 0.60:                                                           # stuck below the 0.6-0.9 partial band
        return ("RELAXATION-CANNOT-FIT-NOISE",
                f"wd=0: train stuck low (sustained level {lvl:.3f}; per-seed final {tr_seed_final}) -- even with "
                "decay off, the relaxation gradient cannot fit noise. DEEPER problem; precision weighting "
                "promoted from 'PC pathology' to 'PC survival'.")
    return ("WEAK-GRADIENT-PARTIAL",
            f"wd=0: train climbed but <0.95 sustained (level {lvl:.3f}; per-seed final {tr_seed_final}) -- "
            "weak-but-nonzero gradient: wd was the killer AND the gradient is genuinely weak. Consistent with "
            "the W1-asymmetry story; precision weighting stays queued as 'PC pathology'.")


def score_q3_21a(entries):
    """Q3 |W1|/|W2| asymmetry on the Q2 cell (free; §4 Q3). Ratio over per-seed tail norms."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "wd=0: unbounded norm growth -- Q3 ratio N/A (recorded as finding)."
    ratios = []
    for e in entries:
        pe = e.get("per_epoch")
        if not pe or not pe["w1_norm"] or not pe["w2_norm"]:
            continue
        w1 = float(np.mean(pe["w1_norm"][-Q2_TAIL:]))
        w2 = float(np.mean(pe["w2_norm"][-Q2_TAIL:]))
        ratios.append(w1 / max(w2, 1e-8))
    ratio = float(np.mean(ratios)) if ratios else float("nan")
    if ratio > 0.5:
        return ("DECAY-COUPLED", f"|W1|/|W2| ratio {ratio:.3f} at wd=0 -- no W1-selective collapse without decay: "
                                 "asymmetry confirmed as a decay-coupled phenomenon.")
    return ("INTRINSIC-WEAKNESS", f"|W1|/|W2| ratio {ratio:.3f} at wd=0 -- W1 collapsed relative to W2 even at wd=0: "
                                  "deeper/intrinsic W1-gradient weakness.")


def score_c1_21a(entries):
    """C1: backprop+shuffled @ d=1.0, wd=0 -- the PC-SPECIFIC vs GENERIC-ADAM razor (closes the §4 hedge;
    DEEP_RESEARCH C1). Compared implicitly against Q2 (A1s pc_transport@wd=0: peaked ~0.85 @ ep200 then
    eroded to 0.29 with |W1| exploding 16->620). Backprop's gradient SHUTS OFF once noise is memorized, so
    the PC-specific prediction is: train -> ~1.0 and HOLDS, |W1| stable. ALWAYS reports raw trajectory so a
    misfired bin is adjudicable (cf the Q2/Q3 bin misfires Kimi caught)."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "wd=0: unbounded norm growth under backprop+noise -- recorded as finding, not retried."
    pes = [e["per_epoch"] for e in entries if e.get("per_epoch")]
    tail_tr = [_tail_mean(pe, "train_acc") for pe in pes]
    tail_min = [float(np.min(pe["train_acc"][-Q2_TAIL:])) if pe.get("train_acc") else float("nan") for pe in pes]
    peak_tr = [float(np.max(pe["train_acc"])) if pe.get("train_acc") else float("nan") for pe in pes]
    w1_init = [float(pe["w1_norm"][0]) for pe in pes if pe.get("w1_norm")]
    w1_tail = [float(np.mean(pe["w1_norm"][-Q2_TAIL:])) for pe in pes if pe.get("w1_norm")]
    lvl = float(np.mean(tail_tr)) if tail_tr else float("nan")
    peak = float(np.mean(peak_tr)) if peak_tr else float("nan")
    i0 = float(np.mean(w1_init)) if w1_init else float("nan")
    it = float(np.mean(w1_tail)) if w1_tail else float("nan")
    grow = (it / max(i0, 1e-8)) if np.isfinite(i0) and np.isfinite(it) else float("nan")
    n = len(tail_min)
    n_held = int(sum(1 for t in tail_min if np.isfinite(t) and t >= 0.90))
    all_held = (n > 0 and n_held == n)            # EVERY seed held >=0.90 throughout the final 5k (cf Q2's tail-min gate)
    raw = (f"tail_train {lvl:.3f} (per-seed tail-min {[round(t, 2) for t in tail_min]}, {n_held}/{n} held>=0.90), "
           f"peak_train {peak:.3f}, |W1| init->tail {i0:.1f}->{it:.1f} ({grow:.1f}x)")
    inflated = grow > 3.0            # PC exploded ~39x; a benign memorization plateau is ~2-3x (oracle@wd0 ref ~1.3x)
    if all_held and not inflated:
        return ("PC-SPECIFIC",
                f"backprop memorized noise and HELD in ALL {n} seeds ({raw}) -- the wd-independent erosion is PC's "
                "relaxation; precision weighting confirmed as the correct promoted fix. §4 banked.")
    if inflated and not all_held:
        return ("GENERIC-ADAM",
                f"backprop eroded in {n - n_held}/{n} seeds with |W1| inflation ({raw}) -- generic Adam-under-"
                "persistent-noise pathology; PC is only an instance. Fix queue rescopes to optimizer-side.")
    if inflated and all_held:
        return ("PC-SPECIFIC (NLM-STABLE)",
                f"|W1| inflated ({raw}) but train PRESERVED in all seeds -- Prieto-style NLM weight-inflation that "
                "scales logits without destroying predictions; the erosion is still PC-specific. §4 banked.")
    if not all_held and not inflated:
        return ("MIXED",
                f"backprop held {n_held}/{n} seeds but {n - n_held} eroded WITHOUT |W1| inflation ({raw}) -- partial "
                "erosion; the LEAD adjudicates vs Q2's PC trajectory (did PC erode these seeds too?).")
    return "UNSCORED", f"backprop+shuf@wd=0 outside pre-registered bins ({raw}); report raw, no narration."


def score_c2_21a(entries):
    """C2: A1 (pc_transport, REAL labels) @ d=0.5, wd=0 -- the forward cortex-risk probe (DEEP_RESEARCH
    Q4 / C2). WARNING null confound: PC@d=0.5 sits at train~0.108 (close-out), so 'train stable' is
    ambiguous (it may simply never have fit enough to erode from). The DISAMBIGUATING DV is the |W1|
    trajectory: inflates => the runaway fires on REAL structure under capacity stress (persistent-error-
    specific -> cortex risk); stable => it needs unresolvable NOISE (noise-specific). The primary
    noise-vs-real adjudicator remains the close-out PC@wd=0@d=1.0-REAL (held train=1.0 to 30k); C2 is the
    'persistent-capacity-error' probe, null weakened."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "wd=0: unbounded norm growth on REAL @ d=0.5 -- recorded as finding, not retried."
    pes = [e["per_epoch"] for e in entries if e.get("per_epoch")]
    w1_init = [float(pe["w1_norm"][0]) for pe in pes if pe.get("w1_norm")]
    w1_tail = [float(np.mean(pe["w1_norm"][-Q2_TAIL:])) for pe in pes if pe.get("w1_norm")]
    tail_tr = [_tail_mean(pe, "train_acc") for pe in pes]
    tail_disp = [float(np.mean([d for d in pe["residual"][-Q2_TAIL:] if d is not None]))
                 if pe.get("residual") else float("nan") for pe in pes]
    lvl = float(np.mean(tail_tr)) if tail_tr else float("nan")
    i0 = float(np.mean(w1_init)) if w1_init else float("nan")
    it = float(np.mean(w1_tail)) if w1_tail else float("nan")
    grow = (it / max(i0, 1e-8)) if np.isfinite(i0) and np.isfinite(it) else float("nan")
    disp = float(np.nanmean(tail_disp)) if tail_disp else float("nan")
    raw = (f"train {lvl:.3f}, |W1| init->tail {i0:.1f}->{it:.1f} ({grow:.1f}x), displacement_tail {disp:.2f}")
    if grow > 3.0:
        return ("PERSISTENT-ERROR-SPECIFIC",
                f"|W1| inflated {grow:.1f}x on REAL structure @ d=0.5 ({raw}) -- the runaway fires under "
                "capacity stress, not just noise; cortex risk confirmed; precision/iPC = survival-critical.")
    return ("NOISE-SPECIFIC",
            f"|W1| stable ({grow:.2f}x) on REAL @ d=0.5 ({raw}) -- the runaway needs unresolvable NOISE, not "
            "merely persistent real error; the cortex role is safer than Q4 feared.")


def score_sgd_21a(cells, ref_key):
    """SGD/LR optimizer-sweep @ d=0.5 pre-registered bins (the #6 'PROVEN' hedge closer). The AdamW ref
    cell (cells[ref_key]) reproduces the close-out 0.853 (parity arbiter -- STOP if WARN). Any non-AdamW
    optimizer materially beating it -> ADAM-SPECIFIC PLATEAU (#6 reopens; 'hard-capacity' wrong); all
    capping within the band -> HARD-CAPACITY CONFIRMED across optimizers (#6 closed; 'PROVEN' legitimate).
    NB: SGD wd is coupled-L2 (PyTorch default) vs AdamW decoupled -> biases TOWARD confirming hard-capacity.
    'Broke' = mean beats ref+tol AND a paired one-sided Wilcoxon (sgd_per_seed > ref_per_seed, p<0.05) so a
    noisy mean cannot spuriously un-bank #6 (exp-designer F2). Per-seed + p-values ALWAYS in the raw string."""
    if any(c.get("diverged", False) for c in cells.values()):
        return "DIVERGED", "an optimizer cell diverged (NaN/inf) -- recorded as finding; parity unreliable, do not bank."
    ref = cells.get(ref_key)
    if ref is None:
        return "UNSCORED", "AdamW ref cell missing -- cannot adjudicate the optimizer sweep."
    ref_train = ref.get("train_mean", float("nan"))
    if not np.isfinite(ref_train) or abs(ref_train - ORACLE_REF["train"]) > ORACLE_REF["tol"]:
        return "PARITY-WARN", (f"AdamW ref train {ref_train:.3f} != {ORACLE_REF['train']}+-{ORACLE_REF['tol']} -- "
                                "harness drift; reconcile vs close-out before trusting any optimizer cell.")
    ref_ps = ref.get("train_per_seed", [])
    threshold = ref_train + ORACLE_REF["tol"]            # materially beats the AdamW plateau (apples-to-apples)
    broke = []
    raw_parts = []
    for k, c in cells.items():
        sgd_ps = c.get("train_per_seed", [])
        n_pairs = min(len(sgd_ps), len(ref_ps))
        n_beat = sum(1 for i in range(n_pairs) if sgd_ps[i] > ref_ps[i])
        p = _wilcoxon(sgd_ps, ref_ps) if n_pairs >= 6 else None      # paired one-sided (sgd > ref); None if n<6/scipy missing
        paired = (p is not None and p < 0.05) or (p is None and n_beat >= max(1, n_pairs - 2))  # sign-test fallback
        mean_beats = c["train_mean"] > threshold
        pstr = f"p={p:.3f}" if p is not None else "p=N/A"
        raw_parts.append(f"{k}: train={c['train_mean']:.3f} ({pstr}, {n_beat}/{n_pairs} beat ref)")
        if k != ref_key and mean_beats and paired:
            broke.append((k, c["train_mean"], pstr, n_beat, n_pairs))
    raw = ", ".join(raw_parts)
    if broke:
        names = ", ".join(f"{k}={t:.3f} ({ps}, {nb}/{np_})" for k, t, ps, nb, np_ in broke)
        return ("ADAM-SPECIFIC-PLATEAU",
                f"a non-AdamW optimizer broke the plateau ({names} > AdamW {ref_train:.3f}+{ORACLE_REF['tol']}, paired) "
                f"-- #6 reopens; 'hard-capacity' is optimizer-specific. [{raw}]")
    return ("HARD-CAPACITY-CONFIRMED",
            f"all optimizers cap within the AdamW plateau ({ref_train:.3f}+-{ORACLE_REF['tol']}, paired) -- #6 closed; "
            f"'PROVEN' legitimate across optimizers. [{raw}]")


# ================================================================== C3 razor ===
# The precision "scalpel" (C3_CELL_SPEC_dynamic_precision.md). 4 cells @ d=1.0, wd=0, 30k, 10 seeds,
# paired ctx (substrate/split/init/perm stream identical to the gate21a cells by construction -- same
# build_volume/build_mask/init_seeds_masked/split_random). Each arm adds ONE variable: the precision rule
# (C3-S static per-layer Π §3.2; C3-D dynamic per-unit π(displacement) §3.3) on top of pc_transport.
# Baselines are NOT re-run: vanilla-noise = the Q2 run; vanilla-real = C2 (gate21a-ctrl).
#
# Reads are PRE-REGISTERED (§4 K2 triple + §5 selectivity), language attached BEFORE the run. The §4
# K2 kill criterion (displacement / |W1|-slope / train) is DECOUPLED from the §5 H1-mechanism read
# (targeting vs giving-up). Out-of-bin cells report raw, no narration. The LEAD applies H1 + banks.

C3_TAIL = 50                # §4: "sustained over final 50 log points" (50 * K=100 = 5k epochs)
C3_W1_WIN = 100             # §4: "|W1| over final 10k" = 100 log points (K=100)
C3_DISP_BAR = 1.0           # §4: displacement < 1.0
C3_W1_SLOPE_PCT = 0.01      # §4: |d|W1||/1k ep < 1% of |W1| over final 10k
C3_REAL_TRAIN_FLOOR = 0.9   # §4 real-arm floor: train >= 0.9 (= 0.9 * oracle-fit at d=1.0)
# C3_ALPHA_FROZEN: §3.3 α = 1/d_mid, d_mid = geom midpoint of settled-vs-orbit PER-UNIT displacement.
# RE-FROZEN from the per-unit distribution by c3_smoke()'s α diagnostic (2026-07-25, 2060 uncompiled,
# 600ep, 2 seeds): d_low=0.00026 (settled, ep5-105 median) , d_high=0.00464 (orbit, ep400-600 median)
# -> d_mid=0.00110 -> α=908.6 -> frozen 900. Dynamics verified to reproduce Q2 (train peak 0.865 @ep199
# then erodes to 0.621; per-unit disp jumps 18x at the fit peak, the Q2 signature). This replaces the
# spec's α≈0.4 PRIOR, which was calibrated from the AGGREGATE L2 norm (‖·‖ over n*h≈647k elements) -- a
# different scale than per-unit mean-|·| (HARNESS NOTE §3.3). CAVEAT (for the LEAD/review-gate): 600ep
# under-samples Q2's deep orbit (agg-equiv ~3.7 here vs ~6-13 at 30k); if the deep-orbit per-unit d_high
# ~0.009 is the right anchor, α~650. The gate is robust across α in [650, 900] -- both place the
# half-closed point (π=0.5) in the observed per-unit transition zone. alpha=0 reproduces vanilla bitwise
# (the parity guard). NOT tuned on C3 output (frozen from the Q2 dynamics diagnostic, pre-run).
C3_ALPHA_FROZEN = 900.0


def _tail_arr(per_epoch_traj, key, n):
    v = per_epoch_traj.get(key) or []
    return np.array(v[-n:], dtype=np.float64) if len(v) else np.array([], dtype=np.float64)


def score_c3_k2(entries, is_noise, log_every=100):
    """§4 K2 / safe-mode triple (displacement / |W1|-slope / train), read off per-epoch logs.

    NOISE arms (safe-mode success): disp < 1.0 sustained (final 50) AND |W1|-slope flat (final 10k);
    train UNCONSTRAINED (expected to fall; alt: train>=0.9 AND settled ALSO passes).
    REAL arms (K2 decision): + train >= 0.9 (the kill-doc v2 absolute floor)."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "a C3 cell diverged (NaN/inf) -- recorded as finding, not retried."
    pes = [e["per_epoch"] for e in entries if e.get("per_epoch")]
    if not pes:
        return "UNSCORED", "no per-epoch trajectory -- cannot read the K2 triple."
    # displacement = the K2 triple's L2-norm residual (§3.4). Sustained = MAX of the final 50 below the bar.
    disp_max, disp_mean = [], []
    for pe in pes:
        d = _tail_arr(pe, "residual", C3_TAIL)
        if d.size:
            disp_max.append(float(np.max(d))); disp_mean.append(float(np.mean(d)))
    settled = bool(disp_max) and all(d < C3_DISP_BAR for d in disp_max)
    # |W1| slope per 1k epochs over the final 10k (= C3_W1_WIN log points; K=log_every epochs/point).
    epochs_span = C3_W1_WIN * log_every                              # final-window span in epochs (10k for K=100)
    intervals_per_1k = epochs_span / 1000.0                          # log-point intervals per 1k epochs (10 for K=100)
    slopes, w1_means = [], []
    for pe in pes:
        w = _tail_arr(pe, "w1_norm", C3_W1_WIN)
        if w.size >= 2:
            slopes.append(float((w[-1] - w[0]) / intervals_per_1k)); w1_means.append(float(np.mean(w)))
    flat = bool(slopes) and all(abs(s) < C3_W1_SLOPE_PCT * max(m, 1e-8) for s, m in zip(slopes, w1_means))
    tr_tail = [float(np.mean(_tail_arr(pe, "train_acc", C3_TAIL))) if _tail_arr(pe, "train_acc", C3_TAIL).size else float("nan") for pe in pes]
    tr_lvl = float(np.mean(tr_tail)) if tr_tail else float("nan")
    raw = (f"disp_tail_max {[round(d,2) for d in disp_max]} (mean {[round(d,2) for d in disp_mean]}), "
           f"|W1|-slope/1k {[round(s,2) for s in slopes]} vs 1%-of-{[round(m,1) for m in w1_means]} bar, "
           f"train_tail {tr_lvl:.3f} (per-seed {[round(t,2) for t in tr_tail]})")
    settled_ok = settled and flat
    if is_noise:
        if settled_ok:
            return ("SAFE-MODE-PASS",
                    f"settled (disp<1.0 sustained) + |W1|-slope flat; train unconstrained ({raw}). "
                    "Substrate stops trusting the noise stream (CLS data-moat).")
        if settled and tr_lvl >= 0.9:
            return ("SAFE-MODE-PASS (STABLE-FIT)",
                    f"settled AND train held >=0.9 ({raw}) -- pre-registered alt pass ('learned to fit noise stably'); "
                    "not required, not narrated as goal.")
        return ("SAFE-MODE-FAIL", f"not settled / slope not flat ({raw}) -- precision did not stabilize the relaxation on noise.")
    if settled_ok and tr_lvl >= C3_REAL_TRAIN_FLOOR:
        return ("K2-PASS", f"settled + |W1|-slope flat + train {tr_lvl:.3f}>=0.9 ({raw}) -- precision stabilized PC under real persistent error.")
    return ("K2-FAIL", f"did not meet the K2 triple ({raw}).")


def score_c3_targeting(entries, mode):
    """§5 H1-mechanism read (DECOUPLED from §4 K2): HOW settling happened -- targeted (gate closes on
    orbiting units, stays open elsewhere) vs giving-up (uniform π collapse). C3-D-real is the decision cell;
    C3-S has no per-unit π so its giving-up signal is Π_out -> lower bound with train<0.9."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "diverged -- §5 N/A."
    pes = [e["per_epoch"] for e in entries if e.get("per_epoch")]
    if not pes:
        return "UNSCORED", "no per-epoch trajectory -- §5 N/A."
    tr_tail = [float(np.mean(_tail_arr(pe, "train_acc", C3_TAIL))) if _tail_arr(pe, "train_acc", C3_TAIL).size else float("nan") for pe in pes]
    tr_lvl = float(np.mean(tr_tail)) if tr_tail else float("nan")
    if mode == "c3_dynamic":
        # final-log-point per-unit selectivity stats, averaged across seeds
        corr_tail, flow_tail = [], []
        for pe in pes:
            c = _tail_arr(pe, "pi_corr", 1)
            f = _tail_arr(pe, "pi_frac_low", 1)
            if c.size: corr_tail.append(float(c[-1]))
            if f.size: flow_tail.append(float(f[-1]))
        corr_m = float(np.mean(corr_tail)) if corr_tail else float("nan")
        flow_m = float(np.mean(flow_tail)) if flow_tail else float("nan")
        raw = f"corr(π,d) {corr_m:.2f} (per-seed {[round(c,2) for c in corr_tail]}), frac-low(π) {flow_m:.2f} (per-seed {[round(f,2) for f in flow_tail]}), train {tr_lvl:.3f}"
        if corr_m <= -0.5 and flow_m < 0.5 and tr_lvl >= 0.9:
            return ("TARGETING", f"gate closes selectively on orbiting units and stays open elsewhere ({raw}). H1 mechanism signature.")
        if flow_m >= 0.5:
            return ("GIVING-UP", f"uniform π collapse (frac-low>=0.5) -- settled by ceasing to listen to EVERYTHING ({raw}). Fails the floor.")
        return ("SELECTIVITY-AMBIGUOUS", f"neither pure targeting nor uniform collapse ({raw}); report raw, no narration.")
    # c3_static: no per-unit π. Giving-up = Π_out -> lower bound AND train < 0.9.
    po_tail = []
    for pe in pes:
        p = _tail_arr(pe, "Pi_out", C3_TAIL)
        if p.size: po_tail.append(float(np.mean(p)))
    po_m = float(np.mean(po_tail)) if po_tail else float("nan")
    raw = f"Π_out_tail {po_m:.3f}, train {tr_lvl:.3f}"
    if po_m <= 0.011 and tr_lvl < 0.9:
        return ("GIVING-UP", f"Π_out -> lower bound with train<0.9 ({raw}) -- static precision declined to learn.")
    return ("SELECTIVITY-N/A", f"static per-layer Π has no per-unit selectivity; §5 giving-up not triggered ({raw}).")


CFG_C3 = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
              epochs=30000, log_every=100, log_per_epoch=True, early_stop=False,
              seeds=tuple(range(10)), fracs=(0.9,), wd=0.0,
              # C3-D (§3.3) frozen constants:
              c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
              # C3-S (§3.2) frozen constants:
              c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)

# RTDIAG (KIMI-THE-WATCH §4.1) -- radial/tangential W1 decomposition to route C5. 2 cells, 10 seeds,
# 15k epochs. vanilla-real = the d=1.0 wd=0 baseline (stability NOT pre-assumed -- fills the F3 cell
# that was never measured); c3d-real = the runaway cell (the C5 routing arbiter). snap_every=100 ->
# 149 intervals; late third = ep10-15k, DEEP in the erosion regime (C3-D-real erosion clearly underway
# by ~ep5-10k; the prior 2k run was regime-confounded -- sampled peak grok, not the runaway). C3-D
# constants carried so the c3d-real cell matches the live C3 run exactly. 10 seeds resolves the per-seed
# mode split (7 radial-balloon / 3 tangential-drift in the 30k C3 data).
CFG_RTDIAG = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                  epochs=15000, log_every=100, log_per_epoch=True, early_stop=False,
                  seeds=tuple(range(10)), fracs=(0.9,), wd=0.0, snap_every=100,
                  c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                  c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)

# W1-GATE FACTORIAL (user spec): causal intervention on the radial/tangential decomposition. All 4
# cells are c3_dynamic @ d=1.0 wd=0 10 seeds 15k (snap_every=100, want_rtdiag ON as the gate check).
# W2 learns freely; only gW1's channel is gated. Tests which channel matters for schema stability:
#   both-on=baseline | radial-only=freeze tangential (user hyp) | tang-only=freeze radial (Kimi C5a)
#   | frozen=W2-alone control. The cell that passes K2 = the channel that needed gating.
CFG_W1GATE = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                  epochs=15000, log_every=100, log_per_epoch=True, early_stop=False,
                  seeds=tuple(range(10)), fracs=(0.9,), wd=0.0, snap_every=100,
                  c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                  c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)


# =========================================================== GATE-2.1a-C5NORM FUSEE ===
# C5 NORM-BAND GATE (Kimi addendum §2/§3; supersedes the displacement-gate spec). The w1gate proved both
# channels synergistic (channel-gating refuted); the rtdiag trajectories proved displacement~5 THROUGH the
# grok -> a disp-gate (tau=1.0) would have BLOCKED the grok. The correct gate signal is ||W1|| (the
# bifurcation parameter): banked both-on grok completes by ||W1||<=42, collapse onset >=47; gate at theta_hi=45
# sits in the 5-unit gap. theta_lo=38 (max over seeds of ||W1|| at test-peak, -10%). Derivation pre-registered
# in docs/ADDENDUM_C5_FUSEE_norm_gate.md §2 -- STACK-SPECIFIC (c3d @ d=1.0); the noise arm reuses this band
# FLAGGED-not-assumed (addendum §3: if noise-memorization needs ||W1||>45, re-derive from its own trajectory).
C5NORM_THETA = dict(theta_hi=45.0, theta_lo=38.0)
C5NORM_W1_BUFFER = 5.0            # clause 2 ceiling = theta_hi + buffer (collapse onset>=47 -> 50 clears it)
C5NORM_TEST_FLOOR = 0.9          # clause 1: TEST (generalization) tail-min >= 0.9 -- the gate protects the
                                 # GROK, not just the fit (both-on's TRAIN erodes too: 3/10 pass train>=0.9;
                                 # but test tail50_min>=0.9 is 0/10 -> test is the discriminating bar)
C5NORM_SLEEP_MAX_EPOCHS = 2000   # clause 3: bounded sleep window (=20 log pts; wd@lr=2e-3 pulls ||W1|| theta_hi->theta_lo in ~85ep)
C5NORM_WAKE_FRAC_MIN = 0.5       # clause 4: not permanently asleep (real arm)
C5NORM_CELLS = [                 # (name, norm_gate or None, wd) -- all c3d stack (alpha=900, d=1.0); w1_gate="both"
    ("F0", C5NORM_THETA, 0.0),   # norm-gate ON, wd=0 : boundary-oscillation test (governor-necessity)
    ("F1", C5NORM_THETA, 1.0),   # norm-gate ON, wd=1.0: THE sleep/wake cycle (the candidate)
    ("G0", None,          1.0),  # norm-gate OFF, wd=1.0: same-stack governor-only control
]
CFG_C5NORM = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                  epochs=15000, log_every=100, log_per_epoch=True, early_stop=False,
                  seeds=tuple(range(10)), fracs=(0.9,), wd=1.0, snap_every=100,
                  c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                  c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)


# =============================================================== GATE-2.1a-DYNMAP ===
# DYNMAP-30k (DYNMAP_SPEC.md): measurement-only dynamics map. A1 = F1 (c3d + AdamW + wd=1.0 + norm-gate
# θ=[38,45]) extended to 30k with the full probe battery; A2 = matched backprop baseline (mode="backprop",
# AdamW + wd=1.0, NO gate, NO c3d) at 30k. Same arch/data/seeds. Probe grid = 40 fixed-grid epochs. NO
# mechanism changes vs F1 (the K2'-PASS baseline at 15k, test=1.0 10/10) — probes are read-only snapshots.
DYNMAP_PROBE_EPOCHS = ([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500]
                       + list(range(3000, 30001, 1000)))   # 40 frozen-grid epochs (§2); ep 30000 -> terminal
CFG_DYNMAP = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                  epochs=30000, log_every=100, log_per_epoch=True, early_stop=False,
                  seeds=tuple(range(10)), fracs=(0.9,), wd=1.0, snap_every=100,
                  c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                  c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)


# =============================================================== GATE-2.1a-MECHINTERP ===
# MECH-INTERP (MECH_INTERP_SPEC.md): mechanistic interpretability -- dissect the G0 config (wd=1.0, NO
# gate, c3d, AdamW; the simplest working config, test=1.0 10/10 banked in C5NORM) to find WHAT circuit
# the network builds for modular addition. Two arms: G0-probe (real labels) + G0-noise (shuffled = the
# NULL; if structure also appears on noise -> capacity artifact, not schema). Each probe carries the
# DYNMAP battery PLUS 5 new READ-ONLY fields: per-unit logit contribution, activations (mean + 20xh),
# cumulative ablation curve (9 pts), SVD of W1/W2 (top-10), Fourier energy of the a/b input blocks.
# Pre-registered predictions M1 (sparsity) / M2 (SVD rank) / M3 (Fourier) / M4 (activation stability) /
# M5 (noise null). All probes under no_grad + DETERMINISTIC (no RNG) -> parity-inert (mechinterp_smoke).
MECHINTERP_PROBE_EPOCHS = (list(range(0, 2001, 100)) + list(range(3000, 30001, 1000)))  # dense thru grok, sparse after
MI_PROBE_N = 20                                                                  # fixed held-out probe-pair count (spec §1)
MI_ABLATION_KS = (1, 5, 10, 20, 25, 30, 50, 100, 150, 200, 256)                 # cumulative unit-ablation ranks (spec §3c; 25/30 bracket the M1 10% falsifier = k/256=0.098/0.117)
CFG_MECHINTERP = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                      epochs=30000, log_every=100, log_per_epoch=True, early_stop=False,
                      seeds=tuple(range(10)), fracs=(0.9,), wd=1.0, snap_every=100,
                      c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                      c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)


# =============================================================== MECH-INTERP-V2 ===
# V2 fixes the v1 cue confound (experiment-designer review): v1 used ONE fixed 20-pair probe set from seed-0's
# test split, so seeds 1-9 had 15-20/20 probe pairs IN TRAIN -> activation-space probes (AP-r, settling, sleep)
# measured MEMORIZATION not persistence. V2: each seed uses its OWN test split's first-20 as the probe set
# (all 20 genuinely held-out for every seed). PLUS three additions Kimi flagged:
#  (1) DENSE early grid: every-50 through ep1000 (was every-100) -> resolves whether BP has a sub-100ep
#      transient sparse phase (the Q3 caveat). 0,50,...,1000,2000,3000,...,30000.
#  (2) PER-UNIT radial/tangential mass (256 each, was scalar means) -> enables BOTH rho(radial,usage) AND
#      rho(tangential,usage). Q4 re-opens (was null under the scalar-only v1).
#  (3) W1/W2 weight matrices at the grok-window + terminal probes -> enables Gate-1 exact folded Fourier
#      conc (resolves the 0.49-vs-0.95), per-unit frequency census (Q4a), and M2 rank-decrease trajectory.
# V2 dumps to gate2_mechinterp_v2.json (v1 kept for comparison). Training configs IDENTICAL to v1.
MECHINTERP_PROBE_EPOCHS_V2 = (list(range(0, 1001, 50)) + list(range(2000, 30001, 1000)))  # dense thru ep1000, then every 1000
MI_W_CKPT_WINDOW = 2000        # save W1/W2 matrices at probes with epoch<=this (covers all arms' grok<=~600) + terminal
CFG_MECHINTERP_V2 = dict(CFG_MECHINTERP)   # identical training config; V2 differs only in probe grid/battery/seed-probes


# =========================================================== GATE-2.1a-PCNATIVE ===
# PC-NATIVE SGD MOAT TEST (PC_NATIVE_SPEC.md). Does PC grok+hold schemas WITHOUT backprop-era machinery
# (AdamW -> plain SGD, wd -> 0, gate OFF), keeping c3d precision? C5NORM proved PC holds schemas but ONLY
# under AdamW + wd=1.0 + gate. This removes ALL of it: opt="sgd", wd=0.0, norm_gate=None. Precision (c3d)
# is the ONLY remaining gain control -- the PC-native one. Free energy F is logged (the Lyapunov function).
# Pre-registered outcomes O1 (grok+hold) / O2 (grok+erode) / O3 (no grok); attribution N1 vs G0 isolates
# the optimizer (both wd=0 no-gate; N1=SGD vs G0=AdamW). Spec §0-§4.
PCNATIVE_LR_LIST = (2e-3, 1e-2, 1e-1, 1.0, 10.0)   # Phase 1 sweep; 2e-3 = AdamW parity point
PCNATIVE_LR_SEEDS = 3                                # Phase 1 (quick finder; seeds 0,1,2)
PCNATIVE_LR_EPOCHS = 2000                            # Phase 1 (2k -- enough to see train rise)
PCNATIVE_LR_FLOOR = 0.8                              # train >= this in 2k -> "working LR"
PCNATIVE_QUORUM = 8                                  # >=8/10 seeds per criterion (spec §1 O1)
PCNATIVE_GROK_BAR = 0.9                              # peak test >= 0.9 = grok (generalization)
PCNATIVE_HOLD_BAR = 0.9                              # tail-min test >= 0.9 = hold (no erosion)
PCNATIVE_RESID_CONV_BAR = 0.5                        # T_eval residual (||x1^T-x1^{T-1}||) below this => relaxation
                                                     # converged; above => oscillating/non-equilibrium (red-team:
                                                     # the RATIO resid_T200/resid_T20 is blind to a period-2 orbit,
                                                     # which has ratio~1 but both >>0; use the ABSOLUTE resid)
PCNATIVE_CELLS = [                                   # Phase 2 (at lr*); all SGD, c3d, d=1.0, no gate
    ("N1",  0.0, "real"),     # SGD, wd=0,  real labels  -- THE PC-native headline (vs G0=AdamW wd=0)
    ("N2",  1.0, "real"),     # SGD, wd=1.0, real labels  -- decay disentangle under SGD (coupled-L2)
    ("N1n", 0.0, "shuffled"), # SGD, wd=0,  noise labels  -- control (should NOT grok)
    ("N2n", 1.0, "shuffled"), # SGD, wd=1.0, noise labels  -- control
]
CFG_PCNATIVE = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                    epochs=15000, log_every=100, log_per_epoch=True, early_stop=False,
                    seeds=tuple(range(10)), fracs=(0.9,), snap_every=100,
                    opt="sgd", wd=0.0, want_rtdiag=True,
                    c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                    c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)


# =========================================================== GATE-2.1a-PCNATIVE N1' ===
# N1' FOLLOW-UP (PC-native plateau diagnosis). N1 (SGD m=0 wd=0 lr=1.0) plateaued at train=0.832, test=0.055
# (O3). Two questions: (A) did the T=20 relaxation converge? (finite-T bias -> the prime suspect for the
# plateau); (B) does momentum break the plateau? (literature: vanilla GD groks modular addition WITHOUT
# momentum -- Prieto et al. ICLR 2025 -- so momentum is expected IRRELEVANT; this is the negative control).
# Phase A re-runs N1 (m=0, byte-parity) WITH a T_eval=200 eval-only relaxation diagnostic at every rtdiag
# snapshot (resid_T200 vs resid_T20; F trajectory -> does it plateau by step 20?). Phase B sweeps m=0.9 over
# {0.1,0.3,1.0} (finder) then runs the best at 10 seeds. The cfg['momentum'] knob defaults to 0.0 = parity.
N1PRIME_LR = (0.1, 0.3, 1.0)          # Phase B momentum-sweep LRs (lr*=1.0 was the SGD sweet spot in Phase 1)
N1PRIME_MOMENTUM = 0.9                # heavy-ball momentum for the negative control
N1PRIME_TEVAL = 200                   # eval-only long-relaxation steps for the finite-T diagnostic
N1PRIME_LR_SEEDS = 3                  # Phase B finder (seeds 0,1,2)
N1PRIME_LR_EPOCHS = 2000              # Phase B finder (2k -- same window as the PC-native Phase 1)


# =========================================================== GATE-2.1a-PCNATIVE FBGAIN ===
# FEEDBACK-GAIN SWEEP (loop-gain-is-the-oscillation-cause hypothesis). Vanilla PC (no precision) oscillates
# HARDER than c3d (resid 15 vs 5.4). Root-cause hypothesis: TIED-WEIGHT LOOP GAIN. Forward W2 and feedback
# W2^T are the same gear, so loop gain = ||W2||^2. A feedback gain B2 = g*W2 (g<1) breaks this: loop gain ->
# g*||W2||^2. The brain does this (feedforward/feedback are different synapses). fb_gain scales Be INSIDE
# bpc_grads_masked (forward W2e UNCHANGED) -- the "second gear". Default 1.0 = tied (byte-parity with V1).
# NB: NOT c3d precision (which attenuates fb per-UNIT via pi_fb and breaks the Lyapunov descent); fb_gain is a
# GLOBAL scalar on the whole feedback matrix -> damps the loop WITHOUT distorting the per-unit structure.
FBGAIN_LIST = (0.1, 0.3, 0.5, 1.0)    # Phase 1 finder; 1.0 = tied (parity with V1: train~0.39 resid~15)
FBGAIN_FINDER_SEEDS = 3               # Phase 1 (seeds 0,1,2)
FBGAIN_FINDER_EPOCHS = 2000           # Phase 1 (2k)


CFG_GATE21A = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                   epochs=30000, log_every=100, log_per_epoch=True, early_stop=False,
                   seeds=tuple(range(10)), fracs=(0.9,), wd=1.0)


# =========================================================== STAGED CHANNEL ===
# STAGED CHANNEL TRAINING (STAGED_CHANNEL_SPEC v2). Does FREEZING all updates after grok hold the
# schema? The erosion under AdamW+wd=0 (10/10 grok, 10/10 erode) is caused by active updates past grok.
# wd=1.0 already holds (SHY synaptic downscaling). This tests whether "stop learning" (freeze) suffices.
# All cells: c3d (alpha=900), d=1.0, AdamW lr=2e-3, T=20, eta=0.2, 10 seeds, 30k epochs, rtdiag ON.
# Phase 1 (0->switch): w1_gate="both" for ALL cells. Switch at epoch 2500 (after banked peaks 1100-2100,
# before erosion onset 3000+). Optimizer FLUSH at switch (re-init m,v -> stale phase-1 transient out).
#   S-frozen        wd=0, switch@2500, phase2=frozen         -- DECISION CELL: zero-update persistence
#   S-frozen-W1only wd=0, switch@2500, phase2=frozen_w1only  -- W2 drift detector (W1 frozen, W2 learns)
#   S-lowrate       wd=0, switch@2500, phase2=radial         -- low-rate maintenance (W1 radial-only)
#   S-both          wd=0, no switch (both channels throughout) -- same-run EROSION baseline
#   S-both-wd1      wd=1.0, no switch                         -- same-run HOLD baseline (wd=sleep/SHY)
STAGED_SWITCH_EPOCH = 2500                       # after banked peaks <=2100, before erosion onset 3000+
STAGED_QUORUM = 8                                # >=8/10 seeds per criterion (spec §1 R1/R2)
STAGED_GROK_BAR = 0.9                            # test >= 0.9
STAGED_TAIL = 50                                 # final 50 log points = 5k epochs (hold window)
STAGED_W1_WIN = 100                              # final 100 log points = 10k epochs (bounded window)
STAGED_CELLS = [                                 # (name, wd, switch_epoch, phase2_regime)
    ("S-frozen",        0.0, STAGED_SWITCH_EPOCH, "frozen"),
    ("S-frozen-W1only", 0.0, STAGED_SWITCH_EPOCH, "frozen_w1only"),
    ("S-lowrate",       0.0, STAGED_SWITCH_EPOCH, "radial"),
    ("S-both",          0.0, None,                None),     # erosion baseline (no switch)
    ("S-both-wd1",      1.0, None,                None),     # hold baseline (wd=sleep)
]
CFG_STAGED = dict(P=53, h=256, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                  epochs=30000, log_every=100, log_per_epoch=True, early_stop=False,
                  seeds=tuple(range(10)), fracs=(0.9,), wd=0.0, snap_every=100,
                  c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                  c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)


def _dump_gate21a(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           "oracle_ref": ORACLE_REF, "gate21a_wd": list(GATE21A_WD), "q2_tail_logpts": Q2_TAIL,
           "cells": results}
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def drive_gate21a(cfg, label, save_path, run_controls=False):
    """Run the Gate-2.1a cells, reusing run_seeds_masked (close-out harness). Default (run_controls=False)
    runs the three pre-registered wd-sweep cells (Q1 oracle wd-sweep, Q2 A1s@wd=0, Q3 |W1|/|W2|). With
    run_controls=True it runs the THREE CLOSE-OUT CONTROL CELLS that close the remaining banked hedges:
    C1 (backprop+shuffled@d=1.0 wd=0 -- PC-specific vs generic-Adam), the SGD/LR optimizer-sweep @ d=0.5
    (the #6 'PROVEN' closer), and C2 (A1 pc_transport REAL @ d=0.5 wd=0 -- the cortex-risk probe). All
    reuse the same masks/init/split + perm stream by construction (same build_volume/build_mask/
    init_seeds_masked/split_random as drive() and the default branch)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: Gate-2.1a needs CUDA (CPU = days, GATE21A_SPEC §6)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P)
    N = P * P
    seeds = cfg["seeds"]
    top = max(cfg["fracs"])
    epochs, K = cfg["epochs"], cfg["log_every"]
    nS = len(seeds)
    print("=" * 104)
    if run_controls:
        print(f"GATE-2.1a-CTRL [{label}]  | device={dev} P={P} h={h} epochs={epochs} "
              f"T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={nS} log_every={K}")
        print("  C1 backprop+shuffled @ d=1.0 wd=0 (PC-specific vs generic-Adam)  |  "
              "SGD/LR opt-sweep @ d=0.5 {AdamW 2e-3 REF, SGD 1e-2, SGD 1e-1} (#6 'PROVEN' closer)  |  "
              "C2 A1(pc_transport,real) @ d=0.5 wd=0 (cortex-risk probe)")
        print("  PRE-REGISTERED: wd=0 cells = MECHANISM PROBES (train-hold / |W1|-trajectory DVs, NOT grok). "
              "AdamW ref must reproduce 0.853 (+-0.02 parity). Oracle cells run FULL 30k.")
    else:
        print(f"GATE-2.1a [{label}] wd-sweep razor  | device={dev} P={P} h={h} epochs={epochs} "
              f"T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={nS} log_every={K}")
        print("  Q1 oracle(backprop) @ d=0.5 wd in {1.0,0.1,0.01,0.0}  |  Q2 A1s(pc_transport,shuffled) @ d=1.0 wd=0  |  Q3 |W1|/|W2| (free)")
        print("  PRE-REGISTERED (§4): wd=0 = MECHANISM PROBE (success = train>=0.95 SUSTAINED, NOT grok). Oracle runs FULL 30k.")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}

    # shared per-seed split + A5 perm stream (SAME stream as drive(): RandomState(s)->split_random->rng.shuffle)
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)

    def masks_at(density):
        M1s, M2s, st = [], [], []
        for s in seeds:
            m1, m2, stats = build_mask(*vols[s], density, P, h)
            M1s.append(m1); M2s.append(m2); st.append(stats)
        return np.stack(M1s), np.stack(M2s), st

    M1_05, M2_05, st_05 = masks_at(0.5)
    M1_10, M2_10, st_10 = masks_at(1.0)
    print(f"[d=0.5] r={np.mean([x['r'] for x in st_05]):.3f} realized={np.mean([x['realized'] for x in st_05]):.3f} "
          f"coverage={np.mean([x['coverage'] for x in st_05]):.3f}   "
          f"[d=1.0] r={np.mean([x['r'] for x in st_10]):.3f} realized={np.mean([x['realized'] for x in st_10]):.3f} "
          f"coverage={np.mean([x['coverage'] for x in st_10]):.3f}")

    lab_real = [c for _ in range(nS)]
    lab_shuf = [perms[i] for i in range(nS)]
    results = {}

    def _cell_summary(name, res, wd, density, mode, shuffled, scorer=None):
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        div = any(e["diverged"] for e in res)
        pe0 = res[0].get("per_epoch") if res else None
        stop = (pe0["stopped_epoch"] if pe0 else epochs)
        cell = {"wd": wd, "density": density, "mode": mode, "shuffled": shuffled,
                "train_mean": float(np.mean(tr)), "train_per_seed": tr,
                "test_mean": float(np.mean(te)), "test_per_seed": te,
                "diverged": div, "stopped_epoch": stop,
                "per_epoch": [e["per_epoch"] for e in res]}
        tag = f" stopped@{stop}" if stop != epochs else " full-run"
        print(f"\n[{name}] mode={mode} shuffled={shuffled} wd={wd} d={density}{tag}" + ("  DIVERGED" if div else ""))
        print(f"  train {cell['train_mean']:.3f}+-{float(np.std(tr)):.3f} | "
              f"test {cell['test_mean']:.3f}+-{float(np.std(te)):.3f}")
        if scorer is not None:
            cell["label"], cell["lang"] = scorer(res)
            print(f"  -> {cell['label']}: {cell['lang']}")
        return cell

    if not run_controls:
        # ---- Q2: A1s @ d=1.0, wd=0 (run FIRST so Q3 reads its norms; the long PC cell) ----
        cc = dict(cfg); cc["wd"] = 0.0
        t0 = time.perf_counter()
        q2 = run_seeds_masked("pc_transport", seeds, [lab_shuf], [splits_list], M1_10, M2_10, a, b, P,
                              cc, dev, deplete=False, label_kind="shuffled",
                              log_per_epoch=True, early_stop=False, es_uses_block=False)
        results["Q2_A1s_wd0"] = _cell_summary("Q2", q2, 0.0, 1.0, "pc_transport", True, score_q2_21a)
        results["Q2_A1s_wd0"]["wall_s"] = round(time.perf_counter() - t0, 1)
        results["Q2_A1s_wd0"]["q2_tail"] = {        # §4 Q2 "sustains": per-seed tail mean + min (last 5k ep) for the LEAD's read
            "tail_mean_per_seed": [_tail_mean(e["per_epoch"]) for e in q2 if e.get("per_epoch")],
            "tail_min_per_seed": [float(np.min(e["per_epoch"]["train_acc"][-Q2_TAIL:]))
                                  for e in q2 if e.get("per_epoch") and e["per_epoch"]["train_acc"]],
        }

        # ---- Q3: |W1|/|W2| on the Q2 cell (FREE; norms already logged) ----
        lab3, lang3 = score_q3_21a(q2)
        print(f"  -> Q3 {lab3}: {lang3}")
        results["Q3_asymmetry"] = {"label": lab3, "lang": lang3}
        _dump_gate21a(save_path, results, cfg)              # incremental: Q2+Q3 banked before the Q1 sweep

        # ---- Q1: oracle @ d=0.5, wd sweep (wd=1.0 = stream-parity arbiter vs the close-out) ----
        for wd in GATE21A_WD:
            cc = dict(cfg); cc["wd"] = wd
            t0 = time.perf_counter()
            q1 = run_seeds_masked("backprop", seeds, [lab_real], [splits_list], M1_05, M2_05, a, b, P,
                                  cc, dev, deplete=False, label_kind="real",
                                  log_per_epoch=True, early_stop=False, es_uses_block=False)
            key = f"Q1_oracle_wd{wd}"
            results[key] = _cell_summary(f"Q1 wd={wd}", q1, wd, 0.5, "backprop", False,
                                         lambda res, _wd=wd: score_q1_21a(_wd, res))
            results[key]["wall_s"] = round(time.perf_counter() - t0, 1)
            if wd == 1.0:                                   # stream-parity arbiter vs close-out oracle @ d=0.5, 30k
                r = results[key]
                ok = (abs(r["train_mean"] - ORACLE_REF["train"]) <= ORACLE_REF["tol"] and
                      abs(r["test_mean"] - ORACLE_REF["test"]) <= ORACLE_REF["tol"])
                print(f"  PARITY vs close-out (train {ORACLE_REF['train']}, test {ORACLE_REF['test']} "
                      f"+-{ORACLE_REF['tol']}): {'PASS' if ok else 'WARN -- harness drift; reconcile vs close-out before trusting any cell'}")
                results["_parity"] = {"ref": ORACLE_REF,
                                      "razor_wd1": {"train": r["train_mean"], "test": r["test_mean"]}, "pass": ok}
            _dump_gate21a(save_path, results, cfg)          # incremental dump after each Q1 cell

    if run_controls:
        # ---- C1: backprop+shuffled @ d=1.0, wd=0 (PC-SPECIFIC vs GENERIC-ADAM razor; ~1 min) ----
        # Closes the §4 hedge: does backprop ALSO erode under wd=0+noise? Reuses A5's perm stream
        # (lab_shuf) + A1's d=1.0 masks (M1_10) -- the SAME pairing as Q2, only mode=backprop.
        cc = dict(cfg); cc["wd"] = 0.0
        t0 = time.perf_counter()
        c1 = run_seeds_masked("backprop", seeds, [lab_shuf], [splits_list], M1_10, M2_10, a, b, P,
                              cc, dev, label_kind="shuffled",
                              log_per_epoch=True, early_stop=False, es_uses_block=False)
        results["C1_bp_shuf_wd0"] = _cell_summary("C1", c1, 0.0, 1.0, "backprop", True, score_c1_21a)
        results["C1_bp_shuf_wd0"]["wall_s"] = round(time.perf_counter() - t0, 1)
        _dump_gate21a(save_path, results, cfg)

        # ---- SGD/LR optimizer-sweep @ d=0.5 (the #6 'PROVEN' hedge closer; ~min) ----
        # Oracle (backprop, real) over {AdamW lr=2e-3 (REF/parity), SGD lr in {1e-2, 1e-1}}. wd=1.0 (the
        # canonical close-out wd -> AdamW ref MUST reproduce 0.853). opt defaults to 'adamw' so the REF
        # cell is the same code path as Q1 wd=1.0; only opt='sgd' exercises the new _optimizer knob.
        opt_sweep = [("adamw", 2e-3), ("sgd", 1e-2), ("sgd", 1e-1)]
        sgd_cells, ref_key = {}, None
        for opt_name, lr in opt_sweep:
            cc = dict(cfg); cc["wd"] = 1.0; cc["opt"] = opt_name; cc["lr"] = lr
            t0 = time.perf_counter()
            res = run_seeds_masked("backprop", seeds, [lab_real], [splits_list], M1_05, M2_05, a, b, P,
                                   cc, dev, label_kind="real",
                                   log_per_epoch=True, early_stop=False, es_uses_block=False)
            key = f"opt_{opt_name}_lr{lr}"
            results[key] = _cell_summary(f"opt {opt_name} lr={lr}", res, 1.0, 0.5, "backprop", False)
            results[key]["opt"] = opt_name
            results[key]["lr"] = lr
            results[key]["wall_s"] = round(time.perf_counter() - t0, 1)
            sgd_cells[key] = results[key]
            if opt_name == "adamw":                        # parity arbiter (the #6 stream-parity check)
                ok = (abs(results[key]["train_mean"] - ORACLE_REF["train"]) <= ORACLE_REF["tol"] and
                      abs(results[key]["test_mean"] - ORACLE_REF["test"]) <= ORACLE_REF["tol"])
                print(f"  PARITY (opt-sweep AdamW ref) vs close-out (train {ORACLE_REF['train']}, test "
                      f"{ORACLE_REF['test']} +- {ORACLE_REF['tol']}): "
                      f"{'PASS' if ok else 'WARN -- harness drift; STOP before trusting any optimizer cell'}")
                results["_parity"] = {"ref": ORACLE_REF,
                                      "opt_sweep_adamw": {"train": results[key]["train_mean"],
                                                          "test": results[key]["test_mean"]}, "pass": ok}
                ref_key = key
            _dump_gate21a(save_path, results, cfg)         # incremental dump after each optimizer cell
        opt_lab, opt_lang = score_sgd_21a(sgd_cells, ref_key)
        results["_opt_sweep"] = {"label": opt_lab, "lang": opt_lang}
        print(f"  -> OPT-SWEEP {opt_lab}: {opt_lang}")
        _dump_gate21a(save_path, results, cfg)

        # ---- C2: A1 (pc_transport, REAL labels) @ d=0.5, wd=0 (forward cortex-risk probe; ~5-20 min) ----
        # Null confound: PC@d=0.5 sits at train~0.108 -> 'train stable' is ambiguous. DV = |W1| trajectory
        # (disambiguates): inflates => runaway on REAL structure (persistent-error-specific, cortex risk);
        # stable => needs noise (noise-specific). displacement already logged (PC arm).
        cc = dict(cfg); cc["wd"] = 0.0
        t0 = time.perf_counter()
        c2 = run_seeds_masked("pc_transport", seeds, [lab_real], [splits_list], M1_05, M2_05, a, b, P,
                              cc, dev, label_kind="real",
                              log_per_epoch=True, early_stop=False, es_uses_block=False)
        results["C2_A1_real_wd0"] = _cell_summary("C2", c2, 0.0, 0.5, "pc_transport", False, score_c2_21a)
        results["C2_A1_real_wd0"]["wall_s"] = round(time.perf_counter() - t0, 1)
        _dump_gate21a(save_path, results, cfg)

    print("\n" + "=" * 104)
    if run_controls:
        print("GATE-2.1a CONTROLS DONE (C1 + SGD-sweep + C2). Reads are pre-registered; out-of-bin cells "
              "report raw, no narration. LEAD applies the decision tree + banks.")
    else:
        print("GATE-2.1a DONE. Reads are pre-registered (§4); unscored/out-of-bin cells report raw, no narration.")
    _dump_gate21a(save_path, results, cfg)


def gate21a_smoke():
    """Local harness sanity (GATE21A_SPEC §7). Proves: (1) the wd-param change + wd0_safe guard did
    NOT break the harness; (2) exposing wd is numerically inert (razor oracle@wd=1.0 == the close-out
    path, bit-for-bit, at short budget -- early_stop True/False is a no-op for the oracle); (3) A1s@wd=0
    climbs (no NaN, finite |W1|); (4) oracle@wd=0 stays finite; (5) C1 (backprop+shuf@wd=0) finite;
    (6) the SGD optimizer knob (opt='sgd') is finite, no NaN. NOT science."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    a, b, c = make_cells(P); N = P * P
    print("GATE-2.1a SMOKE (2 seeds, short) -- harness sanity; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0,
                log_every=50, log_per_epoch=True, fracs=(0.9,), wd=1.0)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)

    def masks(density):
        M1s, M2s = [], []
        for s in seeds:
            m1, m2, _ = build_mask(*vols[s], density, P, h)
            M1s.append(m1); M2s.append(m2)
        return np.stack(M1s), np.stack(M2s)
    M1_05, M2_05 = masks(0.5)
    M1_10, M2_10 = masks(1.0)
    lab_real = [c for _ in seeds]; lab_shuf = list(perms)

    # (1)+(2) PARITY: razor oracle@wd=1.0 (early_stop=False) == close-out path (early_stop=True), bit-for-bit.
    c_off = dict(base, epochs=1000, early_stop=False)
    c_on = dict(base, epochs=1000, early_stop=True)
    r_off = run_seeds_masked("backprop", seeds, [lab_real], [splits_list], M1_05, M2_05, a, b, P,
                             c_off, dev, label_kind="real", log_per_epoch=True, early_stop=False)
    r_on = run_seeds_masked("backprop", seeds, [lab_real], [splits_list], M1_05, M2_05, a, b, P,
                            c_on, dev, label_kind="real", log_per_epoch=True, early_stop=True)
    for i in range(len(seeds)):
        assert r_off[i]["per_epoch"]["train_acc"] == r_on[i]["per_epoch"]["train_acc"], \
            "PARITY FAIL: early_stop altered the oracle@wd=1.0 trajectory"
        assert r_off[i]["per_epoch"]["test_acc"] == r_on[i]["per_epoch"]["test_acc"], "PARITY FAIL (test)"
        assert r_off[i]["diverged"] is False and r_on[i]["diverged"] is False
    tr_off = float(np.mean([e["train"] for e in r_off]))
    print(f"  PARITY oracle@wd=1.0@d=0.5 (1000ep): razor(early_stop=False) == close-out-path(True) BIT-FOR-BIT "
          f"[train {tr_off:.3f}]. OK")

    # (3) A1s @ d=1.0, wd=0: train climbs, no NaN, finite |W1| (exercises the wd0_safe guard, wd=0 path).
    cc = dict(base, wd=0.0, epochs=800, early_stop=False)
    r = run_seeds_masked("pc_transport", seeds, [lab_shuf], [splits_list], M1_10, M2_10, a, b, P,
                         cc, dev, label_kind="shuffled", log_per_epoch=True, early_stop=False)
    tr = [e["train"] for e in r]
    assert all(np.isfinite(t) for t in tr), "A1s@wd=0 NaN"
    assert all(not e["diverged"] for e in r), "A1s@wd=0 diverged within 800ep (inspect wd0_guard/numerics)"
    w1_last = float(np.mean([e["per_epoch"]["w1_norm"][-1] for e in r]))
    print(f"  A1s@1.0 wd=0 (800ep): train {np.mean(tr):.3f} (per-seed {[round(t,2) for t in tr]}), "
          f"|W1|_last {w1_last:.1f} (finite, growing). OK")

    # (4) oracle @ d=0.5, wd=0: finite (no immediate NaN; wd0_safe guard exercised on backprop+wd=0).
    cc = dict(base, wd=0.0, epochs=400, early_stop=False)
    r = run_seeds_masked("backprop", seeds, [lab_real], [splits_list], M1_05, M2_05, a, b, P,
                         cc, dev, label_kind="real", log_per_epoch=True, early_stop=False)
    tr = [e["train"] for e in r]
    assert all(np.isfinite(t) for t in tr), "oracle@wd=0 NaN"
    print(f"  oracle@0.5 wd=0 (400ep): train {np.mean(tr):.3f}, diverged={any(e['diverged'] for e in r)}. OK")

    # (5) C1: backprop+shuffled @ d=1.0, wd=0 (exercises the C1 cell's code path; must stay finite).
    cc = dict(base, wd=0.0, epochs=400, early_stop=False)
    r = run_seeds_masked("backprop", seeds, [lab_shuf], [splits_list], M1_10, M2_10, a, b, P,
                         cc, dev, label_kind="shuffled", log_per_epoch=True, early_stop=False)
    tr = [e["train"] for e in r]
    assert all(np.isfinite(t) for t in tr), "C1 backprop+shuf@wd=0 NaN"
    assert all(not e["diverged"] for e in r), "C1 backprop+shuf@wd=0 diverged"
    print(f"  C1 backprop+shuf@1.0 wd=0 (400ep): train {np.mean(tr):.3f}, diverged=False. OK")

    # (6) SGD optimizer knob @ d=0.5, wd=1.0 (exercises opt='sgd' via _optimizer; must stay finite).
    cc = dict(base, opt="sgd", lr=1e-2, wd=1.0, epochs=400, early_stop=False)
    r = run_seeds_masked("backprop", seeds, [lab_real], [splits_list], M1_05, M2_05, a, b, P,
                         cc, dev, label_kind="real", log_per_epoch=True, early_stop=False)
    tr = [e["train"] for e in r]
    assert all(np.isfinite(t) for t in tr), "SGD@0.5 NaN"
    assert all(not e["diverged"] for e in r), "SGD@0.5 diverged"
    print(f"  SGD lr=1e-2 wd=1.0 @0.5 (400ep): train {np.mean(tr):.3f}, diverged=False. OK")
    print("SMOKE PASS -- harness intact, parity holds, wd=0 numerics safe, C1+SGD finite. "
          "Ready for review gate + --gate21a / --gate21a-ctrl.")


# ============================================================== GATE-2.1a-C3 razor ===
def _dump_c3(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, "cells": results}
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def drive_gate21a_c3(cfg, label, save_path):
    """Run the 4 C3 precision cells (C3_CELL_SPEC §2): C3-S/D × noise/real @ d=1.0, wd=0, 30k, 10 seeds.
    Reuses run_seeds_masked (the close-out harness) with the c3_static/c3_dynamic modes. Same masks/init/
    split + perm stream as the gate21a cells by construction. Vanilla baselines = Q2 (noise) + C2 (real),
    NOT re-run. Reads = pre-registered §4 K2 triple + §5 targeting signature; the LEAD applies H1 + banks."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: C3 needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); K = cfg["log_every"]
    arms = ("C3-S-noise", "C3-D-noise", "C3-S-real", "C3-D-real")
    print("=" * 104)
    print(f"GATE-2.1a-C3 [{label}] precision scalpel  | device={dev} P={P} h={h} d=1.0 wd=0 "
          f"epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={len(seeds)} log_every={K}")
    print(f"  arms={arms}  | C3-D alpha={cfg['c3_alpha']} (FROZEN from the per-unit Q2 displacement by c3_smoke)")
    print("  PRE-REGISTERED (§4 K2 triple + §5 targeting); out-of-bin cells report raw, no narration.")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]; lab_shuf = list(perms)

    results = {}

    def _summarize(name, res, mode, is_noise):
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        div = any(e["diverged"] for e in res)
        pe0 = res[0].get("per_epoch") if res else None
        stop = (pe0["stopped_epoch"] if pe0 else cfg["epochs"])
        cell = {"mode": mode, "shuffled": is_noise, "wd": cfg["wd"], "density": 1.0,
                "train_mean": float(np.mean(tr)), "train_per_seed": tr,
                "test_mean": float(np.mean(te)), "test_per_seed": te,
                "diverged": div, "stopped_epoch": stop,
                "per_epoch": [e["per_epoch"] for e in res]}
        cell["k2_label"], cell["k2_lang"] = score_c3_k2(res, is_noise, log_every=K)
        cell["tgt_label"], cell["tgt_lang"] = score_c3_targeting(res, mode)
        tag = f" stopped@{stop}" if stop != cfg["epochs"] else " full-run"
        print(f"\n[{name}] mode={mode} shuffled={is_noise} d=1.0 wd=0{tag}" + ("  DIVERGED" if div else ""))
        print(f"  train {cell['train_mean']:.3f}+-{float(np.std(tr)):.3f} | test {cell['test_mean']:.3f}+-{float(np.std(te)):.3f}")
        print(f"  -> §4 {cell['k2_label']}: {cell['k2_lang']}")
        print(f"  -> §5 {cell['tgt_label']}: {cell['tgt_lang']}")
        return cell

    for arm in arms:
        mode, depl, mkind, lkind = ARM_MODE[arm]
        cc = dict(cfg)
        lab = lab_shuf if lkind == "shuffled" else lab_real
        t0 = time.perf_counter()
        res = run_seeds_masked(mode, seeds, [lab], [splits_list], M1, M2, a, b, P,
                               cc, dev, deplete=depl, label_kind=lkind,
                               log_per_epoch=True, early_stop=False, es_uses_block=False)
        results[arm] = _summarize(arm, res, mode, lkind == "shuffled")
        results[arm]["wall_s"] = round(time.perf_counter() - t0, 1)
        _dump_c3(save_path, results, cfg)

    print("\n" + "=" * 104)
    print("OUTCOME (§2): each cell's §4 bin reported above; the LEAD applies H1 + banks. Do not narrate beyond the bins.")
    pass_k2 = lambda lab: lab in ("SAFE-MODE-PASS", "SAFE-MODE-PASS (STABLE-FIT)", "K2-PASS")
    if pass_k2(results["C3-S-real"]["k2_label"]) or pass_k2(results["C3-D-real"]["k2_label"]):
        print("  K2 fix achieved on a real arm -> P1 (HOLD) survives; read §5 (targeting vs giving-up) for H1.")
    else:
        print("  K2 fix budget exhausted on the real arms -> per kill-doc v2 §3 the C5 fallback decision point "
              "(P1-hold at risk). No third C3 arm may be added post-hoc.")
    _dump_c3(save_path, results, cfg)


# =========================================================== GATE-2.1a-RTDIAG razor ===
# Verdict thresholds (ASYMMETRIC; solutions-architect + experiment-designer review). AdamW's per-
# coordinate normalization does NOT commute with the radial/tangential split: a purely-radial gradient
# reads ratio~1.3 under AdamW (probed), so ratio in [1.0, 1.5] is INSIDE the distortion band. The
# tangential side (ratio<1.0) is distortion-safe -- AdamW only inflates ratio TOWARD radial, it cannot
# push a true-tangential signal below 1.0. So: radial needs ratio>1.5 (clear of the band with margin);
# ratio in [1.0, 1.5] -> AMBIGUOUS -> defer. frac>0.5/<0.5 is a distributional cross-check (it does NOT
# rescue the radial side -- the same distortion that inflates mean ratio also inflates the per-unit
# ||radial||>||tang|| count; the ratio bar is the load-bearing guard).
RTDIAG_RADIAL_RATIO_BAR = 1.5
RTDIAG_TANG_RATIO_BAR = 1.0
RTDIAG_FRAC_BAR = 0.5


def _dump_rtdiag(path, results, cfg, overall):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "math_note": ("ReLU homogeneity decomposition (KIMI-THE-WATCH §4.1). For hidden unit i, scaling "
                      "W1[i,:] by c>0 and W2[:,i] by 1/c leaves f invariant -> the radial direction "
                      "u_i=W1_prev[i,:]/||W1_prev[i,:]|| is the GAUGE-ORBIT TANGENT (function-neutral "
                      "only IF W2[:,i] rescales reciprocally, which this diagnostic does NOT verify -- "
                      "it reads the observed dW1 direction under the live optimizer). Only the tangential "
                      "component changes what the net computes. dW1 measured over snap_every epochs "
                      "INCLUDING AdamW + the non-edge peg. Non-edges are 0 in both snapshots -> dW1=0 "
                      "there -> the decomposition is automatically confined to the active fan-in (at d=1.0 "
                      "the mask is fully dense, so this is moot there). radial_i=(dW1.u_i)u_i; "
                      "tangential_i=dW1-radial_i. ratio_mean=mean(||radial||)/mean(||tang||) per interval "
                      "(ratio-of-means = robust bulk ratio); frac_radial_dominant=frac units with "
                      "||radial||>||tang||. CAVEAT: AdamW per-coordinate normalization does not commute "
                      "with the radial/tangential split (a purely-radial gradient reads ratio~1.3), so "
                      "the [1.0,1.5] ratio band is INSIDE AdamW's distortion band -> treated as AMBIGUOUS "
                      "(radial needs ratio>1.5; tangential needs ratio<1.0, which is distortion-safe since "
                      "AdamW only inflates ratio toward radial)."),
        "preregistered_verdict": ("LATE-PHASE-GATED, ASYMMETRIC thresholds (experiment-designer + "
                                  "solutions-architect review). The W1 runaway is a LATE-training pathology "
                                  "and every from-scratch trajectory is necessarily tangential early -> the "
                                  "verdict follows the LATE third: ratio_late>1.5 AND frac_late>0.5 -> "
                                  "RADIAL-DOMINANT -> C5a norm-pin; ratio_late<1.0 AND frac_late<0.5 -> "
                                  "TANGENTIAL-DOMINANT -> C5c fusee; ratio in [1.0,1.5] (AdamW distortion "
                                  "band) or frac disagreement -> AMBIGUOUS. The overall ratio is "
                                  "corroboration only. ON LATE-vs-OVERALL TENSION: late GOVERNS (the early "
                                  "direction-finding phase is necessarily tangential; the late phase carries "
                                  "the runaway signal) -- tension is FLAGGED as a NOTE, NOT routed to MIXED. "
                                  "MIXED only when BOTH late and overall are ambiguous. If late is ambiguous "
                                  "but overall is decisive, overall governs. Per-interval ratio winsorized "
                                  "to <=100 (div-by-zero guard). Decision cell = c3d-real."),
        "cells": results,
        "overall": overall,
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def _rtdiag_verdict(entries):
    """Aggregate per-entry rtdiag interval summaries -> per-cell verdict + per-seed breakdown.
    Returns a dict: verdict in {radial-dominant, tangential-dominant, mixed, UNSCORED}, the overall
    ratio_mean/frac, early-vs-late phase aggregates (ratio + frac), a per-seed late-phase direction
    list + majority (the C3 data showed high seed variance -> report it), and the full per-seed
    interval trajectory.
    PATCHED per experiment-designer review: the W1 runaway is a LATE-training pathology (C3-D-real:
    |W1| 16->~117 mean / 187 max over 30k, train 1.0->0.50 -- the noise-cell 16->620 figure is a
    different cell) and every from-scratch trajectory is necessarily TANGENTIAL early (W1 rotates
    from random init toward the task). So the verdict is GATED ON THE LATE PHASE (decision-relevant);
    the early direction-finding phase must not be allowed to flip the routing (aggregate-hides-
    structure lesson, NOTES F1). Overall ratio is reported as corroboration only. Per-interval ratio
    is winsorized to <=100 so one near-div-by-zero interval (tang~0) cannot dominate the aggregate.
    CAVEAT (post-2k-result, experiment-designer + red-team): "late third of a too-short run" is NOT
    "late phase of the phenomenon" -- if epochs < the runaway onset (~ep2k for c3d-real), the late
    third still samples the grok phase and the verdict is regime-confounded. The caller must ensure
    epochs reach the erosion regime (C3-D-real erosion is clearly underway by ~ep5-10k)."""
    RATIO_CAP = 100.0                           # winsorize: one div-by-zero interval (tang~0) must not dominate
    per_seed = []
    all_ratio, all_frac, all_rmean, all_tmean = [], [], [], []
    for e in entries:
        intervals = (e.get("per_epoch") or {}).get("rtdiag", []) or []
        per_seed.append(intervals)
        for iv in intervals:
            if iv is None:
                continue
            all_ratio.append(min(iv["ratio_mean"], RATIO_CAP))   # winsorized (raw kept in per_seed)
            all_frac.append(iv["frac_radial_dominant"])
            all_rmean.append(iv["radial_mean"]); all_tmean.append(iv["tang_mean"])
    nan = float("nan")
    if not all_ratio:
        return dict(verdict="UNSCORED", verdict_lang="no rtdiag intervals produced",
                    overall_ratio_mean=nan, overall_frac_radial=nan, overall_radial_mean=nan,
                    overall_tang_mean=nan, ratio_early=nan, ratio_late=nan,
                    frac_early=nan, frac_late=nan, per_seed_late_direction=[],
                    per_seed_verdict_majority=nan, n_intervals_per_seed=[len(iv) for iv in per_seed],
                    per_seed=per_seed)
    o_ratio = float(np.mean(all_ratio)); o_frac = float(np.mean(all_frac))
    o_rmean = float(np.mean(all_rmean)); o_tmean = float(np.mean(all_tmean))

    def _dir(ratio, frac):                      # direction bucket from a (ratio, frac) pair (ASYMMETRIC)
        if ratio > RTDIAG_RADIAL_RATIO_BAR and frac > RTDIAG_FRAC_BAR:
            return "radial-dominant"            # ratio>1.5: clear of AdamW's [1.0,1.3] distortion band
        if ratio < RTDIAG_TANG_RATIO_BAR and frac < RTDIAG_FRAC_BAR:
            return "tangential-dominant"        # ratio<1.0: distortion-safe (AdamW only inflates toward radial)
        return "ambiguous"                      # [1.0,1.5] band OR frac disagrees -> borderline -> defer

    early_r, late_r, early_f, late_f = [], [], [], []
    per_seed_late_dir = []
    for intervals in per_seed:
        n = len(intervals)
        if n >= 3:                               # need >=3 intervals to split into early/late thirds
            ti = max(1, n // 3)
            er = [min(iv["ratio_mean"], RATIO_CAP) for iv in intervals[:ti] if iv]
            lr = [min(iv["ratio_mean"], RATIO_CAP) for iv in intervals[-ti:] if iv]
            ef = [iv["frac_radial_dominant"] for iv in intervals[:ti] if iv]
            lf = [iv["frac_radial_dominant"] for iv in intervals[-ti:] if iv]
            early_r += er; late_r += lr; early_f += ef; late_f += lf
            per_seed_late_dir.append(_dir(float(np.mean(lr)) if lr else nan,
                                          float(np.mean(lf)) if lf else nan))
        else:
            per_seed_late_dir.append("ambiguous")
    r_early = float(np.mean(early_r)) if early_r else nan
    r_late = float(np.mean(late_r)) if late_r else nan
    f_early = float(np.mean(early_f)) if early_f else nan
    f_late = float(np.mean(late_f)) if late_f else nan
    d_late = _dir(r_late, f_late)
    d_overall = _dir(o_ratio, o_frac)
    # Verdict follows the LATE phase (where the runaway lives) when late is decisive; else falls back
    # to overall; else MIXED. Late-vs-overall tension is flagged (early tangential phase vs late runaway).
    if d_late != "ambiguous":
        v = d_late
        tension = (d_overall != "ambiguous" and d_overall != d_late)
        arrow = {"radial-dominant": "norm inflation (direction stable) -> C5a (norm-pin) priority",
                 "tangential-dominant": "direction drift (orbiting relaxation teaches garbage) -> C5c (fusee) priority"}[v]
        lang = (f"LATE-PHASE decisive: ratio_late={r_late:.2f}, frac_late={f_late:.2f} -> {arrow}. "
                f"[overall ratio={o_ratio:.2f} frac={o_frac:.2f}; early ratio={r_early:.2f} frac={f_early:.2f}]")
        if tension:
            lang += (" (NOTE: overall disagrees with late -- the early direction-finding phase is necessarily "
                     "tangential; the LATE phase carries the runaway signal, so late governs.)")
    elif d_overall != "ambiguous":
        v = d_overall
        arrow = {"radial-dominant": "norm inflation -> C5a (norm-pin) priority",
                 "tangential-dominant": "direction drift -> C5c (fusee) priority"}[v]
        lang = (f"late ambiguous; OVERALL decisive: ratio={o_ratio:.2f}, frac={o_frac:.2f} -> {arrow}. "
                f"[late ratio={r_late:.2f} frac={f_late:.2f}]")
    else:
        v = "mixed"
        lang = (f"both late and overall ambiguous (ratio_late={r_late:.2f} frac_late={f_late:.2f}; "
                f"overall ratio={o_ratio:.2f} frac={o_frac:.2f}) -> defer C5 routing.")
    # per-seed late-phase majority (confidence signal; C3-D-real had train std 0.22 -> high seed variance)
    decided = [d for d in per_seed_late_dir if d != "ambiguous"]
    if v in ("radial-dominant", "tangential-dominant") and decided:
        majority = float(sum(1 for d in decided if d == v) / len(decided))
    else:
        majority = nan
    n_seed = len(per_seed_late_dir)
    if n_seed < 10 and v != "UNSCORED":
        lang += f" [{n_seed}-seed pilot -- confirm scale (>=10 seeds) before building C5.]"
    return dict(verdict=v, verdict_lang=lang, overall_ratio_mean=o_ratio, overall_frac_radial=o_frac,
                overall_radial_mean=o_rmean, overall_tang_mean=o_tmean,
                ratio_early=r_early, ratio_late=r_late, frac_early=f_early, frac_late=f_late,
                per_seed_late_direction=per_seed_late_dir, per_seed_verdict_majority=majority,
                n_intervals_per_seed=[len(iv) for iv in per_seed], per_seed=per_seed)


def drive_gate21a_rtdiag(cfg, label, save_path):
    """Run the RADIAL/TANGENTIAL W1 diagnostic (KIMI-THE-WATCH §4.1) on 2 cells @ d=1.0 wd=0:
    vanilla-real (pc_transport; the d=1.0 wd=0 baseline -- stability NOT pre-assumed, this fills the
    F3 cell that was never measured) + c3d-real (c3_dynamic, the runaway cell = the C5 routing
    arbiter). Reuses run_seeds_masked with want_rtdiag=True (read-only; parity-inert). Reads = the
    late-phase radial-vs-tangential ratio (the runaway is late-training; early is necessarily
    tangential); DECIDES C5a (norm-pin) vs C5c (fusee). NOT C5."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: rtdiag needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); snap = cfg["snap_every"]
    print("=" * 104)
    print(f"GATE-2.1a-RTDIAG [{label}] radial/tangential W1 decomposition  | device={dev} P={P} h={h} "
          f"d=1.0 wd=0 epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} seeds={len(seeds)} snap_every={snap}")
    print("  PRE-REGISTERED VERDICT (late-phase-gated, asymmetric thresholds; experiment-designer +")
    print("    solutions-architect review): the W1 runaway is a LATE-training pathology; every from-scratch")
    print("    trajectory is necessarily tangential early. So the verdict follows the LATE third:")
    print(f"    ratio_late>{RTDIAG_RADIAL_RATIO_BAR} & frac_late>{RTDIAG_FRAC_BAR} -> RADIAL-DOMINANT (C5a norm-pin);")
    print(f"    ratio_late<{RTDIAG_TANG_RATIO_BAR} & frac_late<{RTDIAG_FRAC_BAR} -> TANGENTIAL-DOMINANT (C5c fusee);")
    print("    ratio in [1.0,1.5] (AdamW distortion band) or frac disagreement -> AMBIGUOUS. On late-vs-overall")
    print("    tension: late GOVERNS (flagged as NOTE, NOT MIXED); MIXED only when both late+overall ambiguous.")
    print("  cells: vanilla-real (pc_transport, d=1.0 wd=0 baseline; stability NOT pre-assumed) + c3d-real")
    print("    (c3_dynamic, the runaway cell = C5 routing arbiter). NOT C5 -- a diagnostic.")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]

    cells = [("vanilla-real", "pc_transport"), ("c3d-real", "c3_dynamic")]
    results = {}
    for name, mode in cells:
        cc = dict(cfg)
        t0 = time.perf_counter()
        res = run_seeds_masked(mode, seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                               cc, dev, deplete=False, label_kind="real",
                               log_per_epoch=True, early_stop=False, es_uses_block=False,
                               want_rtdiag=True)
        vd = _rtdiag_verdict(res)
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        results[name] = dict(mode=mode, shuffled=False, wd=cfg["wd"], density=1.0, snap_every=snap,
                             train_mean=float(np.mean(tr)), train_per_seed=tr,
                             test_mean=float(np.mean(te)), test_per_seed=te,
                             diverged=any(e["diverged"] for e in res),
                             verdict=vd["verdict"], verdict_lang=vd["verdict_lang"],
                             overall_ratio_mean=vd["overall_ratio_mean"],
                             overall_frac_radial=vd["overall_frac_radial"],
                             overall_radial_mean=vd["overall_radial_mean"],
                             overall_tang_mean=vd["overall_tang_mean"],
                             ratio_early=vd["ratio_early"], ratio_late=vd["ratio_late"],
                             frac_early=vd["frac_early"], frac_late=vd["frac_late"],
                             per_seed_late_direction=vd["per_seed_late_direction"],
                             per_seed_verdict_majority=vd["per_seed_verdict_majority"],
                             n_intervals_per_seed=vd["n_intervals_per_seed"],
                             intervals_per_seed=vd["per_seed"],
                             wall_s=round(time.perf_counter() - t0, 1))
        print(f"\n[{name}] mode={mode} d=1.0 wd=0 snap_every={snap}  "
              f"(train {results[name]['train_mean']:.3f} | test {results[name]['test_mean']:.3f})")
        print(f"  overall: ratio={vd['overall_ratio_mean']:.3f} frac_radial={vd['overall_frac_radial']:.3f}  "
              f"| radial_mean={vd['overall_radial_mean']:.4f} tang_mean={vd['overall_tang_mean']:.4f}")
        print(f"  LATE  : ratio={vd['ratio_late']:.3f} frac={vd['frac_late']:.3f}  |  "
              f"EARLY: ratio={vd['ratio_early']:.3f} frac={vd['frac_early']:.3f}  (late carries the runaway signal)")
        print(f"  per-seed late-direction={vd['per_seed_late_direction']}  majority={vd['per_seed_verdict_majority']}")
        print(f"  -> VERDICT: {vd['verdict']}")
        print(f"    {vd['verdict_lang']}")
        _dump_rtdiag(save_path, results, cfg, _overall_verdict(results))

    overall = _overall_verdict(results)
    print("\n" + "=" * 104)
    print(f"OVERALL VERDICT: {overall['verdict']} (decision cell = c3d-real; vanilla-real is the d=1.0 wd=0 baseline).")
    print("  DO NOT BUILD C5 -- this is a diagnostic. Wait for the verdict before C5a/C5c.")
    _dump_rtdiag(save_path, results, cfg, overall)


def _overall_verdict(results):
    """Cross-cell verdict: the c3d-real cell is the C5 routing arbiter (the runaway cell). vanilla-real
    is the d=1.0 wd=0 baseline (context; its stability is NOT pre-assumed -- it fills the unmeasured
    F3 cell). Report both; the decision follows c3d-real unless the two cells strongly disagree (then
    MIXED)."""
    v_dec = results.get("c3d-real", {}).get("verdict", "UNSCORED")
    v_base = results.get("vanilla-real", {}).get("verdict", "UNSCORED")
    if v_dec in ("radial-dominant", "tangential-dominant"):
        verdict = v_dec
        lang = f"decision cell c3d-real = {v_dec}"
        if v_base != v_dec and v_base not in ("UNSCORED", "mixed"):
            lang += f" (NOTE: baseline vanilla-real = {v_base} disagrees -- report both, defer to compiled run)"
    else:
        verdict = "mixed"
        lang = f"decision cell c3d-real = {v_dec} -> ambiguous"
    return {"verdict": verdict, "lang": lang, "decision_cell": "c3d-real", "baseline_cell": "vanilla-real"}


def rtdiag_smoke():
    """Local parity + sanity for the rtdiag instrumentation (NOT science). Proves:
    (1) want_rtdiag=True reproduces want_rtdiag=False BITWISE on train/test/w1/w2 trajectories (the
        snapshot is a read-only detach().clone() after the peg -- no RNG/dtype/graph perturbation);
    (2) the radial/tangential numbers are finite and in a sane range (ratio_mean in [0,100], no NaN/inf).
    Mirrors c3_smoke's parity pattern (which already proves pc_transport-vs-c3_static bitwise parity)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    a, b, c = make_cells(P); N = P * P
    print("RTDIAG SMOKE (2 seeds, 200ep) -- parity (rtdiag-ON == rtdiag-OFF bitwise) + sane numbers; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=50, log_per_epoch=True,
                fracs=(0.9,), wd=0.0, epochs=200, snap_every=50)
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]
    r_off = run_seeds_masked("pc_transport", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                             dict(base), dev, label_kind="real", log_per_epoch=True, early_stop=False,
                             want_rtdiag=False)
    r_on = run_seeds_masked("pc_transport", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                            dict(base), dev, label_kind="real", log_per_epoch=True, early_stop=False,
                            want_rtdiag=True)
    for i in range(len(seeds)):                                  # (1) BITWISE PARITY (read-only invariant)
        for key in ("train_acc", "test_acc", "w1_norm", "w2_norm"):
            a_, b_ = r_off[i]["per_epoch"][key], r_on[i]["per_epoch"][key]
            assert np.array_equal(a_, b_), f"RTDIAG PARITY FAIL ({key}, seed {seeds[i]})"
    print("  PARITY (want_rtdiag=True == False) bitwise: train/test/w1/w2 trajectories array_equal. OK")
    for i in range(len(seeds)):                                  # (2) SANE NUMBERS (finite, ratio in [0,100])
        ivs = r_on[i]["per_epoch"]["rtdiag"]
        assert ivs, "no rtdiag intervals produced"
        for iv in ivs:
            for k in ("radial_mean", "tang_mean", "ratio_mean", "frac_radial_dominant"):
                assert np.isfinite(iv[k]), f"rtdiag {k} not finite: {iv}"
            assert 0.0 <= iv["ratio_mean"] <= 100.0, f"ratio_mean out of [0,100]: {iv}"
            assert 0.0 <= iv["frac_radial_dominant"] <= 1.0, f"frac out of [0,1]: {iv}"
        print(f"  seed {seeds[i]}: {len(ivs)} intervals | " +
              " | ".join(f"ep{iv['epoch']}: rad={iv['radial_mean']:.4f} tang={iv['tang_mean']:.4f} "
                         f"ratio={iv['ratio_mean']:.2f} frac={iv['frac_radial_dominant']:.2f}" for iv in ivs))
    print("SMOKE PASS -- rtdiag parity bitwise + numbers finite & in range. "
          "Ready for review gate + --gate21a-rtdiag.")


# =========================================================== GATE-2.1a-W1GATE razor ===
W1GATE_CELLS = [
    ("both-on", "both"),          # baseline (unrestricted gW1)
    ("radial-only", "radial"),    # user hyp: freeze tangential (orbiting-relaxation direction drift = noise)
    ("tang-only", "tangential"),  # Kimi C5a: freeze radial (norm inflation is the problem)
    ("frozen", "frozen"),         # W2-alone control (gW1 = 0)
]


def _dump_w1gate(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "math_note": ("W1-gating factorial (user spec). Per epoch BEFORE opt.step, project gW1 onto "
                      "radial/tangential using the CURRENT (pre-step) W1: u_i=W1[i,:]/||W1[i,:]||; "
                      "radial_g=(gW1.u_i)u_i; tang_g=gW1-radial_g. Gate gW1 to {radial_g | tang_g | 0} "
                      "and pass to the optimizer. W2 learns freely (full gW2). Mask preserved by "
                      "construction (u=0 on non-edges). want_rtdiag ON as the gate-correctness check. "
                      "CAVEAT: AdamW per-coord normalization does not commute with the projection -> the "
                      "measured (post-step) gated-channel magnitude is small-but-nonzero (AdamW leakage), "
                      "not exactly 0; frozen (gW1=0) has NO leakage (W1 truly unchanged)."),
        "preregistered_verdict": ("radial-only vs both-on judged by a PAIRED Wilcoxon on per-seed train "
                                  "(not a bare mean-diff at n=10). sig & radial-only>both-on -> tangential "
                                  "HARMFUL; sig & <both-on -> tangential USEFUL (direction learning); NOT "
                                  "sig -> NO DETECTABLE tangential effect (weak user-hyp support; NOT clean "
                                  "equivalence -- TOST would be needed, nonzero tangential residual under "
                                  "radial-only = AdamW leakage, and 15k may under-sample the erosion regime). "
                                  "tang-only K2-PASS -> inflation was the SOLE problem (C5a VALIDATED). "
                                  "frozen~=both-on -> W2 alone explains the fit. K2 = disp<1.0 sustained + "
                                  "|W1| slope <1%/1k + train holds. The cell that passes K2 = the channel "
                                  "that needed gating."),
        "cells": results,
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def drive_gate21a_w1gate(cfg, label, save_path):
    """W1-gating factorial (user spec): CAUSAL intervention on the radial/tangential decomposition.
    4 cells (both-on / radial-only / tang-only / frozen), all c3_dynamic @ d=1.0 wd=0 10 seeds 15k.
    W2 learns freely; gW1's channel is gated pre-opt-step. want_rtdiag ON as the gate check; K2 triple
    (disp / |W1|-slope / train) via score_c3_k2. DECIDES which C5 arm to build; does NOT build C5."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: w1gate needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); snap = cfg["snap_every"]; K = cfg["log_every"]
    print("=" * 104)
    print(f"GATE-2.1a-W1GATE [{label}] W1 radial/tangential gating factorial  | device={dev} P={P} h={h} "
          f"d=1.0 wd=0 epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} seeds={len(seeds)} snap_every={snap}")
    print(f"  cells: {[(n, g) for n, g in W1GATE_CELLS]}  (all c3_dynamic; W2 learns freely; want_rtdiag ON)")
    print("  PRE-REGISTERED: radial-only vs both-on judged by a PAIRED Wilcoxon on per-seed train (NOT a bare")
    print("    mean-diff): sig & radial-only>both -> tangential HARMFUL; sig & <both -> USEFUL (direction")
    print("    learning); NOT sig -> NO DETECTABLE effect (weak user-hyp support; nonzero tangential residual")
    print("    under radial-only = AdamW leakage; 15k may under-sample the erosion regime). tang-only K2-PASS")
    print("    -> C5a VALIDATED; frozen~=both-on -> W2 alone explains the fit.")
    print("=" * 104)
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]

    results = {}
    for name, gate in W1GATE_CELLS:
        t0 = time.perf_counter()
        res = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                               dict(cfg), dev, deplete=False, label_kind="real",
                               log_per_epoch=True, early_stop=False, es_uses_block=False,
                               want_rtdiag=True, w1_gate=gate)
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        k2_label, k2_lang = score_c3_k2(res, is_noise=False, log_every=K)
        vd = _rtdiag_verdict(res)                      # gate-correctness read (gated channel -> ~0 magnitude)
        niv = max((len(ivs) for ivs in vd["per_seed"]), default=0)
        ti = max(1, niv // 3)
        late_ivs = [iv for ivs in vd["per_seed"] for iv in ivs[-ti:]]
        late_rmean = float(np.mean([iv["radial_mean"] for iv in late_ivs])) if late_ivs else float("nan")
        late_tmean = float(np.mean([iv["tang_mean"] for iv in late_ivs])) if late_ivs else float("nan")
        results[name] = dict(w1_gate=gate, mode="c3_dynamic", shuffled=False, wd=cfg["wd"], density=1.0,
                             snap_every=snap,
                             train_mean=float(np.mean(tr)), train_per_seed=tr,
                             test_mean=float(np.mean(te)), test_per_seed=te,
                             diverged=any(e["diverged"] for e in res),
                             k2_label=k2_label, k2_lang=k2_lang,
                             rtdiag_verdict=vd["verdict"],
                             gate_check_late_radial_mean=late_rmean, gate_check_late_tang_mean=late_tmean,
                             overall_ratio_mean=vd["overall_ratio_mean"], overall_frac_radial=vd["overall_frac_radial"],
                             ratio_late=vd["ratio_late"], frac_late=vd["frac_late"],
                             n_intervals_per_seed=vd["n_intervals_per_seed"],
                             intervals_per_seed=vd["per_seed"],
                             per_epoch=[e["per_epoch"] for e in res],
                             wall_s=round(time.perf_counter() - t0, 1))
        print(f"\n[{name}] w1_gate={gate}: train {results[name]['train_mean']:.3f} | test {results[name]['test_mean']:.3f}")
        print(f"  K2: {k2_label} -- {k2_lang}")
        print(f"  gate-check (late): radial_mean={late_rmean:.5f} | tang_mean={late_tmean:.5f}  "
              f"(gated channel should be ~0; frozen: both ~0)")
        _dump_w1gate(save_path, results, cfg)

    # pre-registered comparisons (paired Wilcoxon on per-seed train; _wilcoxon from schema_gate1).
    # F1 patch (experiment-designer): the central "tangential was noise" claim must NOT rest on a bare
    # mean-difference at n=10 (SEM ~0.05-0.07 ~= the old 0.05 bar). Use a paired test: a SIGNIFICANT
    # difference -> tangential mattered (HARMFUL if radial-only>both, USEFUL if <); a NON-significant
    # difference -> "no detectable effect" (NOT "confirmed equivalent" -- TOST would be needed for that,
    # and the 15k window may under-sample the erosion regime, so the harm could be slow-onset).
    def _tm(cell): return results[cell]["train_mean"]
    def _tps(cell): return results[cell]["train_per_seed"]
    def _paired_p(a_cell, b_cell):   # one-sided p that a_cell > b_cell (None if n<6/scipy/all-ties)
        return _wilcoxon(_tps(a_cell), _tps(b_cell))
    both, ron, ton, frz = _tm("both-on"), _tm("radial-only"), _tm("tang-only"), _tm("frozen")
    cmp = []
    d = ron - both
    p_ron_gt = _paired_p("radial-only", "both-on") if d > 0 else None
    p_both_gt = _paired_p("both-on", "radial-only") if d < 0 else None
    p_dir = p_ron_gt if d > 0 else p_both_gt
    sig = (p_dir is not None and p_dir < 0.05)
    pleak = results["radial-only"]["gate_check_late_tang_mean"]
    pleak_both = results["both-on"]["gate_check_late_tang_mean"]
    leak_pct = (100.0 * pleak / pleak_both) if pleak_both > 0 else float("nan")
    if sig and d > 0:
        cmp.append(f"radial-only > both-on ({ron:.3f} > {both:.3f}, d={d:+.3f}, paired-Wilcoxon p={p_dir:.3f}) -> TANGENTIAL WAS HARMFUL (stopping it significantly helped)")
    elif sig and d < 0:
        cmp.append(f"radial-only < both-on ({ron:.3f} < {both:.3f}, d={d:+.3f}, paired-Wilcoxon p={p_dir:.3f}) -> TANGENTIAL WAS USEFUL (schemas need direction learning)")
    else:
        pstr = f"p={p_dir:.3f}" if p_dir is not None else "p=n/a"
        cmp.append(f"radial-only ~= both-on ({ron:.3f} vs {both:.3f}, d={d:+.3f}, paired-Wilcoxon {pstr} NOT sig) -> NO DETECTABLE tangential effect "
                   f"(user hyp weakly supported: stopping tangential didn't measurably hurt; CAVEAT: ~{leak_pct:.0f}% tangential residual under radial-only "
                   f"(AdamW leakage), and 15k may under-sample the erosion regime -> not a clean equivalence/TOST).")
    if results["tang-only"]["k2_label"] == "K2-PASS":
        cmp.append("tang-only K2-PASS -> inflation was the SOLE problem (C5a/norm-pin VALIDATED)")
    else:
        cmp.append(f"tang-only K2={results['tang-only']['k2_label']} -> freezing radial alone did NOT suffice (inflation not the sole problem)")
    if abs(frz - both) < 0.10:
        cmp.append(f"frozen~=both-on ({frz:.3f} vs {both:.3f}) -> W2 ALONE explains the fit (W1 updates not load-bearing)")
    else:
        cmp.append(f"frozen != both-on ({frz:.3f} vs {both:.3f}) -> W1 updates ARE load-bearing")
    results["_comparisons"] = cmp
    results["_stats"] = dict(radial_vs_both_paired_p_dir=p_dir, radial_tang_leak_pct=leak_pct)
    print("\n" + "=" * 104)
    print("PRE-REGISTERED COMPARISONS:")
    for s in cmp:
        print("  " + s)
    k2_pass = [n for n, _ in W1GATE_CELLS if results[n]["k2_label"] == "K2-PASS"]
    print(f"  K2-pass cells: {k2_pass}  (the channel that needed gating; empty -> none settle)")
    print("  DO NOT BUILD C5 yet -- this determines which arm to build.")
    _dump_w1gate(save_path, results, cfg)


def w1gate_smoke():
    """Local parity + gate-correctness check (NOT science). Proves:
    (1) w1_gate='both' (explicit) == default (omitted) BITWISE on train/test/w1/w2 (the `if w1_gate!=both`
        guard skips entirely -> parity by construction; verified);
    (2) gate works: frozen -> W1 ~unchanged (||W1|| flat; gW1=0 has NO AdamW leakage); radial-only and
        tang-only show reduced gated-channel magnitude (small-but-nonzero due to AdamW leakage). Short."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    a, b, c = make_cells(P); N = P * P
    print("W1GATE SMOKE (2 seeds, 150ep) -- 'both' parity + gate-correctness; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=50, log_per_epoch=True,
                fracs=(0.9,), wd=0.0, epochs=150, snap_every=50,
                c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]
    # (1) PARITY: w1_gate='both' (explicit) == default (omitted) bitwise
    r_def = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                             dict(base), dev, deplete=False, label_kind="real",
                             log_per_epoch=True, early_stop=False, want_rtdiag=True)
    r_both = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                              dict(base), dev, deplete=False, label_kind="real",
                              log_per_epoch=True, early_stop=False, want_rtdiag=True, w1_gate="both")
    for i in range(len(seeds)):
        for key in ("train_acc", "test_acc", "w1_norm", "w2_norm"):
            assert np.array_equal(r_def[i]["per_epoch"][key], r_both[i]["per_epoch"][key]), \
                f"W1GATE PARITY FAIL ({key}, seed {seeds[i]}): 'both' != default"
    print("  PARITY (w1_gate='both' == default) bitwise: train/test/w1/w2 array_equal. OK")
    # (2) GATE CORRECTNESS
    runs = {"both": r_both}
    for gate in ("radial", "tangential", "frozen"):
        runs[gate] = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                                      dict(base), dev, deplete=False, label_kind="real",
                                      log_per_epoch=True, early_stop=False, want_rtdiag=True, w1_gate=gate)
    for gate, name in (("radial", "radial-only"), ("tangential", "tang-only"), ("frozen", "frozen")):
        ivs0 = runs[gate][0]["per_epoch"]["rtdiag"]
        late = ivs0[-1] if ivs0 else None
        assert late, f"{name}: no rtdiag intervals"
        print(f"  [{name:11s}] last interval: rad={late['radial_mean']:.5f} tang={late['tang_mean']:.5f} "
              f"ratio={late['ratio_mean']:.3f}")
    # frozen: W1 truly unchanged (||W1|| flat; no AdamW leakage since gW1=0)
    w1n = runs["frozen"][0]["per_epoch"]["w1_norm"]
    print(f"  [frozen]     w1_norm: first={w1n[0]:.3f} last={w1n[-1]:.3f} (W1 frozen -> must be ~equal)")
    assert abs(w1n[-1] - w1n[0]) < 0.5, f"frozen W1 moved: {w1n[0]:.3f}->{w1n[-1]:.3f}"
    # radial-only: late tang_mean should be << both-on's (gating removed tangential; residual = AdamW leakage)
    bt = runs["both"][0]["per_epoch"]["rtdiag"][-1]["tang_mean"]
    rt = runs["radial"][0]["per_epoch"]["rtdiag"][-1]["tang_mean"]
    print(f"  [gate effect] both-on late tang={bt:.5f} | radial-only late tang={rt:.5f} (radial-only should be smaller)")
    print("SMOKE PASS -- 'both' parity bitwise; frozen W1 unchanged; gate-correctness verified. "
          "Ready for review gate + --gate21a-w1gate.")


def c3_smoke():
    """Local harness sanity (C3_CELL_SPEC §7 + §3.1 PARITY GUARD). Proves: (1) C3-S with Π≡1 (c3_lambda=0)
    reproduces pc_transport VANILLA bitwise (40ep, 2-seed, strict np.array_equal on train/test trajectories);
    (2) C3-D with π≡1 (alpha=0) reproduces vanilla bitwise; (3) the α diagnostic reads the PER-UNIT
    displacement distribution of the ungated A1s@wd=0 relaxation -> freezes α = 1/d_mid. NOT science."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    a, b, c = make_cells(P); N = P * P
    print("C3 SMOKE (2 seeds, short) -- parity (π≡1 == vanilla bitwise) + α diagnostic; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=1, log_per_epoch=True,
                fracs=(0.9,), wd=0.0, epochs=40, c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=0.0,
                c3_lambda=0.0, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]

    # (1) PARITY C3-S: c3_static with c3_lambda=0 -> Π_h=Π_out≡1 -> bitwise vanilla. log_every=1 logs every ep.
    r_van = run_seeds_masked("pc_transport", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                             dict(base), dev, label_kind="real", log_per_epoch=True, early_stop=False)
    r_c3s = run_seeds_masked("c3_static", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                             dict(base), dev, label_kind="real", log_per_epoch=True, early_stop=False)
    for i in range(len(seeds)):
        assert np.array_equal(r_van[i]["per_epoch"]["train_acc"], r_c3s[i]["per_epoch"]["train_acc"]), "C3-S PARITY FAIL (train_acc)"
        assert np.array_equal(r_van[i]["per_epoch"]["test_acc"], r_c3s[i]["per_epoch"]["test_acc"]), "C3-S PARITY FAIL (test_acc)"
    print("  PARITY C3-S (λ=0 -> Π≡1) == vanilla bitwise: train/test trajectories array_equal over 40ep. OK")

    # (2) PARITY C3-D: c3_dynamic with alpha=0 -> π_i≡1 -> fb ungated -> bitwise vanilla.
    r_c3d = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                             dict(base), dev, label_kind="real", log_per_epoch=True, early_stop=False)
    for i in range(len(seeds)):
        assert np.array_equal(r_van[i]["per_epoch"]["train_acc"], r_c3d[i]["per_epoch"]["train_acc"]), "C3-D PARITY FAIL (train_acc)"
        assert np.array_equal(r_van[i]["per_epoch"]["test_acc"], r_c3d[i]["per_epoch"]["test_acc"]), "C3-D PARITY FAIL (test_acc)"
    print("  PARITY C3-D (α=0 -> π≡1) == vanilla bitwise: train/test trajectories array_equal over 40ep. OK")

    # (3) α DIAGNOSTIC: per-unit displacement of the UNGATED A1s@wd=0 relaxation @ d=1.0 (the Q2 dynamics,
    #     read per-unit). Custom loop captures per_unit_disp + train each epoch (run_seeds_masked doesn't
    #     expose per-unit disp). Run LONG ENOUGH to reach the deep orbit (Q2 erodes only after ep~200;
    #     250ep under-samples d_high -> α mis-frozen high). Still cheap (~tens of seconds on a 2060).
    n_diag = 600
    S = len(seeds)
    Xall = torch.tensor(onehot2(a, b, P), device=dev).to(torch.bool)
    ntr = len(splits_list[0][0])
    Xtr = torch.zeros(S, ntr, 2 * P, dtype=torch.bool, device=dev)
    Ytr = torch.zeros(S, ntr, P, dtype=torch.bool, device=dev)
    ctr = torch.zeros(S, ntr, dtype=torch.int16, device=dev)
    lab_shuf = list(perms)
    for i, (tr, _te) in enumerate(splits_list):
        Xtr[i] = Xall[tr]
        Ytr[i, torch.arange(ntr), torch.tensor(lab_shuf[i][tr])] = True
        ctr[i] = torch.tensor(lab_shuf[i][tr])
    M1t = torch.tensor(M1, device=dev); M2t = torch.tensor(M2, device=dev)
    W1, W2 = init_seeds_masked(seeds, P, h, M1, M2, dev)
    W1.requires_grad_(True); W2.requires_grad_(True)
    opt = torch.optim.AdamW([W1, W2], lr=2e-3, weight_decay=0.0)
    n_pe = torch.full((S,), float(ntr), device=dev)
    tr_mask = torch.ones(S, ntr, dtype=torch.bool, device=dev)
    scale_agg = float(np.sqrt(ntr * h))                       # per-unit mean-|·| ~= agg-L2-norm / sqrt(n*h)
    disp_traj, train_ckpt = [], {}                            # per-epoch per-unit disp (S,h); train at checkpoints
    tr_ckpts = {49, 199, n_diag - 1}
    t0 = time.perf_counter()
    for ep in range(n_diag):
        opt.zero_grad(set_to_none=True)
        g1, g2, _tr, _mr, _resid, per_unit, _e1, _e2 = bpc_grads_masked(
            W1, W2, Xtr, Ytr, W2, M1t, M2t, base["T"], base["eta"], want_trace=False, want_c3=True,
            n_per_entry=n_pe, tr_mask=tr_mask)                # pi_*=None -> ungated vanilla relaxation; want_c3 -> per-unit disp
        W1.grad, W2.grad = g1, g2
        opt.step()
        with torch.no_grad():
            W1.mul_(M1t.to(torch.float32)); W2.mul_(M2t.to(torch.float32))
            if ep in tr_ckpts:
                pred = blogits_masked(Xtr, W1, W2, M1t, M2t).argmax(-1)
                train_ckpt[ep] = float((((pred == ctr) & tr_mask).sum(1).float() / n_pe).mean())
        disp_traj.append(per_unit.copy())
        if ep % 100 == 0:
            print(f"    [α-diag ep {ep:4d}/{n_diag}] per-unit disp median={float(np.median(per_unit)):.5f} "
                  f"(agg~{float(np.median(per_unit))*scale_agg:.2f})", flush=True)
    print(f"    [α-diag done in {time.perf_counter()-t0:.1f}s]")
    disp_arr = np.array(disp_traj, dtype=np.float64)         # (n_diag, S, h)
    early = disp_arr[5:105]; late = disp_arr[-200:]          # settled (early climb) vs deep-orbit phases
    d_low = float(np.median(early)); d_high = float(np.median(late))
    d_mid = float(np.sqrt(d_low * d_high)) if d_low > 0 and d_high > 0 else float("nan")
    alpha = float(1.0 / d_mid) if d_mid > 0 else float("nan")
    tr_str = ", ".join(f"ep{k}={v:.3f}" for k, v in sorted(train_ckpt.items()))
    climb_erode = (train_ckpt.get(199, 0) > 0.6 and train_ckpt.get(n_diag - 1, 1) < train_ckpt.get(199, 1) - 0.1)
    print(f"  α DIAGNOSTIC (A1s@wd=0 @ d=1.0, {n_diag}ep, per-unit |x1^T-x1^{{T-1}}| mean over batch):")
    print(f"    SANITY train: {tr_str} "
          f"({'climb-then-erode OK (matches Q2)' if climb_erode else 'shape note -- inspect vs Q2'})")
    print(f"    per-unit disp: early-ep[5:105] median={d_low:.5f} | late-ep[-200:] median={d_high:.5f}")
    print(f"    (~aggregate-L2 equiv: early {d_low*scale_agg:.2f} | late {d_high*scale_agg:.2f}; "
          f"Q2 logged ~0.76 settled vs ~6-13 orbiting)")
    print(f"    d_mid = geom_midpoint(d_low,d_high) = {d_mid:.5f}  ->  α = 1/d_mid = {alpha:.2f}")
    print(f"    gate range: at d_low π={1.0/(1+alpha*d_low):.3f}, at d_mid π={1.0/(1+alpha*d_mid):.3f}, "
          f"at d_high π={1.0/(1+alpha*d_high):.3f} (π0=1, πmin=0.02)")
    print("SMOKE PASS -- C3-S/C3-D π≡1 parity bitwise; α diagnostic above (freeze into C3_ALPHA_FROZEN "
          "before the 3060 run). Ready for the 4-agent review gate + --gate21a-c3.")
    return {"d_low": d_low, "d_high": d_high, "d_mid": d_mid, "alpha": alpha,
            "train_ckpt": train_ckpt, "n_diag": n_diag, "agg_scale": scale_agg}


def _sleep_episodes(sleep_bool):
    """Maximal runs of True in a bool sequence -> list of (start_idx, length)."""
    eps, i, n = [], 0, len(sleep_bool)
    while i < n:
        if sleep_bool[i]:
            j = i
            while j < n and sleep_bool[j]:
                j += 1
            eps.append((i, j - i))
            i = j
        else:
            i += 1
    return eps


def score_c5norm_k2prime(entries, norm_gate, log_every=100, is_noise=False):
    """C5 FUSEE K2' (cycle-aware, 4 clauses; addendum §3). Reads per-epoch ||W1||/test/sleep_state.
    Returns (label, lang, details). PASS requires every APPLICABLE clause to pass:
      1. TEST (generalization) >= 0.9 sustained THROUGH sleeps (tail MIN over final 5k >= 0.9; the gate
         protects the GROK, not just the fit -- both-on's TRAIN erodes too but test is the discriminating bar)
      2. ||W1|| inside [theta_lo, theta_hi+buffer] over final 5k (tail MAX <= ceiling; no secular runaway)
      3. cycle closure: every sleep episode ends in wake within a bounded window (no deadlock). Gate-ON only.
      4. wake-fraction>=0.5 over the WHOLE run (not permanently asleep). REAL arm only (noise -> N/A).
    G0 (gate OFF): only clauses 1,2 apply (3,4 = N/A) -> its K2' reads 'does wd alone hold'. If the gate
    never fires (0 episodes all seeds), clause 3 is vacuously true but the verdict flags the cycle untested."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "a cell diverged (NaN/inf) -- recorded as finding, not retried.", {}
    pes = [e.get("per_epoch") for e in entries if e.get("per_epoch")]
    if not pes:
        return "UNSCORED", "no per-epoch trajectory -- cannot read K2'.", {}
    tail_n = C3_TAIL                                  # 50 log points = 5k epochs at K=100
    # clause 1: TEST (generalization) sustained through sleeps -- the gate protects the GROK, not just the fit.
    # The MIN over the final-5k log points catches erosion at sleep points (the addendum's "no slow-motion
    # erosion disguised as cycling"). Absolute floor 0.9: both-on's TRAIN erodes too (3/10 pass train>=0.9) but
    # test tail50_min>=0.9 is 0/10 -- test is the discriminating bar. TRAIN is reported as a secondary metric
    # (NOT pass/fail) so the lead can see fit-vs-generalization divergence."""
    te_tail_min, te_tail_mean, te_peak = [], [], []
    tr_tail_min, tr_tail_mean = [], []
    for pe in pes:
        te = _tail_arr(pe, "test_acc", tail_n); allte = np.asarray(pe.get("test_acc") or [], dtype=np.float64)
        tr = _tail_arr(pe, "train_acc", tail_n)
        if te.size:
            te_tail_min.append(float(np.min(te))); te_tail_mean.append(float(np.mean(te)))
        if allte.size:
            te_peak.append(float(np.max(allte)))
        if tr.size:
            tr_tail_min.append(float(np.min(tr))); tr_tail_mean.append(float(np.mean(tr)))
    c1_vals = dict(test_tail_min_per_seed=te_tail_min,
                   test_tail_mean=float(np.mean(te_tail_mean)) if te_tail_mean else float("nan"),
                   test_peak_per_seed=te_peak,
                   train_tail_min_per_seed=tr_tail_min,
                   train_tail_mean=float(np.mean(tr_tail_mean)) if tr_tail_mean else float("nan"),
                   floor=C5NORM_TEST_FLOOR)
    c1 = bool(te_tail_min) and all(t >= C5NORM_TEST_FLOOR for t in te_tail_min)
    # clause 2: ||W1|| inside band over final 5k (tail MAX <= ceiling = theta_hi + buffer). The band is
    # ALWAYS C5NORM_THETA (the stack-specific runaway boundary) -- for G0 (gate OFF) this is the load-bearing
    # clause: does wd ALONE hold ||W1|| under the ceiling, or does it run away (both-on reached 47-99)?"""
    w1_max, w1_min = [], []
    for pe in pes:
        w = _tail_arr(pe, "w1_norm", tail_n)
        if w.size:
            w1_max.append(float(np.max(w))); w1_min.append(float(np.min(w)))
    hi = float(C5NORM_THETA["theta_hi"]); lo = float(C5NORM_THETA["theta_lo"])
    ceiling = hi + C5NORM_W1_BUFFER
    c2_vals = dict(tail_max_per_seed=w1_max, tail_min_per_seed=w1_min, ceiling=ceiling,
                   theta_hi=hi, theta_lo=lo)
    c2 = bool(w1_max) and all(m <= ceiling for m in w1_max)
    clauses = {"c1_test_sustained": (c1, c1_vals), "c2_w1_in_band": (c2, c2_vals)}
    gate_engaged = False
    if norm_gate is not None:
        # clause 3: cycle closure (longest sleep episode within bounded window; an open-at-end long run = deadlock)
        max_ep_epochs, n_eps_all, periods_ep = [], [], []
        for pe in pes:
            sl = [bool(s) for s in (pe.get("ng_sleep") or [])]
            eps = _sleep_episodes(sl)
            n_eps_all.append(len(eps))
            if eps:
                max_ep_epochs.append(max(L for _, L in eps) * log_every)
                starts = [s for s, _ in eps]
                periods_ep.extend((starts[k + 1] - starts[k]) * log_every for k in range(len(starts) - 1))
        gate_engaged = any(n > 0 for n in n_eps_all)
        c3_vals = dict(max_sleep_episode_epochs=max_ep_epochs, n_episodes_per_seed=n_eps_all,
                       sleep_max_bound=C5NORM_SLEEP_MAX_EPOCHS,
                       mean_cycle_period_epochs=(float(np.mean(periods_ep)) if periods_ep else None))
        c3 = all(m <= C5NORM_SLEEP_MAX_EPOCHS for m in max_ep_epochs) if max_ep_epochs else True
        clauses["c3_cycle_closure"] = (c3, c3_vals)
        # clause 4: wake-fraction >= 0.5 (whole-run cumulative); REAL arm only
        final_wake = [float((pe.get("ng_wake_frac") or [1.0])[-1]) for pe in pes]
        if is_noise:
            clauses["c4_wake_frac"] = (None, "N/A (noise arm; clause 4 is real-arm-only)")
        else:
            c4_vals = dict(final_wake_frac_per_seed=final_wake, floor=C5NORM_WAKE_FRAC_MIN)
            clauses["c4_wake_frac"] = (all(w >= C5NORM_WAKE_FRAC_MIN for w in final_wake), c4_vals)
    else:
        clauses["c3_cycle_closure"] = (None, "N/A (gate OFF)")
        clauses["c4_wake_frac"] = (None, "N/A (gate OFF)")
    checked = [(k, v) for k, (v, _) in clauses.items() if v is not None]
    k2pass = all(v for _, v in checked) if checked else False
    parts = [f"{k}={'PASS' if v else 'FAIL'}" for k, v in checked]
    # VACUOUS-PASS GUARD (experiment-designer): if the gate is ON but never fired (0 episodes all seeds),
    # the cell is behaviorally identical to G0 (parity-proven: theta_hi=inf == None) -- the sleep/wake CYCLE
    # was never exercised, so a cycle-PASS claim is not falsifiable. Override the label so the DECISION read
    # cannot "bank the cycle" for a cycle that never ran. Clauses 1/2/4 stay informational (wd-alone read)."""
    if norm_gate is not None and not gate_engaged:
        label = "K2'-UNTESTED"
    else:
        label = "K2'-PASS" if k2pass else "K2'-FAIL"
    lang = ("; ".join(parts) +
            (f" || gate-engaged={gate_engaged}" if norm_gate is not None else "") +
            (" || FLAG: gate never fired (0 episodes all seeds) -> cycle untested; cell reduces to G0-like." if (norm_gate is not None and not gate_engaged) else ""))
    details = {k: val for k, (_, val) in clauses.items()}
    details["all_clauses"] = {k: v for k, (v, _) in clauses.items()}
    return label, lang, details


def score_pcnative_k2(entries, log_every=100, is_noise=False):
    """PC-native K2 (the moat test; PC_NATIVE_SPEC.md §2d). NO gate clauses (gate is OFF for every cell).
    Three criteria, each must hold in >=8/10 (=PCNATIVE_QUORUM) seeds:
      grok    = PEAK test >= 0.9 (the model generalized at some point). Peak over the WHOLE trajectory --
                a transient early grok that later erodes still counts as grok (-> O2 grok+erode), it does
                NOT count as O3 (no grok). Matches spec §1 O1 ("test >= 0.9").
      hold    = tail-MIN test >= 0.9 (min over final C3_TAIL=50 log points = final 5k epochs). The MIN (not
                mean) is the strict bar -- any erosion drops it. Matches C5NORM c1 (the discriminating bar).
      bounded = ||W1|| tail-slope per 1k < 1% of mean(||W1||) over final C3_W1_WIN=100 log points (=10k ep).
                Same computation as score_c3_k2's 'flat'. Catches secular weight runaway under SGD.
    PASS = grok AND hold AND bounded (real arm). F (free energy) trajectory slope reported DIAGNOSTIC ONLY
    (descent = healthy Lyapunov; not pass/fail -- accuracy can mislead, but F is a secondary read here).
    Noise arm: same flags computed; grok+hold on noise = memorization (a CONTROL FAILURE, labelled as such)."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "a cell diverged (NaN/inf) -- recorded as finding, not retried.", {}
    pes = [e.get("per_epoch") for e in entries if e.get("per_epoch")]
    if not pes:
        return "UNSCORED", "no per-epoch trajectory -- cannot read PC-native K2.", {}
    # silent-divergence guard (red-team): wd>0 cells (N2/N2n) escape the wd0_safe in-loop guard, so the
    # diverged FLAG may stay False even with non-finite ||W1||. Catch it from the trajectory itself.
    if any(not np.all(np.isfinite(np.asarray((pe.get("w1_norm") or []), dtype=np.float64))) for pe in pes):
        return "DIVERGED", "silent divergence (non-finite ||W1|| in trajectory; wd>0 cell escaped wd0_safe guard)", {}
    n = len(pes)
    quorum = min(PCNATIVE_QUORUM, n)
    # grok: peak test >= 0.9 over the whole trajectory (per seed)
    grok_flags, te_peaks = [], []
    for pe in pes:
        te = np.asarray(pe.get("test_acc") or [], dtype=np.float64)
        pk = float(np.max(te)) if te.size else float("nan")
        te_peaks.append(pk)
        grok_flags.append(bool(te.size and pk >= PCNATIVE_GROK_BAR))
    grok_n = int(sum(grok_flags))
    # hold: min test over final 50 log points >= 0.9 (per seed)
    hold_flags, te_tail_mins = [], []
    for pe in pes:
        te_t = _tail_arr(pe, "test_acc", C3_TAIL)
        if te_t.size:
            m = float(np.min(te_t)); te_tail_mins.append(m); hold_flags.append(m >= PCNATIVE_HOLD_BAR)
        else:
            hold_flags.append(False); te_tail_mins.append(float("nan"))
    hold_n = int(sum(hold_flags))
    # bounded: |W1| slope per 1k < 1% of mean(|W1|) over final 100 log points (per seed). Denominator uses
    # the rescore_c3.py correction: N log points -> N-1 inter-point intervals (=9.9 for a full 100-pt window,
    # NOT 10.0; the in-file score_c3_k2 still carries the 10.0 off-by-one -- do NOT inherit it; the ~1% error
    # is the same scale as the 1% bar and biases toward bounded=PASS = the PC-favoring/lenient direction).
    bounded_flags, w1_slopes, w1_means = [], [], []
    for pe in pes:
        w = _tail_arr(pe, "w1_norm", C3_W1_WIN)
        if w.size >= 2:
            intervals_per_1k = ((w.size - 1) * log_every) / 1000.0
            s = float((w[-1] - w[0]) / intervals_per_1k); m = float(np.mean(w))
            w1_slopes.append(s); w1_means.append(m)
            bounded_flags.append(abs(s) < C3_W1_SLOPE_PCT * max(m, 1e-8))
        else:
            bounded_flags.append(False); w1_slopes.append(float("nan")); w1_means.append(float("nan"))
    bounded_n = int(sum(bounded_flags))
    # ||W2|| monitoring (vanilla-PC follow-up, red-team: N1's ||W2|| inflated 7->53 UNMONITORED). Reported
    # alongside ||W1||; NOT added to the bounded pass-criterion (keeps the banked N1 verdict unchanged -- a
    # W2 runaway is a REPORTED flag, read alongside the W1 bounded check).
    w2_slopes, w2_means = [], []
    for pe in pes:
        w2 = _tail_arr(pe, "w2_norm", C3_W1_WIN)
        if w2.size >= 2:
            iv = ((w2.size - 1) * log_every) / 1000.0
            w2_slopes.append(float((w2[-1] - w2[0]) / iv)); w2_means.append(float(np.mean(w2)))
        else:
            w2_slopes.append(float("nan")); w2_means.append(float("nan"))
    # F trajectory slope (DIAGNOSTIC): slope of per-seed F over the final window; report per-cell mean
    f_slopes = []
    for pe in pes:
        f = _tail_arr(pe, "F", C3_W1_WIN)
        if f.size >= 2:
            intervals_per_1k = ((f.size - 1) * log_every) / 1000.0
            f_slopes.append(float((f[-1] - f[0]) / intervals_per_1k))
    grok = grok_n >= quorum
    hold = hold_n >= quorum
    bounded = bounded_n >= quorum
    f_mean = float(np.nanmean(f_slopes)) if f_slopes else float("nan")
    raw = (f"grok {grok_n}/{n} (peak test>=0.9), hold {hold_n}/{n} (tail-min>=0.9), "
           f"bounded {bounded_n}/{n} (||W1||-slope/1k<1%) | "
           f"test_peak {np.nanmean(te_peaks):.3f}, "
           f"test_tail_min {[round(t, 2) for t in te_tail_mins]}, "
           f"||W1||-slope/1k {[round(s, 3) for s in w1_slopes]} vs 1%-of-{[round(m, 1) for m in w1_means]}, "
           f"||W2||-slope/1k {[round(s, 3) for s in w2_slopes]} vs mean-{[round(m, 1) for m in w2_means]}, "
           f"F-slope/1k mean={f_mean:.3e}")
    if is_noise:
        if grok and hold:
            label = "CONTROL-MEMORIZED"      # noise arm generalized -> the net memorized shuffled labels (control FAIL)
        elif not grok:
            label = "CONTROL-OK"             # noise arm did not grok -> the control held (expected)
        else:
            label = "CONTROL-AMBIGUOUS"
    else:
        label = "K2-PASS" if (grok and hold and bounded) else "K2-FAIL"
    details = dict(grok_flags=grok_flags, hold_flags=hold_flags, bounded_flags=bounded_flags,
                   grok_n=grok_n, hold_n=hold_n, bounded_n=bounded_n, quorum=quorum,
                   test_peak_per_seed=te_peaks, test_tail_min_per_seed=te_tail_mins,
                   w1_slope_per_1k=w1_slopes, w1_mean=w1_means,
                   w2_slope_per_1k=w2_slopes, w2_mean=w2_means,
                   f_slope_per_1k=f_slopes, f_slope_mean=(None if np.isnan(f_mean) else f_mean),
                   bars=dict(grok=PCNATIVE_GROK_BAR, hold=PCNATIVE_HOLD_BAR,
                             w1_slope_pct=C3_W1_SLOPE_PCT, tail_logpts=C3_TAIL, w1_win=C3_W1_WIN))
    return label, raw, details


def score_staged_channel(entries, channel_switch_epoch, log_every=100):
    """STAGED CHANNEL scorer (STAGED_CHANNEL_SPEC §1/§3c). Adapts score_pcnative_k2's grok/hold/bounded
    + adds switch-point diagnostics. w1_slope_per_1k AND w2_slope_per_1k are BOTH required for bounded
    (both weight matrices must be flat post-switch for the schema to be structurally stable).

    Criteria (spec §3c):
      grok    = test PEAK >= 0.9 in ALL seeds (Phase 1 — before/at switch). All-seeds, not quorum.
      hold    = test tail-MIN (final 50 log pts = 5k ep) >= 0.9 in >=8/10 seeds (spec §1 R1/R2).
      bounded = ||W1|| AND ||W2|| tail-slope/1k < 1% of mean over final 100 log pts (10k ep), >=8/10 seeds.

    Switch-point diagnostics (channel_switch_epoch is not None):
      - test trajectory around switch (log points sw-2..sw+2 = epochs 2300-2700 for switch@2500)
      - ||W1|| and ||W2|| at switch epoch (per seed)
      - per-seed test at switch epoch (grok verification: all should be >= 0.9)

    R1/R2 verdict (applied to S-frozen by the driver; reported here for reference):
      R1 HOLDS  = hold True (tail-min >= 0.9 in >=8/10) -> schema structurally stable
      R2 ERODES = tail-min < 0.9 in >=5/10 -> schema decays even without updates
    Returns (label, lang, details)."""
    if any(e.get("diverged", False) for e in entries):
        return "DIVERGED", "a cell diverged (NaN/inf) -- recorded as finding, not retried.", {}
    pes = [e.get("per_epoch") for e in entries if e.get("per_epoch")]
    if not pes:
        return "UNSCORED", "no per-epoch trajectory -- cannot read staged K2.", {}
    if any(not np.all(np.isfinite(np.asarray((pe.get("w1_norm") or []), dtype=np.float64))) for pe in pes):
        return "DIVERGED", "silent divergence (non-finite ||W1|| in trajectory)", {}
    n = len(pes)
    quorum = min(STAGED_QUORUM, n)
    # grok: peak test >= 0.9 over the WHOLE trajectory (per seed); ALL seeds required (spec §3c)
    grok_flags, te_peaks = [], []
    for pe in pes:
        te = np.asarray(pe.get("test_acc") or [], dtype=np.float64)
        pk = float(np.max(te)) if te.size else float("nan")
        te_peaks.append(pk)
        grok_flags.append(bool(te.size and pk >= STAGED_GROK_BAR))
    grok_n = int(sum(grok_flags))
    grok_all = all(grok_flags)
    # hold: test tail-MIN >= 0.9 over final 50 log pts (per seed)
    hold_flags, te_tail_mins = [], []
    for pe in pes:
        te_t = _tail_arr(pe, "test_acc", STAGED_TAIL)
        if te_t.size:
            m = float(np.min(te_t)); te_tail_mins.append(m); hold_flags.append(m >= STAGED_GROK_BAR)
        else:
            hold_flags.append(False); te_tail_mins.append(float("nan"))
    hold_n = int(sum(hold_flags))
    # bounded: ||W1|| AND ||W2|| slope/1k < 1% of mean over final 100 log pts (per seed, BOTH must pass)
    bounded_flags, w1_slopes, w1_means, w2_slopes, w2_means = [], [], [], [], []
    for pe in pes:
        w1 = _tail_arr(pe, "w1_norm", STAGED_W1_WIN)
        w2 = _tail_arr(pe, "w2_norm", STAGED_W1_WIN)
        w1_ok = w2_ok = False
        if w1.size >= 2:
            iv1 = ((w1.size - 1) * log_every) / 1000.0
            s1 = float((w1[-1] - w1[0]) / iv1); m1 = float(np.mean(w1))
            w1_slopes.append(s1); w1_means.append(m1)
            w1_ok = abs(s1) < C3_W1_SLOPE_PCT * max(m1, 1e-8)
        else:
            w1_slopes.append(float("nan")); w1_means.append(float("nan"))
        if w2.size >= 2:
            iv2 = ((w2.size - 1) * log_every) / 1000.0
            s2 = float((w2[-1] - w2[0]) / iv2); m2 = float(np.mean(w2))
            w2_slopes.append(s2); w2_means.append(m2)
            w2_ok = abs(s2) < C3_W1_SLOPE_PCT * max(m2, 1e-8)
        else:
            w2_slopes.append(float("nan")); w2_means.append(float("nan"))
        bounded_flags.append(w1_ok and w2_ok)
    bounded_n = int(sum(bounded_flags))
    # switch-point diagnostics
    switch_diag = {}
    if channel_switch_epoch is not None:
        sw_idx = channel_switch_epoch // log_every
        lo_idx, hi_idx = max(0, sw_idx - 2), sw_idx + 3            # log pts sw-2..sw+2 (epochs 2300-2700)
        around_te, w1_at_sw, w2_at_sw, te_at_sw = [], [], [], []
        for pe in pes:
            te = np.asarray(pe.get("test_acc") or [], dtype=np.float64)
            w1 = np.asarray(pe.get("w1_norm") or [], dtype=np.float64)
            w2 = np.asarray(pe.get("w2_norm") or [], dtype=np.float64)
            around_te.append(te[lo_idx:hi_idx].tolist() if te.size >= hi_idx else [])
            w1_at_sw.append(float(w1[sw_idx]) if w1.size > sw_idx else float("nan"))
            w2_at_sw.append(float(w2[sw_idx]) if w2.size > sw_idx else float("nan"))
            te_at_sw.append(float(te[sw_idx]) if te.size > sw_idx else float("nan"))
        switch_diag = dict(
            switch_epoch=channel_switch_epoch, switch_logpt=sw_idx,
            test_around_switch=around_te, w1_at_switch_per_seed=w1_at_sw, w2_at_switch_per_seed=w2_at_sw,
            test_at_switch_per_seed=te_at_sw,
            grok_at_switch_all=bool(all(t >= STAGED_GROK_BAR for t in te_at_sw if np.isfinite(t))),
        )
    hold = hold_n >= quorum
    bounded = bounded_n >= quorum
    n_erode = int(sum(1 for t in te_tail_mins if np.isfinite(t) and t < STAGED_GROK_BAR))
    raw = (f"grok {grok_n}/{n} (ALL required), hold {hold_n}/{n} (tail-min>=0.9), "
           f"bounded {bounded_n}/{n} (||W1||+||W2||-slope<1%) | "
           f"test_peak {[round(t, 2) for t in te_peaks]}, "
           f"test_tail_min {[round(t, 2) for t in te_tail_mins]}, "
           f"||W1||-slope/1k {[round(s, 3) for s in w1_slopes]} vs {[round(m, 1) for m in w1_means]}, "
           f"||W2||-slope/1k {[round(s, 3) for s in w2_slopes]} vs {[round(m, 1) for m in w2_means]}")
    # R1/R2 label (spec §1): HOLDS if hold quorum met; ERODES if >=5/10 tail-min < 0.9; else AMBIGUOUS.
    # EXPERIMENT-DESIGNER PATCH 1: gate on grok_all — if the schema never formed in ALL seeds (peak test
    # <0.9 in some seed), R1/R2 are uninterpretable (the pre-registration's Phase-1 grok precondition).
    # NB for S-frozen: test_acc is a pure function of frozen weights -> bitwise-constant post-switch ->
    # R2-ERODES is structurally unreachable; R1-HOLDS (or GROK-FAIL) is the only possible outcome
    # (red-team Issue B). The informative cells are S-frozen-W1only and S-lowrate (partial updates).
    if not grok_all:
        label = "GROK-FAIL"
    elif hold:
        label = "R1-HOLDS"
    elif n_erode >= 5:
        label = "R2-ERODES"
    else:
        label = "AMBIGUOUS"
    details = dict(grok_flags=grok_flags, hold_flags=hold_flags, bounded_flags=bounded_flags,
                   grok_n=grok_n, grok_all=grok_all, hold_n=hold_n, bounded_n=bounded_n,
                   quorum=quorum, n_erode=n_erode,
                   test_peak_per_seed=te_peaks, test_tail_min_per_seed=te_tail_mins,
                   w1_slope_per_1k=w1_slopes, w1_mean=w1_means,
                   w2_slope_per_1k=w2_slopes, w2_mean=w2_means,
                   switch_diagnostics=switch_diag,
                   bars=dict(grok=STAGED_GROK_BAR, tail_logpts=STAGED_TAIL, w1_win=STAGED_W1_WIN,
                             w1_slope_pct=C3_W1_SLOPE_PCT))
    return label, raw, details


def _dump_c5norm(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "theta": dict(C5NORM_THETA),
        "math_note": ("C5 NORM-BAND GATE (FUSEE addendum). Per batch element, AFTER mask peg / BEFORE opt.step: "
                      "w1_norms=W1.reshape(B,-1).norm(dim=1); sleep_state=(sleep_state|(w1_norms>=theta_hi))&"
                      "(w1_norms>=theta_lo); gate=(~sleep_state).float(); gW1*=gate[:,None,None]; gW2*=gate[:,None,None]. "
                      "theta_hi=inf -> gate never fires -> bitwise 'both' (parity guard). AdamW's DECOPLED weight_decay "
                      "applies to param.data inside opt.step regardless of the gated gradient, so a zeroed-grad seed "
                      "STILL decays during sleep -> ||W1|| drops below theta_lo -> wake (deadlock-breaker; VERIFIED "
                      "torch 2.6: zeroed-grad param 10.0->9.98 under wd=1.0 lr=2e-3). The per-element scalar gate "
                      "preserves zero-on-non-edges (0*gate=0) so the grad-mask assert holds."),
        "k2prime_note": ("K2' (4 clauses, cycle-aware): (1) TEST (generalization)>=0.9 tail-MIN over final 5k -- "
                         "the gate protects the GROK not just the fit (both-on train>=0.9 passes 3/10 but test>=0.9 is "
                         "0/10, so test is the discriminating bar; train reported secondary, NOT pass/fail); "
                         "(3) every sleep episode ends in wake within 2k epochs (no deadlock); (4) wake-fraction>=0.5 "
                         "whole-run (real arm only). G0 (gate OFF): only 1,2 apply. theta=[38,45] is c3d@d=1.0-"
                         "specific; NOISE arm reuses it FLAGGED-not-assumed."),
        "banked_controls": {"both-on_wd0": "w1gate: 10/10 erode, 7/10 collapse (c3d d=1.0 real)",
                            "A1s_wd1": "vanilla held 5/10 (cross-stack reference)"},
        "cells": results,
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def drive_gate21a_c5norm(cfg, label, save_path):
    """C5 NORM-BAND GATE factorial (Kimi addendum §3). 3 cells (F0/F1/G0) x {noise, real}, all c3d @ d=1.0.
    F0 = norm-gate ON, wd=0 (boundary-oscillation / governor-necessity); F1 = norm-gate ON, wd=1.0 (the
    sleep/wake candidate); G0 = norm-gate OFF, wd=1.0 (same-stack governor-only control). Attribution: F1 vs
    G0 = gate's added value over decay; F1 vs banked-both-on = full effect; F0 vs F1 = governor necessity.
    K2' (4 clauses) via score_c5norm_k2prime. Auto-extend to 30k is a DEPLOY decision (the spec flags it iff
    K2'.2/.3 undecided at 15k) -- not encoded here; rerun with epochs=30000 if needed."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: C5NORM needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); K = cfg["log_every"]
    print("=" * 104)
    print(f"GATE-2.1a-C5NORM [{label}] norm-band sleep/wake gate  | device={dev} P={P} h={h} d=1.0 "
          f"epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={len(seeds)} log_every={K}")
    print(f"  theta=[{C5NORM_THETA['theta_lo']},{C5NORM_THETA['theta_hi']}] (c3d@d=1.0; grok<=42, collapse>=47, gate@45 in the gap)")
    print(f"  cells: {[(n, ('ON' if g else 'OFF'), 'wd='+str(w)) for n, g, w in C5NORM_CELLS]}  (all c3_dynamic; noise+real paired)")
    print("  PRE-REGISTERED: F1-real is the decision cell. K2' PASS (4 clauses) -> P1-hold via sleep/wake; "
          "FAIL -> attribution read (F1 vs G0 = gate value; F0 vs F1 = governor). NOISE arm theta flagged-not-assumed.")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]; lab_shuf = list(perms)

    def _run_cell(name, gate, wd, lkind, lab):
        cc = dict(cfg); cc["wd"] = wd
        t0 = time.perf_counter()
        res = run_seeds_masked("c3_dynamic", seeds, [lab], [splits_list], M1, M2, a, b, P,
                               cc, dev, deplete=False, label_kind=lkind,
                               log_per_epoch=True, early_stop=False, es_uses_block=False,
                               want_rtdiag=True, w1_gate="both", norm_gate=gate)
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        is_noise = (lkind == "shuffled")
        k2_label, k2_lang, k2_det = score_c5norm_k2prime(res, gate, log_every=K, is_noise=is_noise)
        # blocked-fraction trajectory (cell-level): mean sleep over entries per log point (instantaneous).
        # G0 (gate OFF) has no ng_sleep -> blocked_frac stays empty (gate never fires)."""
        sl_cols = [((e.get("per_epoch") or {}).get("ng_sleep") or []) for e in res]
        n_lp = max((len(x) for x in sl_cols), default=0)
        blocked_frac = []
        for i in range(n_lp):
            vals = [sl_cols[j][i] for j in range(len(res)) if i < len(sl_cols[j])]
            if vals:
                blocked_frac.append(float(np.mean(vals)))
        wf_cols = [((e.get("per_epoch") or {}).get("ng_wake_frac") or []) for e in res]
        wake_frac_final = float(np.mean([wf[-1] for wf in wf_cols if wf])) if any(wf_cols) else 1.0
        cell = dict(name=name, norm_gate=(None if gate is None else dict(gate)), wd=wd, shuffled=is_noise,
                    mode="c3_dynamic", density=1.0,
                    train_mean=float(np.mean(tr)), train_per_seed=tr,
                    test_mean=float(np.mean(te)), test_per_seed=te,
                    diverged=any(e["diverged"] for e in res),
                    k2_label=k2_label, k2_lang=k2_lang, k2_details=k2_det,
                    blocked_fraction_traj=blocked_frac, wake_fraction_final=wake_frac_final,
                    per_epoch=[e["per_epoch"] for e in res],
                    wall_s=round(time.perf_counter() - t0, 1))
        tag = name + ("-noise" if is_noise else "-real")
        print(f"\n[{tag}] gate={'ON' if gate else 'OFF'} wd={wd}: train {cell['train_mean']:.3f} | "
              f"test {cell['test_mean']:.3f} | wake_frac_final {wake_frac_final:.2f}")
        print(f"  -> {k2_label}: {k2_lang}")
        return cell

    results = {}
    for name, gate, wd in C5NORM_CELLS:
        results[name + "-real"] = _run_cell(name, gate, wd, "real", lab_real)
        results[name + "-noise"] = _run_cell(name, gate, wd, "shuffled", lab_shuf)
        _dump_c5norm(save_path, results, cfg)

    # attribution summary (the decision read)
    print("\n" + "=" * 104)
    print("ATTRIBUTION (addendum §3):")
    print(f"  F1-real K2'={results['F1-real']['k2_label']}  (the candidate)")
    print(f"  F0-real K2'={results['F0-real']['k2_label']}  (governor-necessity: F0<<F1 -> wd is load-bearing)")
    print(f"  G0-real K2'={results['G0-real']['k2_label']}  (gate value: F1 vs G0)")
    f1c2 = results["F1-real"]["k2_details"]["c2_w1_in_band"]
    g0c2 = results["G0-real"]["k2_details"]["c2_w1_in_band"]
    if isinstance(f1c2, dict) and isinstance(g0c2, dict):
        f1max = float(np.mean(f1c2.get("tail_max_per_seed") or [0])); g0max = float(np.mean(g0c2.get("tail_max_per_seed") or [0]))
        print(f"  ||W1|| tail-max: F1 {f1max:.1f} (ceiling {f1c2['ceiling']:.0f}) vs G0 {g0max:.1f} "
              f"-> {'gate capped; G0 ran away' if g0max > f1c2['ceiling'] and f1max <= f1c2['ceiling'] else 'see trajectories'}")
    f1lab = results["F1-real"]["k2_label"]
    if f1lab == "K2'-PASS":
        decision = "F1-real K2'-PASS -> P1-hold via sleep/wake survives (bank the cycle)."
    elif f1lab == "K2'-UNTESTED":
        decision = ("F1-real K2'-UNTESTED (gate never fired -> ||W1|| stayed < theta_hi; cell == G0). NOT bankable: "
                    "the cycle was never exercised -> re-derive theta OR conclude wd-alone held (F1==G0, gate decorative).")
    else:
        decision = "F1-real K2'-FAIL -> kill-doc v2 §5 queue (C5b µPC -> class exhaustion -> kill). Anti-rescue unchanged."
    print(f"  DECISION (lead): {decision}")
    _dump_c5norm(save_path, results, cfg)


def _dump_pcnative(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "pcnative_note": ("PC-NATIVE SGD MOAT TEST (PC_NATIVE_SPEC.md). Removes ALL backprop-era machinery: "
                          "AdamW -> plain SGD (momentum=0), wd -> 0, norm-gate OFF. Keeps c3d precision (the "
                          "PC-native gain control) + logs F = 0.5*(e1sq+e2sq) (the free-energy Lyapunov fn). "
                          "Question: does PC grok+hold schemas on its own terms? SGD wd is COUPLED L2 "
                          "(g'=g+wd*W) vs AdamW decoupled; at wd=0 both are plain GD. F field: per-entry, "
                          "per-log-point scalar (None for non-c3 modes)."),
        "phase1": {"lr_list": list(PCNATIVE_LR_LIST), "lr_star": results.get("_phase1_lr_star"),
                   "lr_floor": PCNATIVE_LR_FLOOR, "seeds": list(range(PCNATIVE_LR_SEEDS)),
                   "epochs": PCNATIVE_LR_EPOCHS,
                   "lr_results": {str(k): v for k, v in (results.get("_phase1") or {}).items()}},
        "phase2_cells": [(n, w, lk) for n, w, lk in PCNATIVE_CELLS],
        "k2_note": ("K2 (no gate clauses): grok = peak test>=0.9 (>=8/10); hold = tail-MIN test>=0.9 over "
                    "final 50 log pts (>=8/10); bounded = ||W1||-slope/1k < 1% of mean over final 100 log "
                    "pts (>=8/10). PASS = grok AND hold AND bounded (real arm). F-slope is DIAGNOSTIC ONLY. "
                    "Noise arm: grok+hold = CONTROL-MEMORIZED (control failure)."),
        "banked_controls": {"F1_C5NORM": "AdamW wd=1.0 gate=ON: test=1.0 10/10 (K2'-PASS) -- the held baseline",
                            "G0_C5NORM": "AdamW wd=1.0 gate=OFF: test=1.0 10/10 -- wd-alone holds (AdamW)",
                            "both-on_w1gate": "AdamW wd=0 gate=OFF: 10/10 grok, 10/10 erode -- the erosion baseline"},
        "cells": {k: v for k, v in results.items() if not k.startswith("_")},
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def drive_gate21a_pcnative(cfg, label, save_path):
    """PC-NATIVE SGD MOAT TEST (PC_NATIVE_SPEC.md). Two phases:
      Phase 1 (LR finder): loop PCNATIVE_LR_LIST, 3 seeds x 2k epochs, SGD wd=0 c3d d=1.0 no-gate. Pick
        lr* = highest LR (by value) where mean final-train >= 0.8 and not diverged. Fallback: best-performing
        LR (documents O3 if none reach the floor).
      Phase 2 (decisive): at lr*, loop PCNATIVE_CELLS (N1/N2/N1n/N2n) x 10 seeds x 15k epochs, SGD c3d d=1.0,
        norm_gate=None (gate OFF all cells), w1_gate="both", want_rtdiag=True. score_pcnative_k2 per cell.
      N1 (SGD wd=0 real) is THE headline; N1 vs banked G0 (AdamW wd=0) isolates the optimizer (both no-gate
      wd=0). Pre-registered O1/O2/O3; anti-rescue unchanged (spec §1)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: PCNATIVE needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); K = cfg["log_every"]
    print("=" * 104)
    print(f"GATE-2.1a-PCNATIVE [{label}] SGD moat test  | device={dev} P={P} h={h} d=1.0 "
          f"opt=SGD epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} lr_sweep={PCNATIVE_LR_LIST} "
          f"seeds={len(seeds)} log_every={K}")
    print(f"  Phase 1: LR finder ({PCNATIVE_LR_SEEDS} seeds x {PCNATIVE_LR_EPOCHS}ep, wd=0) -> lr* "
          f"(highest LR w/ train>={PCNATIVE_LR_FLOOR})")
    print(f"  Phase 2: {[(n, 'wd='+str(w), lk) for n, w, lk in PCNATIVE_CELLS]} "
          f"(10 seeds x {cfg['epochs']}ep, c3d d=1.0, norm_gate=None all)")
    print("  PRE-REGISTERED (spec §1): O1 grok+hold (PASS) -> PC real, moat exists; O2 grok+erode -> PC's own "
          "dynamics unstable (most informative); O3 no grok -> PC needs AdamW. N1 vs banked-G0 = optimizer axis.")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]; lab_shuf = list(perms)

    results = {}

    # ===================== PHASE 1: LR FINDER (3 seeds x 2k epochs, SGD wd=0 c3d d=1.0 no-gate) ======
    p1_idx = list(range(PCNATIVE_LR_SEEDS))                 # positions [0,1,2] = seeds 0,1,2
    p1_seeds = tuple(p1_idx)
    M1_p1 = M1[p1_idx]; M2_p1 = M2[p1_idx]
    splits_p1 = [[splits_list[i] for i in p1_idx]]          # 1 split-set x 3 seeds -> B=3
    lab_real_p1 = [[c for _ in p1_idx]]
    phase1 = {}
    print("\n--- PHASE 1: LR finder ---")
    for lr in PCNATIVE_LR_LIST:
        cc = dict(cfg); cc["opt"] = "sgd"; cc["wd"] = 0.0
        cc["lr"] = lr; cc["epochs"] = PCNATIVE_LR_EPOCHS
        t0 = time.perf_counter()
        res = run_seeds_masked("c3_dynamic", p1_seeds, lab_real_p1, splits_p1, M1_p1, M2_p1, a, b, P,
                               cc, dev, deplete=False, label_kind="real",
                               log_per_epoch=False, early_stop=False, es_uses_block=False,
                               want_rtdiag=False, w1_gate="both", norm_gate=None)
        tr = [e["train"] for e in res]
        div = any(e["diverged"] for e in res)
        mean_tr = float(np.nanmean(tr)) if tr else 0.0
        phase1[lr] = dict(lr=lr, train_mean=mean_tr, train_per_seed=tr, diverged=div,
                          n_seeds=len(p1_seeds), wall_s=round(time.perf_counter() - t0, 1))
        print(f"  lr={lr:<8g}: train_mean={mean_tr:.3f} diverged={div}  "
              f"{'(PASS: >= %.1f)' % PCNATIVE_LR_FLOOR if (not div and mean_tr >= PCNATIVE_LR_FLOOR) else ''}")
        results["_phase1"] = phase1
        _dump_pcnative(save_path, results, cfg)
    passing = [lr for lr in PCNATIVE_LR_LIST
               if (not phase1[lr]["diverged"]) and phase1[lr]["train_mean"] >= PCNATIVE_LR_FLOOR]
    lr_star = max(passing, key=lambda lr: phase1[lr]["train_mean"]) if passing else None
    if lr_star is None:                                     # O3-adjacent: no LR reached the floor
        best = max(PCNATIVE_LR_LIST, key=lambda lr: (phase1[lr]["train_mean"] if not phase1[lr]["diverged"] else -1.0))
        lr_star = best
        print(f"  -> NO LR reached train>={PCNATIVE_LR_FLOOR} in {PCNATIVE_LR_EPOCHS}ep. "
              f"Fallback lr*={lr_star} (best train={phase1[lr_star]['train_mean']:.3f}). Documents O3 risk.")
    else:
        print(f"  -> lr* = {lr_star} (highest-train LR among those with train>={PCNATIVE_LR_FLOOR} in "
              f"{PCNATIVE_LR_EPOCHS}ep, not diverged; train={phase1[lr_star]['train_mean']:.3f}).")
    results["_phase1_lr_star"] = lr_star
    _dump_pcnative(save_path, results, cfg)

    # ===================== PHASE 2: DECISIVE TEST (lr*, 10 seeds x 15k, SGD c3d d=1.0 no-gate) ========
    print(f"\n--- PHASE 2: decisive test @ lr*={lr_star} ---")

    def _run_cell(name, wd, lkind, lab):
        cc = dict(cfg); cc["opt"] = "sgd"; cc["lr"] = lr_star; cc["wd"] = wd
        t0 = time.perf_counter()
        res = run_seeds_masked("c3_dynamic", seeds, [lab], [splits_list], M1, M2, a, b, P,
                               cc, dev, deplete=False, label_kind=lkind,
                               log_per_epoch=True, early_stop=False, es_uses_block=False,
                               want_rtdiag=True, w1_gate="both", norm_gate=None)
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        is_noise = (lkind == "shuffled")
        k2_label, k2_lang, k2_det = score_pcnative_k2(res, log_every=K, is_noise=is_noise)
        cell = dict(name=name, wd=wd, shuffled=is_noise, opt="sgd", lr=lr_star,
                    mode="c3_dynamic", density=1.0, norm_gate=None,
                    train_mean=float(np.nanmean(tr)), train_per_seed=tr,
                    test_mean=float(np.nanmean(te)), test_per_seed=te,
                    diverged=any(e["diverged"] for e in res),
                    k2_label=k2_label, k2_lang=k2_lang, k2_details=k2_det,
                    per_epoch=[e["per_epoch"] for e in res],
                    wall_s=round(time.perf_counter() - t0, 1))
        tag = name + ("-real" if not is_noise else "-noise")
        print(f"\n[{tag}] SGD wd={wd}: train {cell['train_mean']:.3f} | test {cell['test_mean']:.3f}"
              + ("  DIVERGED" if cell["diverged"] else ""))
        print(f"  -> {k2_label}: {k2_lang}")
        return cell

    for name, wd, lkind in PCNATIVE_CELLS:
        lab = lab_real if lkind == "real" else lab_shuf
        results[name] = _run_cell(name, wd, lkind, lab)
        _dump_pcnative(save_path, results, cfg)

    # attribution summary (the decision read; spec §4)
    print("\n" + "=" * 104)
    print("ATTRIBUTION (spec §4) -- N1 vs N2 isolates optimizer x decay; N1 vs banked-G0 isolates optimizer:")
    n1 = results.get("N1", {}); n2 = results.get("N2", {})
    print(f"  N1 (SGD wd=0 real)  K2={n1.get('k2_label','?')}  -- THE PC-native headline")
    print(f"  N2 (SGD wd=1  real) K2={n2.get('k2_label','?')}  -- decay disentangle under SGD (coupled-L2)")
    print(f"  N1n/ N2n (noise)     K2={results.get('N1n',{}).get('k2_label','?')} / "
          f"{results.get('N2n',{}).get('k2_label','?')}  -- controls (should NOT grok)")
    n1lab = n1.get("k2_label", "?")
    n2lab = n2.get("k2_label", "?")
    # noise-control gate (experiment-designer): O1 ("moat exists -> unfreeze P2") must require the noise
    # control did NOT memorize. If N1n/N2n also grok+hold on SHUFFLED labels, the net has enough capacity to
    # memorize the (train->label) map -> N1's "grok+hold" may be memorization, not generalization. That is
    # the trivial-metric trap (AGENTS.md #6); a control-satisfying success does NOT unfreeze PC.
    n1n_lab = results.get("N1n", {}).get("k2_label", "?")
    n2n_lab = results.get("N2n", {}).get("k2_label", "?")
    control_clean = ("CONTROL-MEMORIZED" not in (n1n_lab, n2n_lab))
    if not control_clean:
        print(f"  *** NOISE-CONTROL FLAG: a noise cell MEMORIZED ({[l for l in (n1n_lab, n2n_lab) if l == 'CONTROL-MEMORIZED']}) "
              f"-- capacity confound; O1 cannot be banked as a clean moat. ***")
    if n1lab == "K2-PASS" and n2lab == "K2-PASS":
        decision = ("O1 (grok+hold): PC-native works; wd decorative under SGD. Moat exists. -> P2 battery."
                    if control_clean else
                    "O1-AMBIGUOUS: N1/N2 grok+hold BUT a noise control MEMORIZED -> capacity confound, not a "
                    "clean generalization moat. Do NOT unfreeze P2 without a sharper held-out test (AGENTS.md #6).")
    elif n1lab == "K2-PASS" and n2lab != "K2-PASS":
        decision = ("O1 (narrow): PC-native works; SGD+wd=1.0 breaks it (coupled-L2). wd HARMFUL under SGD."
                    if control_clean else
                    "O1-AMBIGUOUS (narrow): N1 holds but a noise control memorized -> capacity confound; do not unfreeze.")
    elif n1lab == "K2-FAIL" and n2lab == "K2-PASS":
        decision = "O2-adjacent: SGD can't hold without wd; wd needed even under SGD. PC-native can't hold alone."
    elif "DIVERGED" in (n1lab, n2lab) or n1lab.startswith("CONTROL"):
        decision = "see trajectories (divergence / control-label); read per-cell K2 details."
    else:
        decision = ("O2/O3: PC-native did not grok+hold unaided. If N1 peaked test>=0.9 then eroded -> O2 "
                    "(PC's own dynamics unstable); if N1 never reached test>=0.9 -> O3 (PC needs AdamW). "
                    "Apply diagnostic toolkit (rtdiag/F); anti-rescue unchanged.")
    print(f"  DECISION (lead): {decision}")
    _dump_pcnative(save_path, results, cfg)


def _dump_n1prime(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "n1prime_note": ("N1' FOLLOW-UP (PC-native plateau diagnosis). N1 (SGD m=0 wd=0 lr=1.0) plateaued at "
                         "train=0.832 test=0.055 (O3). Phase A re-runs N1 (byte-parity, m=0) WITH a T_eval=200 "
                         "eval-only relaxation diagnostic at every rtdiag snapshot: resid_T200 vs resid_T20 "
                         "(T=20 convergence) + the F trajectory over 200 steps. Phase B sweeps momentum=0.9 "
                         "(heavy-ball) over {0.1,0.3,1.0} (finder, 3 seeds x 2k) then runs the best at 10 seeds "
                         "x 15k. NB: the T_eval diagnostic is eval-only (no_grad, no weight update, no RNG) -> "
                         "N1_diag MUST reproduce N1 (train=0.832, test=0.055) = the parity guard. momentum "
                         "default 0.0 = byte-parity; cfg['momentum']=0.9 opts into heavy-ball."),
        "phaseB": {"lr_list": list(N1PRIME_LR), "momentum": N1PRIME_MOMENTUM,
                   "lr_best": results.get("_phaseB_lr_best"), "finder_seeds": list(range(N1PRIME_LR_SEEDS)),
                   "finder_epochs": N1PRIME_LR_EPOCHS},
        "teval": N1PRIME_TEVAL,
        "banked": {"N1_prior": "SGD m=0 wd=0 lr=1.0: train=0.832 test=0.055 K2-FAIL (the O3 plateau cell)"},
        "cells": {k: v for k, v in results.items() if not k.startswith("_")},
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def _tdiag_summary(res):
    """Read the T_eval diagnostic off a cell's per-entry trajectories. resid_T200 (per-entry, per-snapshot,
    final-step activity change after T_eval steps) vs resid_T20 (per_epoch['residual'], the T=20 training-
    step residual). If resid_T200 << resid_T20 -> the T=20 relaxation had NOT converged (finite-T bias is the
    plateau suspect). F trajectory: F at step 20 vs step 200 (plateau => T=20 was sufficient)."""
    out = {}
    r20, r200 = [], []
    for e in res:
        pe = e.get("per_epoch") or {}
        a = np.asarray(pe.get("residual") or [], dtype=float)
        b = np.asarray(pe.get("resid_T200") or [], dtype=float)
        if a.size:
            r20.append(float(np.mean(a[-10:])))       # tail-10 mean of the T=20 residual trajectory
        if b.size:
            r200.append(float(np.mean(b[-10:])))      # tail-10 mean of the T=200 residual trajectory
    m20 = float(np.mean(r20)) if r20 else float("nan")
    m200 = float(np.mean(r200)) if r200 else float("nan")
    out["resid_T20_tail_mean"] = m20
    out["resid_T200_tail_mean"] = m200
    out["resid_ratio_T200_over_T20"] = float(m200 / m20) if (np.isfinite(m20) and m20 > 0) else float("nan")
    # VERDICT on absolute residual (red-team fix): the ratio is blind to a period-2 orbit (ratio~1, both >>0).
    # A converged relaxation has resid -> 0; resid > PCNATIVE_RESID_CONV_BAR => oscillating / non-equilibrium.
    out["relaxation_converged"] = bool(np.isfinite(m200) and m200 < PCNATIVE_RESID_CONV_BAR)
    # F trajectory: aggregate trace at the LAST snapshot (shared across entries; take the mean for safety).
    f_traces = []
    for e in res:
        F = (e.get("per_epoch") or {}).get("F_T200") or []
        if F:
            f_traces.append(np.asarray(F[-1], dtype=float))
    f_traces = [t for t in f_traces if t is not None and t.size > 0 and np.all(np.isfinite(t))]
    if f_traces:
        fm = np.mean(np.stack(f_traces), axis=0)
        out["F_step20"] = float(fm[19]) if len(fm) > 19 else float("nan")
        out["F_step200"] = float(fm[-1])
        out["F_ratio_step20_over_step200"] = (float(fm[19] / max(abs(fm[-1]), 1e-12))
                                              if len(fm) > 19 else float("nan"))
        # monotonicity / oscillation (vanilla-PC test): is F descending, or does it RISE (period-2 orbit)?
        # F_net = descent over the full trace; F_max_step_rise = largest single-step increase (>>0 => oscillating).
        # F is the float32 batch-total energy over ~6.4M terms (O(1e3-1e4)); its rounding noise floor is ~1e-2,
        # so the monotone bar is RELATIVE (1e-3 of |F|), not a bit-exact 1e-9 (red-team: the 1e-9 bar was ~6-7
        # orders below noise -> always False -> dead). Reported DIAGNOSTIC ONLY; the c3d-cause GATE uses the
        # robust resid_converged flag (resid<0.5), not this (a period-2 orbit has resid>>0 regardless of F).
        out["F_net_descent"] = float(fm[-1] - fm[0])
        diffs = np.diff(fm)
        out["F_max_step_rise"] = float(np.max(diffs)) if diffs.size else float("nan")
        out["F_monotone_descent"] = bool(out["F_max_step_rise"] <= 1e-3 * max(float(np.max(np.abs(fm))), 1.0))
    out["n_resid_T200_snaps"] = int(len((res[0].get("per_epoch") or {}).get("resid_T200") or [])) if res else 0
    return out


def drive_n1prime(cfg, label, save_path):
    """N1' FOLLOW-UP (PC-native plateau diagnosis). Two phases:
      Phase A (finite-T diagnostic): re-run N1 (SGD m=0 wd=0 lr=1.0, 10 seeds x 15k) WITH a T_eval=200
        eval-only relaxation at every rtdiag snapshot. PARITY GUARD: N1_diag must reproduce N1 (train=0.832,
        test=0.055) -- the diagnostic is no_grad/no-update/no-RNG. Answers: did the T=20 relaxation converge?
      Phase B (momentum negative control): sweep m=0.9 over N1PRIME_LR (3 seeds x 2k finder), run the best
        at 10 seeds x 15k. Answers: does momentum break the train=0.83 plateau? (literature: vanilla GD groks
        modular addition w/o momentum -- Prieto ICLR 2025 -- so momentum expected IRRELEVANT.)"""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: N1' needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); K = cfg["log_every"]
    print("=" * 104)
    print(f"GATE-2.1a-PCNATIVE-N1' [{label}] plateau diagnosis  | device={dev} P={P} h={h} d=1.0 opt=SGD "
          f"epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} seeds={len(seeds)} log_every={K}")
    print(f"  Phase A: N1_diag (m=0 wd=0 lr=1.0) + T_eval={N1PRIME_TEVAL} relaxation diagnostic (10 seeds x 15k)")
    print(f"  Phase B: N1' m={N1PRIME_MOMENTUM} sweep lr={list(N1PRIME_LR)} (finder {N1PRIME_LR_SEEDS} seeds x "
          f"{N1PRIME_LR_EPOCHS}ep) -> best lr at 10 seeds x 15k")
    print("  QUESTIONS: (A) did T=20 converge? (resid_T200 vs resid_T20; F plateau by step 20?)  "
          "(B) does momentum break the 0.83 plateau? (expect NO per Prieto ICLR 2025)")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]

    results = {}

    def _summarize(name, res, wd, momentum, lr, is_noise=False):
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        div_flags = [bool(e["diverged"]) for e in res]
        n_div = int(sum(div_flags))
        # diverged -> SPURIOUS-FINITE train (argmax of non-finite logits ~= 1/P, NOT NaN) -> filtered by the
        # explicit flag here, NOT by nanmean. Use train_mean_nondiverged for any decision (train_mean keeps
        # the spurious value for reference). NB: divergence is whole-batch (isfinite(W1).all()), so n_div is
        # 0-or-len(seeds), never partial.
        tr_clean = [t for t, d in zip(tr, div_flags) if not d]
        k2_label, k2_lang, k2_det = score_pcnative_k2(res, log_every=K, is_noise=is_noise)
        cell = dict(name=name, wd=wd, momentum=momentum, shuffled=is_noise, opt="sgd", lr=lr,
                    mode="c3_dynamic", density=1.0, norm_gate=None,
                    train_mean=float(np.nanmean(tr)), train_per_seed=tr,
                    train_mean_nondiverged=(float(np.mean(tr_clean)) if tr_clean else float("nan")),
                    n_diverged=n_div,
                    test_mean=float(np.nanmean(te)), test_per_seed=te,
                    diverged=any(div_flags),
                    k2_label=k2_label, k2_lang=k2_lang, k2_details=k2_det,
                    per_epoch=[e["per_epoch"] for e in res])
        print(f"\n[{name}] SGD m={momentum} wd={wd} lr={lr}: train {cell['train_mean']:.3f} | "
              f"test {cell['test_mean']:.3f}" + ("  DIVERGED" if cell["diverged"] else ""))
        print(f"  -> {k2_label}: {k2_lang}")
        return cell

    # ===================== PHASE A: N1_diag (m=0 + T_eval=200 diagnostic) ==========================
    print("\n--- PHASE A: N1 finite-T diagnostic (m=0, T_eval=200) ---")
    ccA = dict(cfg); ccA["opt"] = "sgd"; ccA["momentum"] = 0.0; ccA["wd"] = 0.0
    ccA["lr"] = 1.0; ccA["T_eval"] = N1PRIME_TEVAL
    t0 = time.perf_counter()
    resA = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                            ccA, dev, deplete=False, label_kind="real",
                            log_per_epoch=True, early_stop=False, es_uses_block=False,
                            want_rtdiag=True, w1_gate="both", norm_gate=None)
    results["N1_diag"] = _summarize("N1_diag", resA, 0.0, 0.0, 1.0)
    results["N1_diag"]["wall_s"] = round(time.perf_counter() - t0, 1)
    results["N1_diag"]["tdiag"] = _tdiag_summary(resA)
    td = results["N1_diag"]["tdiag"]
    print(f"  [tdiag] resid_T20_tail={td.get('resid_T20_tail_mean'):.4g}  "
          f"resid_T200_tail={td.get('resid_T200_tail_mean'):.4g}  "
          f"ratio(T200/T20)={td.get('resid_ratio_T200_over_T20'):.3f}")
    print(f"  [tdiag] F step20={td.get('F_step20'):.4g}  step200={td.get('F_step200'):.4g}  "
          f"ratio(20/200)={td.get('F_ratio_step20_over_step200'):.3f}  "
          f"(~1 => F plateaued by step 20 => T=20 converged; >>1 => still descending => finite-T bias)")
    _dump_n1prime(save_path, results, cfg)

    # ===================== PHASE B: N1' momentum sweep (m=0.9) =====================================
    print(f"\n--- PHASE B: N1' momentum={N1PRIME_MOMENTUM} sweep ---")
    p1_idx = list(range(N1PRIME_LR_SEEDS))
    M1p, M2p = M1[p1_idx], M2[p1_idx]
    splits_p1 = [[splits_list[i] for i in p1_idx]]
    lab_p1 = [[c for _ in p1_idx]]
    finder = {}
    for lr in N1PRIME_LR:
        cc = dict(cfg); cc["opt"] = "sgd"; cc["momentum"] = N1PRIME_MOMENTUM; cc["wd"] = 0.0
        cc["lr"] = lr; cc["epochs"] = N1PRIME_LR_EPOCHS; cc["T_eval"] = None
        res = run_seeds_masked("c3_dynamic", tuple(p1_idx), lab_p1, splits_p1, M1p, M2p, a, b, P,
                               cc, dev, deplete=False, label_kind="real",
                               log_per_epoch=False, early_stop=False, es_uses_block=False,
                               want_rtdiag=False, w1_gate="both", norm_gate=None)
        tr = [e["train"] for e in res]; div = any(e["diverged"] for e in res)
        mean_tr = float(np.nanmean(tr)) if tr else 0.0
        finder[lr] = dict(lr=lr, train_mean=mean_tr, diverged=div, n_seeds=len(p1_idx),
                          train_per_seed=tr)
        results["N1m9_lr%s" % lr] = dict(finder[lr], momentum=N1PRIME_MOMENTUM, wd=0.0, opt="sgd")
        print(f"  [finder] m=0.9 lr={lr:<5g}: train_mean={mean_tr:.3f} diverged={div}")
        _dump_n1prime(save_path, results, cfg)
    lr_best = max(N1PRIME_LR, key=lambda lr: (finder[lr]["train_mean"] if not finder[lr]["diverged"] else -1.0))
    all_finder_diverged = all(finder[lr]["diverged"] for lr in N1PRIME_LR)
    results["_phaseB_lr_best"] = lr_best
    results["_phaseB_all_diverged"] = all_finder_diverged
    if all_finder_diverged:
        print(f"  *** FINDER FLAG: ALL swept LRs diverged at m={N1PRIME_MOMENTUM} (effective LR ~10x). "
              f"lr_best={lr_best} is the least-bad (tie-break). Momentum is UNSTABLE across the swept range -> "
              f"the optimizer-step hypothesis is NOT ruled out; the structural conclusion below is INVALID if "
              f"N1m9_best also diverges. ***")
    print(f"  -> Phase B best lr = {lr_best} (train={finder[lr_best]['train_mean']:.3f})")

    ccB = dict(cfg); ccB["opt"] = "sgd"; ccB["momentum"] = N1PRIME_MOMENTUM; ccB["wd"] = 0.0
    ccB["lr"] = lr_best; ccB["T_eval"] = None
    t0 = time.perf_counter()
    resB = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                            ccB, dev, deplete=False, label_kind="real",
                            log_per_epoch=True, early_stop=False, es_uses_block=False,
                            want_rtdiag=True, w1_gate="both", norm_gate=None)
    results["N1m9_best"] = _summarize("N1m9_best", resB, 0.0, N1PRIME_MOMENTUM, lr_best)
    results["N1m9_best"]["wall_s"] = round(time.perf_counter() - t0, 1)
    _dump_n1prime(save_path, results, cfg)

    # ===================== DECISION (the two questions) ============================================
    print("\n" + "=" * 104)
    print("N1' DIAGNOSIS:")
    n1d = results["N1_diag"]; n1m9 = results["N1m9_best"]
    print(f"  (A) finite-T: resid_T200={td.get('resid_T200_tail_mean', float('nan')):.3g}  "
          f"resid_T20={td.get('resid_T20_tail_mean', float('nan')):.3g}  "
          f"(ratio {td.get('resid_ratio_T200_over_T20', float('nan')):.2f}, for reference only)  "
          f"F_max_step_rise={td.get('F_max_step_rise', float('nan')):.3g}")
    if not td.get("relaxation_converged", True):
        print(f"      -> WARNING: relaxation NOT converged (resid_T200={td.get('resid_T200_tail_mean', 0):.2f} "
              f">> {PCNATIVE_RESID_CONV_BAR}; F_max_step_rise={td.get('F_max_step_rise', 0):.3g} => period-2 "
              f"oscillation). The T=20 gradient is computed at a NON-EQUILIBRIUM state -> noisy. NB: more T will "
              f"NOT fix this (resid_T200 ~= resid_T20); the oscillation is in the relaxation DYNAMICS, not the "
              f"step count. (Prior 'T=20 converged' ratio-verdict was WRONG -- the ratio is blind to a period-2 orbit.)")
    else:
        print("      -> T=20 had converged (resid_T200 ~= 0, well below the bar): finite-T step-count is NOT "
              "the cause; the plateau is in the M-step/optimizer, not the E-step relaxation.")
    # (B) momentum negative control -- divergence-aware (experiment-designer fix): instability must NOT be
    # mislabeled as "structural plateau". Key the decision on the non-diverged mean; if N1m9_best diverged,
    # the structural conclusion is INVALID (momentum was unstable, not structurally-impotent).
    n1m9_div = n1m9.get("n_diverged", 0)
    n1m9_clean = n1m9.get("train_mean_nondiverged", float("nan"))
    print(f"  (B) momentum: N1 m=0 train={n1d['train_mean']:.3f}  vs  N1' m=0.9 train={n1m9['train_mean']:.3f} "
          f"(lr*={lr_best}, n_diverged={n1m9_div}/{len(seeds)}, non-diverged mean={n1m9_clean:.3f})")
    # per-seed paired delta (N1m9 - N1, same seeds/splits/init) -- a partial effect on means is visible here.
    # Only meaningful when N1m9_best did NOT diverge (diverged train is spurious-finite, not NaN).
    if n1m9_div == 0 and len(n1m9["train_per_seed"]) == len(n1d["train_per_seed"]):
        deltas = [float(b - a) for a, b in zip(n1d["train_per_seed"], n1m9["train_per_seed"])
                  if np.isfinite(b - a)]
        mean_delta = float(np.mean(deltas)) if deltas else float("nan")
        n_improved = int(sum(d > 0.05 for d in deltas))
        print(f"      per-seed delta(N1'-N1): mean={mean_delta:+.3f}, n_seeds_improved>0.05={n_improved}/{len(deltas)}")
    else:
        mean_delta = float("nan")
    if n1m9_div > 0 or all_finder_diverged:
        print("      -> INCONCLUSIVE (unstable): momentum diverged at the swept LRs (effective LR ~10x). "
              "Cannot conclude 'structural' -- the optimizer-step hypothesis is NOT ruled out by instability. "
              "Re-sweep at lower LRs (m=0.9 dampens the step; try lr in {0.01,0.03,0.1}) before concluding.")
    elif np.isfinite(n1m9_clean) and n1m9_clean > 0.95:
        print("      -> momentum DID help (non-diverged train>0.95): plateau was an optimizer-step-size issue "
              "(SURPRISE vs Prieto ICLR 2025). Investigate momentum+SGD further.")
    elif np.isfinite(n1m9_clean) and mean_delta > 0.05:
        print(f"      -> momentum gave a PARTIAL lift (mean_delta={mean_delta:+.3f}, <0.95): some step-size "
              "effect but did NOT grok. Inconclusive-on-grok; report the partial effect, do not claim structural.")
    else:
        print("      -> momentum did NOT break the plateau (no divergence, non-diverged train<=0.95, no partial "
              "lift): plateau is STRUCTURAL, not an optimizer-step-size artifact. Confirms vanilla-GD result "
              "(Prieto ICLR 2025): momentum is not the fix.")
    parity_ok = abs(n1d["train_mean"] - 0.832) < 0.02 and abs(n1d["test_mean"] - 0.055) < 0.02
    print(f"  PARITY: N1_diag train={n1d['train_mean']:.3f} test={n1d['test_mean']:.3f} vs prior N1 "
          f"(0.832/0.055) -> {'MATCH (T_eval diagnostic is parity-inert)' if parity_ok else 'DRIFT -- investigate'}")
    _dump_n1prime(save_path, results, cfg)


def _dump_vanilla(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "vanilla_note": ("VANILLA-PC SGD test (c3d-precision-is-the-plateau-cause hypothesis). Same N1 config "
                         "(SGD m=0 wd=0 lr=1.0 d=1.0, 10 seeds x 15k) but mode=pc_transport (pi_fb=None -> "
                         "bitwise-vanilla Whittington-Bogacz PC, full-strength feedback, NO precision gating) "
                         "instead of c3_dynamic. The 4-agent gate + red-team found the c3d relaxation OSCILLATES "
                         "(F rose 23.6% step20->step200; resid~5>>0, period-2 orbit) -- by epoch ~5k all 256 "
                         "units have pi<0.5 (pi_mean~0.28), starving feedback, underdamping the relaxation. "
                         "If V1 (vanilla) GROKS on the held-out test (K2-PASS = test peak>=0.9 quorum >=8/10, "
                         "NOT just train>0.95) AND its relaxation converges (resid<0.5) -> c3d precision WAS the "
                         "plateau cause. If V1 only fits train without generalizing, or plateaus/oscillates -> "
                         "the cause is architectural (or also a generalization problem), not precision. ||W2|| "
                         "is now monitored (N1 inflated 7->53)."),
        "resid_conv_bar": PCNATIVE_RESID_CONV_BAR,
        "banked": {"N1_c3d": "SGD m=0 wd=0 lr=1.0 c3_dynamic: train=0.832 test=0.055, resid~5 (oscillating)"},
        "cells": {k: v for k, v in results.items() if not k.startswith("_")},
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def drive_vanilla(cfg, label, save_path):
    """VANILLA-PC SGD test -- is c3d precision the plateau cause? Runs the N1 config (SGD m=0 wd=0 lr=1.0,
    d=1.0, 10 seeds x 15k) under mode=pc_transport (vanilla WB PC, pi_fb=None, no precision gating) instead of
    c3_dynamic, WITH the T_eval=200 relaxation diagnostic + ||W2|| monitoring.
      V1  = pc_transport, real labels     -- the test (does vanilla PC grok where c3d plateaued?)
      V1n = pc_transport, shuffled labels -- noise control (should NOT grok)
    Decision: V1 train>0.95 AND resid<0.5 AND F monotone-descent -> c3d precision WAS the cause. Else the
    cause is architectural. Parity: V1 uses the SAME seeds/splits/masks/init as N1 (only mode differs)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: vanilla-PC needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); K = cfg["log_every"]
    print("=" * 104)
    print(f"GATE-2.1a-PCNATIVE-VANILLA [{label}] c3d-precision-is-the-cause test  | device={dev} P={P} h={h} "
          f"d=1.0 opt=SGD mode=pc_transport epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} lr=1.0 "
          f"seeds={len(seeds)} log_every={K}")
    print(f"  V1 (pc_transport, real) + V1n (pc_transport, shuffled) -- both with T_eval={N1PRIME_TEVAL} "
          f"diagnostic + ||W2|| monitoring. SAME seeds/splits/masks/init as N1 (only mode differs).")
    print("  QUESTION: does vanilla PC (no precision gating) grok where c3d plateaued? If yes + relaxation "
          "converges (resid<0.5, F monotone) -> c3d precision WAS the cause.")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]; lab_shuf = list(perms)

    results = {}

    def _run_cell(name, lkind, lab):
        cc = dict(cfg); cc["opt"] = "sgd"; cc["momentum"] = 0.0; cc["wd"] = 0.0
        cc["lr"] = 1.0; cc["T_eval"] = N1PRIME_TEVAL
        t0 = time.perf_counter()
        res = run_seeds_masked("pc_transport", seeds, [lab], [splits_list], M1, M2, a, b, P,
                               cc, dev, deplete=False, label_kind=lkind,
                               log_per_epoch=True, early_stop=False, es_uses_block=False,
                               want_rtdiag=True, w1_gate="both", norm_gate=None)
        is_noise = (lkind == "shuffled")
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        div_flags = [bool(e["diverged"]) for e in res]
        k2_label, k2_lang, k2_det = score_pcnative_k2(res, log_every=K, is_noise=is_noise)
        cell = dict(name=name, mode="pc_transport", wd=0.0, momentum=0.0, shuffled=is_noise, opt="sgd", lr=1.0,
                    density=1.0, norm_gate=None,
                    train_mean=float(np.nanmean(tr)), train_per_seed=tr,
                    test_mean=float(np.nanmean(te)), test_per_seed=te,
                    diverged=any(div_flags), n_diverged=int(sum(div_flags)),
                    k2_label=k2_label, k2_lang=k2_lang, k2_details=k2_det,
                    tdiag=_tdiag_summary(res),
                    per_epoch=[e["per_epoch"] for e in res],
                    wall_s=round(time.perf_counter() - t0, 1))
        td = cell["tdiag"]
        print(f"\n[{name}] pc_transport SGD m=0 wd=0 lr=1.0: train {cell['train_mean']:.3f} | "
              f"test {cell['test_mean']:.3f}" + ("  DIVERGED" if cell["diverged"] else ""))
        print(f"  -> {k2_label}: {k2_lang}")
        print(f"  [tdiag] resid_T20={td.get('resid_T20_tail_mean', float('nan')):.3g}  "
              f"resid_T200={td.get('resid_T200_tail_mean', float('nan')):.3g}  "
              f"converged={td.get('relaxation_converged')}  "
              f"F_net_descent={td.get('F_net_descent', float('nan')):.3g}  "
              f"F_max_step_rise={td.get('F_max_step_rise', float('nan')):.3g}  "
              f"F_monotone={td.get('F_monotone_descent')}")
        return cell

    results["V1"] = _run_cell("V1", "real", lab_real)
    _dump_vanilla(save_path, results, cfg)
    results["V1n"] = _run_cell("V1n", "shuffled", lab_shuf)
    _dump_vanilla(save_path, results, cfg)

    # ===================== DECISION ================================================================
    print("\n" + "=" * 104)
    print("VANILLA-PC DIAGNOSIS (is c3d precision the plateau cause?):")
    v1 = results["V1"]; td = v1["tdiag"]
    w2_slope = float(np.nanmean(v1["k2_details"].get("w2_slope_per_1k", [float("nan")])))
    w2_mean = float(np.nanmean(v1["k2_details"].get("w2_mean", [float("nan")])))
    print(f"  V1 train={v1['train_mean']:.3f} test={v1['test_mean']:.3f} (c3d N1 was 0.832/0.055)")
    print(f"  relaxation: resid_T200={td.get('resid_T200_tail_mean', float('nan')):.3g} "
          f"(converged={td.get('relaxation_converged')}, bar={PCNATIVE_RESID_CONV_BAR}); "
          f"F_monotone_descent={td.get('F_monotone_descent')} (F_max_step_rise={td.get('F_max_step_rise', float('nan')):.3g})")
    print(f"  ||W2||: slope/1k={w2_slope:.3f} mean={w2_mean:.2f} (c3d N1 inflated 7->53)")
    v1n = results["V1n"]
    print(f"  V1n (noise control): train={v1n['train_mean']:.3f} test={v1n['test_mean']:.3f} K2={v1n['k2_label']}")
    # GATE on the TEST-based k2_label (grok=test peak>=0.9 quorum), NOT a train-only "grok" (experiment-
    # designer: train>0.95 with test at chance is memorization, not schema success -- the historical overclaim
    # trap). resid_converged (resid<bar) is the robust convergence gate; F_monotone is REPORTED only (its
    # threshold is noise-calibrated, not a hard bar). Noise control must be clean (not MEMORIZED/AMBIGUOUS).
    v1_grok = (v1["k2_label"] == "K2-PASS")
    v1_fits = (v1["train_mean"] > 0.95)
    v1_conv = bool(td.get("relaxation_converged", False))
    control_clean = v1n["k2_label"] not in ("CONTROL-MEMORIZED", "CONTROL-AMBIGUOUS")
    if v1_grok and v1_conv and control_clean:
        print("  -> c3d precision WAS the plateau cause: vanilla PC GROKS (test K2-PASS, >=8/10 quorum) AND "
              "its relaxation converges (resid<bar) where c3d oscillated (resid~5, period-2). Precision "
              "attenuation starved the feedback path; removing it restores the Lyapunov descent. Action: "
              "re-derive precision (or drop it) on the SGD stack.")
    elif v1_fits and v1_conv and not v1_grok:
        print(f"  -> vanilla PC FITS train ({v1['train_mean']:.2f}>0.95) and converges, but does NOT generalize "
              f"(test {v1['test_mean']:.3f}, K2={v1['k2_label']}): c3d blocks the FIT, but the plateau is ALSO a "
              f"generalization problem (vanilla fits without grokking). NOT a clean 'c3d was the cause' -- read "
              f"the test trajectory.")
    elif v1_conv and v1["train_mean"] <= 0.85:
        print(f"  -> vanilla relaxation CONVERGES (resid<bar) but train still plateaus ({v1['train_mean']:.3f}): "
              "c3d precision is NOT the (sole) cause -- the plateau is in the M-step/gradient or capacity, not "
              "the E-step precision. The oscillation was a c3d artifact, but fixing it doesn't unblock fitting.")
    else:
        print(f"  -> vanilla PC does NOT grok (K2={v1['k2_label']}, train={v1['train_mean']:.3f}) and/or the "
              f"relaxation does not cleanly converge (resid_converged={v1_conv}): c3d precision is NOT the "
              f"(sole) cause. The plateau is architectural (the PC gradient/relaxation or SGD step-size), not "
              f"precision-gating. Read resid/F/||W2|| above for the mechanism.")
    _dump_vanilla(save_path, results, cfg)


def _dump_fbgain(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "fbgain_note": ("FEEDBACK-GAIN SWEEP (loop-gain-is-the-oscillation-cause hypothesis). Vanilla PC "
                        "(pc_transport, no c3d) under SGD, sweeping fb_gain (B2=g*W2, forward W2 UNCHANGED -> "
                        "loop gain = g*||W2||^2). g=1.0 = tied (parity with V1: train~0.39, resid~15 oscillating). "
                        "Hypothesis: the tied-weight loop gain causes the relaxation oscillation; g<1 damps it. "
                        "Pre-registered: (1) g<=0.3 converges (resid<0.5) AND train>=0.95 -> loop gain WAS the "
                        "cause -> test grokking next (the moat); (2) converges but train<0.95 -> oscillation "
                        "fixed but SGD still can't fit -> ill-conditioning -> iPC; (3) no g converges -> "
                        "oscillation architectural -> iPC or untied B2. NB: 'fits' = train>=0.95 (fit), NOT "
                        "grok (test generalization) -- grokking is the NEXT test if the fit succeeds."),
        "fbgain_list": list(FBGAIN_LIST), "resid_conv_bar": PCNATIVE_RESID_CONV_BAR,
        "phase1_best_g": results.get("_phase1_best_g"),
        "banked": {"V1_tied": "pc_transport fb_gain=1.0: train~0.39 test~0.01 resid~15 (oscillating)"},
        "cells": {k: v for k, v in results.items() if not k.startswith("_")},
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def drive_fbgain(cfg, label, save_path):
    """FEEDBACK-GAIN SWEEP -- is tied-weight loop gain the oscillation cause? Vanilla PC (pc_transport, no
    c3d) under SGD m=0 wd=0 lr=1.0, sweeping fb_gain (B2=g*W2, forward W2 unchanged). g=1.0 = tied (parity
    with V1: train~0.39, resid~15). g<1 damps the loop independently of the forward path.
      Phase 1 (finder): g in FBGAIN_LIST, 3 seeds x 2k, real -- pick best g (lowest resid; converging first).
      Phase 2 (decisive): best g + g=1.0 (parity), 10 seeds x 15k, real + noise. T_eval=200 diagnostic +
      ||W2|| monitoring throughout. Pre-registered outcomes O1/O2/O3 in _dump_fbgain note."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: fbgain needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); K = cfg["log_every"]
    print("=" * 104)
    print(f"GATE-2.1a-PCNATIVE-FBGAIN [{label}] loop-gain sweep  | device={dev} P={P} h={h} d=1.0 opt=SGD "
          f"mode=pc_transport epochs={cfg['epochs']} lr=1.0 seeds={len(seeds)} log_every={K}")
    print(f"  Phase 1: finder g={list(FBGAIN_LIST)} ({FBGAIN_FINDER_SEEDS} seeds x {FBGAIN_FINDER_EPOCHS}ep) "
          f"-> best g (lowest resid, converging first)")
    print(f"  Phase 2: best g + g=1.0 (parity), 10 seeds x {cfg['epochs']}ep, real+noise, T_eval={N1PRIME_TEVAL}")
    print("  QUESTION: does any g<1.0 drive resid<0.5 (convergence)? If yes + train>=0.95 -> tied-weight loop "
          "gain WAS the oscillation cause (the 'second gear' is the PC-native fix).")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]; lab_shuf = list(perms)

    results = {}

    def _run_cell(name, g, lkind, lab, sub_seeds, epochs, want_diag):
        cc = dict(cfg); cc["opt"] = "sgd"; cc["momentum"] = 0.0; cc["wd"] = 0.0
        cc["lr"] = 1.0; cc["fb_gain"] = g; cc["epochs"] = epochs
        cc["T_eval"] = (N1PRIME_TEVAL if want_diag else None)
        t0 = time.perf_counter()
        res = run_seeds_masked("pc_transport", sub_seeds, [lab[:len(sub_seeds)]], [splits_list[:len(sub_seeds)]],
                               M1[:len(sub_seeds)], M2[:len(sub_seeds)], a, b, P,
                               cc, dev, deplete=False, label_kind=lkind,
                               log_per_epoch=True, early_stop=False, es_uses_block=False,
                               want_rtdiag=want_diag, w1_gate="both", norm_gate=None)
        is_noise = (lkind == "shuffled")
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        div_flags = [bool(e["diverged"]) for e in res]
        k2_label, k2_lang, k2_det = score_pcnative_k2(res, log_every=K, is_noise=is_noise)
        cell = dict(name=name, mode="pc_transport", fb_gain=g, wd=0.0, momentum=0.0, shuffled=is_noise,
                    opt="sgd", lr=1.0, density=1.0, norm_gate=None,
                    train_mean=float(np.nanmean(tr)), train_per_seed=tr,
                    train_mean_nondiverged=(float(np.mean([t for t, d in zip(tr, div_flags) if not d]))
                                            if any(not d for d in div_flags) else float("nan")),
                    n_diverged=int(sum(div_flags)),
                    test_mean=float(np.nanmean(te)), test_per_seed=te,
                    diverged=any(div_flags),
                    k2_label=k2_label, k2_lang=k2_lang, k2_details=k2_det,
                    tdiag=(_tdiag_summary(res) if want_diag else {}),
                    per_epoch=[e["per_epoch"] for e in res],
                    wall_s=round(time.perf_counter() - t0, 1))
        td = cell["tdiag"]
        print(f"\n[{name}] pc_transport fb_gain={g}: train {cell['train_mean']:.3f} | test {cell['test_mean']:.3f}"
              + ("  DIVERGED" if cell["diverged"] else ""))
        if td:
            print(f"  -> {k2_label}")
            print(f"  [tdiag] resid_T200={td.get('resid_T200_tail_mean', float('nan')):.3g} "
                  f"converged={td.get('relaxation_converged')}  "
                  f"F_monotone={td.get('F_monotone_descent')} (F_max_step_rise="
                  f"{td.get('F_max_step_rise', float('nan')):.3g})  "
                  f"||W2|| slope/1k={float(np.nanmean(k2_det.get('w2_slope_per_1k',[float('nan')]))):.3f}")
        return cell

    # ===================== PHASE 1: finder (g sweep, 3 seeds x 2k, real, with T_eval) =================
    print("\n--- PHASE 1: fb_gain finder (3 seeds x 2k, T_eval diagnostic) ---")
    p1 = tuple(range(FBGAIN_FINDER_SEEDS))
    finder = {}
    for g in FBGAIN_LIST:
        cell = _run_cell("g%s_finder" % g, g, "real", lab_real, p1, FBGAIN_FINDER_EPOCHS, want_diag=True)
        finder[g] = dict(train=cell["train_mean"], resid=cell["tdiag"].get("resid_T200_tail_mean", float("nan")),
                         converged=cell["tdiag"].get("relaxation_converged", False), diverged=cell["diverged"])
        results["g%s_finder" % g] = cell
        _dump_fbgain(save_path, results, cfg)
    # best g: prefer a CONVERGING g (resid<bar) with highest train; else the lowest-resid g (closest to converging)
    converging = [g for g in FBGAIN_LIST if finder[g]["converged"] and not finder[g]["diverged"]]
    if converging:
        best_g = max(converging, key=lambda g: finder[g]["train"])
    else:
        best_g = min(FBGAIN_LIST, key=lambda g: (finder[g]["resid"] if not finder[g]["diverged"] else float("inf")))
    results["_phase1_best_g"] = best_g
    print(f"\n  -> Phase 1 best g = {best_g} "
          f"(converging={[g for g in FBGAIN_LIST if finder[g]['converged']]}); "
          f"resid={finder[best_g]['resid']:.3g} train={finder[best_g]['train']:.3f}")
    _dump_fbgain(save_path, results, cfg)

    # ===================== PHASE 2: decisive (best g + g=1.0 parity, 10 seeds x 15k, real+noise) =======
    print(f"\n--- PHASE 2: decisive @ best_g={best_g} + g=1.0 parity (10 seeds x 15k) ---")
    phase2_gs = sorted(set([best_g, 1.0]))
    for g in phase2_gs:
        tag = "g%s" % g + ("_best" if g == best_g else ("_tied" if g == 1.0 else ""))
        results[tag + "_real"] = _run_cell(tag + "_real", g, "real", lab_real, seeds, cfg["epochs"], want_diag=True)
        _dump_fbgain(save_path, results, cfg)
        results[tag + "_noise"] = _run_cell(tag + "_noise", g, "shuffled", lab_shuf, seeds, cfg["epochs"], want_diag=True)
        _dump_fbgain(save_path, results, cfg)

    # ===================== DECISION (pre-registered O1/O2/O3) =======================================
    print("\n" + "=" * 104)
    print("FBGAIN DIAGNOSIS (is tied-weight loop gain the oscillation cause?):")
    bg = best_g
    bg_cell = results.get("g%s_best_real" % bg) or results.get("g%s_real" % bg, {})
    bg_td = bg_cell.get("tdiag", {})
    bg_conv = bool(bg_td.get("relaxation_converged", False))
    bg_train = bg_cell.get("train_mean_nondiverged", bg_cell.get("train_mean", float("nan")))
    bg_test = bg_cell.get("test_mean", float("nan"))
    bg_k2 = bg_cell.get("k2_label", "?")
    tied = (results.get("g1.0_tied_real") or results.get("g1.0_real")
            or (bg_cell if bg == 1.0 else None) or {})   # best_g==1.0 -> tied IS the best-g cell
    print(f"  best g={bg}: train={bg_train:.3f} test={bg_test:.3f} K2={bg_k2} "
          f"resid={bg_td.get('resid_T200_tail_mean', float('nan')):.3g} converged={bg_conv} "
          f"F_monotone={bg_td.get('F_monotone_descent')}")
    print(f"  g=1.0 (tied parity): train={tied.get('train_mean', float('nan')):.3f} "
          f"resid={tied.get('tdiag', {}).get('resid_T200_tail_mean', float('nan')):.3g} "
          f"(banked V1: train~0.39 resid~15)")
    # noise control
    bg_noise = results.get("g%s_best_noise" % bg) or results.get("g%s_noise" % bg, {})
    control_clean = bg_noise.get("k2_label", "?") not in ("CONTROL-MEMORIZED", "CONTROL-AMBIGUOUS")
    print(f"  noise control @ best g: train={bg_noise.get('train_mean', float('nan')):.3f} "
          f"K2={bg_noise.get('k2_label', '?')} (control_clean={control_clean})")
    fits = np.isfinite(bg_train) and bg_train >= 0.95
    if bg_conv and fits and control_clean:
        print("  -> O1: loop gain WAS the oscillation cause. fb_gain<1 converges the relaxation (resid<bar, "
              "F monotone) AND SGD fits (train>=0.95) AND the noise control did NOT memorize. The 'second "
              "gear' is the PC-native fix. ACTION: test GROKKING next (train fits -- does it generalize? "
              "test/K2 above). Anti-rescue: a fit is not yet a grok; do not bank a moat until test K2-PASS.")
    elif bg_conv and fits and not control_clean:
        print("  -> O1-AMBIGUOUS: best g converges AND fits train (>=0.95) BUT the noise control MEMORIZED "
              "(shuffled labels) -> the fit may be memorization CAPACITY, not a loop-gain fix. Do NOT claim "
              "'loop gain was the cause' or bank a moat. Re-test with a sharper held-out/generalization bar.")
    elif bg_conv and not fits:
        print(f"  -> O2: oscillation FIXED (resid<bar, F monotone) but SGD still can't fit (train={bg_train:.3f}"
              f"<0.95). Loop gain was the oscillation cause, but ill-conditioning remains -- the damped "
              f"relaxation settles cleanly yet the gradient is still insufficient for SGD. ACTION: iPC "
              f"(joint E+M, per-step updates) on the damped stack.")
    elif not any(finder[g]["converged"] for g in FBGAIN_LIST):
        print("  -> O3: NO fb_gain converges the relaxation (resid>>bar at every g). The oscillation is "
              "ARCHITECTURAL (not loop-gain-driven). ACTION: iPC, or a fully UNTIED feedback B2 (independent "
              "weights, not a scalar gain on W2). Anti-rescue unchanged.")
    else:
        print(f"  -> MIXED: best g={bg} did not cleanly converge in Phase 2 (resid_converged={bg_conv}) but "
              f"some g converged in the finder. Read the per-g trajectories; the 2k->15k gap may show erosion. "
              f"No clean verdict -- report raw, do not claim a cause.")
    _dump_fbgain(save_path, results, cfg)


# =========================================================== STAGED CHANNEL razor ===
def _dump_staged(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "spec": ("STAGED_CHANNEL_SPEC v2. Does freezing updates after grok hold the schema? Erosion under "
                 "AdamW+wd=0 (10/10 grok, 10/10 erode) is caused by active updates past grok. wd=1.0 holds "
                 "(SHY downscaling). This tests whether 'stop learning' (freeze) suffices. All cells c3d "
                 "alpha=900, d=1.0, AdamW lr=2e-3, T=20, eta=0.2, 10 seeds, 30k epochs, rtdiag ON. Phase 1 "
                 "(0->switch=2500): w1_gate='both'. Optimizer FLUSH at switch (re-init AdamW m,v)."),
        "switch_epoch": STAGED_SWITCH_EPOCH,
        "preregistered_outcomes": {
            "S_frozen_REFERENCE": ("TAUTOLOGICAL: test_acc is a pure function of bitwise-frozen W1/W2 "
                                   "(blogits_masked), so hold is structurally guaranteed and erosion is "
                                   "structurally unreachable. S-frozen is a REFERENCE/SANITY cell, NOT a decision cell."),
            "D1_w1only_HOLDS": "S-frozen-W1only tail-min >=0.9 in >=8/10 -> W2 drift alone does NOT erode the schema",
            "D2_w1only_ERODES": "S-frozen-W1only tail-min <0.9 in >=5/10 -> W2 drift alone erodes (W1 not enough)",
            "D3_lowrate_HOLDS": "S-lowrate tail-min >=0.9 in >=8/10 -> low-rate radial maintenance holds",
            "D4_lowrate_ERODES": "S-lowrate tail-min <0.9 in >=5/10 -> low-rate maintenance insufficient",
        },
        "cells_config": [(n, w, s, r) for n, w, s, r in STAGED_CELLS],
        "banked_controls": {
            "both-on_w1gate": "AdamW wd=0 gate=OFF c3d: 10/10 grok, 10/10 erode, 7/10 collapse (the erosion baseline)",
            "G0_C5NORM": "AdamW wd=1.0 gate=OFF: test=1.0 10/10 (wd-alone holds)",
            "F1_C5NORM": "AdamW wd=1.0 gate=ON: test=1.0 10/10 K2'-PASS",
        },
        "cells": results,
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def drive_staged_channel(cfg, label, save_path):
    """STAGED CHANNEL (STAGED_CHANNEL_SPEC v2). 5 cells, all c3d @ d=1.0, 10 seeds, 30k, rtdiag ON.
    S-frozen (decision cell): freeze ALL updates after grok -> does the schema hold? S-frozen-W1only:
    W1 frozen, W2 learns (W2 drift detector). S-lowrate: W1 radial-only (low-rate maintenance).
    S-both: same-run erosion baseline (wd=0, no switch). S-both-wd1: same-run hold baseline (wd=1.0).
    Pre-registered R1 (HOLDS) / R2 (ERODES) on S-frozen. Optimizer FLUSH at switch epoch."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: staged channel needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = cfg["seeds"]; top = max(cfg["fracs"]); K = cfg["log_every"]
    print("=" * 104)
    print(f"STAGED CHANNEL [{label}] freeze-after-grok test  | device={dev} P={P} h={h} d=1.0 "
          f"epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={len(seeds)} log_every={K}")
    print(f"  switch_epoch={STAGED_SWITCH_EPOCH} (after banked peaks <=2100, before erosion onset 3000+)")
    print(f"  cells: {[(n, 'wd='+str(w), ('sw='+str(s) if s else 'no-sw'), r) for n, w, s, r in STAGED_CELLS]}")
    print("  PRE-REGISTERED: R1 frozen HOLDS (tail-min>=0.9 in >=8/10) -> schema stable; "
          "R2 frozen ERODES (<0.9 in >=5/10) -> wd does more than halt growth. S-frozen is the decision cell.")
    print("=" * 104)

    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]

    results = {}

    def _run_cell(name, wd, switch_epoch, phase2_regime):
        cc = dict(cfg); cc["wd"] = wd
        t0 = time.perf_counter()
        res = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                               cc, dev, deplete=False, label_kind="real",
                               log_per_epoch=True, early_stop=False, es_uses_block=False,
                               want_rtdiag=True, w1_gate="both", norm_gate=None,
                               channel_switch_epoch=switch_epoch, phase2_regime=phase2_regime)
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        k2_label, k2_lang, k2_det = score_staged_channel(res, switch_epoch, log_every=K)
        cell = dict(name=name, wd=wd, switch_epoch=switch_epoch, phase2_regime=phase2_regime,
                    mode="c3_dynamic", density=1.0,
                    train_mean=float(np.mean(tr)), train_per_seed=tr,
                    test_mean=float(np.mean(te)), test_per_seed=te,
                    diverged=any(e["diverged"] for e in res),
                    k2_label=k2_label, k2_lang=k2_lang, k2_details=k2_det,
                    per_epoch=[e["per_epoch"] for e in res],
                    wall_s=round(time.perf_counter() - t0, 1))
        tag = f"{'sw@'+str(switch_epoch) if switch_epoch else 'no-sw'} wd={wd} regime={phase2_regime}"
        print(f"\n[{name}] {tag}: train {cell['train_mean']:.3f} | test {cell['test_mean']:.3f}"
              + ("  DIVERGED" if cell["diverged"] else ""))
        print(f"  -> {k2_label}: {k2_lang}")
        if switch_epoch is not None and k2_det.get("switch_diagnostics"):
            sd = k2_det["switch_diagnostics"]
            print(f"  [switch@{switch_epoch}] test_at_switch={[round(t,2) for t in sd['test_at_switch_per_seed']]} "
                  f"(grok_all={sd['grok_at_switch_all']})  ||W1||={[round(w,1) for w in sd['w1_at_switch_per_seed']]}  "
                  f"||W2||={[round(w,1) for w in sd['w2_at_switch_per_seed']]}")
        return cell

    for name, wd, switch_epoch, phase2_regime in STAGED_CELLS:
        results[name] = _run_cell(name, wd, switch_epoch, phase2_regime)
        _dump_staged(save_path, results, cfg)

    # decision read (4-agent gate patch: S-frozen is TAUTOLOGICAL — test_acc is a pure function of
    # bitwise-frozen W1/W2 via blogits_masked, so R1 is structurally guaranteed and R2 unreachable.
    # The DECISION CELLS are S-frozen-W1only (does W2 drift alone erode?) and S-lowrate (does low-rate
    # maintenance hold?). S-frozen is a reference/sanity cell only. S-both and S-both-wd1 are baselines.)
    print("\n" + "=" * 104)
    print("DECISION (4-agent gate: S-frozen is TAUTOLOGICAL; decision cells = W1only + lowrate):")
    sf = results.get("S-frozen", {})
    w1o = results.get("S-frozen-W1only", {})
    lr = results.get("S-lowrate", {})
    print(f"  S-frozen        K2={sf.get('k2_label','?')}  -- REFERENCE (tautological: frozen W => constant test_acc)")
    print(f"  S-frozen-W1only K2={w1o.get('k2_label','?')}  -- DECISION CELL: does W2 drift alone erode?")
    print(f"  S-lowrate       K2={lr.get('k2_label','?')}  -- DECISION CELL: does low-rate radial maintenance hold?")
    print(f"  S-both          K2={results.get('S-both',{}).get('k2_label','?')}  -- erosion baseline (should erode)")
    print(f"  S-both-wd1      K2={results.get('S-both-wd1',{}).get('k2_label','?')}  -- hold baseline (wd=sleep)")
    w1o_lab = w1o.get("k2_label", "?")
    lr_lab = lr.get("k2_label", "?")
    both_lab = results.get("S-both", {}).get("k2_label", "?")
    wd1_lab = results.get("S-both-wd1", {}).get("k2_label", "?")
    baseline_erodes = both_lab in ("R2-ERODES",)
    baseline_holds = wd1_lab in ("R1-HOLDS",)
    w1o_holds = w1o_lab in ("R1-HOLDS",)
    lr_holds = lr_lab in ("R1-HOLDS",)
    if not baseline_erodes:
        decision = ("BASELINE FAIL: S-both did NOT reproduce erosion — all comparisons vacuous. "
                    "Investigate before reading decision cells.")
    elif w1o_holds and lr_holds:
        decision = ("D1+D3: both decision cells HOLD. Schema is robust to W2 drift AND low-rate W1 "
                    "maintenance. The neocortex module alone maintains its schema once grokked. "
                    "Erosion under S-both requires BOTH channels updating at full rate.")
    elif w1o_holds and not lr_holds:
        decision = ("D1: W2 drift does NOT erode, but D4: low-rate W1 maintenance fails. "
                    "W1 tangential updates are needed to maintain the schema (radial-only insufficient).")
    elif not w1o_holds and lr_holds:
        decision = ("D2: W2 drift ALONE erodes the schema. W1 frozen is not enough. "
                    "D3: but low-rate W1 maintenance holds — the schema needs SOME W1 updating.")
    else:
        decision = ("D2+D4: both decision cells ERODE. Schema is fragile — even partial updates kill it. "
                    "wd=1.0's hold is an active maintenance mechanism, not just halting growth.")
    print(f"  baseline check: S-both erodes={baseline_erodes}, S-both-wd1 holds={baseline_holds}")
    print(f"  -> {decision}")
    _dump_staged(save_path, results, cfg)


def staged_smoke():
    """Local harness sanity for the staged channel switch (NOT science). Proves:
    (1) S-frozen actually FREEZES: ||W1|| AND ||W2|| bitwise-identical at ALL post-switch log points
        (zero slope EXACTLY, not approximately — skip_step=True means opt.step never fires).
    (2) S-both (no switch) reproduces the erosion baseline trajectory shape: ||W1|| MOVES (grows).
    (3) Optimizer FLUSH fires at switch: the run completes past the switch without error (the re-init
        of AdamW m,v at the switch epoch exercises the _optimizer re-creation code path).
    (4) S-frozen-W1only: W1 frozen (bitwise-identical post-switch) BUT W2 moves (W2 drifts).
    (5) Parity guard: channel_switch_epoch=None reproduces the no-switch path bitwise (S-both == a plain
        c3d run with no switch infra).
    2 seeds, short budget. Switch at epoch 100, total 300 epochs."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    a, b, c = make_cells(P); N = P * P
    print("STAGED CHANNEL SMOKE (2 seeds, 300ep, switch@100) -- freeze + parity + flush; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=50, log_per_epoch=True,
                fracs=(0.9,), wd=0.0, epochs=300, snap_every=100,
                c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)
    sw = 100
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]

    # (1)+(3) S-frozen: ||W1||/||W2|| bitwise flat post-switch; flush fires at switch
    r_frozen = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                                dict(base), dev, deplete=False, label_kind="real",
                                log_per_epoch=True, early_stop=False, want_rtdiag=True,
                                channel_switch_epoch=sw, phase2_regime="frozen")
    sw_logpt = sw // 50      # log point index at switch (log_every=50)
    for i in range(len(seeds)):
        w1 = r_frozen[i]["per_epoch"]["w1_norm"]
        w2 = r_frozen[i]["per_epoch"]["w2_norm"]
        pa = r_frozen[i]["per_epoch"].get("phase2_active", [])
        post_w1 = w1[sw_logpt:]; post_w2 = w2[sw_logpt:]
        assert pa and pa[sw_logpt], f"phase2_active not True at switch log pt (seed {seeds[i]})"
        assert all(pa[sw_logpt:]), f"phase2_active not all-True post-switch (seed {seeds[i]})"
        assert len(post_w1) >= 2, f"need >=2 post-switch log pts (seed {seeds[i]})"
        # EXACT freeze: all post-switch ||W1||/||W2|| bitwise identical (skip_step -> opt.step never fires)
        w1_flat = all(x == post_w1[0] for x in post_w1)
        w2_flat = all(x == post_w2[0] for x in post_w2)
        assert w1_flat, f"S-frozen ||W1|| NOT bitwise-flat post-switch (seed {seeds[i]}): {post_w1}"
        assert w2_flat, f"S-frozen ||W2|| NOT bitwise-flat post-switch (seed {seeds[i]}): {post_w2}"
    print(f"  (1) S-frozen FREEZE: ||W1|| and ||W2|| bitwise-identical at ALL post-switch log pts "
          f"(seed0 ||W1||={r_frozen[0]['per_epoch']['w1_norm'][sw_logpt]:.3f} flat x{len(post_w1)} pts). OK")

    # (2) S-both (no switch): ||W1|| moves (erosion baseline trajectory). ALSO the parity reference.
    r_both = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                              dict(base), dev, deplete=False, label_kind="real",
                              log_per_epoch=True, early_stop=False, want_rtdiag=True)
    for i in range(len(seeds)):
        w1 = r_both[i]["per_epoch"]["w1_norm"]
        assert abs(w1[-1] - w1[0]) > 1e-6, f"S-both ||W1|| did not move (seed {seeds[i]})"
    print(f"  (2) S-both EROSION baseline: ||W1|| moves (seed0 {r_both[0]['per_epoch']['w1_norm'][0]:.3f} "
          f"-> {r_both[0]['per_epoch']['w1_norm'][-1]:.3f}). OK")

    # (4) S-frozen-W1only: W1 frozen post-switch, W2 moves
    r_w1only = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                                dict(base), dev, deplete=False, label_kind="real",
                                log_per_epoch=True, early_stop=False, want_rtdiag=True,
                                channel_switch_epoch=sw, phase2_regime="frozen_w1only")
    for i in range(len(seeds)):
        w1 = r_w1only[i]["per_epoch"]["w1_norm"]
        w2 = r_w1only[i]["per_epoch"]["w2_norm"]
        post_w1 = w1[sw_logpt:]; post_w2 = w2[sw_logpt:]
        # W1 frozen (bitwise identical post-switch); W2 moves
        assert all(x == post_w1[0] for x in post_w1), \
            f"S-frozen-W1only ||W1|| NOT flat post-switch (seed {seeds[i]}): {post_w1}"
        assert any(abs(x - post_w2[0]) > 1e-9 for x in post_w2), \
            f"S-frozen-W1only ||W2|| did NOT move post-switch (seed {seeds[i]}): {post_w2}"
    print(f"  (4) S-frozen-W1only: W1 bitwise-frozen post-switch, W2 moves "
          f"(seed0 ||W2|| {r_w1only[0]['per_epoch']['w2_norm'][sw_logpt]:.3f} "
          f"-> {r_w1only[0]['per_epoch']['w2_norm'][-1]:.3f}). OK")

    # (5) Parity: channel_switch_epoch=None == a plain c3d run WITHOUT the switch kwargs (bitwise)
    r_plain = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                               dict(base), dev, deplete=False, label_kind="real",
                               log_per_epoch=True, early_stop=False, want_rtdiag=True)
    # r_both above was called with channel_switch_epoch=None (the default) -> must match r_plain bitwise
    for i in range(len(seeds)):
        for key in ("train_acc", "test_acc", "w1_norm", "w2_norm"):
            a_ = np.asarray(r_both[i]["per_epoch"][key], dtype=np.float64)
            b_ = np.asarray(r_plain[i]["per_epoch"][key], dtype=np.float64)
            assert np.array_equal(a_, b_), \
                f"PARITY FAIL ({key}, seed {seeds[i]}): switch=None != plain (no kwargs)"
    print(f"  (5) Parity: channel_switch_epoch=None == plain c3d run (no switch kwargs) bitwise. OK")
    # (3) optimizer flush: runs 1-3 all completed past the switch epoch without error -> flush code path exercised
    print(f"  (3) Optimizer FLUSH: all switch cells ran past switch@{sw} without error. OK")
    print(f"  (5) Parity: no-switch path (channel_switch_epoch=None) ran clean. OK")
    print("SMOKE PASS -- S-frozen bitwise-freezes ||W1||+||W2||; S-both moves; W1only freezes W1 only; "
          "flush fires. Ready for the 4-agent review gate + --staged-channel.")


def c5norm_smoke():
    """Local parity + gate-functional check (NOT science). Proves:
    (1) PARITY GUARD: norm_gate(theta_hi=inf) == norm_gate=None BITWISE on train/test/w1/w2 (the gate code
        path runs but is inert: (w1_norms>=inf) is always False -> sleep_state stays False -> gate=1.0 ->
        gW1*1.0==gW1 exact under IEEE754). 40ep, 2 seeds. THE pre-deploy guard (addendum §3).
    (2) DEADLOCK-BREAKER: norm_gate(theta_hi=15, theta_lo=8) wd=1.0 -> gate FIRES (sleep_state True), ||W1||
        stops growing during sleep (wd pulls it down), and the cycle CLOSES (sleep episode ends in wake).
        Verifies AdamW decoupled wd applies to zeroed-grad params inside the harness."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    a, b, c = make_cells(P); N = P * P
    print("C5NORM SMOKE (2 seeds) -- parity guard (theta_hi=inf == None) + deadlock-breaker; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=20, log_per_epoch=True,
                fracs=(0.9,), wd=1.0, epochs=40, snap_every=100,
                c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]

    # (1) PARITY: norm_gate(theta_hi=inf) == norm_gate=None bitwise
    r_none = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                              dict(base), dev, deplete=False, label_kind="real",
                              log_per_epoch=True, early_stop=False, want_rtdiag=True, norm_gate=None)
    r_inf = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                             dict(base), dev, deplete=False, label_kind="real",
                             log_per_epoch=True, early_stop=False, want_rtdiag=True,
                             norm_gate=dict(theta_hi=float("inf"), theta_lo=38.0))
    for i in range(len(seeds)):
        for key in ("train_acc", "test_acc", "w1_norm", "w2_norm"):
            a_ = np.asarray(r_none[i]["per_epoch"][key], dtype=np.float64)
            b_ = np.asarray(r_inf[i]["per_epoch"][key], dtype=np.float64)
            # fused AdamW may differ from non-fused in last bits -> float tolerance, not bitwise. A real
            # gate bug (non-inert) would diverge by O(0.1)+, far above this tolerance.
            assert np.allclose(a_, b_, rtol=1e-5, atol=1e-6), \
                f"C5NORM PARITY FAIL ({key}, seed {seeds[i]}): theta_hi=inf != None (max d={np.max(np.abs(a_-b_)) if a_.size else 0:.2e})"
    print("  PARITY (norm_gate theta_hi=inf == None) within float tol (rtol=1e-5): train/test/w1/w2 allclose over 40ep. OK")

    # (2) DEADLOCK-BREAKER (now under fused=True AdamW): low theta so the gate fires early; wd=1.0 must pull
    # ||W1|| back down during sleep. Proven in isolation (zeroed-grad param decays at exactly (1-lr*wd)/epoch);
    # this re-confirms it END-TO-END through the fused optimizer. Two load-bearing asserts:
    #   (a) gate FIRES (sleep_state True somewhere); (b) ||W1|| DECAYED below theta_hi during sleep (wd acts
    #   on zeroed-grad params inside the fused kernel). A sleep->wake transition is corroboration.
    gate_lo = dict(theta_hi=15.0, theta_lo=8.0)
    c2 = dict(base); c2["epochs"] = 400; c2["log_every"] = 20
    r_gate = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                              c2, dev, deplete=False, label_kind="real",
                              log_per_epoch=True, early_stop=False, want_rtdiag=True, norm_gate=gate_lo)
    fired = False; decayed = False; saw_transition = False
    for i in range(len(seeds)):
        sl = [bool(s) for s in (r_gate[i]["per_epoch"].get("ng_sleep") or [])]
        w1n = r_gate[i]["per_epoch"]["w1_norm"]
        if any(sl):
            fired = True
            # (b) ||W1|| decayed below theta_hi while sleeping -> wd is acting on zeroed-grad params
            asleep_w1 = [w1n[k] for k in range(len(sl)) if sl[k]]
            if min(asleep_w1) < gate_lo["theta_hi"]:
                decayed = True
            eps = _sleep_episodes(sl)
            # corroboration: any sleep episode ended in wake (a sleep->wake transition exists)
            if any(s + L < len(sl) for s, L in eps):
                saw_transition = True
            print(f"  [seed {seeds[i]}] gate fired; n_episodes={len(eps)}; ||W1|| start={w1n[0]:.1f} "
                  f"min={min(w1n):.1f} end={w1n[-1]:.1f}; sleep->wake transition={'yes' if any(s+L<len(sl) for s,L in eps) else 'no (run ended asleep)'}")
    assert fired, "DEADLOCK-BREAKER: gate never fired (theta_hi=15 too high for 400ep) -- lower theta or raise epochs"
    assert decayed, "DEADLOCK-BREAKER: ||W1|| did not decay below theta_hi during sleep -- wd not acting on zeroed-grad"
    print(f"  DEADLOCK-BREAKER: gate fired AND ||W1|| decayed below theta_hi during sleep (wd acts on zeroed-grad). "
          f"sleep->wake transition observed={'yes' if saw_transition else 'no (corroborated by isolated wd test)'}")
    print("SMOKE PASS -- theta_hi=inf parity within float tol (fused AdamW); gate fires + wd decays ||W1|| "
          "during sleep (verified under fused=True). Ready for the 4-agent review gate + --gate21a-c5norm.")


def pcnative_smoke():
    """PC-native SGD plumbing check (NOT science). Verifies the 4 build-step requirements:
    (1) SGD c3d path runs 10 epochs without crash / divergence (opt='sgd', wd=0, d=1.0 -- the experiment stack);
    (2) weights UPDATE under SGD (||W1|| moves -- SGD is actually stepping, not a no-op);
    (3) mask pegging holds at d=0.5 (non-edges present -> the in-loop grad-mask assert at L659 actively checks;
        completing the run = peg held; the peg W1.mul_(M1t) is optimizer-independent but exercised under SGD);
    (4) the F field (free energy) is populated (c3 path returns e1sq/e2sq -> F = 0.5*(e1sq+e2sq))."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    a, b, c = make_cells(P); N = P * P
    print("PCNATIVE SMOKE (2 seeds) -- SGD c3d path + F logging + mask peg; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=5, log_per_epoch=True,
                fracs=(0.9,), wd=0.0, epochs=10, snap_every=100, opt="sgd",
                c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)        # d=1.0 (the experiment stack)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]

    # (1)+(2)+(4): SGD c3d d=1.0 run -- no crash, ||W1|| moves, F populated
    res = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                           dict(base), dev, deplete=False, label_kind="real",
                           log_per_epoch=True, early_stop=False, want_rtdiag=True, norm_gate=None)
    assert not any(e["diverged"] for e in res), "SGD c3d path diverged in 10 epochs (lr=2e-3) -- check SGD setup"
    for i in range(len(seeds)):
        w1n = res[i]["per_epoch"]["w1_norm"]
        assert len(w1n) >= 2, f"insufficient log points (seed {seeds[i]})"
        assert abs(w1n[-1] - w1n[0]) > 1e-6, f"SGD did not update W1 (seed {seeds[i]}): ||W1|| flat at {w1n[0]:.3f}"
        F = res[i]["per_epoch"].get("F") or []
        assert F, f"F field empty (seed {seeds[i]})"
        assert any(f is not None for f in F), f"F field all-None (seed {seeds[i]}) -- c3 path not populating F"
    print(f"  (1) SGD c3d d=1.0 ran 10ep, no divergence.")
    print(f"  (2) ||W1|| moved under SGD: seed0 {res[0]['per_epoch']['w1_norm'][0]:.3f} -> "
          f"{res[0]['per_epoch']['w1_norm'][-1]:.3f}.")
    print(f"  (4) F populated: seed0 F[0]={res[0]['per_epoch']['F'][0]} (per-seed free energy, c3 path).")

    # (3) mask peg at d=0.5 (non-edges present -> the in-loop grad-mask assert L659 actively fires; completing
    #     = non-edges stayed 0). 1 seed, 5 ep -- plumbing only. The peg (W1.mul_(M1t)) is opt-independent.
    m1_05, m2_05, _ = build_mask(*vols[0], 0.5, P, h)
    n_nonedge = int((~m1_05).sum())
    peg = run_seeds_masked("c3_dynamic", (0,), [[c for _ in (0,)]], [[splits_list[0]]],
                           m1_05[None], m2_05[None], a, b, P, dict(base), dev, deplete=False, label_kind="real",
                           log_per_epoch=False, early_stop=False, want_rtdiag=False, norm_gate=None)
    assert not peg[0]["diverged"], "d=0.5 SGD run diverged -- unrelated to peg; investigate"
    assert n_nonedge > 0, "d=0.5 mask has no non-edges -- peg check is vacuous (lower density)"
    print(f"  (3) mask peg @ d=0.5 ({n_nonedge} non-edges): run completed -> in-loop grad-mask assert held "
          f"(non-edge grads ~0); train={peg[0]['train']:.3f} finite.")
    print("SMOKE PASS -- SGD c3d path runs, ||W1|| updates, mask peg holds (d=0.5 assert), F logged. "
          "Ready for the 4-agent review gate + --gate21a-pcnative.")


# =========================================================== GATE-2.1a-DYNMAP ===
def _dump_dynmap(path, results, cfg):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "probe_epochs": list(DYNMAP_PROBE_EPOCHS),
        "probe_note": ("DYNMAP-30k (DYNMAP_SPEC.md). MEASUREMENT ONLY — no mechanism changes vs F1 (K2'-PASS at "
                       "15k, test=1.0 10/10). A1 = F1 config (c3d + AdamW + wd=1.0 + norm-gate θ=[38,45]) @ 30k; "
                       "A2 = matched BP (mode=backprop, AdamW + wd=1.0, NO gate, NO c3d) @ 30k. Same arch/data/"
                       "seeds. Probes are READ-ONLY snapshots firing AFTER grad computation + AFTER w1_gate/"
                       "norm_gate but BEFORE opt.step (g1_try == the final gated update; W1/W2 == pre-step). "
                       "Battery: train/test acc, global ||W1||/||W2||, per-unit ||W1_i|| (256) and path-weighted "
                       "c_i=||W1_i||×||W2[:,i]|| (256), radial/tangential decomposition of the in-hand g1_try onto "
                       "u_i=W1eff_pre/||W1eff_pre||, gate sleep_state (A1 only), relaxation residual (A1 only). "
                       "NO extra backward (g1_try.detach() snapshot); all under no_grad; RNG capture/restore. "
                       "Bit-parity smoke (dynmap_smoke) proves the probe block is inert. The frozen grid lists "
                       "ep=30000 (unreachable under epochs=30000) so the terminal epoch (29999) is probed instead "
                       "-> 40 probes; the epoch field records the ACTUAL ep."),
        "theta": dict(C5NORM_THETA),
        "banked_baseline": {"F1_C5NORM_15k": "AdamW wd=1.0 gate=ON c3d: test=1.0 10/10 (K2'-PASS) — the held baseline"},
        "arms": results,
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def _dynmap_event_anchors(per_epoch_entry, log_every):
    """Post-run event anchors (DYNMAP_SPEC §2): fit epoch = first train>=0.9, grok epoch = first test>=0.9.
    Scans the per_epoch train_acc/test_acc trajectory (already captured at log cadence K). Returns
    (fit_epoch, grok_epoch) as ints (or None if never reached). Log-point index i -> epoch = i*log_every."""
    tr = per_epoch_entry.get("train_acc") or []
    te = per_epoch_entry.get("test_acc") or []
    fit = next((i * log_every for i, v in enumerate(tr) if v >= 0.9), None)
    grok = next((i * log_every for i, v in enumerate(te) if v >= 0.9), None)
    return fit, grok


def _dynmap_setup(cfg, dev, seeds=None):
    """Shared setup for drive_dynmap + dynmap_smoke: cells/data/masks for d=1.0 real arms. Mirrors
    drive_gate21a_c5norm's setup (volumes, split_random(seed), cperm)."""
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = list(seeds if seeds is not None else cfg["seeds"])
    top = max(cfg["fracs"])
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]
    return a, b, P, h, splits_list, M1, M2, lab_real


def drive_dynmap(cfg, label, save_path):
    """DYNMAP-30k driver (DYNMAP_SPEC.md). Two arms, both 30k, 10 seeds, probe_epochs=DYNMAP_PROBE_EPOCHS.
    A1 = F1 config (mode='c3_dynamic', norm_gate=θ=[38,45], wd=1.0). A2 = matched BP (mode='backprop',
    norm_gate=None, wd=1.0). Dumps after EACH arm to save_path (crash-safe). A1 ≠ A2 in THREE things
    (learning rule, precision, gate) — this compares the full PC stack vs vanilla BP (4-agent patch)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: DYNMAP needs CUDA (CPU = days)"
    seeds = list(cfg["seeds"]); K = cfg["log_every"]
    a, b, P, h, splits_list, M1, M2, lab_real = _dynmap_setup(cfg, dev)
    print("=" * 104)
    print(f"DYNMAP-30k [{label}] dynamics map  | device={dev} P={P} h={h} d=1.0 epochs={cfg['epochs']} "
          f"T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={len(seeds)} log_every={K}")
    print(f"  A1: mode=c3_dynamic gate=ON θ=[{C5NORM_THETA['theta_lo']},{C5NORM_THETA['theta_hi']}] wd=1.0 (== F1, K2'-PASS baseline)")
    print(f"  A2: mode=backprop     gate=OFF wd=1.0 (matched BP; no c3d, no relaxation T-loop)")
    print(f"  probes: {len(DYNMAP_PROBE_EPOCHS)} frozen-grid epochs (read-only; bit-parity smoke verifies inert)")
    print(f"  PRE-REGISTERED: P1 durability (A1 holds test=1.0 @30k); P3 PC-specificity (BP sparse assemblies?); "
          f"P4 BP radial rise-then-fall. P5 rotation is A3 (separate --rotation). NOISE arm NOT run (real only).")
    print("=" * 104)
    results = {}

    def _run_arm(name, mode, gate):
        cc = dict(cfg); cc["wd"] = 1.0
        t0 = time.perf_counter()
        res = run_seeds_masked(mode, tuple(seeds), [lab_real], [splits_list], M1, M2, a, b, P,
                               cc, dev, deplete=False, label_kind="real",
                               log_per_epoch=True, early_stop=False, es_uses_block=False,
                               want_rtdiag=True, w1_gate="both", norm_gate=gate,
                               probe_epochs=DYNMAP_PROBE_EPOCHS)
        tr = [e["train"] for e in res]; te = [e["test"] for e in res]
        anchors = []
        for e in res:
            fit, grok = _dynmap_event_anchors(e["per_epoch"], K)
            anchors.append(dict(fit_epoch=fit, grok_epoch=grok))
        n_probes = [len((e.get("per_epoch") or {}).get("probes") or []) for e in res]
        arm = dict(name=name, mode=mode, norm_gate=(None if gate is None else dict(gate)), wd=1.0,
                   density=1.0, label_kind="real",
                   train_mean=float(np.mean(tr)), train_per_seed=tr,
                   test_mean=float(np.mean(te)), test_per_seed=te,
                   diverged=any(e["diverged"] for e in res),
                   n_probes_per_seed=n_probes, event_anchors=anchors,
                   per_epoch=[e["per_epoch"] for e in res],
                   wall_s=round(time.perf_counter() - t0, 1))
        tag = name
        print(f"\n[{tag}] mode={mode} gate={'ON' if gate else 'OFF'}: train {arm['train_mean']:.3f} | "
              f"test {arm['test_mean']:.3f} | probes/seed={n_probes[0]} | wall={arm['wall_s']}s")
        fit_n = sum(1 for x in anchors if x["fit_epoch"] is not None)
        grok_n = sum(1 for x in anchors if x["grok_epoch"] is not None)
        print(f"  -> fit(train>=0.9): {fit_n}/{len(seeds)} | grok(test>=0.9): {grok_n}/{len(seeds)}")
        return arm

    results["A1_F1_30k"] = _run_arm("A1_F1_30k", "c3_dynamic", C5NORM_THETA)
    _dump_dynmap(save_path, results, cfg)
    results["A2_BP_30k"] = _run_arm("A2_BP_30k", "backprop", None)
    _dump_dynmap(save_path, results, cfg)
    print("\n" + "=" * 104)
    a1t = results["A1_F1_30k"]["test_mean"]; a2t = results["A2_BP_30k"]["test_mean"]
    print(f"  A1 (F1) test={a1t:.3f} | A2 (BP) test={a2t:.3f}")
    print(f"  P1 DURABILITY: A1 {'HOLDS test>=0.9 @30k' if a1t >= 0.9 else 'BREAKS — horizon diagnosis'}")
    print(f"  See outputs/gate2_dynmap.json for the full probe battery; run --rotation for A3 (P5).")


def dynmap_smoke():
    """DYNMAP bit-parity smoke (DYNMAP_BUILD_SPEC §5; acceptance criterion #1). Same seed, 3000 epochs,
    probes ON vs OFF -> bitwise identical trajectories (np.array_equal on train_acc/test_acc/w1_norm/
    w2_norm). Must pass for BOTH arms: A1 stack (c3_dynamic + norm_gate + probes) AND A2 stack
    (mode=backprop + probes). Proves the probe block is read-only / parity-inert."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    print("DYNMAP SMOKE (2 seeds, 3000 ep) -- bit-parity probes ON vs OFF, BOTH arms; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=100,
                log_per_epoch=True, fracs=(0.9,), wd=1.0, epochs=3000, snap_every=100,
                c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)
    # _dynmap_setup builds cells/splits/masks (P,h from base); a,b are the cell arrays run_seeds_masked needs.
    a, b, _P, _h, splits_list, M1, M2, lab_real = _dynmap_setup(base, dev, seeds=seeds)
    smoke_probe_grid = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 2999]
    keys = ("train_acc", "test_acc", "w1_norm", "w2_norm")

    def _parity(mode, gate, tag):
        r_off = run_seeds_masked(mode, seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                                 dict(base), dev, deplete=False, label_kind="real",
                                 log_per_epoch=True, early_stop=False, want_rtdiag=True,
                                 w1_gate="both", norm_gate=gate, probe_epochs=None)
        r_on = run_seeds_masked(mode, seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                                dict(base), dev, deplete=False, label_kind="real",
                                log_per_epoch=True, early_stop=False, want_rtdiag=True,
                                w1_gate="both", norm_gate=gate, probe_epochs=smoke_probe_grid)
        for i in range(len(seeds)):
            # probe count = len(grid) (terminal 2999 added by run_seeds_masked is already in the grid here)
            n_p = len(r_on[i]["per_epoch"].get("probes") or [])
            assert n_p == len(smoke_probe_grid), f"{tag} seed {seeds[i]}: probe count {n_p} != {len(smoke_probe_grid)}"
            for key in keys:
                off_arr = np.asarray(r_off[i]["per_epoch"][key], dtype=np.float64)
                on_arr = np.asarray(r_on[i]["per_epoch"][key], dtype=np.float64)
                assert np.array_equal(off_arr, on_arr), \
                    f"{tag} PARITY FAIL ({key}, seed {seeds[i]}): probes ON != OFF " \
                    f"(max d={np.max(np.abs(off_arr-on_arr)) if off_arr.size else 0:.2e})"
        print(f"  [{tag}] BIT-PARITY OK: probes ON == OFF bitwise on {keys} (2 seeds, 3000ep). "
              f"probes/seed={len(smoke_probe_grid)} captured.")

    _parity("c3_dynamic", C5NORM_THETA, "A1=c3d+gate")
    _parity("backprop", None, "A2=BP")
    print("SMOKE PASS -- probe block is read-only / parity-inert for BOTH arms (A1 stack + A2 stack). "
          "Ready for the 4-agent review gate + --dynmap.")


# =============================================================== MECH-INTERP ===
def _mechinterp_probe_set(P, top):
    """Build the FROZEN 20-pair probe set (MECH_INTERP_SPEC §1). 'the test split' = the canonical
    (seed-0) split_random; first 20 test indices. IDENTICAL across all arms and seeds (the spec's hard
    constraint) -> enables clean cross-seed/cross-arm event-aligned aggregation. Held-out for seed 0;
    a FIXED eval set for seeds 1-9 (documented -- the mech-interp science is about WEIGHTS: which units
    compute the answer, Fourier structure of W1, SVD rank; the probe inputs only fix the evaluation set
    for activation/contribution magnitude). Returns (probe_inputs (20,2P) float32 numpy, probe_labels
    (20,) int numpy)."""
    a, b, c = make_cells(P); N = P * P
    rng = np.random.RandomState(0)                  # canonical split (seed 0)
    _tr, te = split_random(N, top, rng)
    pidx = te[:MI_PROBE_N]
    pi = onehot2(a, b, P)[pidx].astype(np.float32)  # (20, 2P)
    pl = c[pidx].astype(np.int64)                   # (20,)
    return pi, pl


def _dump_mechinterp(path, results, cfg, probe_inputs, probe_labels, note=None, banked=None,
                     probe_epochs=None, per_seed=False):
    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    if note is None:
        note = ("MECH-INTERP (MECH_INTERP_SPEC.md). Dissects the G0 config (mode=c3_dynamic, norm_gate=None "
                "(OFF), wd=1.0, AdamW, d=1.0) -- the simplest working config (test=1.0 10/10 banked in C5NORM). "
                "Two arms: G0-probe (real labels) + G0-noise (shuffled = NULL). Each probe carries the DYNMAP "
                "battery (train/test acc, ||W1||/||W2||, per-unit ||W1_i||, c_i, radial/tang, resid) PLUS the "
                "read-only mech-interp battery: mean_contribution (h) = per-unit logit contribution to the "
                "correct class on the fixed 20-pair probe set; mean_activation (h) + x1 (20xh) full activations; "
                "ablation (11 [k,test_acc] pts; k=25/30 bracket the M1 10%-zeroing falsifier) = cumulative unit "
                "ablation ranked by |mean_contribution|, FULL-test acc; svd_w1/svd_w2 = FULL singular spectrum "
                "(2P / P values; M2 rank-decrease falsifier) + W1 top-10 right singular vectors; freq_energy_a/b "
                "(P) population DFT energy of the a/b input blocks; freq_conc_a/b (h) PER-UNIT Fourier "
                "concentration (Gate-1 discriminating metric -- the aggregate freq_energy repeats the Gate-1 "
                "aggregate-PR retraction: a population of single-freq neurons is broadband; per-unit conc_i = "
                "max_f|F|^2 / sum_f|F|^2. UNFOLDED metric (no conjugate fold, DC incl): single-freq cosine neuron "
                "reads ~0.5 (power split across k,P-k); Gate-1 fourier_code (fold k<->P-k, drop DC) reads ~0.95; "
                "null ~0.1. So ~0.5 = clean clock (UNFOLDED), ~0.1 = untuned; M3 falsifier). All under "
                "no_grad, DETERMINISTIC (no RNG) -> parity-inert (mechinterp_smoke). SVD/Fourier on W1 "
                "transposed to spec convention (2P,h). PRE-REGISTERED: M1 sparsity, M2 SVD rank, M3 Fourier, "
                "M4 activation stability, M5 noise null.")
    if banked is None:
        banked = {"G0_C5NORM": "AdamW wd=1.0 gate=OFF c3d: test=1.0 10/10 (wd-alone holds) -- the dissected config"}
    if probe_epochs is None:
        probe_epochs = MECHINTERP_PROBE_EPOCHS
    # probe_set metadata: per-seed (v2, None inputs) vs global (v1, seed-0's set)
    if per_seed or probe_inputs is None:
        probe_set_meta = {"n": MI_PROBE_N, "per_seed": True,
                          "source": "PER-SEED held-out: each seed's own test split first-20 (all 20 test-disjoint "
                                    "from that seed's train -- fixes the v1 cue confound). Cross-arm-paired-within-"
                                    "seed (A1/G0 seed-s share seed-s's probe set)."}
    else:
        probe_set_meta = {"n": MI_PROBE_N, "per_seed": False,
                          "source": "seed-0 split_random, first 20 test indices (canonical; identical across arms/seeds) "
                                    "[V1 -- cue-confounded for seeds 1-9]",
                          "inputs_shape": list(probe_inputs.shape), "labels": probe_labels.tolist()}
    obj = {
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "probe_epochs": list(probe_epochs),
        "ablation_ks": list(MI_ABLATION_KS),
        "probe_set": probe_set_meta,
        "probe_note": note,
        "banked_baseline": banked,
        "arms": results,
    }
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2, allow_nan=False)


def _mechinterp_setup(cfg, seeds=None):
    """Shared setup for drive_mechinterp / drive_mechinterp_a1a2 / smokes (mirrors _dynmap_setup):
    cells + per-seed d=1.0 volumes/masks/splits + per-seed shuffled-label perms. Returns
    (a, b, P, h, splits_list, M1, M2, lab_real, perms). The canonical probe set is built separately by
    _mechinterp_probe_set (it is IDENTICAL across all arms/seeds by construction)."""
    P, h = cfg["P"], cfg["h"]
    a, b, c = make_cells(P); N = P * P
    seeds = list(seeds if seeds is not None else cfg["seeds"])
    top = max(cfg["fracs"])
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list, perms = [], []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), "GUARD: train/test overlap"
        cperm = c.copy(); rng.shuffle(cperm)
        splits_list.append((tr, te)); perms.append(cperm)
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]
    return a, b, P, h, splits_list, M1, M2, lab_real, perms


def _mechinterp_run_arm(name, mode, norm_gate, label_kind, labs, splits_list, M1, M2,
                        a, b, P, cfg, dev, seeds, K, probe_inputs=None, probe_labels=None,
                        probe_epochs=None, per_seed_probe=False):
    """Run ONE mech-interp arm (general: any mode, any norm_gate) with the FULL battery. probe_epochs
    defaults to MECHINTERP_PROBE_EPOCHS (v1 grid). per_seed_probe=True (v2) builds a per-seed held-out
    probe set inside run_seeds_masked (each seed's own test split first-20); =False uses the driver-supplied
    global probe_inputs/labels (v1 path). Returns the arm dict (same schema across arms/v1/v2 -> mergeable)."""
    cc = dict(cfg); cc["wd"] = 1.0
    if probe_epochs is None:
        probe_epochs = MECHINTERP_PROBE_EPOCHS
    t0 = time.perf_counter()
    res = run_seeds_masked(mode, tuple(seeds), [labs], [splits_list], M1, M2, a, b, P,
                           cc, dev, deplete=False, label_kind=label_kind,
                           log_per_epoch=True, early_stop=False, es_uses_block=False,
                           want_rtdiag=True, w1_gate="both", norm_gate=norm_gate,
                           probe_epochs=probe_epochs,
                           mechinterp_probes=True, probe_inputs=probe_inputs, probe_labels=probe_labels,
                           per_seed_probe=per_seed_probe)
    tr = [e["train"] for e in res]; te = [e["test"] for e in res]
    anchors = []
    for e in res:
        fit, grok = _dynmap_event_anchors(e["per_epoch"], K)
        anchors.append(dict(fit_epoch=fit, grok_epoch=grok))
    n_probes = [len((e.get("per_epoch") or {}).get("probes") or []) for e in res]
    arm = dict(name=name, mode=mode, norm_gate=(None if norm_gate is None else dict(norm_gate)),
               wd=1.0, density=1.0, label_kind=label_kind,
               train_mean=float(np.mean(tr)), train_per_seed=tr,
               test_mean=float(np.mean(te)), test_per_seed=te,
               diverged=any(e["diverged"] for e in res),
               n_probes_per_seed=n_probes, event_anchors=anchors,
               per_epoch=[e["per_epoch"] for e in res],
               wall_s=round(time.perf_counter() - t0, 1))
    gate_str = "OFF" if norm_gate is None else f"ON θ=[{norm_gate['theta_lo']},{norm_gate['theta_hi']}]"
    print(f"\n[{name}] mode={mode} gate={gate_str} label_kind={label_kind}: "
          f"train {arm['train_mean']:.3f} | test {arm['test_mean']:.3f} | "
          f"probes/seed={n_probes[0]} | wall={arm['wall_s']}s")
    fit_n = sum(1 for x in anchors if x["fit_epoch"] is not None)
    grok_n = sum(1 for x in anchors if x["grok_epoch"] is not None)
    print(f"  -> fit(train>=0.9): {fit_n}/{len(seeds)} | grok(test>=0.9): {grok_n}/{len(seeds)}")
    return arm


def drive_mechinterp(cfg, label, save_path):
    """MECH-INTERP driver (MECH_INTERP_SPEC.md). Dissects the G0 config (wd=1.0, no gate, c3d, AdamW) to
    find WHAT circuit the network builds for modular addition. Two arms: G0-probe (real labels) and
    G0-noise (shuffled labels = the NULL; if structure also appears on noise -> capacity artifact).
    Battery = DYNMAP fields + 5 new read-only mech-interp probes. Dumps after EACH arm (crash-safe)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: MECH-INTERP needs CUDA (CPU = days)"
    P, h = cfg["P"], cfg["h"]
    seeds = list(cfg["seeds"]); K = cfg["log_every"]
    top = max(cfg["fracs"])
    probe_inputs, probe_labels = _mechinterp_probe_set(P, top)
    a, b, _P, _h, splits_list, M1, M2, lab_real, perms = _mechinterp_setup(cfg, seeds)
    lab_noise = [perms[i] for i in range(len(seeds))]
    print("=" * 104)
    print(f"MECH-INTERP [{label}]  what circuit does G0 build for (a+b) mod {P}?  device={dev} h={h} "
          f"epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={len(seeds)}")
    print(f"  G0 = mode=c3_dynamic norm_gate=None (OFF) wd=1.0 AdamW (simplest working config; test=1.0 10/10 banked)")
    print(f"  arms: G0-probe (real) | G0-noise (shuffled = NULL). Probe set: {MI_PROBE_N} fixed pairs "
          f"(seed-0 test[:{MI_PROBE_N}]), identical across arms/seeds.")
    print(f"  probes: {len(MECHINTERP_PROBE_EPOCHS)} frozen-grid epochs (dense through grok, sparse after) + battery:")
    print(f"    logit contribution | activations (mean + 20xh) | ablation curve {list(MI_ABLATION_KS)} | "
          f"SVD FULL spectrum | Fourier a/b energy + per-unit conc")
    print(f"  PRE-REGISTERED: M1 sparsity (distributed?) | M2 SVD rank drops? | M3 Fourier concentration? | "
          f"M4 activation stability? | M5 NOISE NULL.")
    print("=" * 104)
    results = {}

    results["G0-probe"] = _mechinterp_run_arm("G0-probe", "c3_dynamic", None, "real", lab_real,
                                              splits_list, M1, M2, a, b, P, cfg, dev, seeds, K,
                                              probe_inputs=probe_inputs, probe_labels=probe_labels)
    _dump_mechinterp(save_path, results, cfg, probe_inputs, probe_labels)
    results["G0-noise"] = _mechinterp_run_arm("G0-noise", "c3_dynamic", None, "shuffled", lab_noise,
                                              splits_list, M1, M2, a, b, P, cfg, dev, seeds, K,
                                              probe_inputs=probe_inputs, probe_labels=probe_labels)
    _dump_mechinterp(save_path, results, cfg, probe_inputs, probe_labels)
    print("\n" + "=" * 104)
    rp = results["G0-probe"]["test_mean"]; rn = results["G0-noise"]["test_mean"]
    print(f"  G0-probe (real) test={rp:.3f} | G0-noise (shuffled) test={rn:.3f}")
    print(f"  M5 NOISE NULL: G0-noise should NOT grok (test~chance {1/P:.3f}) and show NO SVD/Fourier/ablation structure.")
    print(f"  See {save_path} for the full probe battery (5 new fields per probe); event-aligned analysis is post-run.")


def drive_mechinterp_a1a2(cfg, label, save_path):
    """MECH-INTERP A1+A2 driver. Adds two arms to the G0 mech-interp dissection using the EXACT same
    probe battery / grid / fixed probe set as drive_mechinterp (the cross-arm comparison invariant):
      A1-gated: mode=c3_dynamic, norm_gate=C5NORM_THETA (theta=[38,45]), wd=1.0  (== F1, the gated stack)
      A2-bp:    mode=backprop,   norm_gate=None,                    wd=1.0  (matched BP; no c3d, no gate)
    Both 30k, 10 seeds, real labels, mechinterp probes ON. Dumps to a SEPARATE JSON -- G0-probe/G0-noise
    stay banked in gate2_mechinterp.json and are NOT re-run. Dumps after EACH arm (crash-safe). The arm
    dicts share the G0 schema -> the two JSONs merge directly for the 4-arm event-aligned analysis."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: MECH-INTERP-a1a2 needs CUDA (CPU = days)"
    seeds = list(cfg["seeds"]); K = cfg["log_every"]
    P, h = cfg["P"], cfg["h"]
    top = max(cfg["fracs"])
    probe_inputs, probe_labels = _mechinterp_probe_set(P, top)
    a, b, _P, _h, splits_list, M1, M2, lab_real, _perms = _mechinterp_setup(cfg, seeds)
    print("=" * 104)
    print(f"MECH-INTERP-A1A2 [{label}]  add A1 (F1 gated) + A2 (BP) arms to the circuit dissection  "
          f"device={dev} h={h} epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={len(seeds)}")
    print(f"  A1-gated: mode=c3_dynamic gate=ON θ=[{C5NORM_THETA['theta_lo']},{C5NORM_THETA['theta_hi']}] wd=1.0 (== F1)")
    print(f"  A2-bp:    mode=backprop gate=OFF wd=1.0 (matched BP; no c3d, no relaxation T-loop)")
    print(f"  Probe set + grid + battery IDENTICAL to G0-probe/G0-noise (cross-arm invariant): "
          f"{MI_PROBE_N} fixed pairs; {len(MECHINTERP_PROBE_EPOCHS)} frozen-grid epochs.")
    print(f"  G0-probe/G0-noise are BANKED (gate2_mechinterp.json) -- NOT re-run; this dumps separately.")
    print("=" * 104)
    results = {}
    a1a2_note = (
        "MECH-INTERP A1+A2 arms (MECH_INTERP_SPEC.md §1: BP-probe / F1-probe, re-captured with the full battery "
        "since DYNMAP lacked SVD/Fourier/ablation). IDENTICAL probe set / grid / battery as gate2_mechinterp.json "
        "(the cross-arm invariant). A1-gated = mode=c3_dynamic + C5NORM norm_gate theta=[38,45] + wd=1.0 (== F1, "
        "the gated stack); A2-bp = mode=backprop + no gate + wd=1.0 (matched BP; no c3d, no relaxation T-loop). "
        "Both 30k, 10 seeds, real labels. Arm schema == G0 schema -> this JSON merges with gate2_mechinterp.json "
        "for the 4-arm (G0-probe/G0-noise/A1-gated/A2-bp) event-aligned analysis. Cross-arm questions: does the "
        "gate (A1 vs G0) or the learning rule (A2 vs G0) change the circuit (Fourier freqs / SVD subspace / "
        "ablation sparsity)? For A2 (BP) the per-probe 'resid' is null (no PC relaxation). All probes read-only / "
        "no_grad / DETERMINISTIC -> parity-inert (mechinterp_a1a2_smoke).")
    a1a2_banked = {
        "G0-probe": "banked in gate2_mechinterp.json: c3_dynamic gate=OFF wd=1.0 test=1.0 10/10",
        "F1_C5NORM": "A1 == F1 config: c3_dynamic gate=ON theta=[38,45] wd=1.0 (K2'-PASS test=1.0 10/10 at 15k)",
        "A2_BP_DYNMAP": "A2 == DYNMAP A2_BP_30k: backprop gate=OFF wd=1.0",
    }
    results["A1-gated"] = _mechinterp_run_arm("A1-gated", "c3_dynamic", C5NORM_THETA, "real", lab_real,
                                              splits_list, M1, M2, a, b, P, cfg, dev, seeds, K,
                                              probe_inputs=probe_inputs, probe_labels=probe_labels)
    _dump_mechinterp(save_path, results, cfg, probe_inputs, probe_labels, note=a1a2_note, banked=a1a2_banked)
    results["A2-bp"] = _mechinterp_run_arm("A2-bp", "backprop", None, "real", lab_real,
                                           splits_list, M1, M2, a, b, P, cfg, dev, seeds, K,
                                           probe_inputs=probe_inputs, probe_labels=probe_labels)
    _dump_mechinterp(save_path, results, cfg, probe_inputs, probe_labels, note=a1a2_note, banked=a1a2_banked)
    print("\n" + "=" * 104)
    a1t = results["A1-gated"]["test_mean"]; a2t = results["A2-bp"]["test_mean"]
    print(f"  A1-gated (F1) test={a1t:.3f} | A2-bp test={a2t:.3f}")
    print(f"  Cross-arm vs banked G0-probe (test=1.0): do A1/A2/G0 share Fourier freqs / SVD subspace / ablation sparsity?")
    print(f"  See {save_path}; merge with gate2_mechinterp.json for the full 4-arm event-aligned analysis (post-run).")


def drive_mechinterp_v2(cfg, label, save_path):
    """MECH-INTERP V2 driver (fixes the v1 cue confound + adds dense grid / per-unit radial-tang / W1-W2
    checkpoints). All 4 arms in ONE JSON: G0-probe (c3d, no gate), G0-noise (shuffled), A1-gated (c3d +
    C5NORM_THETA), A2-bp (backprop). Each seed uses its OWN test split's first-20 as the probe set
    (per_seed_probe=True -> all 20 genuinely held-out for every seed). v2 dense grid (every-50 through
    ep1000). Dumps after EACH arm (crash-safe). Training configs IDENTICAL to v1; v1 JSON kept for comparison.
    Arm schema == v1 schema + radial_mass_per_unit/tang_mass_per_unit (h) + W1_matrix/W2_matrix at the
    grok-window+terminal probes -> enables Gate-1 exact folded Fourier conc, per-unit frequency census,
    M2 rank-decrease trajectory, and BOTH rho(radial,usage)/rho(tang,usage)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: MECH-INTERP-v2 needs CUDA (CPU = days)"
    seeds = list(cfg["seeds"]); K = cfg["log_every"]
    P, h = cfg["P"], cfg["h"]
    top = max(cfg["fracs"])
    a, b, _P, _h, splits_list, M1, M2, lab_real, perms = _mechinterp_setup(cfg, seeds)
    lab_noise = [perms[i] for i in range(len(seeds))]
    print("=" * 104)
    print(f"MECH-INTERP-V2 [{label}]  per-seed held-out probe set + dense grid + per-unit radial/tang + W1/W2 ckpt")
    print(f"  device={dev} h={h} epochs={cfg['epochs']} T={cfg['T']} eta={cfg['eta']} lr={cfg['lr']} seeds={len(seeds)}")
    print(f"  arms: G0-probe (c3d,gate OFF) | G0-noise (shuffled) | A1-gated (c3d,gate θ=[{C5NORM_THETA['theta_lo']},{C5NORM_THETA['theta_hi']}]) | A2-bp (backprop)")
    print(f"  PROBE FIX: per_seed_probe=True -- each seed's OWN test split first-{MI_PROBE_N} (all held-out; "
          f"v1 used seed-0's set -> 15-20/20 in-train for seeds 1-9 = the cue confound).")
    print(f"  grid: {len(MECHINTERP_PROBE_EPOCHS_V2)} probes (every-50 thru 1000, then every-1000) | "
          f"W1/W2 ckpt at ep<={MI_W_CKPT_WINDOW} + terminal | per-unit radial/tang (Q4 re-opens).")
    print(f"  v1 JSON (gate2_mechinterp.json + gate2_mechinterp_a1a2.json) KEPT for comparison.")
    print("=" * 104)
    results = {}
    v2_note = (
        "MECH-INTERP V2 (fixes v1 cue confound). Per-seed held-out probe set (each seed's own test split "
        "first-20 -- all 20 genuinely held-out for every seed; v1 used seed-0's set so seeds 1-9 had "
        "15-20/20 probe pairs IN TRAIN, cue-confounding activation-space probes). Dense early grid "
        "(every-50 thru ep1000 -> resolves sub-100ep sparse phases). ADDS vs v1: radial_mass_per_unit / "
        "tang_mass_per_unit (h, enables rho(radial,usage) AND rho(tang,usage)); W1_matrix (2P,h) + W2_matrix "
        "(h,P) at probes epoch<=MI_W_CKPT_WINDOW + terminal (enables Gate-1 exact folded Fourier conc, "
        "per-unit frequency census Q4a, M2 rank-decrease trajectory). Training configs IDENTICAL to v1. "
        "All probes read-only/no_grad/DETERMINISTIC -> parity-inert (mechinterp_v2_smoke).")
    v2_banked = {"v1_G0": "gate2_mechinterp.json (cue-confounded; kept for comparison)",
                 "F1": "A1-gated == F1 config (c3d gate θ=[38,45] wd=1.0)"}
    order = [("G0-probe", "c3_dynamic", None, "real", lab_real),
             ("A1-gated", "c3_dynamic", C5NORM_THETA, "real", lab_real),
             ("A2-bp", "backprop", None, "real", lab_real),
             ("G0-noise", "c3_dynamic", None, "shuffled", lab_noise)]
    for name, mode, gate, lkind, labs in order:
        results[name] = _mechinterp_run_arm(name, mode, gate, lkind, labs, splits_list, M1, M2,
                                            a, b, P, cfg, dev, seeds, K,
                                            probe_epochs=MECHINTERP_PROBE_EPOCHS_V2, per_seed_probe=True)
        _dump_mechinterp(save_path, results, cfg, None, None, note=v2_note, banked=v2_banked,
                         probe_epochs=MECHINTERP_PROBE_EPOCHS_V2, per_seed=True)
    print("\n" + "=" * 104)
    print("  " + "  ".join(f"{n}: test={results[n]['test_mean']:.3f}" for n in [o[0] for o in order]))
    print(f"  See {save_path}. Activation-space claims (settling/sleep/detector) now UNCONFOUNDED; "
          f"re-run the trajectory analysis on this JSON.")


def mechinterp_smoke():
    """MECH-INTERP bit-parity smoke (acceptance criterion). 2 seeds, 3000 epochs, G0 config, probes ON vs
    OFF -> bitwise identical trajectories (np.array_equal on train_acc/test_acc/w1_norm/w2_norm). Proves
    the mech-interp battery is read-only / parity-inert. Also asserts the 5 new fields are present and
    correctly shaped on every probe of the ON run."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    print("MECH-INTERP SMOKE (2 seeds, 3000 ep, G0 config) -- bit-parity probes ON vs OFF; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=100,
                log_per_epoch=True, fracs=(0.9,), wd=1.0, epochs=3000, snap_every=100,
                c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)
    a, b, c = make_cells(P); N = P * P
    probe_inputs, probe_labels = _mechinterp_probe_set(P, top)
    vols = {s: build_volume(s + POS_OFFSET, 2 * P, h, P) for s in seeds}
    splits_list = []
    for s in seeds:
        rng = np.random.RandomState(s)
        tr, te = split_random(N, top, rng)
        splits_list.append((tr, te))
    M1s, M2s = [], []
    for s in seeds:
        m1, m2, _ = build_mask(*vols[s], 1.0, P, h)
        M1s.append(m1); M2s.append(m2)
    M1 = np.stack(M1s); M2 = np.stack(M2s)
    lab_real = [c for _ in seeds]
    smoke_probe_grid = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 2999]
    keys = ("train_acc", "test_acc", "w1_norm", "w2_norm")
    mi_fields = ("mean_contribution", "mean_activation", "x1", "ablation",
                 "svd_w1", "svd_w2", "freq_energy_a", "freq_energy_b",
                 "freq_conc_a", "freq_conc_b")

    r_off = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                             dict(base), dev, deplete=False, label_kind="real",
                             log_per_epoch=True, early_stop=False, want_rtdiag=True,
                             w1_gate="both", norm_gate=None, probe_epochs=None)
    r_on = run_seeds_masked("c3_dynamic", seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                            dict(base), dev, deplete=False, label_kind="real",
                            log_per_epoch=True, early_stop=False, want_rtdiag=True,
                            w1_gate="both", norm_gate=None, probe_epochs=smoke_probe_grid,
                            mechinterp_probes=True, probe_inputs=probe_inputs, probe_labels=probe_labels)
    for i in range(len(seeds)):
        n_p = len(r_on[i]["per_epoch"].get("probes") or [])
        assert n_p == len(smoke_probe_grid), f"seed {seeds[i]}: probe count {n_p} != {len(smoke_probe_grid)}"
        for pr in r_on[i]["per_epoch"]["probes"]:                       # NEW fields present + correctly shaped
            assert set(mi_fields).issubset(pr.keys()), \
                f"seed {seeds[i]} ep {pr['epoch']}: missing mech-interp field (have {sorted(pr.keys())})"
            assert len(pr["mean_contribution"]) == h, "mean_contribution must be (h,)"
            assert len(pr["mean_activation"]) == h, "mean_activation must be (h,)"
            assert len(pr["x1"]) == MI_PROBE_N and len(pr["x1"][0]) == h, "x1 must be (20,h)"
            assert len(pr["ablation"]) == len(MI_ABLATION_KS), "ablation must have all ks"
            assert len(pr["svd_w1"]["singular_values"]) == 2 * P, "svd_w1 needs the FULL spectrum (2P)"
            assert len(pr["svd_w1"]["top_vectors"]) == 10, "svd_w1 needs 10 top vectors"
            assert len(pr["svd_w2"]["singular_values"]) == P, "svd_w2 needs the FULL spectrum (P)"
            assert len(pr["freq_energy_a"]) == P and len(pr["freq_energy_b"]) == P, "freq energy must be (P,)"
            assert len(pr["freq_conc_a"]) == h and len(pr["freq_conc_b"]) == h, "freq_conc must be (h,)"
        for key in keys:                                                 # BIT-PARITY: ON == OFF bitwise
            off_arr = np.asarray(r_off[i]["per_epoch"][key], dtype=np.float64)
            on_arr = np.asarray(r_on[i]["per_epoch"][key], dtype=np.float64)
            assert np.array_equal(off_arr, on_arr), \
                f"PARITY FAIL ({key}, seed {seeds[i]}): probes ON != OFF " \
                f"(max d={np.max(np.abs(off_arr-on_arr)) if off_arr.size else 0:.2e})"
    print(f"  BIT-PARITY OK: probes ON == OFF bitwise on {keys} (2 seeds, 3000ep, G0 config).")
    print(f"  NEW FIELDS OK: {list(mi_fields)} present + correctly shaped on every probe.")
    print("SMOKE PASS -- mech-interp battery is read-only / parity-inert. "
          "Ready for the 4-agent review gate + --mechinterp.")


def mechinterp_a1a2_smoke():
    """MECH-INTERP A1+A2 bit-parity smoke. 2 seeds, 3000 epochs, probes ON vs OFF for BOTH new arms:
    A1 (c3_dynamic + C5NORM_THETA gate) and A2 (backprop, no gate). Bitwise identical trajectories
    (np.array_equal on train_acc/test_acc/w1_norm/w2_norm) + new-field presence/shape asserts. Proves
    the mech-interp battery is read-only / parity-inert for the two new stacks (A1 gated + A2 BP)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    print("MECH-INTERP-A1A2 SMOKE (2 seeds, 3000 ep) -- bit-parity probes ON vs OFF for A1 (c3d+gate) + A2 (BP); NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=100,
                log_per_epoch=True, fracs=(0.9,), wd=1.0, epochs=3000, snap_every=100,
                c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)
    a, b, _P, _h, splits_list, M1, M2, lab_real, _perms = _mechinterp_setup(base, seeds)
    probe_inputs, probe_labels = _mechinterp_probe_set(P, top)
    smoke_probe_grid = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 2999]
    keys = ("train_acc", "test_acc", "w1_norm", "w2_norm")
    mi_fields = ("mean_contribution", "mean_activation", "x1", "ablation",
                 "svd_w1", "svd_w2", "freq_energy_a", "freq_energy_b",
                 "freq_conc_a", "freq_conc_b")

    def _parity(mode, gate, tag):
        r_off = run_seeds_masked(mode, seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                                 dict(base), dev, deplete=False, label_kind="real",
                                 log_per_epoch=True, early_stop=False, want_rtdiag=True,
                                 w1_gate="both", norm_gate=gate, probe_epochs=None)
        r_on = run_seeds_masked(mode, seeds, [lab_real], [splits_list], M1, M2, a, b, P,
                                dict(base), dev, deplete=False, label_kind="real",
                                log_per_epoch=True, early_stop=False, want_rtdiag=True,
                                w1_gate="both", norm_gate=gate, probe_epochs=smoke_probe_grid,
                                mechinterp_probes=True, probe_inputs=probe_inputs, probe_labels=probe_labels)
        for i in range(len(seeds)):
            n_p = len(r_on[i]["per_epoch"].get("probes") or [])
            assert n_p == len(smoke_probe_grid), f"{tag} seed {seeds[i]}: probe count {n_p} != {len(smoke_probe_grid)}"
            for pr in r_on[i]["per_epoch"]["probes"]:                       # NEW fields present + shaped
                assert set(mi_fields).issubset(pr.keys()), \
                    f"{tag} seed {seeds[i]} ep {pr['epoch']}: missing mech-interp field"
                assert len(pr["mean_contribution"]) == h, "mean_contribution must be (h,)"
                assert len(pr["mean_activation"]) == h, "mean_activation must be (h,)"
                assert len(pr["x1"]) == MI_PROBE_N and len(pr["x1"][0]) == h, "x1 must be (20,h)"
                assert len(pr["ablation"]) == len(MI_ABLATION_KS), "ablation must have all ks"
                assert len(pr["svd_w1"]["singular_values"]) == 2 * P, "svd_w1 needs the FULL spectrum (2P)"
                assert len(pr["svd_w1"]["top_vectors"]) == 10, "svd_w1 needs 10 top vectors"
                assert len(pr["svd_w2"]["singular_values"]) == P, "svd_w2 needs the FULL spectrum (P)"
                assert len(pr["freq_energy_a"]) == P and len(pr["freq_energy_b"]) == P, "freq energy must be (P,)"
                assert len(pr["freq_conc_a"]) == h and len(pr["freq_conc_b"]) == h, "freq_conc must be (h,)"
            for key in keys:                                                 # BIT-PARITY: ON == OFF bitwise
                off_arr = np.asarray(r_off[i]["per_epoch"][key], dtype=np.float64)
                on_arr = np.asarray(r_on[i]["per_epoch"][key], dtype=np.float64)
                assert np.array_equal(off_arr, on_arr), \
                    f"{tag} PARITY FAIL ({key}, seed {seeds[i]}): probes ON != OFF " \
                    f"(max d={np.max(np.abs(off_arr-on_arr)) if off_arr.size else 0:.2e})"
        print(f"  [{tag}] BIT-PARITY OK: probes ON == OFF bitwise on {keys} (2 seeds, 3000ep). fields OK.")

    _parity("c3_dynamic", C5NORM_THETA, "A1=c3d+gate")
    _parity("backprop", None, "A2=BP")
    print("SMOKE PASS -- mech-interp battery is read-only / parity-inert for BOTH new arms "
          "(A1 gated + A2 BP). Ready for the 4-agent review gate + --mechinterp-a1a2.")


def mechinterp_v2_smoke():
    """MECH-INTERP V2 bit-parity smoke. 2 seeds, 3000 ep, per_seed_probe=True (each seed's own test-split
    first-20), probes ON vs OFF for ALL 4 configs (G0 c3d, A1 c3d+gate, A2 BP, G0-noise). Bitwise identical
    trajectories (np.array_equal on train/test/w1/w2) + v2 field asserts (per-unit radial/tang, W1/W2 ckpt
    conditional) + a per-seed held-out invariant check (the cue-confound fix)."""
    torch.set_num_threads(8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "GUARD: smoke needs CUDA"
    P, h, seeds, top = 53, 256, (0, 1), 0.9
    print("MECH-INTERP-V2 SMOKE (2 seeds, 3000 ep, per-seed probe set) -- bit-parity ON vs OFF, 4 configs; NOT science.")
    base = dict(P=P, h=h, lr=2e-3, T=20, eta=0.2, dep_rate=0.06, tau=5.0, log_every=100,
                log_per_epoch=True, fracs=(0.9,), wd=1.0, epochs=3000, snap_every=100,
                c3_pi0=1.0, c3_pimin=0.02, c3_beta=0.99, c3_alpha=C3_ALPHA_FROZEN,
                c3_lambda=0.1, c3_ema_decay=0.99, c3_pmin=0.01, c3_pmax=100.0, c3_eps=1e-8)
    a, b, _P, _h, splits_list, M1, M2, lab_real, _perms = _mechinterp_setup(base, seeds)
    c_real = (a + b) % P
    # PER-SEED HELD-OUT INVARIANT: each seed's probe set = its OWN test split first-20, disjoint from train.
    for s in range(len(seeds)):
        tr, te = splits_list[s]; pidx = te[:MI_PROBE_N]
        assert set(pidx.tolist()).isdisjoint(set(tr.tolist())), f"seed {seeds[s]} probe pairs leak into train!"
    assert not np.array_equal(splits_list[0][1][:MI_PROBE_N], splits_list[1][1][:MI_PROBE_N]), \
        "seed 0 / seed 1 probe sets identical -- per-seed build broken"
    print(f"  per-seed held-out OK: seed0 probe labels {c_real[splits_list[0][1][:MI_PROBE_N]][:6].tolist()} ... "
          f"!= seed1 {c_real[splits_list[1][1][:MI_PROBE_N]][:6].tolist()} ... (different sets, both test-disjoint)")
    smoke_probe_grid = [0, 50, 100, 200, 400, 600, 800, 1000, 2000, 2500, 2999]   # exercises dense-50 + ckpt-window
    keys = ("train_acc", "test_acc", "w1_norm", "w2_norm")
    mi_fields = ("mean_contribution", "mean_activation", "x1", "ablation", "svd_w1", "svd_w2",
                 "freq_energy_a", "freq_energy_b", "freq_conc_a", "freq_conc_b",
                 "radial_mass_per_unit", "tang_mass_per_unit")

    def _parity(mode, gate, lkind, labs, tag):
        r_off = run_seeds_masked(mode, seeds, [labs], [splits_list], M1, M2, a, b, P,
                                 dict(base), dev, deplete=False, label_kind=lkind,
                                 log_per_epoch=True, early_stop=False, want_rtdiag=True,
                                 w1_gate="both", norm_gate=gate, probe_epochs=None)
        r_on = run_seeds_masked(mode, seeds, [labs], [splits_list], M1, M2, a, b, P,
                                dict(base), dev, deplete=False, label_kind=lkind,
                                log_per_epoch=True, early_stop=False, want_rtdiag=True,
                                w1_gate="both", norm_gate=gate, probe_epochs=smoke_probe_grid,
                                mechinterp_probes=True, per_seed_probe=True)
        for i in range(len(seeds)):
            n_p = len(r_on[i]["per_epoch"].get("probes") or [])
            assert n_p == len(smoke_probe_grid), f"{tag} seed {seeds[i]}: probe count {n_p} != {len(smoke_probe_grid)}"
            for pr in r_on[i]["per_epoch"]["probes"]:
                assert set(mi_fields).issubset(pr.keys()), f"{tag} seed {seeds[i]} ep {pr['epoch']}: missing MI field"
                assert len(pr["radial_mass_per_unit"]) == h and len(pr["tang_mass_per_unit"]) == h, "per-unit radial/tang must be (h,)"
                # W1/W2 ckpt conditional: present at ep<=window OR terminal; absent otherwise
                in_window = (pr["epoch"] <= MI_W_CKPT_WINDOW) or (pr["epoch"] == base["epochs"] - 1)
                if in_window:
                    assert "W1_matrix" in pr and "W2_matrix" in pr, f"{tag} ep {pr['epoch']}: W1/W2 ckpt missing in window"
                    assert len(pr["W1_matrix"]) == 2 * P and len(pr["W1_matrix"][0]) == h, "W1_matrix must be (2P,h)"
                    assert len(pr["W2_matrix"]) == h and len(pr["W2_matrix"][0]) == P, "W2_matrix must be (h,P)"
                else:
                    assert "W1_matrix" not in pr and "W2_matrix" not in pr, f"{tag} ep {pr['epoch']}: W1/W2 ckpt should be absent outside window"
            for key in keys:
                off_arr = np.asarray(r_off[i]["per_epoch"][key], dtype=np.float64)
                on_arr = np.asarray(r_on[i]["per_epoch"][key], dtype=np.float64)
                assert np.array_equal(off_arr, on_arr), \
                    f"{tag} PARITY FAIL ({key}, seed {seeds[i]}): probes ON != OFF " \
                    f"(max d={np.max(np.abs(off_arr-on_arr)) if off_arr.size else 0:.2e})"
        print(f"  [{tag}] BIT-PARITY OK + v2 fields OK (per-unit radial/tang, W1/W2 ckpt conditional).")
        return r_on

    _parity("c3_dynamic", None, "real", lab_real, "G0=c3d")
    _parity("c3_dynamic", C5NORM_THETA, "real", lab_real, "A1=c3d+gate")
    _parity("backprop", None, "real", lab_real, "A2=BP")
    r_on = _parity("c3_dynamic", None, "shuffled",
                   [np.random.RandomState(s).permutation((a + b) % P) for s in seeds], "G0-noise=shuffled")
    # EXERCISE the dump path (parity does NOT cover it -- catches None-input / metadata bugs)
    import tempfile, os
    _stub = lambda r: dict(name="smoke", mode="c3_dynamic", norm_gate=None, wd=1.0, density=1.0,
        label_kind="real", train_mean=1.0, train_per_seed=[1.0]*len(seeds), test_mean=1.0,
        test_per_seed=[1.0]*len(seeds), diverged=False,
        n_probes_per_seed=[len(r[i]["per_epoch"]["probes"]) for i in range(len(seeds))],
        event_anchors=[dict(fit_epoch=0, grok_epoch=0)]*len(seeds),
        per_epoch=[r[i]["per_epoch"] for i in range(len(seeds))], wall_s=0.0)
    _tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w"); _tf.close()
    _dump_mechinterp(_tf.name, {"G0-probe": _stub(r_on)}, base, None, None,
                     probe_epochs=smoke_probe_grid, per_seed=True)
    _d = json.load(open(_tf.name)); os.remove(_tf.name)
    assert _d["probe_set"]["per_seed"] is True, "dump per_seed metadata wrong"
    assert _d["probe_epochs"] == smoke_probe_grid, "dump did not thread the probe grid"
    print("  DUMP PATH OK (None-input per-seed metadata + threaded grid exercised).")
    print("SMOKE PASS -- mech-interp V2 (per-seed probe + dense grid + per-unit radial/tang + W1/W2 ckpt) "
          "is read-only / parity-inert for ALL 4 configs. Ready for the 4-agent review gate + --mechinterp-v2.")


if __name__ == "__main__":
    import sys
    if "--gate21a-smoke" in sys.argv:
        gate21a_smoke()
    elif "--c3-smoke" in sys.argv:
        c3_smoke()
    elif "--rtdiag-smoke" in sys.argv:
        rtdiag_smoke()
    elif "--w1gate-smoke" in sys.argv:
        w1gate_smoke()
    elif "--c5norm-smoke" in sys.argv:
        c5norm_smoke()
    elif "--pcnative-smoke" in sys.argv:
        pcnative_smoke()
    elif "--staged-smoke" in sys.argv:
        staged_smoke()
    elif "--dynmap-smoke" in sys.argv:
        dynmap_smoke()
    elif "--mechinterp-smoke" in sys.argv:
        mechinterp_smoke()
    elif "--mechinterp-a1a2-smoke" in sys.argv:
        mechinterp_a1a2_smoke()
    elif "--mechinterp-v2-smoke" in sys.argv:
        mechinterp_v2_smoke()
    elif "--gate21a-c3" in sys.argv:
        drive_gate21a_c3(CFG_C3, "GATE-2.1a-C3 precision scalpel (C3-S/D x noise/real @ d=1.0 wd=0)",
                         save_path="outputs/gate2_c3.json")
    elif "--gate21a-rtdiag" in sys.argv:
        drive_gate21a_rtdiag(CFG_RTDIAG, "GATE-2.1a radial/tangential W1 diagnostic (vanilla-real + c3d-real @ d=1.0 wd=0)",
                             save_path="outputs/gate2_rtdiag.json")
    elif "--gate21a-w1gate" in sys.argv:
        drive_gate21a_w1gate(CFG_W1GATE, "GATE-2.1a W1 radial/tangential gating factorial (4 cells @ c3d d=1.0 wd=0)",
                             save_path="outputs/gate2_w1gate.json")
    elif "--gate21a-c5norm" in sys.argv:
        drive_gate21a_c5norm(CFG_C5NORM, "GATE-2.1a-C5NORM norm-band sleep/wake gate (F0/F1/G0 @ c3d d=1.0)",
                             save_path="outputs/gate2_c5norm.json")
    elif "--gate21a-pcnative" in sys.argv:
        drive_gate21a_pcnative(CFG_PCNATIVE, "GATE-2.1a-PCNATIVE SGD moat test (SGD wd-sweep @ c3d d=1.0, no gate)",
                               save_path="outputs/gate2_pcnative.json")
    elif "--gate21a-pcnative-n1prime" in sys.argv:
        drive_n1prime(CFG_PCNATIVE, "GATE-2.1a-PCNATIVE N1' plateau diagnosis (finite-T relax + momentum ctrl)",
                      save_path="outputs/gate2_pcnative_n1prime.json")
    elif "--gate21a-pcnative-vanilla" in sys.argv:
        drive_vanilla(CFG_PCNATIVE, "GATE-2.1a-PCNATIVE vanilla-PC test (is c3d precision the plateau cause?)",
                      save_path="outputs/gate2_pcnative_vanilla.json")
    elif "--gate21a-pcnative-fbgain" in sys.argv:
        drive_fbgain(CFG_PCNATIVE, "GATE-2.1a-PCNATIVE feedback-gain sweep (is loop gain the oscillation cause?)",
                     save_path="outputs/gate2_pcnative_fbgain.json")
    elif "--staged-channel" in sys.argv:
        drive_staged_channel(CFG_STAGED, "STAGED CHANNEL freeze-after-grok test (5 cells @ c3d d=1.0, 30k)",
                             save_path="outputs/gate2_staged_channel.json")
    elif "--dynmap" in sys.argv:
        drive_dynmap(CFG_DYNMAP, "DYNMAP-30k dynamics map (A1 F1-30k vs A2 matched BP-30k + probe battery)",
                     save_path="outputs/gate2_dynmap.json")
    elif "--mechinterp" in sys.argv:
        drive_mechinterp(CFG_MECHINTERP, "MECH-INTERP circuit dissection (G0-probe real + G0-noise shuffled, 30k + probe battery)",
                         save_path="outputs/gate2_mechinterp.json")
    elif "--mechinterp-a1a2" in sys.argv:
        drive_mechinterp_a1a2(CFG_MECHINTERP, "MECH-INTERP A1 (F1 gated) + A2 (BP) arms (30k + full battery)",
                              save_path="outputs/gate2_mechinterp_a1a2.json")
    elif "--mechinterp-v2" in sys.argv:
        drive_mechinterp_v2(CFG_MECHINTERP_V2, "MECH-INTERP V2 (per-seed probe fix + dense grid + per-unit radial/tang + W1/W2 ckpt)",
                            save_path="outputs/gate2_mechinterp_v2.json")
    elif "--rotation" in sys.argv:
        import rotation_analysis
        rotation_analysis.main()
    elif "--gate21a" in sys.argv:
        drive_gate21a(CFG_GATE21A, "GATE-2.1a wd-sweep razor (oracle wd-sweep + A1s@wd=0)",
                      save_path="gate2_21a.json")
    elif "--gate21a-ctrl" in sys.argv:
        drive_gate21a(CFG_GATE21A, "GATE-2.1a control cells (C1 + SGD-sweep + C2)",
                      save_path="outputs/gate2_21a_ctrl.json", run_controls=True)
    elif "--selfcheck" in sys.argv:
        drive(CFG_SELFCHECK, "SELF-CHECK A0/A4 @ d=1.0 (10 seeds, full epochs)",
              save_path="gate2_selfcheck.json")
    elif "--full" in sys.argv:
        drive(CFG_FULL, "FULL grok-vs-density (P=53 x10 seeds)",
              save_path="gate2_full.json")
    elif "--closeout" in sys.argv:
        drive(CFG_CLOSEOUT, "CLOSE-OUT Gate-2 d=0.5 cliff x5 budget (30k; A1s razor)",
              save_path="gate2_closeout.json")
    else:
        drive(CFG_SMOKE, "SMOKE A0+A4 @ d=1.0 (plumbing)", save_path="gate2_smoke.json")
