#!/usr/bin/env python3
"""coexistence_add_v14.py — SPEC_SCHEMA_COEXISTENCE_ADD_v1.4

2-schema coexistence (add mod-53 + add mod-59) on L=6 laminar 3D TBT.

Card t_823511d6.  Implements:
  - Dual-modulus encoder IN_DIM=220 (Block A mod-53 K_FREQ=26, Block B mod-59 K59=29)  §4
  - Stratified-quad conn assignment (PATCH A: l_B(i)=(i mod 29)+1)                      §4.2
  - Triple-head swap (add53 [N,53], add59 [N,59], mult control)                          §5
  - C-index (f_spec × ρ_spec) + full-depth cross-schema lesion                           §3
  - C3 tied shared readout positive control (W_shared[:,:53]/[:,:59])                    §6.2
  - C·C spreading control (k-NN local spreading)                                         §6.3
  - Per-area ε_l survival at EVERY layer (P4)                                            §6.4
  - Per-column diagnostics (CC3, 3 independent columns)                                  §3.8

REUSES the proven AblationCortexOpt engine (ablation_cortex_v14_1_opt.py).
The ONLY engine change is the conn sampling line (stratified-quad override
after construction — §4.2 "the ONLY line this design changes").

Constitution: P1-P8 (all PASS). L=6 genuine 3D (P6), >=2 layers (P7).
Sparse firing via divisive normalization + per-neuron homeostatic thresholds (P2)
— NO global top-k. EP contrastive (P1), B_fb separate (P3), B_hc broadcast (P5).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from ablation_cortex_v14_1_opt import AblationCortexOpt, DEVICE

# ================================================================
# CONSTANTS (spec §2.1)
# ================================================================
P_MOD = 53
P_MOD59 = 59
K_FREQ = 26          # Block A per-variable freqs (mod-53)
K59 = 29             # Block B per-variable freqs (mod-59, <=(59-1)/2)
IN_DIM_A = 4 * K_FREQ   # 104
IN_DIM_B = 4 * K59      # 116
IN_DIM = IN_DIM_A + IN_DIM_B  # 220
CHANCE_53 = 1.0 / P_MOD
CHANCE_59 = 1.0 / P_MOD59

N_COL = 3
L_LAMINAR = 6
HIDDEN = 512
SHEET = 23
T_INF = 10
ETA_W = 0.01
ETA_OUT = 0.01
ETA_THETA = 0.001
BATCH = 128

# C-index thresholds (§3.2-3.3)
TAU_FRAC = 0.1    # tau = 0.1 * max contribution
DELTA = 0.25      # 4x dominance for specialisation

OUT_DIR = os.environ.get('COEX_OUT_DIR', '/root/gate2/outputs_coex')
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================
# DUAL-MODULUS FEATURE BUILDER (§4)
# ================================================================

def make_dual_modulus_features(aa, bb):
    """Build IN_DIM=220 dual-modulus Fourier features.

    Block A [0:104]: cos/sin(2*pi*k*a/53), cos/sin(2*pi*k*b/53) for k=1..26
    Block B [104:220]: cos/sin(2*pi*l*a/59), cos/sin(2*pi*l*b/59) for l=1..29

    Both blocks use the SAME (a,b) input values.
    """
    aa = np.asarray(aa, dtype=np.float32)
    bb = np.asarray(bb, dtype=np.float32)
    n = len(aa)

    freqs_53 = np.arange(1, K_FREQ + 1, dtype=np.float32)
    ta53 = 2.0 * np.pi * np.outer(aa, freqs_53) / P_MOD
    tb53 = 2.0 * np.pi * np.outer(bb, freqs_53) / P_MOD

    freqs_59 = np.arange(1, K59 + 1, dtype=np.float32)
    ta59 = 2.0 * np.pi * np.outer(aa, freqs_59) / P_MOD59
    tb59 = 2.0 * np.pi * np.outer(bb, freqs_59) / P_MOD59

    X = np.empty((n, IN_DIM), dtype=np.float32)
    # Block A
    X[:, 0:IN_DIM_A:4] = np.cos(ta53)
    X[:, 1:IN_DIM_A:4] = np.sin(ta53)
    X[:, 2:IN_DIM_A:4] = np.cos(tb53)
    X[:, 3:IN_DIM_A:4] = np.sin(tb53)
    # Block B
    off = IN_DIM_A
    X[:, off + 0:off + IN_DIM_B:4] = np.cos(ta59)
    X[:, off + 1:off + IN_DIM_B:4] = np.sin(ta59)
    X[:, off + 2:off + IN_DIM_B:4] = np.cos(tb59)
    X[:, off + 3:off + IN_DIM_B:4] = np.sin(tb59)
    return X


def make_task_data(prime, n_train=None, n_test=None, seed=42):
    """Build train/test split for modular addition mod prime. Disjoint."""
    rng = np.random.RandomState(seed)
    aa = np.repeat(np.arange(prime), prime)
    bb = np.tile(np.arange(prime), prime)
    cc = (aa + bb) % prime
    X = make_dual_modulus_features(aa, bb)
    Y = cc.astype(np.int64)
    n = prime * prime
    if n_train is None:
        n_train = n // 2
    if n_test is None:
        n_test = n - n_train
    perm = rng.permutation(n)
    Xtr, Ytr = X[perm[:n_train]], Y[perm[:n_train]]
    Xte, Yte = X[perm[n_train:n_train + n_test]], Y[perm[n_train:n_train + n_test]]
    return (torch.from_numpy(Xtr), torch.from_numpy(Ytr),
            torch.from_numpy(Xte), torch.from_numpy(Yte))


def to_onehot(Y, n_classes):
    Yoh = torch.zeros(len(Y), n_classes, device=DEVICE)
    Yoh[torch.arange(len(Y)), Y] = 1.0
    return Yoh


# ================================================================
# STRATIFIED-QUAD CONN OVERRIDE (§4.2, PATCH A)
# ================================================================

def stratified_quad_conn(hidden_dim, k_conn=8, seed=0):
    """§4.2 stratified-quad support: one block-A 4-quad + one block-B 4-quad.

    k_A(i) = (i mod 26)+1, then shuffle (decorrelate columns/seeds)
    l_B(i) = (i mod 29)+1, then shuffle  (PATCH A — covers ALL 29 mod-59 freqs)

    Each frequency quad: [cos(ta), sin(ta), cos(tb), sin(tb)] for that freq.
    Block-A quad for freq k: indices 4*(k-1) .. 4*(k-1)+3
    Block-B quad for freq l: indices IN_DIM_A + 4*(l-1) .. IN_DIM_A + 4*(l-1)+3
    """
    rng = np.random.RandomState(seed)
    assert k_conn == 8, f"stratified-quad requires k_conn=8, got {k_conn}"

    # Frequency assignment
    k_A = np.array([(i % K_FREQ) + 1 for i in range(hidden_dim)])
    l_B = np.array([(i % K59) + 1 for i in range(hidden_dim)])

    # Shuffle frequency->neuron maps (decorrelate across columns/seeds)
    rng.shuffle(k_A)
    rng.shuffle(l_B)

    conn = np.zeros((hidden_dim, k_conn), dtype=np.int64)
    for i in range(hidden_dim):
        k = int(k_A[i])
        l = int(l_B[i])
        # Block-A quad: 4 features of frequency k
        qa = np.array([4 * (k - 1), 4 * (k - 1) + 1,
                       4 * (k - 1) + 2, 4 * (k - 1) + 3])
        # Block-B quad: 4 features of frequency l (offset by IN_DIM_A)
        qb = np.array([IN_DIM_A + 4 * (l - 1), IN_DIM_A + 4 * (l - 1) + 1,
                       IN_DIM_A + 4 * (l - 1) + 2, IN_DIM_A + 4 * (l - 1) + 3])
        conn[i] = np.concatenate([qa, qb])
    return conn


# ================================================================
# COLUMN + HEAD CONSTRUCTION
# ================================================================

def make_column(in_dim, out_dim, seed, n_layers=L_LAMINAR, hidden=HIDDEN,
                stratified_seed=None):
    """Construct an AblationCortexOpt column, then override conn with stratified-quad.

    IMPORTANT: n_hc is always P_MOD59=59 so B_hc [N,59] matches the 59-dim
    score from the mod-59 head. For the mod-53 head, we pad Yoh to 59 dims
    (cols 53-58 = 0); since head_add53 cols 53-58 are frozen zero, yhat cols
    53-58 = 0, so score cols 53-58 = 0 → those columns get zero gradient and
    stay frozen. No engine modification needed.
    """
    if stratified_seed is None:
        stratified_seed = seed + 5000  # decorrelate conn shuffle from main rng

    net = AblationCortexOpt(
        in_dim=in_dim, hidden_dim=hidden, out_dim=out_dim, n_layers=n_layers,
        n_hc=P_MOD59,  # B_hc must match max schema dim (59) for score matmul
        sheet_size=SHEET,
        target_rate=0.10, sigma_norm=1.0, beta_softplus=4.0,
        beta_a=1.0, beta_out=2.0,
        rho=1.0, alpha_dual=0.1, lambda_max=1.0, beta_hc=0.1,
        T_inference=T_INF,
        eta_h=0.5, eta_W=ETA_W, eta_out=ETA_OUT, eta_theta=ETA_THETA,
        k_conn=8, lambda_wd=0.001, w_clip=5.0, gamma_rms=0.9,
        seed=seed,
    )
    # Override conn with stratified-quad (§4.2 — "the ONLY line this design changes")
    conn_np = stratified_quad_conn(hidden, k_conn=8, seed=stratified_seed)
    net.conn = torch.from_numpy(conn_np).to(DEVICE)
    return net


def _make_frozen_mask(N, n_active):
    """RMSProp G_out mask: freeze cols n_active..P_MOD59-1 (zero G_out → no update)."""
    mask = torch.ones(N, P_MOD59, device=DEVICE)
    mask[:, n_active:] = 0.0
    return mask


class TripleHeadColumn:
    """Wraps a column with three output heads: add53, add59, mult (control).

    All heads are [N, P_MOD59=59] to match B_hc [N,59] (n_hc=59).

    Head swap: before training/evaluating task k, swap self.col.W_out to head_k.
    The cortex's train_step always updates self.W_out.

    §5.3 init: head_add59 = clone of head_add53 padded with zero cols 53..58.
    head_add53: cols 53-58 frozen at zero (frozen via G_out mask → no RMSProp update).
    """

    def __init__(self, col):
        self.col = col
        # head_add53 = cortex initial W_out [N, 53] padded to [N, 59] with zero cols
        self.head_add53 = torch.zeros(col.N, P_MOD59, device=DEVICE)
        self.head_add53[:, :P_MOD] = col.W_out[:, :P_MOD].clone()
        # G mask: freeze cols 53-58 (zero G → RMSProp won't update frozen cols)
        self.G_add53 = col.G_out.new_ones(col.N, P_MOD59) * 1e-8
        self.G_add53[:, P_MOD:] = 0.0  # frozen
        # head_add59 = clone of head_add53 padded to [N, 59] with zero cols (§5.3 Patch 6)
        self.head_add59 = self.head_add53.clone()
        self.G_add59 = col.G_out.new_ones(col.N, P_MOD59) * 1e-8
        # head_mult = control only (clone, not trained in coexistence arms)
        self.head_mult = self.head_add53.clone()
        self.G_mult = col.G_out.new_ones(col.N, P_MOD59) * 1e-8
        self._active = 'add53'
        self._heads = {
            'add53': (self.head_add53, self.G_add53),
            'add59': (self.head_add59, self.G_add59),
            'mult': (self.head_mult, self.G_mult),
        }
        # Initialize col.W_out to the active head
        col.W_out = self.head_add53.clone()
        col.G_out = self.G_add53.clone()

    def _swap(self, task):
        self.sync_save()
        h, g = self._heads[task]
        self.col.W_out = h.clone()
        self.col.G_out = g.clone()
        self._active = task

    def sync_save(self):
        h, g = self._heads[self._active]
        self._heads[self._active] = (self.col.W_out.clone(), self.col.G_out.clone())

    def get_head(self, task):
        self.sync_save()
        return self._heads[task][0]


class C3TiedColumn:
    """C3 control: single tied shared readout W_shared [N, 59].

    head_add53 = W_shared[:, :53], head_add59 = W_shared[:, :59].
    Both schemas train the SAME W_shared (views share memory).
    """

    def __init__(self, col):
        self.col = col
        # W_shared [N, 59], init = clone of W_out padded with zero cols (§5.3 PATCH B)
        self.W_shared = torch.zeros(col.N, P_MOD59, device=DEVICE)
        self.W_shared[:, :P_MOD] = col.W_out[:, :P_MOD].clone()
        self.G_shared = torch.ones(col.N, P_MOD59, device=DEVICE) * 1e-8
        self._active = 'add53'
        # Initialize col.W_out to the full W_shared (both schemas use it)
        col.W_out = self.W_shared
        col.G_out = self.G_shared

    def _swap(self, task):
        # C3 tied: ALWAYS use the full W_shared [N,59] for both schemas.
        # For mod-53, Yoh cols 53-58 = 0 → extra cols get gradient only from
        # mod-59 → this IS the tied shared readout (both schemas train W_shared).
        self.col.W_out = self.W_shared
        self.col.G_out = self.G_shared
        self._active = task

    def sync_save(self):
        # Views share memory — already updated in-place. No explicit save needed.
        pass

    def get_head(self, task):
        # For attribution: return the schema-appropriate slice
        if task == 'add53':
            return self.W_shared[:, :P_MOD]
        else:
            return self.W_shared


# ================================================================
# COEXISTENCE ENSEMBLE
# ================================================================

class CoexEnsemble:
    """3 independent columns, each with triple-head. CC3: full-view, no voting."""

    def __init__(self, seed, c3_mode=False):
        self.seed = seed
        self.c3_mode = c3_mode
        self.columns = []
        self.dh_cols = []
        self.masks = []
        rng_masks = np.random.RandomState(seed + 1000)
        for c in range(N_COL):
            net = make_column(in_dim=IN_DIM, out_dim=P_MOD,
                              seed=seed * 100 + c)
            if c3_mode:
                dh = C3TiedColumn(net)
            else:
                dh = TripleHeadColumn(net)
            self.columns.append(net)
            self.dh_cols.append(dh)
            # CC3: full view (all features visible to all columns)
            m = np.ones(IN_DIM, dtype=np.float32)
            self.masks.append(torch.from_numpy(m).to(DEVICE))

        # Calibrate thresholds on mod-53 data
        Xtr53, _, _, _ = make_task_data(P_MOD, n_train=200, seed=seed)
        Xtr53 = Xtr53.to(DEVICE)
        for c in range(N_COL):
            self.columns[c].calibrate_thresholds(Xtr53[:200] * self.masks[c])

    def _view(self, X, c):
        return X * self.masks[c]

    def _swap_all(self, task):
        for dh in self.dh_cols:
            dh._swap(task)

    def _sync_all(self):
        for dh in self.dh_cols:
            dh.sync_save()

    def predict(self, X, task):
        """Ensemble-averaged logits → argmax. For mod-53, only cols 0-52."""
        self._swap_all(task)
        with torch.no_grad():
            y_list = []
            for c in range(N_COL):
                x, _ = self.columns[c].forward_init(self._view(X, c))
                y = x[self.columns[c].L - 1] @ self.columns[c].W_out
                y_list.append(y)
        self._sync_all()
        y_avg = sum(y_list) / len(y_list)
        # For mod-53: argmax over cols 0-52 only (frozen cols 53-58 = 0 or
        # C3 mode where 53-58 are nonzero from mod-59 training)
        if task in ('add53', 'mult'):
            return y_avg[:, :P_MOD].argmax(dim=-1)
        return y_avg.argmax(dim=-1)

    def evaluate(self, X, Y, task):
        return float((self.predict(X, task) == Y).float().mean().item())

    def train_step_task(self, X, Yoh, task, return_gates=False):
        """Train one step on one task."""
        self._swap_all(task)
        gate_logs = []
        for c in range(N_COL):
            gl = self.columns[c].train_step(
                self._view(X, c), Yoh, return_gates=return_gates)
            gate_logs.append(gl)
        self._sync_all()
        return gate_logs

    def get_per_area_diagnostics(self, Xte, Yoh, task):
        """Per-area ε_l survival at EVERY layer (P4). Returns per-column gate_log."""
        self._swap_all(task)
        per_col = []
        for c in range(N_COL):
            with torch.no_grad():
                result = self.columns[c].infer(
                    self._view(Xte[:64], c), Yoh[:64], return_gates=True)
                per_col.append(result.get('gate_log', {}))
        self._sync_all()
        return per_col


# ================================================================
# C-INDEX + SPECIALISATION (§3.1-3.3)
# ================================================================

def compute_attribution(ensemble, X53_test, X59_test):
    """Per-column per-unit attribution c_u^(s) (§3.1).

    c_u^(s) = act_u^(s) * mean|W_out_s[u,:s]|

    where act_u^(s) = mean over schema-s test set of x_{L-1}[u].
    Uses mean|W_out[u,:s]| restricted to the schema's active columns (0:s)
    — dimension-consistent (§3.1 Patch 5).
    """
    results = []
    for c in range(N_COL):
        col = ensemble.columns[c]
        dh = ensemble.dh_cols[c]
        L = col.L

        # Get head weights (sync first)
        W53 = dh.get_head('add53')  # [N, 59], only first 53 active
        W59 = dh.get_head('add59')  # [N, 59], all 59 active

        # mean|W_out[u,:]| restricted to the schema's active columns
        w53_mag = W53[:, :P_MOD].abs().mean(dim=1)  # [N], cols 0-52
        w59_mag = W59.abs().mean(dim=1)               # [N], cols 0-58

        # Activity at readout layer (L-1) for each schema's test set
        with torch.no_grad():
            x_view53 = ensemble._view(X53_test, c)
            x0_53, _ = col.forward_init(x_view53)
            act53 = x0_53[L - 1].abs().mean(dim=0)  # [N]

            x_view59 = ensemble._view(X59_test, c)
            x0_59, _ = col.forward_init(x_view59)
            act59 = x0_59[L - 1].abs().mean(dim=0)  # [N]

        # Attribution
        c53 = act53 * w53_mag  # [N]
        c59 = act59 * w59_mag  # [N]

        results.append({
            'c53': c53.cpu().numpy(),
            'c59': c59.cpu().numpy(),
            'act53': act53.cpu().numpy(),
            'act59': act59.cpu().numpy(),
        })
    return results


def compute_specialisation(attrib_results, tau_frac=TAU_FRAC, delta=DELTA):
    """§3.2 specialisation sets + §3.3 C-index per column.

    mod-53-specialised: c_u^53 >= tau AND c_u^59 <= delta*c_u^53
    mod-59-specialised: c_u^59 >= tau AND c_u^53 <= delta*c_u^59
    shared: otherwise

    C = f_spec * rho_spec  (§3.3 variant formula)
    """
    per_col = []
    for res in attrib_results:
        c53 = res['c53']
        c59 = res['c59']
        N = len(c53)

        tau = tau_frac * max(c53.max(), c59.max())

        # Non-silent: contribution above 1% of tau
        non_silent = (c53 + c59) > 0.01 * tau

        spec53 = (c53 >= tau) & (c59 <= delta * c53)
        spec59 = (c59 >= tau) & (c53 <= delta * c59)

        S53 = np.where(spec53)[0]
        S59 = np.where(spec59)[0]
        n_spec = len(S53) + len(S59)
        n_nonsilent = int(non_silent.sum())

        f_spec = n_spec / max(n_nonsilent, 1)

        total_energy = (c53 + c59).sum()
        spec_energy = (c53[S53].sum() + c59[S53].sum() +
                       c53[S59].sum() + c59[S59].sum())
        rho_spec = spec_energy / max(total_energy, 1e-12)

        C = f_spec * rho_spec

        per_col.append({
            'S53': S53.tolist(),
            'S59': S59.tolist(),
            'n_spec53': len(S53),
            'n_spec59': len(S59),
            'n_nonsilent': n_nonsilent,
            'f_spec': float(f_spec),
            'rho_spec': float(rho_spec),
            'C': float(C),
            'tau': float(tau),
        })
    return per_col


# ================================================================
# FULL-DEPTH CROSS-SCHEMA LESION (§3.4)
# ================================================================

def lesioned_forward(col, X, S_set):
    """Forward pass with S_set units clamped to 0 at EVERY layer.

    §3.4: clamp S at every layer 0..L-1, re-read through W_out.
    Full-depth clamping propagates through the forward pass.
    """
    with torch.no_grad():
        u0, _, _ = col._dendritic_fwd(X)
        x = col.phi_norm(u0, 0)
        if S_set is not None and len(S_set) > 0:
            x[:, S_set] = 0.0
        for l in range(col.L - 1):
            u = x + col.s_L * (x @ col.W_ff[l])
            x = col.phi_norm(u, l + 1)
            if S_set is not None and len(S_set) > 0:
                x[:, S_set] = 0.0
        return x


def full_depth_lesion(ensemble, X53_test, Y53_test, X59_test, Y59_test,
                      spec_results, lesion_set='S53'):
    """Full-depth cross-schema lesion (§3.4).

    lesion_set: 'S53' (clamp mod-53-specialised), 'S59' (clamp mod-59-specialised),
                'joint' (clamp joint active set — for C3).

    Returns: dict with normal + lesioned accuracy for both schemas.
    """
    results = {
        'lesion_set': lesion_set,
        'acc53_normal': 0.0, 'acc59_normal': 0.0,
        'acc53_lesioned': 0.0, 'acc59_lesioned': 0.0,
    }

    # Normal accuracy
    results['acc53_normal'] = ensemble.evaluate(X53_test, Y53_test, 'add53')
    results['acc59_normal'] = ensemble.evaluate(X59_test, Y59_test, 'add59')

    # Lesioned accuracy: clamp S_set at every layer, ensemble-average logits
    for schema, X_test, Y_test in [('53', X53_test, Y53_test),
                                    ('59', X59_test, Y59_test)]:
        # Determine which head to read through
        read_task = 'add53' if schema == '53' else 'add59'

        # Get lesioned logits per column
        lesioned_y_list = []
        for c in range(N_COL):
            col = ensemble.columns[c]
            dh = ensemble.dh_cols[c]

            # Get the S_set for this column
            if lesion_set == 'S53':
                S = spec_results[c]['S53']
            elif lesion_set == 'S59':
                S = spec_results[c]['S59']
            elif lesion_set == 'joint':
                # Joint active set = all non-silent units (S_shared under §3.2)
                c53 = None  # will be set below
                S = spec_results[c].get('S_joint', [])
            else:
                S = []

            dh._swap(read_task)
            with torch.no_grad():
                x_view = ensemble._view(X_test, c)
                x_lesioned = lesioned_forward(col, x_view, S)
                y = x_lesioned @ col.W_out
                lesioned_y_list.append(y)
        ensemble._sync_all()

        y_avg = sum(lesioned_y_list) / len(lesioned_y_list)
        preds = y_avg.argmax(dim=-1)
        acc = float((preds == Y_test).float().mean().item())
        if schema == '53':
            results['acc53_lesioned'] = acc
        else:
            results['acc59_lesioned'] = acc

    # Verdict (§3.4 decision rule)
    a53_drop = results['acc53_normal'] - results['acc53_lesioned']
    a59_drop = results['acc59_normal'] - results['acc59_lesioned']
    if lesion_set == 'S53':
        if a53_drop > 0.3 and a59_drop < 0.1:
            results['verdict'] = 'H_A_disjoint'
        elif a53_drop > 0.3 and a59_drop > 0.3:
            results['verdict'] = 'H_B_collision'
        else:
            results['verdict'] = 'inconclusive'
    results['a53_drop'] = float(a53_drop)
    results['a59_drop'] = float(a59_drop)
    return results


def full_depth_lesion_both(ensemble, X53_test, Y53_test, X59_test, Y59_test,
                           spec_results):
    """Run full-depth lesion for S53 and symmetrically for S59 (§3.4)."""
    lesion53 = full_depth_lesion(ensemble, X53_test, Y53_test,
                                 X59_test, Y59_test, spec_results, 'S53')
    lesion59 = full_depth_lesion(ensemble, X53_test, Y53_test,
                                 X59_test, Y59_test, spec_results, 'S59')
    return {'lesion_S53': lesion53, 'lesion_S59': lesion59}


# ================================================================
# C·C SPREADING CONTROL (§6.3)
# ================================================================

def cc_spreading_control(Xtr, Ytr, Xte, Yte, prime, k=5):
    """C·C spreading control: k-NN local spreading baseline (§6.3).

    For each test input, find k nearest training inputs (cosine distance on
    the appropriate block features), predict the most common label.
    Cannot learn a GLOBAL modular-arity regularity → plateaus well below 1.0.
    """
    block = 'A' if prime == P_MOD else 'B'
    if block == 'A':
        sl = slice(0, IN_DIM_A)
    else:
        sl = slice(IN_DIM_A, IN_DIM)

    Xtr_f = Xtr[:, sl].cpu().numpy().astype(np.float64)
    Xte_f = Xte[:, sl].cpu().numpy().astype(np.float64)
    Ytr_np = Ytr.cpu().numpy()

    # Normalize for cosine distance
    tr_norm = np.linalg.norm(Xtr_f, axis=1, keepdims=True) + 1e-8
    te_norm = np.linalg.norm(Xte_f, axis=1, keepdims=True) + 1e-8
    Xtr_n = Xtr_f / tr_norm
    Xte_n = Xte_f / te_norm

    # Cosine similarity
    sims = Xte_n @ Xtr_n.T  # [n_test, n_train]

    preds = np.zeros(len(Xte_f), dtype=np.int64)
    for i in range(len(Xte_f)):
        top_k_idx = np.argpartition(sims[i], -min(k, len(Ytr_np)))[-k:]
        labels = Ytr_np[top_k_idx]
        preds[i] = np.bincount(labels, minlength=prime).argmax()

    acc = float((preds == Yte.cpu().numpy()).mean())
    return acc


# ================================================================
# ARM RUNNERS
# ================================================================

def run_single_schema(seed, prime, steps=3000, eval_every=100,
                      progress_path=None, verbose=True):
    """C1 (mod-53-only) or C2 (mod-59-only) sanity gate (§4.5).

    Same per-schema budget as interleaved arms. Must grok or encoder re-specified.
    """
    arm_name = 'C1' if prime == P_MOD else 'C2'
    t0 = time.time()
    Xtr, Ytr, Xte, Yte = make_task_data(prime, seed=seed)
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    # Always use P_MOD59=59 for one-hot — head cols above `prime` are frozen
    Yoh = to_onehot(Ytr, P_MOD59)
    task = 'add53' if prime == P_MOD else 'add59'

    model = CoexEnsemble(seed)
    rng = np.random.RandomState(seed)
    history = []
    grok_step = None
    best = 0.0

    for step in range(1, steps + 1):
        idx = rng.randint(0, len(Xtr), BATCH)
        model.train_step_task(Xtr[idx], Yoh[idx], task,
                              return_gates=(step % eval_every == 0))

        if step % eval_every == 0 or step == 1:
            acc = model.evaluate(Xte, Yte, task)
            best = max(best, acc)
            if acc >= 0.9 and grok_step is None:
                grok_step = step
            history.append({'step': step, 'acc': acc, 'best': best})

            if verbose and (step % 500 == 0 or step == eval_every):
                print(f"  [{arm_name} s{seed}] step {step:5d}: acc={acc:.3f} "
                      f"best={best:.3f} [{time.time()-t0:.0f}s]", flush=True)

            if progress_path:
                try:
                    with open(progress_path, 'w') as pf:
                        json.dump({'arm': arm_name, 'seed': seed, 'step': step,
                                   'acc': acc, 'best': best, 'grok': grok_step,
                                   'elapsed_s': time.time() - t0}, pf)
                except Exception:
                    pass

    # C·C control
    cc_acc = cc_spreading_control(Xtr, Ytr, Xte, Yte, prime)

    # Per-area ε_l survival
    Yoh_te = to_onehot(Yte[:64], P_MOD59)
    per_col_gates = model.get_per_area_diagnostics(Xte[:64], Yoh_te, task)

    dt = time.time() - t0
    result = {
        'arm': arm_name, 'seed': seed, 'prime': prime, 'steps': steps,
        'final_acc': history[-1]['acc'] if history else 0.0,
        'best_acc': best, 'grok_step': grok_step,
        'cc_control_acc': cc_acc, 'chance': 1.0 / prime,
        'history': history,
        'per_area_eps': [
            {k: gl.get(k) for k in ['eps_a_norms_clamped', 'eps_a_norms_free',
             'dh_norms', 'firing_rates', 'energy']}
            for gl in per_col_gates
        ],
        'time': dt,
    }
    print(f"  [DONE] {arm_name} s{seed}: final={result['final_acc']:.3f} "
          f"best={best:.3f} grok={grok_step} CC={cc_acc:.3f} "
          f"({dt:.0f}s)", flush=True)
    return result


def run_coex1(seed, steps=6000, eval_every=100, progress_path=None, verbose=True):
    """Coex-1: fresh-joint interleave (PRIMARY, §6.2).

    Both schemas from step 0, 50/50 interleave. 6000 total (3000/schema).
    """
    t0 = time.time()
    Xtr53, Ytr53, Xte53, Yte53 = make_task_data(P_MOD, seed=seed)
    Xtr59, Ytr59, Xte59, Yte59 = make_task_data(P_MOD59, seed=seed)
    Xtr53, Ytr53 = Xtr53.to(DEVICE), Ytr53.to(DEVICE)
    Xte53, Yte53 = Xte53.to(DEVICE), Yte53.to(DEVICE)
    Xtr59, Ytr59 = Xtr59.to(DEVICE), Ytr59.to(DEVICE)
    Xte59, Yte59 = Xte59.to(DEVICE), Yte59.to(DEVICE)
    Yoh53 = to_onehot(Ytr53, P_MOD59)
    Yoh59 = to_onehot(Ytr59, P_MOD59)

    model = CoexEnsemble(seed)
    rng = np.random.RandomState(seed)
    history = []
    grok53 = grok59 = None
    best53 = best59 = 0.0

    for step in range(1, steps + 1):
        task = 'add53' if step % 2 == 1 else 'add59'
        if task == 'add53':
            idx = rng.randint(0, len(Xtr53), BATCH)
            model.train_step_task(Xtr53[idx], Yoh53[idx], 'add53')
        else:
            idx = rng.randint(0, len(Xtr59), BATCH)
            model.train_step_task(Xtr59[idx], Yoh59[idx], 'add59')

        if step % eval_every == 0 or step == 1:
            a53 = model.evaluate(Xte53, Yte53, 'add53')
            a59 = model.evaluate(Xte59, Yte59, 'add59')
            best53 = max(best53, a53)
            best59 = max(best59, a59)
            if a53 >= 0.9 and grok53 is None:
                grok53 = step
            if a59 >= 0.9 and grok59 is None:
                grok59 = step
            history.append({'step': step, 'acc53': a53, 'acc59': a59,
                            'best53': best53, 'best59': best59})

            if verbose and (step % 500 == 0 or step == eval_every):
                print(f"  [Coex1 s{seed}] step {step:5d}: "
                      f"53={a53:.3f} 59={a59:.3f} "
                      f"best=[{best53:.3f},{best59:.3f}] "
                      f"[{time.time()-t0:.0f}s]", flush=True)

            if progress_path:
                try:
                    with open(progress_path, 'w') as pf:
                        json.dump({'arm': 'Coex1', 'seed': seed, 'step': step,
                                   'acc53': a53, 'acc59': a59,
                                   'grok53': grok53, 'grok59': grok59,
                                   'elapsed_s': time.time() - t0}, pf)
                except Exception:
                    pass

    # C·C controls
    cc53 = cc_spreading_control(Xtr53, Ytr53, Xte53, Yte53, P_MOD)
    cc59 = cc_spreading_control(Xtr59, Ytr59, Xte59, Yte59, P_MOD59)

    # Coexistence diagnostics (per column)
    attrib = compute_attribution(model, Xte53[:200], Xte59[:200])
    spec = compute_specialisation(attrib)

    # Full-depth lesion (both S53 and S59)
    lesion = full_depth_lesion_both(model, Xte53[:200], Yte53[:200],
                                    Xte59[:200], Yte59[:200], spec)

    # Per-area ε_l for both schemas
    Yoh_te53 = to_onehot(Yte53[:64], P_MOD59)
    Yoh_te59 = to_onehot(Yte59[:64], P_MOD59)
    gates53 = model.get_per_area_diagnostics(Xte53[:64], Yoh_te53, 'add53')
    gates59 = model.get_per_area_diagnostics(Xte59[:64], Yoh_te59, 'add59')

    dt = time.time() - t0
    result = {
        'arm': 'Coex1', 'seed': seed, 'steps': steps,
        'final53': history[-1]['acc53'] if history else 0.0,
        'final59': history[-1]['acc59'] if history else 0.0,
        'best53': best53, 'best59': best59,
        'grok53': grok53, 'grok59': grok59,
        'cc53': cc53, 'cc59': cc59,
        'chance53': CHANCE_53, 'chance59': CHANCE_59,
        'coexistence': {
            'per_column': spec,
            'C_values': [s['C'] for s in spec],
            'C_median': float(np.median([s['C'] for s in spec])),
            'f_spec_values': [s['f_spec'] for s in spec],
        },
        'lesion': lesion,
        'per_area_eps_53': [
            {k: gl.get(k) for k in ['eps_a_norms_clamped', 'eps_a_norms_free',
             'dh_norms', 'firing_rates', 'energy']}
            for gl in gates53
        ],
        'per_area_eps_59': [
            {k: gl.get(k) for k in ['eps_a_norms_clamped', 'eps_a_norms_free',
             'dh_norms', 'firing_rates', 'energy']}
            for gl in gates59
        ],
        'history': history,
        'time': dt,
    }
    print(f"  [DONE] Coex1 s{seed}: 53={result['final53']:.3f} "
          f"59={result['final59']:.3f} C_med={result['coexistence']['C_median']:.3f} "
          f"({dt:.0f}s)", flush=True)
    return result


def run_c3(seed, steps=6000, eval_every=100, progress_path=None, verbose=True):
    """C3: tied shared readout positive control (§6.2).

    Single W_shared [N,59]. head_add53=W_shared[:,:53], head_add59=W_shared[:,:59].
    Calibrates the C-index discriminator (expected C<=0.25 + collision-row lesion).
    """
    t0 = time.time()
    Xtr53, Ytr53, Xte53, Yte53 = make_task_data(P_MOD, seed=seed)
    Xtr59, Ytr59, Xte59, Yte59 = make_task_data(P_MOD59, seed=seed)
    Xtr53, Ytr53 = Xtr53.to(DEVICE), Ytr53.to(DEVICE)
    Xte53, Yte53 = Xte53.to(DEVICE), Yte53.to(DEVICE)
    Xtr59, Ytr59 = Xtr59.to(DEVICE), Ytr59.to(DEVICE)
    Xte59, Yte59 = Xte59.to(DEVICE), Yte59.to(DEVICE)
    Yoh53 = to_onehot(Ytr53, P_MOD59)
    Yoh59 = to_onehot(Ytr59, P_MOD59)

    model = CoexEnsemble(seed, c3_mode=True)
    rng = np.random.RandomState(seed)
    history = []
    grok53 = grok59 = None
    best53 = best59 = 0.0

    for step in range(1, steps + 1):
        task = 'add53' if step % 2 == 1 else 'add59'
        if task == 'add53':
            idx = rng.randint(0, len(Xtr53), BATCH)
            model.train_step_task(Xtr53[idx], Yoh53[idx], 'add53')
        else:
            idx = rng.randint(0, len(Xtr59), BATCH)
            model.train_step_task(Xtr59[idx], Yoh59[idx], 'add59')

        if step % eval_every == 0 or step == 1:
            a53 = model.evaluate(Xte53, Yte53, 'add53')
            a59 = model.evaluate(Xte59, Yte59, 'add59')
            best53 = max(best53, a53)
            best59 = max(best59, a59)
            if a53 >= 0.9 and grok53 is None:
                grok53 = step
            if a59 >= 0.9 and grok59 is None:
                grok59 = step
            history.append({'step': step, 'acc53': a53, 'acc59': a59,
                            'best53': best53, 'best59': best59})

            if verbose and (step % 500 == 0 or step == eval_every):
                print(f"  [C3 s{seed}] step {step:5d}: "
                      f"53={a53:.3f} 59={a59:.3f} "
                      f"best=[{best53:.3f},{best59:.3f}] "
                      f"[{time.time()-t0:.0f}s]", flush=True)

            if progress_path:
                try:
                    with open(progress_path, 'w') as pf:
                        json.dump({'arm': 'C3', 'seed': seed, 'step': step,
                                   'acc53': a53, 'acc59': a59,
                                   'elapsed_s': time.time() - t0}, pf)
                except Exception:
                    pass

    # Coexistence diagnostics
    attrib = compute_attribution(model, Xte53[:200], Xte59[:200])
    spec = compute_specialisation(attrib)

    # PATCH 3: compute S_53 directly, clamp separately if non-empty
    # Joint active set = all non-silent units
    for c in range(N_COL):
        c53 = attrib[c]['c53']
        c59 = attrib[c]['c59']
        tau = TAU_FRAC * max(c53.max(), c59.max())
        non_silent = (c53 + c59) > 0.01 * tau
        spec[c]['S_joint'] = np.where(non_silent)[0].tolist()

    # Joint active set lesion (expected: both schemas collapse together — H_B)
    lesion_joint = full_depth_lesion(model, Xte53[:200], Yte53[:200],
                                     Xte59[:200], Yte59[:200], spec, 'joint')

    # If S_53 non-empty: separate S_53-only lesion (PATCH 3)
    any_s53 = any(len(spec[c]['S53']) > 0 for c in range(N_COL))
    lesion_s53 = None
    if any_s53:
        lesion_s53 = full_depth_lesion(model, Xte53[:200], Yte53[:200],
                                       Xte59[:200], Yte59[:200], spec, 'S53')

    dt = time.time() - t0
    result = {
        'arm': 'C3', 'seed': seed, 'steps': steps,
        'final53': history[-1]['acc53'] if history else 0.0,
        'final59': history[-1]['acc59'] if history else 0.0,
        'best53': best53, 'best59': best59,
        'grok53': grok53, 'grok59': grok59,
        'coexistence': {
            'per_column': spec,
            'C_values': [s['C'] for s in spec],
            'C_median': float(np.median([s['C'] for s in spec])),
        },
        'lesion_joint': lesion_joint,
        'lesion_S53': lesion_s53,
        'S53_nonempty': any_s53,
        'history': history,
        'time': dt,
    }
    print(f"  [DONE] C3 s{seed}: 53={result['final53']:.3f} "
          f"59={result['final59']:.3f} C_med={result['coexistence']['C_median']:.3f} "
          f"S53_nonempty={any_s53} ({dt:.0f}s)", flush=True)
    return result


# ================================================================
# MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser(
        description='2-schema coexistence (add mod-53 + mod-59) L=6 laminar 3D TBT')
    ap.add_argument('--arm', choices=['C1', 'C2', 'Coex1', 'C3', 'all'],
                    default='C1')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--steps', type=int, default=3000,
                    help='per-schema steps (Coex1/C3 use 2x this)')
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    arms = ['C1', 'C2', 'Coex1', 'C3'] if args.arm == 'all' else [args.arm]
    progress_path = os.path.join(OUT_DIR, 'coex_PROGRESS.json')

    base_name = args.output or os.path.join(
        OUT_DIR, f'coex_add_v14_{args.arm}_s{"-".join(str(s) for s in args.seeds)}.json')

    print(f"COEXISTENCE ADD v1.4 (L={L_LAMINAR}, N={HIDDEN}, IN_DIM={IN_DIM})")
    print(f"Arms: {arms}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {DEVICE}")
    print(f"Output: {base_name}")
    print(f"Progress: {progress_path}")
    print()

    results = {}
    for arm in arms:
        print(f"\n{'='*70}\n  ARM: {arm}\n{'='*70}")
        results[arm] = []
        for seed in args.seeds:
            print(f"\n  --- {arm} seed {seed} ---", flush=True)
            if arm == 'C1':
                r = run_single_schema(seed, P_MOD, steps=args.steps,
                                      eval_every=args.eval_every,
                                      progress_path=progress_path)
            elif arm == 'C2':
                r = run_single_schema(seed, P_MOD59, steps=args.steps,
                                      eval_every=args.eval_every,
                                      progress_path=progress_path)
            elif arm == 'Coex1':
                r = run_coex1(seed, steps=args.steps * 2,
                              eval_every=args.eval_every,
                              progress_path=progress_path)
            elif arm == 'C3':
                r = run_c3(seed, steps=args.steps * 2,
                           eval_every=args.eval_every,
                           progress_path=progress_path)
            results[arm].append(r)
            try:
                with open(base_name, 'w') as f:
                    json.dump(results, f, indent=1, default=str)
            except Exception:
                pass

    # Summary
    print(f"\n{'='*80}\n  SUMMARY: COEXISTENCE ADD v1.4\n{'='*80}")
    for arm, rs in results.items():
        if not rs:
            continue
        if arm in ('C1', 'C2'):
            finals = [r['final_acc'] for r in rs]
            bests = [r['best_acc'] for r in rs]
            n_grok = sum(1 for r in rs if r['grok_step'] is not None)
            print(f"  {arm}: final={np.median(finals):.3f} "
                  f"best={np.median(bests):.3f} grok={n_grok}/{len(rs)} "
                  f"CC={rs[0].get('cc_control_acc', 0):.3f}")
        else:
            f53 = [r['final53'] for r in rs]
            f59 = [r['final59'] for r in rs]
            cs = [r['coexistence']['C_median'] for r in rs]
            print(f"  {arm}: 53={np.median(f53):.3f} 59={np.median(f59):.3f} "
                  f"C_med={np.median(cs):.3f}")
    print(f"\n  Results: {base_name}")


if __name__ == '__main__':
    main()
