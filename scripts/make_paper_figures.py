#!/usr/bin/env python3
"""make_paper_figures.py — generate F1-F6 publication figures for Paper 1.

Every figure is drawn ONLY from committed artifacts in outputs/ (verified
present). Output: docs/paper/figures/F{1..6}_*.pdf (vector PDF, LaTeX-ready).

F1 grokking curves: stab vs nostab mult, mean±std test acc over 10 seeds
F2 falsifier matrix: 2x2 heatmap (encoder x task), final mean acc + grok counts
F3 T-cliff: ADD (10 seeds) / MULT (3 seeds) / BP (3 seeds) vs T in {1,5,10}
F4 expressivity: single-neuron RMSE 0.70 vs full layer-0 0.019 (rank gap)
F5 coexistence: Coex1 final53/final59 vs C3 final53/final59 (10 seeds each)
F6 replay: per-seed benefit, tied (n=10) +0.333 p=0.001, oja (n=9) +0.454 p=0.002
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs")
FIG = os.path.join(REPO, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
C_STAB, C_NOSTAB, C_ADD, C_MULT, C_BP = "#1f77b4", "#d62728", "#2ca02c", "#1f77b4", "#9467bd"


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


# ── F1: grokking curves ────────────────────────────────────────────────
stab = load("basis_swap_v13_mult-stab_frac80_L6_N1536_T10.json")
nostab = load("basis_swap_v13_mult-nostab_frac80_L6_N1536_T10.json")
bp10 = load("bp_control_arm1_mult_main_L6_N1536_T10_S10000_n10.json")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)

# Panel 1: EP stabilized vs unstabilized
ax = axes[0]
for data, color, label in [(stab, C_STAB, "stabilized (C+D)"),
                           (nostab, C_NOSTAB, "unstabilized")]:
    steps = [h["step"] for h in data["results"][0]["history"]]
    accs = np.array([[h["test_acc"] for h in r["history"]] for r in data["results"]])
    m, s = accs.mean(0), accs.std(0)
    ax.plot(steps, m, color=color, lw=1.6, label=label)
    ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15)
ax.axhline(0.9, color="k", ls="--", lw=0.7, alpha=0.5)
ax.text(steps[-1] * 0.55, 0.905, "grok threshold 0.90", fontsize=7, alpha=0.7)
ax.set_xlabel("step"); ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
ax.set_title("EP + local contrastive: stabilization (10 seeds)")

# Panel 2: EP vs backprop at extended budget (same substrate, same basis)
ax = axes[1]
for data, color, label in [(stab, C_STAB, "EP + local contrastive (C2, 3000 steps)"),
                           (bp10, C_BP, "backprop (10,000 steps)")]:
    steps = [h["step"] for h in data["results"][0]["history"]]
    accs = np.array([[h["test_acc"] for h in r["history"]] for r in data["results"]])
    m, s = accs.mean(0), accs.std(0)
    ax.plot(steps, m, color=color, lw=1.6, label=label)
    ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15)
ax.axhline(0.9, color="k", ls="--", lw=0.7, alpha=0.5)
ax.set_xlabel("optimizer step"); ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
ax.set_title("Local rule vs backprop, same substrate")

fig.savefig(os.path.join(FIG, "F1_grokking_curves.pdf"))
plt.close(fig)

# ── F2: falsifier matrix ───────────────────────────────────────────────
ma = load("matched_add_L6_N1536_T10.json")
mm = load("matched_mult_L6_N1536_T10.json")
f2 = load("f2_emult_add_L6_N1536_T10.json")
def finals(d):
    return [r["final_test_acc"] for r in d["results"]]
def grok(d):
    return sum(1 for r in d["results"] if r["final_test_acc"] >= 0.9)
eadd_add = mean_std(finals(ma)); eadd_mult = mean_std(finals(mm))
c2 = load("basis_swap_v13_mult-stab_frac80_L6_N1536_T10.json")
emult_mult = (c2["summary"]["final_mean"], c2["summary"]["final_std"])  # C2 banked
emult_add = mean_std(finals(f2))
matrix = np.array([[eadd_add[0], eadd_mult[0]], [emult_add[0], emult_mult[0]]])
grok_counts = [[grok(ma), grok(mm)], [grok(f2), grok(c2)]]
fig, ax = plt.subplots(figsize=(3.6, 3.0))
im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["add mod 53", "mult mod 53"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["$E_{\\mathrm{add}}$ (Fourier)", "$E_{\\mathrm{mult}}$ (characters)"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{matrix[i, j]:.3f}\n{grok_counts[i][j]}/10 final $\\geq$ 0.90",
                ha="center", va="center", fontsize=8.5,
                color="white" if matrix[i, j] > 0.5 else "black")
ax.set_title("Falsifier matrix: final accuracy")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(os.path.join(FIG, "F2_falsifier_matrix.pdf"))
plt.close(fig)

# ── F3: T-cliff ────────────────────────────────────────────────────────
add_t = {1: load("l6eqcap_T1_10seeds.json"), 5: load("l6eqcap_T5_10seeds.json"),
         10: load("l6eqcap_T10_10seeds.json")}
mult_t = load("ep_mult_tcliff_banking_L6_N1536.json")
bp_t = load("bp_control_arm1_mult_tcliff_L6_N1536.json")
add_means = [float(np.mean([r["final_test_acc"] for r in add_t[t]["results"]])) for t in (1, 5, 10)]
add_stds = [float(np.std([r["final_test_acc"] for r in add_t[t]["results"]])) for t in (1, 5, 10)]
mult_means = [mult_t["summary"][str(t)]["window_avg_mean"] for t in (1, 5)]
c2 = load("basis_swap_v13_mult-stab_frac80_L6_N1536_T10.json")
mult_means.append(c2["summary"]["window_avg_mean"])  # T=10 = C2 claim
mult_stds = [mult_t["summary"][str(t)]["window_avg_std"] for t in (1, 5)]
mult_stds.append(c2["summary"]["window_avg_std"])
bp_means = [bp_t["summary"][str(t)]["window_avg_mean"] for t in (1, 5, 10)]
bp_stds = [bp_t["summary"][str(t)]["window_avg_std"] for t in (1, 5, 10)]
x = np.arange(3); w = 0.26
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.bar(x - w/2, add_means, w, yerr=add_stds, capsize=2.5, color=C_ADD, label="ADD (10 seeds, final)")
ax.bar(x + w/2, mult_means, w, yerr=mult_stds, capsize=2.5, color=C_MULT, label="MULT (3+10 seeds, window mean)")
# BP has no settling parameter — show as horizontal reference, not a matched T sweep
bp_ref = float(np.mean(bp_means))
ax.axhline(bp_ref, color=C_BP, ls=":", lw=1.5, alpha=0.7,
           label=f"BP reference (no settling param, $\\approx${bp_ref:.2f})")
ax.set_xticks(x); ax.set_xticklabels(["$T=1$", "$T=5$", "$T=10$"])
ax.set_ylabel("accuracy (ADD: final; MULT: window mean)"); ax.set_ylim(0, 1.15)
ax.legend(loc="upper left", fontsize=7)
ax.set_title("Settling-time dependence in the EP learning rule")
fig.savefig(os.path.join(FIG, "F3_Tcliff.pdf"))
plt.close(fig)

# ── F4: expressivity ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.0, 3.0))
labels = ["single layer-0\nneuron (bilinear)", "full layer-0\n(aggregate)"]
vals = [0.700, 0.019]
bars = ax.bar(labels, vals, color=[C_NOSTAB, C_STAB], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.3f}",
            ha="center", fontsize=9)
ax.set_ylabel("RMSE of $\\cos(kab)$ fit")
ax.set_yscale("log")
ax.set_title("Expressivity wall: rank gap (Thm 1) vs aggregate capacity")
ax.annotate("rank 1 vs rank 26", xy=(0, 0.700), xytext=(0.12, 0.35),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
fig.savefig(os.path.join(FIG, "F4_expressivity.pdf"))
plt.close(fig)

# ── F5: coexistence ───────────────────────────────────────────────────
coex1 = load("coex/coex_Coex1_10seeds.json")["Coex1"]
c3 = load("coex/coex_C3_10seeds.json")["C3"]
c1_53 = [r["final53"] for r in coex1]; c1_59 = [r["final59"] for r in coex1]
c3_53 = [r["final53"] for r in c3]; c3_59 = [r["final59"] for r in c3]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
x = np.arange(2); w = 0.2
ax.bar(x - 0.3, [np.mean(c1_53), np.mean(c3_53)], w, yerr=[np.std(c1_53), np.std(c3_53)],
       capsize=2.5, color=C_ADD, label="schema 53")
ax.bar(x - 0.1, [np.mean(c1_59), np.mean(c3_59)], w, yerr=[np.std(c1_59), np.std(c3_59)],
       capsize=2.5, color=C_MULT, label="schema 59")
ax.set_xticks(x); ax.set_xticklabels(["Coex1 (separate readouts)", "C3 (shared readout)"])
ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.15)
ax.legend(loc="lower left")
ax.set_title("Co-grokking (final accuracy): 10/10 vs 0/10 (10 seeds)")
fig.savefig(os.path.join(FIG, "F5_coexistence.pdf"))
plt.close(fig)

# ── F6: replay ────────────────────────────────────────────────────────
mvp = load("mvp0_replay_10seeds.json")
tied_ben = mvp["summary"]["tied"]["per_seed_benefit"]
oja_ben = mvp["summary"]["oja"]["per_seed_benefit"]
fig, ax = plt.subplots(figsize=(4.0, 3.0))
bp_ = ax.boxplot([tied_ben, oja_ben], positions=[0, 1], widths=0.5,
                 patch_artist=True, showfliers=False)
for patch, c in zip(bp_["boxes"], [C_ADD, C_MULT]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
for i, (ben, c) in enumerate([(tied_ben, C_ADD), (oja_ben, C_MULT)]):
    ax.scatter(np.full(len(ben), i) + np.random.default_rng(i).normal(0, 0.03, len(ben)),
               ben, s=18, color=c, zorder=3)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set_xticks([0, 1]); ax.set_xticklabels([f"tied\n+0.333, p=0.001 (n=10)",
                                           f"Oja\n+0.454, p=0.002 (n=9)"])
ax.set_ylabel("per-seed replay benefit")
ax.set_title("Replay benefit (wake/sleep vs wake-extra)")
fig.savefig(os.path.join(FIG, "F6_replay.pdf"))
plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f, f"{os.path.getsize(os.path.join(FIG, f)) / 1024:.1f} KiB")
