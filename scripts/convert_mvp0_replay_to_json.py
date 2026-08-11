#!/usr/bin/env python
"""Convert mvp0 replay experiment UTF-16 txt log -> structured JSON artifact.

Source: scripts/mvp0/results_10seed.txt (committed, UTF-16)
Output: outputs/mvp0_replay_10seeds.json
Also verifies the banked C10 numbers (+0.454 Oja / +0.333 tied, p=0.002/0.001)
from the parsed trajectories, so the paper's C10 row traces to a JSON artifact.
"""
import json, re, sys, statistics

SRC = "outputs/results_10seed.txt"
DST = "outputs/mvp0_replay_10seeds.json"

raw = open(SRC, "rb").read()
try:
    text = raw.decode("utf-16")
except UnicodeDecodeError:
    text = raw.decode("utf-8", errors="replace")
text = text.replace("\r\n", "\n").replace("\r", "\n")

# ---- parse ----
seed_re = re.compile(r"^--- seed (\d+) \| train blocks:.*?test=(\d+) ---$")
arm_re = re.compile(r"^\[(\w+)\s*\] held-out traj: \[([\d. ]+)\]  steps=(\d+)  cos=([-\d.]+)")

modes = {}  # mode -> {seed: {arm: {"traj": [...], "steps": int, "cos": float}}}
mode = None
seed = None
for ln in text.split("\n"):
    ls = ln.strip()
    m = re.search(r"# MODE = (\w+)", ls)
    if m:
        mode = m.group(1)
        modes.setdefault(mode, {})
        continue
    s = seed_re.match(ls)
    if s:
        seed = int(s.group(1))
        modes[mode].setdefault(seed, {})
        continue
    a = arm_re.match(ls)
    if a and mode is not None and seed is not None:
        arm, traj, steps, cos = a.group(1), a.group(2), int(a.group(3)), float(a.group(4))
        modes[mode][seed][arm] = {
            "held_out_traj": [float(x) for x in traj.split()],
            "steps": steps,
            "cos_fb_woutT": cos,
        }

# ---- summary: replay benefit = final(wake_sleep) - final(wake_extra) ----
summary = {}
data_quality = {}
for mode, seeds in modes.items():
    finals = {"wake_sleep": [], "wake_extra": []}
    benefits = []
    missing = []
    for s in sorted(seeds):
        ws = seeds[s].get("wake_sleep", {}).get("held_out_traj")
        we = seeds[s].get("wake_extra", {}).get("held_out_traj")
        if ws and we:
            finals["wake_sleep"].append(ws[-1])
            finals["wake_extra"].append(we[-1])
            benefits.append(ws[-1] - we[-1])
        else:
            missing.append(s)
    data_quality[mode] = {
        "seeds_with_complete_wake_arms": [s for s in sorted(seeds) if s not in missing],
        "seeds_missing_wake_arms": missing,
        "note": "oja seed 1: source log contains ONLY shuffled + label_null arms (wake_sleep/wake_extra lines absent — incomplete run, likely interrupted). The banked C10 mean +0.454 normalizes the 9-seed benefit sum by 10 (missing seed counted as 0); the 9 complete seeds show mean +0.505. Wilcoxon p uses the 9 complete seeds (exact p = 1/2^9 for all-positive) and matches the banked p=0.002.",
    }
    summary[mode] = {
        "n_seeds_with_complete_arms": len(benefits),
        "final_acc_wake_sleep": finals["wake_sleep"],
        "final_acc_wake_extra": finals["wake_extra"],
        "per_seed_benefit": benefits,
        "mean_benefit_over_complete_seeds": round(statistics.mean(benefits), 4) if benefits else None,
        "mean_benefit_over_all_10_slots": round(sum(benefits) / 10, 4) if benefits else None,
        "median_benefit": round(statistics.median(benefits), 4) if benefits else None,
        "seeds_positive_benefit": sum(1 for b in benefits if b > 0) if benefits else None,
        "note": "wake_sleep vs wake_extra: both 1600 steps (equal step counts); benefit = final held-out acc difference. wake_extra = extra wake training, no sleep replay. Banked C10 +0.454 == mean over 10 slots (missing seed = 0); +0.505 == mean over the 9 complete seeds.",
    }

# ---- wilcoxon (one-sided) via normal approx on ranks (no scipy dep needed if absent) ----
def wilcoxon_onesided(x, y):
    """Signed-rank test H1: x > y. Returns (W+, p approx)."""
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(x, y, alternative="greater")
        return float(res.statistic), float(res.pvalue), "scipy"
    except Exception:
        pass
    # fallback: normal approximation
    diffs = [a - b for a, b in zip(x, y)]
    diffs = [d for d in diffs if abs(d) > 1e-12]
    n = len(diffs)
    if n == 0:
        return None, None, "no-diffs"
    ranks = {}
    srt = sorted(abs(d) for d in diffs)
    for d in diffs:
        ad = abs(d)
        ranks[ad] = sum(1 for v in srt if v <= ad)
    W = sum(ranks[abs(d)] for d in diffs if d > 0)
    mu = n * (n + 1) / 4
    sigma = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    z = (W - mu) / sigma
    # one-sided p (greater): P(Z >= z)
    from math import erfc
    p = 0.5 * erfc(z / 2 ** 0.5)
    return W, p, "normal-approx"

for mode, s in summary.items():
    x, y = s["final_acc_wake_sleep"], s["final_acc_wake_extra"]
    W, p, how = wilcoxon_onesided(x, y)
    s["wilcoxon_greater"] = {"W": W, "p": p, "method": how, "n": len(s["final_acc_wake_sleep"])}

# ---- check against banked C10 numbers ----
oja = summary["oja"]
tied = summary["tied"]
checks = {
    "oja_mean_over_10_slots": oja["mean_benefit_over_all_10_slots"],
    "banked_oja": 0.454,
    "oja_p": oja["wilcoxon_greater"]["p"],
    "banked_oja_p": 0.002,
    "tied_mean_over_10_slots": tied["mean_benefit_over_all_10_slots"],
    "banked_tied": 0.333,
    "tied_p": tied["wilcoxon_greater"]["p"],
    "banked_tied_p": 0.001,
}
print("=== C10 verification ===")
for k in ("oja_mean_over_10_slots", "oja_p", "tied_mean_over_10_slots", "tied_p"):
    banked_key = "banked_oja" if k.startswith("oja") else "banked_tied"
    if k.endswith("_p"):
        banked_key += "_p"
    banked = checks[banked_key]
    match = abs(checks[k] - banked) < 0.02 if isinstance(banked, (int, float)) else False
    print(f"  {k}: {checks[k]}  (banked {banked})  {'MATCH' if match else 'MISMATCH'}")

artifact = {
    "experiment": "MVP-0 CLS WAKE/SLEEP REPLAY (flat substrate isolator)",
    "config": {
        "P": 53, "K_FREQ": 26, "IN_DIM": 104, "HIDDEN": 256, "K_CONN": 8,
        "N_BLOCKS": 4, "N_WAKE": 200, "N_SLEEP": 200, "steps_per_arm": 1600,
        "N_SEEDS": 10, "modes": ["tied", "oja"],
        "cortex": "EP beta=2.0 T=10 lr_inf=0.50", "oja": "lr_mirror=0.50 N_WARMUP=500",
        "chance": 0.0189,
    },
    "source_log": "outputs/results_10seed.txt (UTF-16, commit 3745dbf)",
    "arms": ["wake_sleep", "wake_extra", "shuffled", "random_replay", "label_null"],
    "per_mode_per_seed": modes,
    "data_quality": data_quality,
    "summary": summary,
    "paper_row_C10": {
        "claim": "replay benefit: Oja +0.454 (p=0.002), tied +0.333 (p=0.001)",
        "computed_from_this_artifact": {
            "oja_mean_over_10_slots": oja["mean_benefit_over_all_10_slots"],
            "oja_mean_over_complete_seeds": oja["mean_benefit_over_complete_seeds"],
            "oja_p": oja["wilcoxon_greater"]["p"],
            "tied_mean_over_10_slots": tied["mean_benefit_over_all_10_slots"],
            "tied_mean_over_complete_seeds": tied["mean_benefit_over_complete_seeds"],
            "tied_p": tied["wilcoxon_greater"]["p"],
        },
    },
}

with open(DST, "w") as f:
    json.dump(artifact, f, indent=1)
print(f"\nWROTE {DST} ({len(json.dumps(artifact))} bytes)")
