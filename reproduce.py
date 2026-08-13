#!/usr/bin/env python3
"""
reproduce.py — map every outputs/ artifact to its producing script + exact command.

The paper (Appendix A) footnotes each claim to an artifact. This script is the
reverse index: given an artifact, find the command that regenerates it. It is the
entry point a technical reviewer should run first.

Usage:
    python reproduce.py                     # list every artifact → command
    python reproduce.py <pattern>           # show commands matching pattern
    python reproduce.py --check             # verify all cited artifacts exist on disk
    python reproduce.py --verify-hashes     # verify provenance.json hashes match files

Honesty note: not every artifact in outputs/ has a surviving producer script.
Artifacts whose generating script is lost or was container-side are marked
[NO SCRIPT] explicitly — they are still byte-verified in provenance.json, but
their regeneration path is not available. We report this rather than hide it.
"""

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
SCRIPTS = os.path.join(HERE, "scripts")
PROVENANCE = os.path.join(HERE, "provenance.json")

# artifact-prefix → (script, example command)
# Each command is the literal invocation used (or a documented reconstruction).
REPRODUCE_MAP = {
    # ── Paper 1 headline arms ──────────────────────────────────────────────
    "basis_swap_v13_mult-stab_frac80_L6_N1536_T10.json":
        ("run_basis_swap_v13.py",
         "python scripts/run_basis_swap_v13.py --task mult-stab --seeds 0-9 --steps 3000 --train-fractions 0.8"),
    "basis_swap_v13_mult-nostab_frac80_L6_N1536_T10.json":
        ("run_basis_swap_v13.py",
         "python scripts/run_basis_swap_v13.py --task mult-nostab --seeds 0-9 --steps 3000 --train-fractions 0.8"),
    "basis_swap_v13_add-sanity_frac80_L6_N1536_T10.json":
        ("run_basis_swap_v13.py",
         "python scripts/run_basis_swap_v13.py --task add-sanity --seeds 0-9 --steps 3000 --train-fractions 0.8"),
    # ── BP controls ────────────────────────────────────────────────────────
    "bp_control_arm1_mult_main_L6_N1536_T10.json":
        ("run_bp_control_arm1.py",
         "python scripts/run_bp_control_arm1.py --task mult --seeds 0-9 --steps 3000"),
    "bp_control_arm1_mult_main_L6_N1536_T10_S10000.json":
        ("run_bp_control_arm1.py",
         "python scripts/run_bp_control_arm1.py --task mult --seeds 0-9 --steps 10000"),
    "bp_control_arm1_mult_main_L6_N1536_T10_S10000_n10.json":
        ("run_bp_control_arm1.py",
         "python scripts/run_bp_control_arm1.py --task mult --seeds 0-9 --steps 10000 --n 10"),
    "bp_control_arm1_mult_tcliff_L6_N1536.json":
        ("run_bp_control_arm1.py",
         "python scripts/run_bp_control_arm1.py --task mult-tcliff --T 1,5,10"),
    "e4_wd_sweep_bp_arm.json":
        ("e4_wd.py", "python scripts/e4_wd.py --wd 0.0,0.1,1.0"),
    # ── T-cliff / settling ─────────────────────────────────────────────────
    "ep_mult_tcliff_banking_L6_N1536.json":
        ("run_basis_swap_v13.py",
         "python scripts/run_basis_swap_v13.py --task mult-tcliff --T 1,5,10"),
    "l6eqcap_T10_10seeds.json":
        ("run_basis_swap_v13.py", "python scripts/run_basis_swap_v13.py --task eqcap --T 10 --seeds 0-9"),
    "l6eqcap_T5_10seeds.json":
        ("run_basis_swap_v13.py", "python scripts/run_basis_swap_v13.py --task eqcap --T 5 --seeds 0-9"),
    "l6eqcap_T1_10seeds.json":
        ("run_basis_swap_v13.py", "python scripts/run_basis_swap_v13.py --task eqcap --T 1 --seeds 0-9"),
    # ── Coexistence / multi-schema ─────────────────────────────────────────
    "coex/coex_Coex1_10seeds.json":
        ("coexistence_add_v14.py", "python scripts/coexistence_add_v14.py --coex Coex1 --seeds 0-9"),
    "coex/coex_C3_10seeds.json":
        ("coexistence_add_v14.py", "python scripts/coexistence_add_v14.py --coex C3 --seeds 0-9"),
    # ── C+D boundary map ───────────────────────────────────────────────────
    "boundary_map_arm3_stage1_seed0_all.json":
        ("run_boundary_map_arm3.py", "python scripts/run_boundary_map_arm3.py --stage 1"),
    "boundary_map_arm3_stage2_seeds0-7_all.json":
        ("run_boundary_map_arm3.py", "python scripts/run_boundary_map_arm3.py --stage 2 --seeds 0-7"),
    # ── Expressivity / rank gap ────────────────────────────────────────────
    "f4_expressivity_audit.json":
        ("make_paper_figures.py", "python scripts/make_paper_figures.py --f4-only"),
    "c6_exactness_verify.json":
        ("schema_gate2.py", "python scripts/schema_gate2.py --verify-exactness"),
    "matched_mult_L6_N1536_T10.json":
        ("run_matched_mult_add.py", "python scripts/run_matched_mult_add.py --task mult --seeds 0-9"),
    "matched_mult_L6_N1536_T10_alignment.json":
        ("run_matched_mult_add.py", "python scripts/run_matched_mult_add.py --task mult --alignment"),
    "matched_add_L6_N1536_T10.json":
        ("run_matched_mult_add.py", "python scripts/run_matched_mult_add.py --task add --seeds 0-9"),
    # ── Fresh seeds / held-out ─────────────────────────────────────────────
    "fresh_seed_confirm_100-104_L6_N1536_T10.json":
        ("run_fresh_seed_confirm.py",
         "python scripts/run_fresh_seed_confirm.py --seeds 100-104 --steps 3000"),
    # ── Oja mirror (C11) ───────────────────────────────────────────────────
    "c11_oja_run_artifact.json":
        ("ep_oja.py", "python scripts/ep_oja.py --mode c11-oja"),
    # ── Replay (C10) ───────────────────────────────────────────────────────
    "results_10seed.txt":
        ("convert_mvp0_replay_to_json.py",
         "(source log; converted to mvp0_replay_10seeds.json)"),
    "mvp0_replay_10seeds.json":
        ("convert_mvp0_replay_to_json.py",
         "python scripts/convert_mvp0_replay_to_json.py"),
    # ── p97 scaling wall ───────────────────────────────────────────────────
    "p97wall_I1_p29_N1536_S3000.json":
        ("run_p97_scaling_wall.py", "python scripts/run_p97_scaling_wall.py --p 29 --N 1536 --steps 3000"),
    "p97wall_I2_p41_N1536_S3000.json":
        ("run_p97_scaling_wall.py", "python scripts/run_p97_scaling_wall.py --p 41 --N 1536 --steps 3000"),
    "p97wall_D2_p97_N1536_S8000.json":
        ("run_p97_scaling_wall.py", "python scripts/run_p97_scaling_wall.py --p 97 --N 1536 --steps 8000"),
    "prime_sweep_summary.json":
        ("aggregate_prime_sweep.py", "python scripts/aggregate_prime_sweep.py"),
    # ── Sparse matmul benchmark ────────────────────────────────────────────
    "sparse_matmul_benchmark.json":
        ("run_basis_swap_v13.py", "python scripts/run_basis_swap_v13.py --benchmark-sparse"),
    # ── Learned encoder (falsified alternative) ────────────────────────────
    "learned_enc_L6_1_s0-1-2.json":
        ("launch_learned_enc.sh", "bash scripts/launch_learned_enc.sh"),
    # ── Dense MLP control ──────────────────────────────────────────────────
    "rca_e1_dense_mlp_h512_nh2_relu_wd1.0_S20000.json":
        ("rca_e5_skip_connections.py",
         "python scripts/rca_e5_skip_connections.py --e1-dense --wd 1.0 --steps 20000"),
    # ── Temporal / v14 (falsified paradigm) ────────────────────────────────
    "results_v14_temporal_dendritic_v3_N4096_L2.json":
        ("run_temporal_v14.py", "python scripts/run_temporal_v14.py"),
    "v13_14_L4_clamped_seeds5-10.json":
        ("cortex_v13_13.py", "python scripts/cortex_v13_13.py --L 4 --clamped --seeds 5-10"),
    "smoothgate_step0_L2.json":
        ("run_smoothgate_step0.py", "python scripts/run_smoothgate_step0.py"),
    "smoothgate_step0_L2_beta12.json":
        ("run_smoothgate_step0.py", "python scripts/run_smoothgate_step0.py --beta 1.2"),
    # ── TBT track ──────────────────────────────────────────────────────────
    "tbt_laminar_L6_N512_opt.json":
        ("tbt_laminar_opt.py", "python scripts/tbt_laminar_opt.py --L 6 --N 512"),
    "tbt_laminar_L6_N512_opt_hsfvss.json":
        ("tbt_laminar_opt.py", "python scripts/tbt_laminar_opt.py --L 6 --N 512 --hsfvss"),
    "TBT_LAMINAR_FINAL_LEDGER_t_5bfe1989.json":
        ("tbt_laminar_opt.py", "python scripts/tbt_laminar_opt.py --ledger"),
    # ── Neighborhood sweep ─────────────────────────────────────────────────
    "neighborhood_sweep_stage0.json":
        ("neighborhood_sweep_L6.py", "python scripts/neighborhood_sweep_L6.py --stage 0"),
    # ── Alphabet acquisition / precision floor (follow-up, EV artifacts) ───
    "alphabet_arm2_confirm_frequency_seeds0-9.json":
        ("launch_alphabet_arm2_probe.sh", "bash scripts/launch_alphabet_arm2_probe.sh"),
    "bfp_ste_latent.json":
        ("run_lowbit_falsifier.py", "python scripts/run_lowbit_falsifier.py --int4-latent"),
    "phi_matters_control_results.json":
        ("phi_matters_control.py", "python scripts/phi_matters_control.py --seeds 0-9"),
    "firing_rate_ablation_L6.json":
        ("firing_rate_ablation_L6.py", "python scripts/firing_rate_ablation_L6.py"),
    "living_ec_exp0_real_results.json":
        ("living_ec_exp0_real.py", "python scripts/living_ec_exp0_real.py --seeds 10"),
    "living_ec_exp5_results.json":
        ("living_ec_exp5.py", "python scripts/living_ec_exp5.py"),
    # ── CFG / language track (honest negatives) ────────────────────────────
    "cfg_L1c_fixed_enc_seeds0-9.json":
        ("launch_cfg_L1c.sh", "bash scripts/launch_cfg_L1c.sh"),
    "cfg_L1_learned_enc_seeds0-9.json":
        ("launch_cfg_L1b.sh", "bash scripts/launch_cfg_L1b.sh"),
    # ── two-scale / DG ─────────────────────────────────────────────────────
    "two_scale_L6_stage0.json":
        ("two_scale_L6.py", "python scripts/two_scale_L6.py --stage 0"),
    # explicitly lost producers — reported, not hidden
    "dg_cfg_results.json":
        (None, "NO SCRIPT — produced container-side (t_259f36f7 era); script not synced back"),
    "dg_cfg_cosine.json":
        (None, "NO SCRIPT — produced container-side (t_259f36f7 era); script not synced back"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = sys.argv[1:]
    mode = "list"
    pattern = None
    if args and args[0] == "--check":
        mode = "check"
    elif args and args[0] == "--verify-hashes":
        mode = "hashes"
    elif args:
        mode = "filter"
        pattern = args[0]

    if mode == "hashes":
        prov = json.load(open(PROVENANCE))
        arts = prov.get("artifacts", {})
        bad = []
        for rel, expected in sorted(arts.items()):
            fp = os.path.join(HERE, rel)
            if not os.path.isfile(fp):
                bad.append(f"{rel}: MISSING")
                continue
            if sha256(fp) != expected:
                bad.append(f"{rel}: HASH MISMATCH")
        print(f"provenance artifacts: {len(arts)}, mismatches/missing: {len(bad)}")
        for b in bad:
            print(f"  [FAIL] {b}")
        if not bad:
            print("ALL HASHES MATCH")
        return 0 if not bad else 1

    if mode == "check":
        # every artifact cited in the paper resolves to disk
        tex = open(os.path.join(HERE, "paper", "main.tex"), encoding="utf-8",
                   errors="replace").read()
        refs = set(re.findall(r"\\path\{outputs/([^}]+)\}", tex))
        refs |= set(re.findall(r"outputs/([a-zA-Z0-9_\-\./]+\.(?:json|txt))", tex))
        refs = {r.replace("\\_", "_") for r in refs}
        missing = [r for r in sorted(refs) if not os.path.isfile(os.path.join(OUTPUTS, r))]
        print(f"paper-cited artifacts: {len(refs)}, missing: {len(missing)}")
        for m in missing:
            print(f"  [MISSING] outputs/{m}")
        return 0 if not missing else 1

    rows = []
    for f in sorted(os.listdir(OUTPUTS)):
        full = os.path.join(OUTPUTS, f)
        if not os.path.isfile(full):
            continue
        entry = REPRODUCE_MAP.get(f)
        if entry:
            script, cmd = entry
            rows.append((f, script, cmd))
        else:
            rows.append((f, "(unmapped)", "python reproduce.py <name> — add mapping"))

    if mode == "filter":
        rows = [r for r in rows if pattern in r[0]]

    n_script = sum(1 for r in rows if r[1] != "(unmapped)" and r[1] is not None)
    n_noscript = sum(1 for r in rows if r[1] is None)
    print(f"outputs: {len(rows)} | mapped-to-script: {n_script} | explicit NO-SCRIPT: {n_noscript}")
    print()
    for f, script, cmd in rows:
        s = script if script else "NO SCRIPT"
        print(f"{f}")
        print(f"    {s}: {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
