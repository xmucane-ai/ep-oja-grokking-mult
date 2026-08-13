# HISTORY-CHECKER AUDIT: Paper-2 Gap Sweep

**Task:** t_b56f24c4
**Scope:** git 542866d (Paper 1 final, 2026-08-10) → HEAD 21c9bb5 (2026-08-12)
**Mode:** READ-ONLY audit. No code changed, no cards created.
**Method:** diffed PAPER2_OUTLINE_v0.1.md against all 25 commits in range, the run ledger (NOTES.md), STATE_OF_THE_PROGRAMv2.md (the FLOOR), RETRACTIONS INDEX, the paper board, and the five SCOPE docs.

---

## A. BANKED-FOR-LATER ITEMS MISSING FROM OUTLINE

These are results, theorems, or mechanisms proven since Paper 1 ended that Paper 2 should claim or cite but the outline (v0.1) does not mention.

### A1. EC is LIVING — non-stationary tracking (G-LIVING-6 PASS) — banked-verified

The outline's C1 cites "alignment 1.0000" but this is the STATIONARY result (G-LIVING-1, streaming=frozen). The outline does NOT mention G-LIVING-6: the streaming EC TRACKS a continuously rotating eigenspace where the frozen EC is stuck (streaming min_drift=0.918 vs frozen 0.327, gate PASS). This is the result that separates a "dead" frozen alphabet from a genuinely "living" one — the headline differentiator for Paper 2's "acquiring alphabets" thesis.

- **Commit:** bf6472e (NOTES.md + artifacts synced from container)
- **Code:** scripts/living_ec_exp5.py
- **Results:** outputs/living_ec_exp5_results.json
- **NOTES ledger:** G-LIVING-6 (line ~10)
- **Status:** BANKED-VERIFIED

### A2. Composition failure (0/3) — the stronger negative the outline conflates with C3

The outline's C3 is "cortex-alone FORGETS under sequential learning" (train add, then mult corrupts). But there is a BANKED result that is STRONGER and DIFFERENT: with a FROZEN single basis, the cortex cannot compose add+mult AT ALL — composed-add 0/3 grok, best mean 0.047, window mean 0.035 (chance=0.019). This is representational impossibility, not forgetting. The outline's C4 says "add survives mult at ~99%" but does not acknowledge that the COMPOSITION prerequisite (a*(b+c) = add-then-mult in one pass) has NEVER been demonstrated — only SEQUENTIAL retention is targeted.

The EC must deliver BOTH algebraically distinct alphabets (additive ψ AND multiplicative χ) as a prerequisite for composition. The outline must distinguish:
- C3 (sequential forgetting): train add fully → switch to mult → does add survive?
- COMPOSITION (simultaneous): a*(b+c) in one pass — 0/3 banked negative with frozen basis

- **Commit:** e5aa1ac (NOTES.md + composition-test falsifier)
- **Code:** scripts/run_composed_task.py
- **Results:** outputs/composed_task_composed-add_L6_10seeds.json
- **NOTES ledger:** t_94b65ab8 (line ~26)
- **Status:** BANKED-VERIFIED (negative result)

### A3. EC bottleneck compression theorem (CPC) — banked-math, directly feeds C6

The outline's C6 is "minimum architecture preserving add under mult" (📐 math, parallel). But we already have MATH_EC_BOTTLENECK_v1.1 with 6 theorems proving:
- Minimum EC size = O(K characters), NOT O(cortical dimension)
- Biological 22:1 compression gives subspace alignment 0.991
- The cortex:EC ratio is NOT a hard bottleneck for algebra preservation
- Chain is governed by narrowest link (BP3: narrowest-dominance)

This is directly on-point for C6 and the outline should cite it.

- **Commits:** 6066713 (v1.0), 1ccd722 (v1.1 math-review patches)
- **Spec:** docs/MATH_EC_BOTTLENECK_v1.1.md
- **Verification code:** docs/verify_ec_bottleneck_v11.py (5-seed averaged, QR-corrected)
- **Status:** BANKED-MATH (reviewed, 4 MAJOR+2 MINOR patches applied, all theorems qualitatively unchanged)

### A4. DG decorrelation law (β-model, Theorem SEP v1.1) — banked-math, directly feeds C5

The outline's C5 is "spectral orthogonality theorem (📐 CLS interleaving theorem)". But we already have the PROVEN decorrelation law: DG expansion + k-WTA transforms input cosine s into output satisfying 1-s' ≈ (1-s)^0.53 (square-root-type compression of dissimilarity). This is the mathematical foundation for WHY DG separation prevents cross-task interference. The outline should cite Theorem SEP and the β-model.

- **Commit:** acec1bb (4-reviewer ALL APPROVE)
- **Spec:** docs/MATH_DG_ORTHOGONALIZATION_v1.1.md
- **Status:** BANKED-MATH (reviewed APPROVE by all 4 reviewers)

### A5. DG is HARDWIRED, not learned — banked-research, feeds §3 architecture

The outline correctly says "DG: hardwired phase-diverse eigen projection" (§3) and cites "research t_db4ecd8e". But the banked research is deeper than a citation: it proves the EC→DG perforant path is DEVELOPMENTALLY SPECIFIED (ephrin-A3/EphA5, SDF-1α guidance cues), and that the correct model is a FIXED SPARSE random projection (k=3-5 entries/row), NOT a learned eigen projection. The learned eigen projection was biologically WRONG and should be explicitly flagged as replaced. The outline should also mention Cayco-Gajic 2017 (sparse connectivity is ESSENTIAL for separation) as the computational justification.

- **Commit:** bf6472e
- **Spec:** docs/RESEARCH_DG_LEARNED_VS_HARDWIRED_v1.0.md
- **Status:** BANKED-RESEARCH (synthesis with citations)

### A6. Depth×rule sweep — banked-verified, feeds §1 "why local rules"

The outline's §1 says "why local rules: biologically grounded, hardware-amenable" but doesn't cite the depth×rule sweep that showed: on a SIMPLIFIED shared architecture, EP groks ONLY at L=1 (0/10 at L=2-6), NO simple local rule (Hebbian/Oja/Competitive) groks ANYWHERE, and BP is the only multi-depth grokker. The CRITICAL nuance: the banked L=6×EP=10/10 was on the FULL cortex_v14_7 (phi_norm, per-neuron thresholds, dendritic products at every layer). Paper claim REFINED: "architecture × depth determines learning rule success — the full cortical machinery is a prerequisite for EP at depth."

- **Commit:** 21c9bb5
- **Report:** docs/DEPTH_RULE_SWEEP_REPORT.md
- **Code:** scripts/depth_rule_sweep.py
- **Results:** outputs/results_depth_rule_sweep.json
- **Status:** BANKED-VERIFIED

### A7. Dual-schema coexistence (F-NEW3) — banked-verified, feeds C4

STATE_OF_THE_PROGRAMv2 F-NEW3 (commit c8bffc3, 2026-08-10 — just inside Paper 1's scope but banked as a FLOOR result): two addition schemas (add mod-53 + add mod-59) GROK SIMULTANEOUSLY in ONE L=6 cortex via SHARED REPRESENTATION (C_median≈0, no collision, 10/10 both schemas). This shows the cortex CAN hold multiple schemas without interference — directly relevant to C4 (the loop preserves). The outline should cite this as evidence that the cortex substrate supports multi-schema coexistence.

- **Commit:** c8bffc3
- **Results:** outputs/coex/coex_Coex1_10seeds.json, coex_C3_10seeds.json
- **STATE_OF_THE_PROGRAMv2:** F-NEW3
- **Status:** BANKED-VERIFIED (10-seed PASS, Constitution-compliant)

---

## B. RISKS/CAVEATS BANKED THAT THE OUTLINE IGNORES

### B1. Depth-determined learning rule hypothesis REFUTED

The outline does NOT make this claim, but the research (t_24d92922, commit 21c9bb5) explicitly refutes "depth determines which local learning rule is viable." If Paper 2's narrative leans toward "hippocampus uses simpler rules because it's too shallow for EP," a reviewer with the banked research will flag it. The defensible claim is: "computational FUNCTION determines the rule (CLS framework), not depth." The hippocampus is NOT shallow (trisynaptic loop = 4 areas + massive CA3 recurrence).

- **Doc:** docs/RESEARCH_DEPTH_DETERMINED_LEARNING_RULE_v1.0.md
- **Status:** BANKED-RESEARCH (refutation)

### B2. Substrate gap / PREMISE BROKEN (VET re-assessment)

The VET re-assessment (commit 7d97812) maintains the verdict: "PREMISE BROKEN — EP instability research does not close substrate gap; learned-encoder failed twice." The outline's §5 Discussion mentions "toy scale (N≤4096)" but does not acknowledge the substrate gap: the current results are on modular arithmetic, and the path to language (the project's real goal) has known obstacles (CFG below Δ_c, encoder is the wall). A reviewer will ask: "what is φ for language?" — the outline lists this as an open question (§5 Limitations) but doesn't cite the banked CFG failure mode.

- **Commit:** 7d97812
- **Status:** BANKED-CAVEAT (negative, VET-endorsed)

### B3. CFG effective gap 5× below Δ_c — known negative for language claims

MATH_DG_ORTHOGONALIZATION_v1.1 §6: at the d8-d9 depth boundary (train/test split), the effective bilinear gap Δ_eff ≈ 0.020, which is ~5× below Δ_c = 0.10. G3 FAIL is qualified: DG alone does NOT cross Δ_c for deep pairs (n≥5). CA3 completion MIGHT bridge the 5× gap, but this is OPEN. If Paper 2 claims language applicability, the CFG spectral-gap result is a direct counterexample.

- **Doc:** docs/MATH_DG_ORTHOGONALIZATION_v1.1.md §6.2
- **Status:** BANKED-MATH (qualified negative)

### B4. End-of-run instability — confound for C3

The matched ADD-vs-MULT 2×2 (t_81a4f9ea, HANDOFF.md 2026-08-10) showed: ADD 6/10 decayed 20-40 pts by final eval, composed mult 4/10 decayed. The outline's C3 money run (sequential add→mult) must control for this: if add decays on its OWN (without mult training), how do we distinguish sequential forgetting from engine instability? The C+D stabilization (γ_W=0.5, γ_α=0.25, T_decay=1500) fixed this for mult-stab (10/10 stable), so C3 should use C+D stabilization. But the outline's Methods section does not mention C+D.

- **Status:** BANKED-CAVEAT (confound identified, fix known)

### B5. Sparse compute is NOT a GPU win — hardware framing risk

Banked (t_52713334, STATE_OF_THE_PROGRAMv2 era but cited in HANDOFF): torch.sparse.mm/cuSPARSE is ~14× SLOWER forward (1.98ms vs 27.6ms) and ~34× SLOWER learning (0.25ms vs 8.77ms) at N=1536, 9.4% density. Sparse NEVER wins across the density sweep [0.5%-50%]. The outline's §5 Discussion says "honest crossbar framing (measured 14-34× sparse tax on GPU)" — this is correct but the outline frames it as a GPU limitation, not a sparsity-is-representation-only result. The crossbar energy story (physical crossbar skips zero rows) is unaffected, but the "hardware-amenable" claim in §1 must be scoped to compute-in-memory, NOT GPU.

- **Commit:** 8bcd0cf (benchmark); folded into paper at b167c02
- **Results:** outputs/sparse_matmul_benchmark.json
- **Status:** BANKED-VERIFIED

### B6. Dead-fraction probe confounds (C7)

The outline correctly notes C7 is RUNNING (probe t_892f8f81) and mentions "not neurons — gate confound." But the HANDOFF.md C2 precision-protection spec (2026-07-30, now historical) flagged that precision≈freeze is required by AGENTS rule #6, and that W2-GAP was code-confirmed (precision protects W1 only). If the dead-fraction probe measures WEIGHT deadness, the gate confound (hard gate u>θ zeroes 90% by construction) must be separated from representational deadness.

- **Status:** BANKED-CAVEAT (probe in progress, confounds identified)

### B7. EC streaming alignment is 0.972, not 1.000 (verification honesty)

SPEC_LIVING_EC_v1.2 M1 patch: the "Verified: 1.000" labels are BATCH-LIMIT results. The TRUE streaming alignment (10k steps, η=0.1) is 0.972 (stationary), 0.958 (domain shift). The outline's C1 says "alignment 1.0000" — this is the frozen/batch number. The streaming number is 0.972. A reviewer checking the verification code will flag the discrepancy. C1 should report both: "batch-limit alignment 1.0000; true streaming 0.972 (stationary), 0.958 (domain shift)."

- **Commit:** 56fa8a2 (v1.2 folds this honesty patch)
- **Spec:** docs/SPEC_LIVING_EC_v1.2.md §0 (M1 caveat)
- **Status:** BANKED-VERIFIED (honesty-corrected)

### B8. EC model ratio 2.2:1 vs biological 22:1

The model uses cortex:EC ratio ~2.2:1; biology is ~22:1 (MATH_EC_BOTTLENECK §0). The CPC theorem proves the biological ratio provides subspace alignment 0.991, and the minimum is O(K characters). The model's 2.2:1 is sufficient for modular arithmetic (K≈7-14) but is a known scale gap. The outline's §5 Limitations should mention this.

- **Doc:** docs/MATH_EC_BOTTLENECK_v1.1.md §0, docs/RESEARCH_DG_LEARNED_VS_HARDWIRED §Bonus
- **Status:** BANKED-CAVEAT (known gap, theorem shows margin)

---

## C. GAPS THE OUTLINE SHOULD COVER

### C1. The encoder is the wall (the matched ADD-vs-MULT result)

The outline's architecture section describes the cortex and EC/DG but does not state the banked finding (HANDOFF.md 2026-08-10): the input REPRESENTATION, not the engine, is the wall. Standard additive-Fourier bilinear input: ADD groks 9/10, MULT fails 0/10. E_mult character-basis (discrete-log) encoder: SAME engine groks mult 10/10. Paper 2's C4 (the loop prevents forgetting) DEPENDS on the EC delivering the right encoder for each task. This must be explicit in the architecture section.

- **Commits:** 41ff3cf + 3c5559b (Paper 1 era but the finding is load-bearing for Paper 2)
- **Results:** outputs/matched_{add,mult}_L6_N1536_T10.json
- **Status:** BANKED-VERIFIED

### C2. Sequential vs interleaved acquisition — C3's design implication

The outline's C3 is "sequential add→mult" (forgetting). F-NEW3 shows INTERLEAVED dual-schema coexistence works (C≈0, no interference). The paper must address both:
- Sequential (C3): the hard case — does forgetting occur?
- Interleaved (F-NEW3): already shown to work — no interference

If C3 shows forgetting and C4 shows the loop prevents it, the mechanism (spectral orthogonality via DG) must explain WHY sequential fails but interleaved works. The outline's R1 mechanism probe (per-channel Fourier energy) is the right tool but doesn't cite F-NEW3 as the interleaved control.

### C3. EWC/SI baseline commitment

The outline's OQ4 asks "EWC on the same engine?" but doesn't commit. Paper 2's core claim is "beats regularization." Without an EWC/SI baseline on the SAME engine, the claim is unsupported. Recommend: EWC adapted to EP (Fisher information from the contrastive signal) as the primary baseline. The CLS literature (McClelland 1995, Kumaran 2016) is already in the repo (hippocampus_research.md, RESEARCH_DEPTH_DETERMINED §4.1).

### C4. Related-work hooks already in repo

The repo contains extensive literature that Paper 2's Related Work section should cite:
- CLS theory: McClelland, McNaughton & O'Reilly 1995 (RESEARCH_DEPTH_DETERMINED §4.1)
- EP/PC depth scaling: µPC (Innocenti 2025), Error Highways (Mohammadi 2026), Deep PC (Anwar 2025) — all in RESEARCH_DEPTH_DETERMINED §2.2
- T-axis (Song 2024 prospective configuration) — RESEARCH_DEPTH_DETERMINED §2.4
- PC-vs-BP critical evaluation (Zahid 2023) — RESEARCH_DEPTH_DETERMINED §2.3
- DG sparse connectivity (Cayco-Gajic 2017) — RESEARCH_DG_LEARNED_VS_HARDWIRED §4.1
- CHL≡BP equivalence (Xie-Seung 2003) — RESEARCH_DEPTH_DETERMINED §2.1

### C5. Retraction guard: λ=myelination, global top-k, tied-weight cascades

The RETRACTIONS INDEX contains ~30 retracted claims. Paper 2 must NOT re-import:
- λ = myelination (retracted twice, NOTES §RETRACTIONS)
- Global top-k as "biological competition" (k-WTA is a mean-field approximation; P2 requires local inhibition — RESEARCH_DG_LEARNED_VS_HARDWIRED §6)
- "1-layer grokking is meaningful" (the depth×rule sweep confirms EP at L=1 is BP-equivalent/trivial)
- Tied-weight cascades (B_fb ≠ W_ff.T is P3; Gate2 "real PC" rests on code inspection, NOT the Gate2 metric — STATE_OF_THE_PROGRAMv2 FRONTIER)

---

## D. PRIORITIZED ACTION LIST

In order of importance for the outline:

1. **[CRITICAL] Add composition failure (0/3) as a distinct result from C3.** The frozen-basis composition negative is banked and a reviewer will know about it. Distinguish sequential forgetting (C3, untested) from composition impossibility (0/3, banked). Cite commit e5aa1ac.

2. **[CRITICAL] Add G-LIVING-6 (non-stationary tracking) to C1.** The outline cites only the stationary alignment (1.0000). The genuinely "living" property (streaming tracks rotating eigenspace, 0.918 vs frozen 0.327) is the headline differentiator and is banked. Cite commit bf6472e.

3. **[HIGH] Cite the EC bottleneck theorem (CPC) in C6.** The outline marks C6 as 📐 (math, parallel) but 6 theorems are already proven. Minimum EC size = O(K), biological 22:1 gives alignment 0.991. Cite commit 1ccd722.

4. **[HIGH] Cite the DG decorrelation law (β-model, Theorem SEP) in C5.** The outline marks C5 as 📐 but the decorrelation law 1-s' ≈ (1-s)^0.53 is proven. Cite commit acec1bb.

5. **[HIGH] Add C+D stabilization to the Methods section.** The outline does not mention step-decay (γ_W=0.5) and decayed-α (γ_α=0.25, T_decay=1500). Without this, C3's money run will hit the known end-of-run instability (ADD 6/10 decay 20-40pts). Cite commits 52e928b/ac17205.

6. **[HIGH] Add F-NEW3 (dual-schema coexistence) as the interleaved control for C3/C4.** Shows cortex CAN hold two schemas without interference when interleaved — the complement to C3's sequential test. Cite commit c8bffc3.

7. **[MEDIUM] Cite the encoder-is-the-wall finding in §3 Architecture.** Paper 2's loop depends on the EC delivering the right encoder per task. Cite commits 41ff3cf/3c5559b.

8. **[MEDIUM] Cite DG-hardwired research (RESEARCH_DG_LEARNED_VS_HARDWIRED) in §3.** The eigen projection is biologically wrong; fixed sparse random (k=3-5) is correct. Cite commit bf6472e.

9. **[MEDIUM] Correct C1's alignment to report streaming (0.972) alongside batch-limit (1.000).** Verification honesty per SPEC_LIVING_EC_v1.2 M1. Cite commit 56fa8a2.

10. **[MEDIUM] Commit to an EWC/SI baseline (OQ4).** Paper 2's "beats regularization" claim needs the baseline on the same engine.

11. **[LOW] Add EC ratio gap (2.2:1 vs 22:1) to §5 Limitations.** Known gap, theorem shows margin.

12. **[LOW] Cite depth×rule sweep in §1 "why local rules."** EP groks at depth ONLY with full cortical machinery. Cite commit 21c9bb5.

13. **[LOW] Scope "hardware-amenable" to compute-in-memory, not GPU.** Sparse compute is 14-34× slower on GPU (banked benchmark). Cite commit 8bcd0cf.

---

## VERDICT: Is C3 (cortex-alone forgets) consistent with everything banked?

**CONSISTENT — but the weakest-justified claim in the outline, and here is the precise risk.**

C3 is UNTESTED on the current substrate (it's the money run, ⏳). No banked result CONTRADICTS it, but two banked results COMPLICATE it:

1. **F-NEW3 (dual-schema coexistence):** the cortex CAN represent two schemas simultaneously WITHOUT interference when they are acquired INTERLEAVED (C_median≈0, 10/10 both schemas). This means the cortex substrate does NOT inherently reject a second schema. C3's "forgetting" must be specific to SEQUENTIAL acquisition, not a general multi-schema limitation. The outline correctly scopes C3 as "sequential add→mult on ONE substrate" — this is the right design. But the paper must explicitly explain WHY interleaved works (F-NEW3) but sequential might not (C3 untested): the hypothesis is that sequential training overwrites φ₁'s weight channels, while interleaved lets both co-represent from the start.

2. **Composition failure (0/3):** with a frozen single basis, the cortex cannot compose add+mult at all. This is NOT forgetting (the cortex never learned the composed task). But a reviewer will conflate the two. The outline must distinguish: C3 measures whether LEARNED add survives SUBSEQUENT mult training; the composition negative measures whether the cortex can learn add+mult SIMULTANEOUSLY. Different phenomena, different mechanisms.

**No banked result contradicts C3.** The HANDOFF.md historical C2/C4 data (sequential forgets add 0.017; interleave retains 1.0) is from the OLD composition chapter substrate (pre-engine-pivot, ~2026-07-30), NOT the current L=6 AblationCortex. So the closest precedent is on a different engine. C3 is a genuine open question on the current substrate — which is exactly why it's "THE MONEY RUN."

**The spine holds.** The outline's narrative (C3 = cortex-alone forgets → C4 = the loop prevents it → C5 = spectral orthogonality explains why) is logically sound and consistent with all banked results. The risk is not contradiction but INCOMPLETENESS: the outline doesn't cite the banked math (DG decorrelation law, EC bottleneck theorem), doesn't distinguish composition failure from sequential forgetting, and doesn't mention the stabilization machinery (C+D) that C3's money run will need.
