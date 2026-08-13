# PAPER 2 OUTLINE — v0.3 (working draft, 2026-08-13 — results landed)

> Status: Skeleton. Banked results marked ✅, to-run marked ⏳, math marked 📐.
> Spine: CLS via multi-region local learning. The dead-fraction pruning result is FALSIFIED
> (honest negative in Discussion), NOT the spine.
> **v0.3 (2026-08-13): PHI-MATTERS control PASSED (EC alphabet load-bearing at scale, real 10/10 vs shuffled/random/onehot 0/10, Δ+0.93 — the toy 'shuffled≈real' was a shallow-1-layer artifact). 64×64 baseline seed 0 GROKS (best 0.9982) with end-of-run decay → C+D confirmed necessary, N=4096 arms unblocked. DG-on-CFG NEGATIVE (decorrelates as predicted but accuracy drops 0.75→0.48 — expansion recoding hurts CFG). P97 scaling wall banked (0/10 at p=97, N≤2048).**
> **v0.2: audit-folded (13 actions), dead-weight falsified (C7→honest negative), EWC→SI-on-EP, spectral clip fix, terminology aligned with Paper 1.**

## Working title

**"Acquiring Alphabets: Continuous Learning without Catastrophic Forgetting under
Locally Plastic, Backprop-Free Rules"**

*Alt: "Cortex-Alone Forgets: A Multi-Region Local-Learning Architecture for Continual Learning"*

Target: cs.LG / cs.NE. Follows Paper 1 (grokking under local rules) as its natural sequel:
Paper 1 = one region learns. Paper 2 = two regions compose and learn *continuously*.

---

## The core claim (one paragraph for the abstract)

**The claim is NOT "the loop prevents forgetting." It is: a one-alphabet cortex cannot
acquire a second task — the hippocampal loop is what makes it possible.** C3 tests the
first half (cortex-alone fails); C4 tests the second (the loop enables); the composition
failure (0/3, commit e5aa1ac) already proves half of it.

Neural networks trained on task A catastrophically forget task B — the field's answer is
regularization (EWC, SI) or rehearsal, both of which *constrain updates to old weights*.
We take the opposite path: **never touch old weights — acquire the new task's
representation (alphabet φ₂) in a spectrally-disjoint subspace.** A hippocampal loop
(EC: streaming Oja self-organization; DG: hardwired sparse random projection)
delivers φ₂ to a cortex that already learned φ₁, under purely local rules (no backprop,
no weight transport). The EC is *living*: it tracks a continuously rotating eigenspace
(streaming min_drift 0.918 vs frozen stuck 0.327 — G-LIVING-6, commit bf6472e), not a
dead frozen alphabet. We show: (R1) a cortex that grokked task 1 cannot acquire task 2
under sequential training — and prove *why*; (R2) the loop is what makes second-task
acquisition possible — add survives mult at ~99%; (R3) a spectral-orthogonality theorem
explains the mechanism; (R4) the minimum EC+DG architecture that preserves add under
mult; (R5) dead weights are prunable (58.7%) but NOT predictably — an honest negative
for crossbar hardware routing (inference-time zero-skipping stands).

---

## Claims table (the honest spine)

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | EC streaming Oja self-organizes alphabet = hand-designed | ✅ BANKED | batch-limit alignment 1.0000; **true streaming 0.972 stationary / 0.958 domain shift** (SPEC_LIVING_EC_v1.2 M1, commit 56fa8a2); **G-LIVING-6: streaming TRACKS rotating eigenspace (min_drift 0.918) vs frozen stuck (0.327) — the living property** (commit bf6472e); **PHI-MATTERS CONTROL (2026-08-13, t_aee4395a): the EC alphabet is LOAD-BEARING at scale — real φ 10/10 grok (window 0.9435±0.020, best 0.9931±0.009) vs shuffled 0/10, random 0/10, onehot 0/10 (all chance, Δ=+0.935). The toy 'shuffled≈real' result was a shallow-1-layer memorization artifact; deep L=6 REQUIRES the correct algebraic structure** (outputs/PHI_MATTERS_REPORT.md) |
| C2 | Cortex groks schemas under EP+Oja local rules | ✅ BANKED | Paper 1: 10/10 grok mod-mult, T-cliff anti-BP |
| C3 | Cortex-alone **cannot acquire** task 2 after learning task 1 (sequential) | ⏳ THE MONEY RUN | sequential add→mult, same substrate; **C+D stabilization (γ_W=0.5, γ_α=0.25, T_decay=1500, commits 52e928b/ac17205) REQUIRED to control end-of-run instability (B4)**; run at N=1536 AND N=4096 with same relative EC:DG:cortex ratios. **UNBLOCKED (2026-08-13): 64×64 baseline seed 0 GROKS (best 0.9982) — the bigger cortex works; end-of-run decay to 0.562 confirmed the C+D requirement (outputs/cortex_baseline_N4096_s0-9.log)** |
| C4 | The loop **enables** second-task acquisition (add survives mult) | ⏳ | C3's control arm WITH EC+DG; run at N=1536 AND N=4096 with same relative ratios |
| C5 | Mechanism: spectral orthogonality (φ₁⊥φ₂ → zero corruption) | 📐 **BANKED-MATH: cite DG decorrelation law (audit D4)** | 1-s′≈(1-s)^0.53 (Theorem SEP, commit acec1bb) |
| C6 | Minimum architecture preserving add under mult | 📐 **BANKED-MATH: cite EC bottleneck theorem (audit D3)** | min EC = O(K characters), 22:1 → alignment 0.991 (commit 1ccd722) |
| C7 | Static dead-fraction is high & predictable (crossbar) | ❌ **FALSIFIED** | probe t_892f8f81 COMPLETED: 58.7% dead weights prunable w/ zero acc loss ✅, BUT cross-seed Jaccard ~0.45 (NOT structural) ❌, early predictability ~0.2 (dead set forms DURING grokking) ❌. Hard-mapping (static sub-crossbar before training) is FALSIFIED. Crossbar story weakens to inference-time zero-skipping only. |
| C8 | DG expansion recoding improves cortex learning | ❌ **FALSIFIED on CFG (2026-08-13)** | DG-on-CFG (outputs/dg_cfg_results.json): DG decorrelates inputs as predicted (cosine −0.25 → 0.0 post-projection) but accuracy DROPS (baseline 0.75 → dg4 0.48 window, dg8 similar). Expansion recoding HURTS CFG learning on the real engine — the 20:1 expansion benefit does not transfer from mod-arithmetic to grammar. **Report as honest negative in Discussion; the separation mechanism is task-dependent** |
| C9 | Scaling wall at p=97 (capacity, not algorithm) | ✅ **BANKED (2026-08-13)** | P97 sweep D1-D4 (outputs/P97_SCALING_WALL_v2.2_VERDICT_REPORT.md): 0/10 grok at p=97 for N=1536-2048, 5k-10k steps (winMed 0.40-0.76, TRANSIENT); controls p=29 10/10, p=41 10/10, p=53 8/10. **The wall localizes to p∈(41,97] — a capacity/representation limit, not a learning-rule failure. Relevant to C3/C4 task choice (avoid p=97 for the money run) and to the 100M scaling discussion** |

**C3 is the paper.** Everything else orbits it. If cortex-alone CAN acquire task 2
without the loop, the paper collapses to a negative — and we publish the mechanism
study instead.

**CRITICAL DISTINCTION (audit D1):** composition failure (a·(b+c) = 0.037 = chance,
banked, commit e5aa1ac) is a DIFFERENT result from C3. Composition = representational
impossibility with a single frozen basis (add period P vs mult period P−1). C3 = does
LEARNED add survive SUBSEQUENT mult training. Never conflate them in the text.

**KEY REFRAME (user-directed, 2026-08-13):** The claim is NOT "the loop prevents
forgetting." It is: **a one-alphabet cortex cannot acquire a second task — the
hippocampal loop is what makes it possible.** C3 tests the first half (cortex-alone
cannot acquire task 2); C4 tests the second (the loop enables acquisition). The
composition failure (0/3, e5aa1ac) already proves the representational-impossibility
half. Scale-fragility is a publishable negative, not a reroute.

---

## Section-by-section outline

### 1. Introduction (~2 pages)
- The CLS problem: catastrophic forgetting, why it blocks lifelong learning
- The field's approach: regularization (EWC, Kirkpatrick 2017; SI, Zenke 2017),
  rehearsal/replay (Rolnick 2019), progressive nets (Rusu 2016) — **all protect old weights**
- Our claim: don't protect — **avoid**. New task → new subspace, delivered by a
  hippocampal loop that grows representations locally
- Why local rules: biologically grounded, hardware-amenable (compute-in-memory), no weight transport.
  **Depth×rule sweep (commit 21c9bb5): EP groks at depth ONLY with full cortical machinery (phi_norm,
  per-neuron thresholds, dendritic products at every layer). On simplified architecture, EP groks only
  at L=1 (0/10 at L=2-6), no simple local rule (Hebbian/Oja/Competitive) groks anywhere. BP is the
  only multi-depth grokker on simplified architecture (audit A6, D12)**
- Contributions C1–C7

### 2. Background & Related Work (~3 pages)
- EP + Oja local learning (self-contained summary of Paper 1)
- Complementary Learning Systems (McClelland 1995; Kumaran 2016) — the theoretical
  antecedent: hippocampus fast, cortex slow. We implement it with local rules
- Regularization-based CLS and its limits (forgetting bounds, plasticity loss)
- Sparse/distributed representations and pattern separation (DG literature)
- Our positioning: first *working* CLS system where the hippocampus **creates** the
  representation, rather than just replaying old episodes

### 3. System Architecture (~3 pages)
- **Cortex:** L=6 sparse 3D, EP contrastive, phi_norm 10% firing (Paper 1, summary)
- **EC:** streaming Oja, sliding-window transition correlation, self-organizes φ
  (C1 banked: batch 1.0000 / streaming 0.972 on real L=6; G-LIVING-6 living property)
- **DG:** hardwired **fixed sparse random projection (k=3-5 entries/row) + k-WTA sparsening
  (commit bf6472e). BIOLOGICALLY CORRECT: EC→DG perforant path is DEVELOPMENTALLY SPECIFIED
  (ephrin-A3/EphA5, SDF-1α guidance cues — Cayco-Gajic 2017: sparse connectivity is ESSENTIAL
  for separation). The earlier learned eigen projection was BIOLOGICALLY WRONG and is replaced
  (audit A5, D8)**
- **The encoder is the wall (audit C1, D7):** input representation, not the engine, is the
  load-bearing choice. Additive-Fourier input: ADD groks 9/10, MULT fails 0/10. E_mult
  character basis: SAME engine groks mult 10/10. Paper 2's loop DEPENDS on EC delivering
  the right encoder per task (commits 41ff3cf/3c5559b)
- **The loop:** input → EC → DG → cortex; task switch = EC acquires φ₂ → DG separates →
  cortex groks in new subspace. Old subspace untouched
- Figures: architecture diagram; alignment curves; 10% firing visualization

### 4. Results (~5 pages)
- **R1 (C3): cortex-alone cannot acquire task 2.** Sequential add→mult on ONE substrate.
  Forgetting metric: add accuracy before/after mult training. Mechanism probe:
  per-channel Fourier energy through training → overlap vs gradient-interference.
  *If overlap: mult's updates land on add's channels. If interference: disjoint
  channels still corrupt.* **C+D stabilization REQUIRED (γ_W=0.5, γ_α=0.25,
  T_decay=1500) — without it, end-of-run instability confounds forgetting vs
  engine decay (audit B4). Interleaved control: F-NEW3 dual-schema coexistence
  (10/10, commit c8bffc3) — why interleaved works but sequential might not (D6)**
- **R2 (C4): the loop enables acquisition.** Same sequence WITH EC+DG. Add retention target ≥0.95.
  Show φ₁/φ₂ channel occupancy (disjoint by construction of DG).
- **R3 (C5): theorem.** Spectral orthogonality → zero cross-task weight corruption.
  **BANKED: DG decorrelation law 1-s′≈(1-s)^0.53 (Theorem SEP, commit acec1bb).**
  Formal statement + proof sketch + empirical confirmation (channel occupancy).
- **R4 (C6): minimum architecture.** N_DG, channel count, k-WTA k that preserve add.
  Parameter sweep + theory. **BANKED: EC bottleneck theorem — min EC = O(K),
  22:1 → alignment 0.991 (commit 1ccd722).**
- **R5 (C7): dead-fraction — FALSIFIED, report as honest negative.** Dead WEIGHTS are
  prunable (58.7%, zero acc loss), BUT the set is NOT structural across seeds (Jaccard ~0.45)
  and NOT predictable early (~0.2). The dead set forms DURING grokking, not from fixed
  connectivity. **Report: "dead weights are a training artifact, not a topological
  property — inference-time zero-skipping is valid; pre-training crossbar routing is not."**
- **R6 (C8): DG expansion recoding — FALSIFIED on CFG, task-dependent.** DG decorrelates
  inputs as the β-model predicts (cosine −0.25 → 0.0) but accuracy drops (0.75 → 0.48).
  The 20:1 expansion benefit proven on mod-arithmetic does NOT transfer to grammar —
  the separation mechanism is task-dependent. **Honest negative: DG helps separation
  where the task needs orthogonalized codes (arithmetic), hurts where it needs
  sequential structure (CFG).**
- **R7 (C9): the p=97 scaling wall.** 0/10 grok at p=97 across N=1536-2048 and 5k-10k
  steps; p=29/41 grok 10/10, p=53 8/10. Wall localizes to p∈(41,97] — capacity/
  representation limit, not rule failure. **Banks the task-choice constraint for C3/C4
  (stay at p=53) and frames the 100M-scaling discussion honestly.**

### 5. Discussion (~2 pages)
- Why this beats regularization: no plasticity-stability tradeoff, no EWC penalty,
  no rehearsal scheduling. Old weights untouched → zero forgetting *by construction*
- **Baseline commitment (audit D10): SI on EP as the primary baseline — NOT EWC.
  EWC's Fisher information matrix requires likelihood gradients (∂log p/∂θ); EP gives
  contrastive updates (free-clamped differences), not likelihood gradients. The empirical
  Fisher is the worst approximation (van de Ven 2025). SI's path-integral importance metric
  IS computable from EP's contrastive gradient × weight change (formalization under math
  review, MATH_SI_ON_EP_v1.0). The 'beats regularization' evidence = SI-on-EP arm.**
- Hardware: honest crossbar framing — sparse compute is 14-34× SLOWER on GPU (banked,
  commit 8bcd0cf); the architecture is built for **compute-in-memory, NOT GPU** (audit D13);
  static dead-fraction → ~~fixed mapping~~ **C7 FALSIFIED: dead set not stable across seeds (Jaccard ~0.45), not predictable early — inference-time zero-skipping only, NOT pre-training crossbar routing**. Scope all 'hardware-amenable' claims to CiM
- Limitations: arithmetic tasks only; alphabet = frequency channels (what is φ for
  language? open — CFG Δ_eff=0.020 is 5× below Δ_c=0.10, known negative B3);
  toy scale (N≤4096); **EC:cortex ratio 2.2:1 vs biological 22:1 (known gap, theorem
  shows margin, audit D11)**; no hippocampus/PFC yet in the loop
- **Learning-rule framing (audit B1): computational FUNCTION determines the rule
  (CLS framework), NOT depth. The depth-determined hypothesis is refuted (banked).**
  Depth×rule sweep: EP at depth needs the FULL cortical machinery (phi_norm,
  thresholds, dendritic products at every layer) — cite in §1 'why local rules' (D12)
- Future: language, 100M scale, full brain (hippocampus replay + PFC routing)

### 6. Methods (~3 pages)
- Engine, hyperparameters, gates (G-LIVING-1, T-cliff), 10-seed protocol
- **C+D stabilization protocol (γ_W=0.5, γ_α=0.25, T_decay=1500, commits 52e928b/ac17205) — REQUIRED for C3 (D5)**
- **Spectral norm clipping: engine currently runs 30 power iterations from RANDOM init
  every call (no warm-start). Fix = persistent buffers + 1-2 warm-started iterations per
  step (Miyato 2018, standard practice). Speeds up all runs; must be applied before C3/C4
  batch (t_7e5add47 research DONE).**
- **Ratio-consistency protocol (user-directed, 2026-08-13): C3/C4 run at BOTH N=1536 and N=4096
  with the SAME relative EC:DG:cortex ratios (2.2:1 EC bottleneck, 20:1 DG expansion).
  Ratio-consistency across scale validates the EC bottleneck theorem (CPC, commit 1ccd722).
  Scale-fragility is a publishable negative, not a reroute.**
- Reproducibility: artifacts + commit SHAs (Paper 1 convention)

### Appendix A: Retraction guard (claims Paper 2 must NOT re-import)

The project's RETRACTIONS INDEX contains ~30 retracted claims. Paper 2 must explicitly
NOT claim the following, and reviewers familiar with the repo will check:

1. **λ = myelination.** Retracted twice (NOTES §RETRACTIONS). The conduction-delay
   parameter λ is a simulation convenience, NOT a biological myelination correlate.
   Do not use "myelination" anywhere in the paper.
2. **Global top-k as "biological competition."** k-WTA is a mean-field approximation;
   P2 requires LOCAL inhibition (per-neuron, lateral). Global top-k is a code
   convenience, not the biological claim. (RESEARCH_DG_LEARNED_VS_HARDWIRED §6)
3. **"1-layer grokking is meaningful."** The depth×rule sweep (commit 21c9bb5)
   confirms EP at L=1 is BP-equivalent/trivial. Only L≥2 grokking counts as evidence
   for local learning; L=1 results must not be cited as support.
4. **Tied-weight cascades (B_fb = W_ffᵀ).** Weight transport is explicitly rejected
   (constitution P3). Gate2 "real PC" rests on code inspection (B_fb ≠ W_ffᵀ
   verified), NOT the Gate2 metric — do not claim symmetry. (STATE_OF_THE_PROGRAMv2
   FRONTIER)

---

## The experiment matrix (what must run, in order)

| Priority | Experiment | Blocks | Effort |
|---|---|---|---|
| P0 | Sequential add→mult, cortex-alone (C3) at N=1536 | everything | ~1-2 days (64×64 baseline pending) |
| P0 | Same + EC+DG loop (C4) at N=1536 | C3 | ~1-2 days |
| P0 | **Ratio-consistency arm: C3+C4 at N=4096 with SAME relative EC:DG:cortex ratios** | C3 result at N=1536 | ~2-3 days (validates EC bottleneck theorem across scale; scale-fragility = publishable negative). **64×64 baseline seed 0 GROKS (best 0.9982) — arms unblocked (2026-08-13)** |
| P0 | **Interleaved control: F-NEW3 dual-schema coexistence (already banked 10/10, commit c8bffc3) — the interleaved counterpart to C3's sequential test** | — | BANKED (cite, no rerun) |
| P0 | **PHI-MATTERS control (t_aee4395a) — EC alphabet load-bearing at scale** | C1 | ✅ DONE 2026-08-13 — PASSED (real 10/10 vs shuffled/random/onehot 0/10, Δ+0.935) |
| P1 | Channel-energy mechanism probe | C3 | same run, logged |
| P1 | Dead-fraction probe (C7) | — | ✅ DONE — FALSIFIED (see claims table) |
| P1 | **DG-on-CFG (C8) — expansion recoding task-dependence** | — | ✅ DONE 2026-08-13 — FALSIFIED (0.75→0.48; honest negative) |
| P1 | **P97 scaling wall (C9)** | — | ✅ DONE 2026-08-13 — 0/10 at p=97, wall in (41,97] (task-choice constraint for C3) |
| P2 | Minimum-arch sweep (C6) | C4 works | ~2-3 days |
| P2 | Spectral orthogonality proof (C5) | C4 | math, parallel |

---

## Open questions for the author (decide before drafting)

1. **Does the loop include hippocampus replay + PFC routing in this paper, or is
   Paper 2 strictly EC+DG+cortex?** (Recommend: strict EC+DG+cortex — hippocampus/PFC
   are Paper 3. The loop here is "acquisition + separation," not full binding.)
2. **Is the dead-fraction result (C7) a section or a separate short paper?**
   **RESOLVED (2026-08-12): FALSIFIED — not a paper.** Dead weights are prunable (58.7%)
   but not predictable or stable (Jaccard ~0.45). Report as honest negative in Discussion
   only. Crossbar story = inference-time zero-skipping, not pre-training routing.
3. **What is "forgetting" measured as?** (Recommend: add test-acc before/after mult
   training, plus per-channel Fourier energy — the mechanism + the number.)
4. **Baselines:** EWC on the same engine? Or just BP + regularization as reference?
   **RESOLVED (2026-08-12): SI-on-EP, NOT EWC.** EWC's Fisher is ill-posed on EP
   (empirical Fisher = worst approximation per van de Ven 2025; true Fisher needs
   likelihood EP doesn't provide). SI's path-integral importance IS computable from
   EP contrastive gradients (MATH_SI_ON_EP_v1.0, under math review). The
   'beats regularization' evidence = SI-on-EP arm vs no-protection arm vs EC+DG loop.

---

## The funding sentence (from this outline)

> "A brain-like loop that learns task 2 without forgetting task 1 — under purely
> local rules, with a provable mechanism and a hardware path. Demo: watch add survive
> mult. That's the paper; the DGX Spark is the machine that scales it."
