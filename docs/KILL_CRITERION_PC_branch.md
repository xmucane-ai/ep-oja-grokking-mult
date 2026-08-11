# KILL-CRITERION — the PC branch (pre-registered exit, not a rescue)

**v2 — patched after the 3-agent validation gate.** Changes: (1) P1 restructured — now tests FIT
as well as HOLD (the gate's fatal finding: v1 could not fire); (2) C2/C3/C5 moved to d=1.0 real
labels wd=0, deconfounding the instability from the cliff; (3) absolute fit floors added;
(4) §0(a) relabeled per #6 (wd-disentangled, optimizer-axis pending); (5) cite nits fixed.
Status: PRE-REGISTRATION DRAFT v2 for re-validation. Nothing below may be edited after C1 starts.
Per AGENTS.md: every claim carries a falsifier; gates kill claims, not routes; pivots narrow
claims; one variable per arm; paired arms, matched compute.

## 0. The ledger being judged

(a) d=0.5 **test-level** cliff — **wd-disentangled; optimizer-axis pending** (wd swept, Adam
fixed; the SGD/LR sweep never ran — per #6 this is not "proven"). EXCLUDED from this document's
kill logic. NOTE: the exclusion covers the TEST-level datum ONLY. The d=0.5 **train-level** gap
(PC train ≈0.108 vs oracle 0.830 — finite-T relaxation degradation, wd-independent across the
Gate-2.1a oracle sweep) is PC-specific and is IN kill-scope via K0 (§3).
(b) wd-independent relaxation instability on noise @ d=1.0 (A1s: fit-then-erode 0.85→0.29,
‖W1‖ 16→620, displacement orbit 6–13, consistent with a Neimark–Sacker picture [^1^]).
Potentially fixable via the conditioning family.
(c) The actual reason to keep PC: capabilities backprop lacks — associative memory, continual
learning, OOD — UNTESTED in this substrate.

**Scope note (per gate):** Gate-2.1 volume-scaling (C4) and the SGD/LR optimizer-sweep are
OUT of kill-scope — not dropped; they sit in the roadmap, not in this document.

## 1. Claim ledger

- **P1 — VIABILITY.** "PC can serve as the credit-assignment engine on learnable structure in
  this architecture." P1 has TWO legs, each with its own falsifier:
  - **P1-fit:** PC can FIT what backprop fits (train-level), at the same density, modulo the
    pre-registered fix budget.
  - **P1-hold:** PC can HOLD what it fits (no erosion/runaway), modulo the same budget.
  P1 alive ⇔ both legs alive. (v1 tested only the hold leg — the gate's Hole 1.)
- **P2 — MOAT.** "PC delivers at least one capability backprop lacks — associative memory,
  continual learning, or OOD detection — at matched compute, in this substrate."
- **P3 — META.** "Keep investing in the PC branch." P3 holds ⇔ P1 alive AND P2 alive (or
  decidable within the §5 budget).

## 2. The kill rule

**Kill the PC branch iff (P1 falsified on either leg) OR (P2 falsified) OR (§5 budget exhausted
without both alive).** A kill retires the CLAIMS, not the route: the volume/mask machinery,
grokking harness, clock/Fourier metrics, oracle lineage, and the measured instability dataset
(a publishable PC-pathology measurement in its own right) all survive and carry over to the
backprop-family engine. The exit is a claim funeral, not a project funeral.

Adjudication of the three candidate criteria as originally proposed (carried from v1):
1. *"Precision fails to restore settling even on real labels"* — NECESSARY but NOT SUFFICIENT.
   One failed fix ≠ unfixable; the falsifier is exhaustion of the pre-registered fix budget,
   read off the settling metric AND the absolute fit floor (§3 K2), not accuracy alone.
2. *"Volume-scaling shows PC merely tracks the backprop oracle"* — **REJECTED as a kill
   criterion.** Tracking the oracle on learnable structure IS the P1 viability bar, not a
   failure. Killing PC for tracking backprop on grokking is a category error (the moat question
   is P2's battery, not the grokking curve). Logged to prevent post-hoc mission-creep in the
   kill direction.
3. *"Instability fires on real/compositional structure"* — NECESSARY but NOT SUFFICIENT, and now
   instrumented at d=1.0 where PC actually fits (peak ≈1.0), so "fires" is measurable rather
   than vacuous (the gate's Hole 2). If a conditioning fix restores settling+fit on real
   structure, the instability becomes a design condition, not a kill.

## 3. P1 falsifiers — FIT (K0) and HOLD (K1∧K2)

**K0 — the FIT falsifier (new, closes Hole 1).** For each density d in the pre-registered
viability set {1.0, 0.5}: the best PC arm available within the fix budget must reach
**train-fit ≥ 0.8 × oracle-train-fit at the same density** (floors: d=1.0 → ≥0.80;
d=0.5 → ≥0.664 against oracle 0.830). **Current standing: the vanilla arm FAILS K0 at d=0.5
today** (train ≈0.108 < 0.664, Gate-2 sweep) and passes at d=1.0. P1-fit at d=0.5 is therefore
on PROBATION, not banked: cells C3d/C5d (the fix arms run @ d=0.5, real labels, wd=0) decide
whether the fit gap is fixable. **P1 dead on FIT ⇔ no fix cell, within budget, lifts d=0.5
train-fit to floor** (while d=1.0 stays at floor). If PC passes at d=1.0 but never at d≤0.5,
P1 survives only as the narrowed claim "viable at near-dense connectivity" — which collides with
the project's locality/edge-decay thesis; whether a dense-only engine serves the CLS design is
an owner decision at the next crossroads, pre-noted here per the pivot-narrows-claims rule.

**K1 — the HOLD-risk falsifier (moved to d=1.0, closes Hole 2).** Cell C2: A1 @ d=1.0, REAL
labels, wd=0, vanilla arm, 30k. The instability "fires on real structure" ⇔ peak train ≈1.0
followed by **erosion ≥0.1 from peak** AND **‖W1‖ late-slope positive** over the final 10k
epochs AND **displacement >1.0 sustained**. (At d=1.0 the peak is ≈1.0, so the erosion bar is
meaningful — at d=0.5 it was vacuous: 0.108→0.008. Hole 2 closed by construction.)
- ¬K1 → P1-hold alive; the instability is quarantined as a NOISE-STREAM problem (banked as a
  CLS data-moat design input: garbage/adversarial streams need a guard regardless of engine).
- K1 → P1-hold alive only if K2 passes.

**K2 — the FIX-EFFICACY falsifier (absolute floor added, closes Hole 3).** Cells C3 (precision
arm) and, only on C3 failure, C5 (ONE pre-chosen fallback: µPC-style reparametrization OR an
iPC-style per-step schedule — chosen at design time, not after seeing data). All @ d=1.0, real
labels, wd=0, 30k. A fix cell PASSES ⇔ all of:
  - mean per-seed displacement < 1.0 sustained over the final 50 log points (5k epochs), AND
  - ‖W1‖ late-slope flat (|Δ‖W1‖| per 1k epochs < 1% of ‖W1‖ over the final 10k), AND
  - **train ≥ 0.9 × oracle-train-fit at the same density** (d=1.0 floor: ≥0.90 — an ABSOLUTE
    bar, not 0.9×own-peak; a fix that settles PC at 0.097 now FAILS. Hole 3 closed.)
**P1 dead on HOLD ⇔ K1 AND (no fix cell passes K2 within budget).**
- K1 ∧ fixes pass → P1-hold alive-with-condition: "PC requires conditioning fixes." The fix
  becomes load-bearing architecture; P2's battery (§4) is then run WITH the fix in place.

**C1's adjudication (run first, unchanged):** backprop+shuffled @ d=1.0, wd=0, 30k. If backprop
ALSO erodes on noise, "PC-specific relaxation instability" dies and re-scopes to "generic
Adam-under-persistent-noise pathology" — the fix moves from PC-side to optimizer-side; P1/P2
survive. This kills one mechanism claim, nothing else.

## 4. P2 falsifier — the moat battery (the gate that does not exist yet)

P2's falsifier requires a pre-registered battery with baselines stated NOW (baseline-shopping
after the fact is the rescue-reframe of kill documents). All arms paired, matched compute,
10 seeds, this substrate:

- **M1 — associative memory.** Hetero-associative recall (partial or noisy pattern → full
  pattern). Baselines: modern Hopfield network AND a backprop autoencoder, matched parameter
  count. Bar: PC recall ≥ best baseline + 5 points, or > 2σ across seeds, at ≥1 capacity level.
- **M2 — continual learning.** Sequential task battery (split modular-arithmetic family or
  permuted-input analog at this scale). Baselines: backprop+EWC and backprop+replay, matched
  replay/regularization budget. Bar: mean forgetting < best baseline by ≥ 20% relative at
  matched compute.
- **M3 — OOD detection.** Held-out structure detection, energy/AUROC vs backprop softmax-baseline
  and BP-energy baseline. Bar: AUROC ≥ best baseline + 0.05.

**P2 dead ⇔ PC fails to clear the bar on ALL THREE in ONE run of the battery** (no re-rolls,
no margin re-negotiation post-hoc). Field priors for calibration only: associative memory is
PC's most-cited small-scale win [^6^][^7^][^8^]; continual has multiple positive small-scale
results [^9^][^10^][^11^]; OOD is the thinnest claim in the W&B-PCN literature — do not let M3
carry the moat alone.

**P2 dead kills the branch even if P1 lives.** A PC that works but does nothing special is a
cost-center with an inference-phase runtime overhead (structural — see §7, [^5^]) and no moat.
This project's reason for PC is the capabilities; "PC ≈ backprop + overhead" is a stop.

## 5. Budget kill (P3 time-box)

Pre-registered envelope for the entire decision: C1 (~1 GPU-min) + C2 (~25 GPU-min eager) +
C3 (~1 GPU-hr) + optional C5 (~1 GPU-hr) + K0's d=0.5 fix arms C3d/C5d (~2 GPU-hr) + battery
(≤ 24 GPU-hr) = **≤ 50 GPU-hours on the 3060** (adjust once, here, before C1 starts: ____).
If the envelope is exhausted with P1 or P2 undecided → kill by budget. Rationale: an engine
whose viability cannot be established inside ~two days of a 3060 will not carry a 1B-parameter
CLS program.

Run order is mandatory: C1 → C2 → C3 (→ C5 only on C3 fail; C3d/C5d run if the d=1.0 fix works,
to attempt K0's d=0.5 floor) → battery ONLY if P1 (both legs) alive.

**Out of scope (not dropped):** Gate-2.1 volume-scaling (C4 in the roadmap) and the SGD/LR
optimizer-sweep sit outside this document; §0(a) is relabeled accordingly per #6.

## 6. What does NOT kill (anti-traps, logged pre-emptively)

- PC tracking the oracle on grokking at any density/width (that is the viability bar, §2.2).
- The d=0.5 TEST-level cliff (wd-disentangled, optimizer-pending; excluded, §0a).
- **The d=0.5 TRAIN-level fit gap being waved into "the shared cliff"** — v1's own scope error,
  caught by the gate. The exclusion covers test-level only; the fit gap is scored by K0. Logged
  as the survival-side anti-trap: exclusions may not swallow PC-specific failures.
- C1 showing backprop also erodes on noise (re-scopes mechanism, §3).
- A capability edge on only ONE of M1/M2/M3 — that is P2 ALIVE (narrow but alive); the branch
  continues with the claim narrowed to that capability.
- Instability on noise streams if real structure holds (quarantined to data-moat design, §3).
- K0 failure at d≤0.5 with d=1.0 passing — claim narrows to "near-dense viable" (§3 K0), with
  the architectural consequence pre-noted; not a letter-kill of P1.

## 7. Field prior — fixable engine or dead-end substrate? (calibration, not evidence)

**Grounded prior: fixable-engine with an unproven-at-scale upside; the dead-end signal exists
but the live trajectory points the other way.**

- **The field's central named pathology is the one you measured.** µPC (Innocenti et al. 2025):
  standard PCNs are "inherently unscalable" because (i) the **inference landscape becomes
  increasingly ill-conditioned with model size AND training time** (measured via the
  activity-Hessian condition number) and (ii) forward activations vanish/explode with depth [^2^].
  Your non-settling + W1-runaway under persistent error is the 2-layer, noise-driven instance of
  pathology (i). The same paper mitigates both by reparametrization, training 128-layer networks
  on simple classification tasks — **a first step, explicitly not a scale proof** [^2^]; the
  companion theory reads PC learning as an approximate trust-region (second-order) method [^3^].
  Your fix family (precision / schedule / reparametrization) is literally the field's fix family.
- **The failure→fix cadence is the fixable-engine signature.** Deterioration named (2022) →
  stability fix (iPC, 2024) → scale fix (µPC, 2025) → benchmarking infrastructure (PCX, 2025)
  [^2^][^4^][^14^]. Dead-end fields re-state the same failure for a decade; this one keeps
  converting failures into fixed instances within 2–3 years each. Frieder & Lukasiewicz's
  divergence result is a REGIME boundary (small-weights converge), i.e. an engineering constraint
  to stay inside — not an impossibility theorem [^1^].
- **The fading evidence, honestly:** 25+ years and nothing past simple-classification scale;
  pre-µPC, adding layers made PC *worse* [^4^]; where PC works it reliably ≈ backprop on standard
  discriminative tasks, so raw performance is not the moat [^12^]; and Millidge's own assessment
  is that PC will not beat BP on GPUs — the inference phase is structurally slower [^5^]. (Song
  et al. 2024 is NOT an ≈BP result: its thesis is that prospective configuration is SUPERIOR to
  BP in online, low-data, and continual regimes — which belongs to the moat column, not the
  parity column [^13^].) The residual dead-end probability lives at scale: nobody has shown the
  fixes hold past toy scale.
- **Net for this project:** your CLS does not need PC to beat backprop on GPUs — it needs
  associative memory, continual learning, and OOD-robustness on consumer hardware, which is
  exactly the quadrant where PC's small-scale evidence is real [^6^][^7^][^8^][^9^][^10^][^11^][^13^]
  and backprop is weakest. The bet is aligned with the field's actual edge. That is why the kill
  weight sits on the P2 battery, not on the performance question — and why the fix budget is
  spent on the conditioning family the field has repeatedly vindicated.

## 8. Sign-off

- [ ] v2 patches re-validated by the 3-agent gate — date:
- [ ] C1–C5 cells + battery spec frozen (this document) — date:
- [ ] Budget number filled (§5) — date:
- [ ] C5 fallback lever chosen (precision-fail branch) — date:
- [ ] Battery baselines + margins frozen (§4) — date:
- [ ] Agent A (Kimi) reviewed; Agent B reviewed; owner approved.

---

[^1^]: Frieder, S. & Lukasiewicz, T. (2022). (Non-)Convergence Results for Predictive Coding Networks. ICML, PMLR 162:6793–6810. https://proceedings.mlr.press/v162/frieder22a.html
[^2^]: Innocenti, F., Achour, E.M., Buckley, C.L. (2025). µPC: Scaling Predictive Coding to 100+ Layer Networks. NeurIPS 2025 / arXiv:2505.13124. https://arxiv.org/abs/2505.13124 — NOTE: this is the depth/reparametrization paper; do NOT conflate with the separate Innocenti-2026 capacity-equivalence paper (arXiv:2602.07697) cited in pc_research.md.
[^3^]: Innocenti, F. (2025). Towards Scaling Deep Neural Networks with Predictive Coding: Theory and Practice. arXiv:2510.23323. https://arxiv.org/abs/2510.23323
[^4^]: Pinchetti, L., Qi, C., Lokshyn, O., Emde, C., et al. (2025). Benchmarking Predictive Coding Networks — Made Simple. ICLR 2025; see also VERSES research blog (2025). https://www.verses.ai/research-blog/benchmarking-predictive-coding-networks-made-simple
[^5^]: Millidge, B. (2023). Thoughts on the Future of Predictive Coding. https://www.beren.io/2023-03-30-Thoughts-on-future-of-PC/
[^6^]: Salvatori, T., Song, Y., Hong, Y., Sha, L., et al. (2021). Associative Memories via Predictive Coding. NeurIPS 2021. https://proceedings.neurips.cc/paper/2021/hash/1fb36c4ccf88f7e67ead155496f02338-Abstract.html
[^7^]: Tang, M., Salvatori, T., Millidge, B., Song, Y., et al. (2023). Recurrent Predictive Coding Models for Associative Memory Employing Covariance Learning. PLoS Computational Biology. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010719
[^8^]: Yoo, J. & Wood, F. (2022). BayesPCN: A Continually Learnable Predictive Coding Associative Memory. NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/c13d5a10028586fdc15ee7da97b7563f-Abstract-Conference.html
[^9^]: Ororbia, A., Mali, A., Giles, C.L., et al. (2022). Lifelong Neural Predictive Coding: Learning Cumulatively Online Without Forgetting. NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/26f5a4e26c13d1e0a47f46790c999361-Abstract-Conference.html
[^10^]: Lee, J., Jo, J., Lee, B., Lee, J.H., Yoon, S. (2022). Brain-Inspired Predictive Coding Improves the Performance of Machine Challenging Tasks. Frontiers in Computational Neuroscience. https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.1062678/full
[^11^]: Annabi, L., Pitti, A., Quoy, M. (2022). Continual Sequence Modeling with Predictive Coding. Frontiers in Neurorobotics. https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.845955/full
[^12^]: Millidge, B., Tschantz, A., Buckley, C.L. (2022). Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs. Neural Computation 34(6):1329. https://direct.mit.edu/neco/article-abstract/34/6/1329/110646
[^13^]: Song, Y., Millidge, B., Salvatori, T., Lukasiewicz, T., et al. (2024). Inferring Neural Activity Before Plasticity as a Foundation for Learning Beyond Backpropagation. Nature Neuroscience. https://www.nature.com/articles/s41593-023-01514-1 — cited here for its actual thesis: prospective configuration OUTPERFORMS BP in online / low-data / continual regimes; the ≈BP claim in this document rests on [^12^].
[^14^]: Salvatori, T., Song, Y., Yordanov, Y., et al. (2024). A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks (iPC). ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/554414e570a85eb3118e988c5d77986f-Abstract-Conference.html
