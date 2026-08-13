# Grokking Modular Arithmetic under a Local, Backprop-Free Learning Rule

**The multiplicative-character basis as the load-bearing ingredient.**

This repository contains the paper, the engine, and every artifact cited in the
manuscript — the complete evidence trail for:

> **To our knowledge, the first grokking of modular arithmetic under a locally
> plastic, backprop-free learning rule.**

## The claim, in one paragraph

> **A deep ($L{=}6$) sparse predictive-coding cortex trained with a local
> contrastive Equilibrium Propagation rule (backprop-free, no weight transport)
> groks modular multiplication (mod 53) persistently: 10/10 seeds, mean final 0.989**
> (3000 steps). It does not with the additive-Fourier basis, and we prove why: a
> rigorous rank gap (Theorem 1: rank = (p+1)/2). The multiplicative-character basis
> linearizes multiplication; the C+D stabilization schedule makes grokking persistent;
> the T-cliff is a characteristic feature of the settling-based EP implementation tested
> here.

## What's new since the paper (2026-08-13)

The paper reports grokking given a hand-designed alphabet. Since then, the alphabet
itself has been **learned, not given** — under the same local rules:

- **The EC self-organizes the alphabet.** A streaming Oja entorhinal-cortex module
  converges to the *same* multiplicative-character basis the paper hand-designed:
  batch-limit alignment 1.0000, true streaming alignment 0.972 (stationary) /
  0.958 (domain shift). Artifacts: `outputs/phi_matters_control_results.json`,
  `outputs/living_ec_exp0_real_results.json`.
- **The EC is "living," not frozen.** Under a continuously rotating eigenspace, the
  streaming EC tracks (min drift alignment 0.918) where a frozen basis is stuck
  (0.327). Artifact: `outputs/living_ec_exp5_results.json`.
- **The alphabet is load-bearing at depth.** Control sweep on the real L=6 engine:
  the correct character basis groks 10/10 (window 0.9435); shuffled, random, and
  one-hot codes all fail at chance (0/10, Δ = +0.935) — the toy "shuffled ≈ real"
  result was a shallow-1-layer artifact. Artifact: `outputs/phi_matters_control_results.json`.
- **Honest negatives are banked as artifacts, not hidden:** a DG expansion recoding
  that decorrelates as theory predicts yet *hurts* grammar learning (0.75 → 0.48,
  `outputs/dg_cfg_results.json`), and a scaling wall at prime p=97 (0/10 grok,
  `outputs/p97wall_*`).
- **A continual-learning paper is in progress** — the hippocampal loop (EC alphabet
  acquisition + DG pattern separation, both local) is the next milestone. See
  `docs/PAPER2_OUTLINE_v0.1.md`.

All results above are reproducible from `outputs/` with the same commit-SHA convention
as the paper's claims table.

## Repository layout

```
paper/            LaTeX source (main.tex), compiled PDF, figures F1-F6
scripts/          Engine (ablation_cortex_v14_1), figure generator, run scripts
outputs/          All artifacts cited in the paper, each with its commit SHA
docs/             Follow-up analyses (Paper 2 outline, audits)
```

## Reproducing

1. `pip install -r requirements.txt` (torch, numpy, matplotlib)
2. `python scripts/run_basis_swap_v13.py --task mult-stab --seeds 0-9 --steps 3000 --train-fractions 0.8`
   — the headline C2 arm (writes `basis_swap_v13_mult-stab_frac80_L6_N1536_T10.json`,
   the artifact cited in the paper)
3. `python scripts/make_paper_figures.py` — regenerates figures F1-F6 from the
   committed artifacts (F4's single-neuron value is backed by
   `outputs/f4_expressivity_audit.json`, which reproduces RMSE 0.7004, R² 0.036
   from the kernel directly; the layer-0 empirical fit 0.019 is documented in
   the same file with provenance)

Every number in the paper traces to a committed artifact in `outputs/`; the
claims table (Appendix A of the paper) footnotes each claim to its artifact
and commit SHA. `provenance.json` lists the SHA256 hash of every artifact in
this repository, so a reviewer can verify that a downloaded file is
byte-identical to the one used to generate the paper's figures and tables —
independently of the private-history commit SHAs cited in the paper.

## Honesty notes

- The BP control arm is **budget-matched, not an impossibility claim**:
  backprop-trained transformers grok modular multiplication in the literature
  (Power et al. 2022; Nguyen 2026), and a dense MLP control groks at ~3900
  steps in a 20k run. On the sparse L=6 substrate, BP has not grokked by 10k
  steps (0.70, 0/2) — consistent with a delay, magnitude not quantified.
- Sparsity (10% firing, ~9% structural weight density) is a *representation*
  lever, not a GPU-compute lever: measured, cuSPARSE is 14-34x slower than
  dense cuBLAS at this scale. The crossbar story is structural/energy, not
  current-GPU. A dead-weight probe (58.7% prunable) showed the prunable set is
  a training artifact (not stable across seeds), so savings are inference-time
  zero-skipping, not pre-training routing.
- **Alphabet status, precisely:** the *paper* reports a hand-designed alphabet
  (that was the correct scope for its claim). Follow-up work demonstrates the
  alphabet is **learned, not given** under local rules (see "What's new").
  Co-grokking is accuracy-only; the replay result is a single configuration;
  C11 (Oja mirror) is partial.

## Citation

```bibtex
@misc{bhular2026grokking,
  title={Grokking Modular Arithmetic under a Local, Backprop-Free Learning Rule},
  author={Bhular, Omar},
  year={2026},
  note={arXiv preprint}
}
```

## License

MIT (see LICENSE). Paper text © the authors; third-party PDFs © their respective
authors, included for review convenience only.
