# Grokking Modular Arithmetic under a Local, Backprop-Free Learning Rule

**The multiplicative-character basis as the load-bearing ingredient.**

This repository contains the paper, the engine, and every artifact cited in the
manuscript — the complete evidence trail for:

> **To our knowledge, the first grokking of modular arithmetic under a purely
> local, backprop-free learning rule.**

## The claim, in one paragraph

> **A deep ($L{=}6$) sparse predictive-coding cortex trained with a local
contrastive Equilibrium Propagation rule (backprop-free, no weight transport)
groks modular multiplication (mod 53) persistently: **10/10 seeds, mean final 0.989**
(3000 steps). It does not with the additive-Fourier basis, and we prove why: a
rigorous rank gap (Theorem 1: rank = (p+1)/2). The multiplicative-character basis
linearizes multiplication; the C+D stabilization schedule makes grokking persistent;
the T-cliff is a characteristic feature of the settling-based EP implementation tested
here.

## Repository layout

```
paper/            LaTeX source (main.tex), compiled PDF, figures F1-F6
scripts/          Engine (ablation_cortex_v14_1), figure generator, run scripts
outputs/          All artifacts cited in the paper, each with its commit SHA
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

## Commit SHAs

The paper's claims table cites commit SHAs (e.g. 87c7250, 38c6702) that refer
to the private development history of this project. This public repository is
a single squashed commit, so those SHAs do not resolve here; each artifact's
content is present in `outputs/` and is reproducible from the scripts in
`scripts/`.

## Honesty notes

- The BP control arm is **budget-matched, not an impossibility claim**:
  backprop-trained transformers grok modular multiplication in the literature
  (Power et al. 2022; Nguyen 2026), and a dense MLP control groks at ~3900
  steps in a 20k run. On the sparse L=6 substrate, BP has not grokked by 10k
  steps (0.70, 0/2) — consistent with a delay, magnitude not quantified.
- Sparsity (10% firing, ~9% structural weight density) is a *representation*
  lever, not a GPU-compute lever: measured, cuSPARSE is 14-34x slower than
  dense cuBLAS at this scale. The crossbar story is structural/energy, not
  current-GPU.
- The alphabet is **given, not acquired**; co-grokking is accuracy-only; the
  replay result is a single configuration; C11 (Oja mirror) is partial.

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
