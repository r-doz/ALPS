"""Structural-complexity proxy for a compiled DeGAS program.

DeGAS compiles every program to one closed-form joint Gaussian mixture; every additional
`if`/`else` branch and every extra component in an explicit `gm([pi...], [mu...], [sigma...])`
combinatorially multiplies the number of components in that final mixture. Component count is
therefore a cheap, already-available proxy for how structurally complex a program is --
requiring no new parsing, just reading GaussianMix.n_comp() off the already-compiled
distribution.

Motivation: investigating why ALPS's mutation search sometimes converges on candidates that
win on NLL but lose badly on GW2 (Gaussian-Wasserstein distance to the real program) relative
to a simpler candidate. On mog1, the real program and a plain single-component-per-variable
candidate both compile to n_comp=1; a full-ALPS mutated candidate with similar/better NLL but
much worse GW2 compiled to n_comp=12 (nested if/else) or n_comp=27 (three independent
3-component gm's) -- i.e. mutation-driven NLL-only selection found candidates an order of
magnitude more complex than the true structure warrants, with nothing in the selection
criterion penalizing that. This module makes that complexity gap directly measurable, and
provides pick_best_tradeoff() as a selection rule that accounts for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEGAS_SRC = Path(__file__).resolve().parents[2] / "DeGAS" / "src"
if str(_DEGAS_SRC) not in sys.path:
    sys.path.insert(0, str(_DEGAS_SRC))

from optimization import compile2SOGA_text, produce_cfg_text, smooth_cfg, start_SOGA  # noqa: E402


def n_components(dist) -> int:
    """Component count of an already-compiled distribution's joint Gaussian mixture."""
    return dist.gm.n_comp()


def program_complexity(program_text: str) -> int:
    """Compiles a fully-literal DeGAS program (no unresolved `_paramN` placeholders -- see
    helpers/param_extractor.py; substitute fitted values in first, e.g. via
    aggregate_results_table.py's substitute_params) and returns its joint mixture's
    component count.

    Raises whatever compile2SOGA_text/produce_cfg_text/start_SOGA raise on a malformed or
    numerically-degenerate program -- callers should catch and treat that the same way they'd
    treat any other compile failure (no complexity score available), not silently return a
    sentinel value that could be mistaken for a real (e.g. 0 or -1) component count.
    """
    compiled = compile2SOGA_text(program_text)
    cfg = produce_cfg_text(compiled)
    smooth_cfg(cfg)
    dist = start_SOGA(cfg)
    return n_components(dist)


def pick_best_tradeoff(
    candidates: list[dict],
    nll_key: str = "NLL",
    complexity_key: str = "n_comp",
    margin_floor: float = 0.05,
    margin_frac: float = 0.01,
) -> dict:
    """Selects the best NLL/complexity tradeoff among a population of already-scored
    candidates (each a dict with at least `nll_key` and `complexity_key`).

    Rule: find the population's own best NLL, then among every candidate within a small
    margin of it, return the SIMPLEST one (lowest complexity). The margin is
    `max(margin_floor, margin_frac * abs(best_nll))` -- deliberately shaped after
    RefineStat's own held_out_nll_margin_floor/held_out_nll_margin_frac tolerance
    (refine_loop.py), rather than an arbitrary linear penalty (NLL + lambda*n_comp): a fixed
    per-component "nats" exchange rate would need separate tuning for every benchmark's very
    different true complexity (mog1's real program has 1 component, csi's has 16), whereas a
    margin scaled to the NLL's own magnitude does not.

    This never trades away real fit quality -- it only breaks near-ties (candidates whose NLL
    difference is within noise) in favor of simplicity. Concretely, it's what would have fixed
    mog1 run 20260830_165747: four candidates tied at NLL 6.7072-6.7076 (n_comp=8) plus one
    outlier at NLL=6.7091 (n_comp=27, 0.0015 "better" NLL, but worse on every structural
    measure) -- pure-NLL selection picked the outlier; this rule picks one of the n_comp=8 ones.

    The defaults (margin_floor=0.05 nats, margin_frac=1% of |best_nll|) are deliberately
    tighter than RefineStat's own (1.0 / 50%), which govern a different question (how much of
    a train/held-out generalization gap to tolerate) -- here the candidates being compared have
    already survived the same selection funnel and are typically already close, so a looser
    margin risks trading away real NLL gains, not just noise. Not yet validated on a benchmark
    that genuinely needs high complexity (e.g. csi) -- check there before trusting these
    defaults broadly.
    """
    best_nll = min(c[nll_key] for c in candidates)
    margin = max(margin_floor, margin_frac * abs(best_nll))
    eligible = [c for c in candidates if c[nll_key] <= best_nll + margin]
    return min(eligible, key=lambda c: c[complexity_key])
