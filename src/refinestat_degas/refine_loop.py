"""D||P||L iterative refinement loop for DeGAS, adapted from RefineStat's Definition 3.2.

INPUT: Rmax (max refinements), alpha (max likelihood resamples), beta (target valid
programs), K (min passing diagnostics)
OUTPUT: best valid program by held-out NLL (lower is better; DeGAS's exact analytic
likelihood makes this a strictly cleaner analog of ELPD-LOO than the original's PSIS
approximation -- see diagnostics.py's module docstring).

    r, ell, V = 0, 0, []
    while r < Rmax and len(V) < beta:
        generate a program statement-by-statement (checked, backtrack on failure)
        if generation didn't finish within budget: r += 1; restart
        fit params by gradient descent against train_data (DeGAS's own optimize())
        score, diags = forward-sample reliability + held-out NLL check
        if score >= K: V.append(candidate)
        elif last toplevel statement assigns an observed (var_names) variable and
             ell < alpha: backward 1 toplevel unit ("resample likelihood"); ell += 1
        else: backward further ("resample prior"); r += 1
    return argmin(V, key=held_out_nll)

"resample likelihood" vs "resample prior": RefineStat keys off the literal `observed=`
keyword in PyMC calls. DeGAS has no such marker, so an assignment to a benchmark var_names
variable is treated as likelihood-like (an observed/output variable); an assignment to any
other (latent/temp) variable is prior-like. Note that in these 5 benchmarks every program
variable is typically also a dataset var_names variable (no inherent latent variables), so
in practice most backtracking classifies as "resample likelihood" -- this is a documented,
expected consequence of the benchmarks' structure, not a bug.
"""

from __future__ import annotations

import functools
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "DeGAS" / "src", _REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from optimization import (  # noqa: E402
    compile2SOGA_text,
    initialize_params,
    optimize,
    produce_cfg_text,
    smooth_cfg,
    start_SOGA,
)
from PROGRAMS.likelihood import compute_likelihood  # noqa: E402
from helpers.param_extractor import extract_params  # noqa: E402
from helpers.generation_prompt import DEGAS_GRAMMAR  # noqa: E402

from . import diagnostics  # noqa: E402
from .checkers.degas_checker import DegasChecker, _get_parser  # noqa: E402
from .text_utils import fix_uniform_trailing_literal  # noqa: E402

SYSTEM_PROMPT_RAW = (
    "You are an expert in probabilistic programming and Bayesian modelling.\n"
    "You write programs exclusively in DeGAS, following the reference below.\n\n"
    + DEGAS_GRAMMAR
    + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Numbers: exactly 2 decimal places, range [-100.00, 100.00]
• gm weights must sum to exactly 1.00
• At most one * per assignment line
• Coefficient always before variable in products
• No division
• Do NOT put underscores or other characters in variable names, use the names of the dataset variables exactly as they appear in the stats
• Reply with DeGAS statements only -- no prose, no markdown fences, no JSON
"""
)


@dataclass
class RefineConfig:
    unit_name: str = "toplevel"
    max_units: int = 20
    Rmax: int = 100  # RefineStat's own paper value (Appendix E), shared across both backends
    alpha: int = 2   # RefineStat's own value, shared across both backends
    beta: int = 4    # RefineStat's own value, shared across both backends
    K: int = 4  # cutoff; meaning depends on inference backend (see main.py): 4 of 5 checks
    # (r_hat, ess_bulk, ess_tail, no_divergences, held_out_ok) for --inference gradient, or
    # RefineStat's own zeta=5 of 7 (mcmc_diagnostics.py's checks) for --inference mcmc
    n_opt_steps: int = 100
    lr: float = 0.01
    n_samples: int = 500
    n_chains: int = 4
    held_out_nll_margin_floor: float = 1.0
    held_out_nll_margin_frac: float = 0.5
    max_wall_seconds: float | None = None  # give up on this seed (return best found so far,
    # possibly None) once the attempt loop has run this long. Checked between attempts, not
    # during one -- an in-flight fit_fn call (e.g. a slow NUTS chain) is allowed to finish
    # rather than being interrupted mid-computation.


def _stats(data, var_names: list[str]) -> dict:
    arr = np.array(data)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return {
        "var_names": var_names,
        "n": len(data),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "skewness": (np.mean((arr - mean) ** 3, axis=0) / (std**3)).tolist(),
        "kurtosis": (np.mean((arr - mean) ** 4, axis=0) / (std**4)).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
    }


def build_prompt_messages(stats: dict, benchmark: str) -> list[dict]:
    stats_lines = "\n".join(f"  {k}: {v}" for k, v in stats.items())
    user = f"""Data summary for the "{benchmark}" dataset:
{stats_lines}

Write ONE DeGAS program that models this data-generating process, using every variable
listed in var_names. Follow the DeGAS Language Reference exactly. Output only the program
statements -- no prose, no markdown, no JSON wrapper."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_RAW},
        {"role": "user", "content": user},
    ]


def _assigned_vars(stmt_tree) -> set[str]:
    inner = stmt_tree.children[0]
    if inner.data == "assignment":
        return {str(inner.children[0])}
    if inner.data == "conditional":
        _, ifblock, elseblock = inner.children
        out: set[str] = set()
        for s in ifblock.children:
            out |= _assigned_vars(s)
        for s in elseblock.children:
            out |= _assigned_vars(s)
        return out
    return set()


def _last_toplevel_kind(program_text: str, var_names: set[str]) -> str:
    """Returns 'likelihood' if the last completed toplevel statement assigns a benchmark
    var_names variable, else 'prior'."""
    if not program_text.strip():
        return "prior"
    tree = _get_parser().parse(program_text)
    if not tree.children:
        return "prior"
    last_stmt = tree.children[-1].children[0]
    assigned = _assigned_vars(last_stmt)
    return "likelihood" if assigned & var_names else "prior"


def _fill_remaining(itergen, checker: DegasChecker, unit_name: str, max_units: int) -> str:
    for _ in range(max_units):
        if checker.finished():
            break
        itergen.forward(units=[unit_name], num=1)
        checker.code = itergen.generated_text
        if not checker.check():
            itergen.backward(unit_name, num=1)
    return itergen.generated_text


def generate_candidate(
    itergen, var_names: list[str], messages: list[dict], unit_name: str, max_units: int
) -> tuple[str, DegasChecker]:
    itergen.start(messages)
    checker = DegasChecker("", {}, var_names)
    program_text = _fill_remaining(itergen, checker, unit_name, max_units)
    return program_text, checker


_FAILED_CHECKS = {
    "r_hat": False,
    "ess_bulk": False,
    "ess_tail": False,
    "no_divergences": False,
    "held_out_ok": False,
}


def fit_and_diagnose(
    program_text: str, train_data, held_out_data, var_names: list[str], cfg: RefineConfig
) -> dict:
    """DeGAS's compile/optimize/likelihood pipeline can raise on a structurally-valid-but-
    numerically-degenerate program (e.g. gradient descent driving a component's covariance
    singular/NaN) -- exactly the class of failure RefineStat's run_pymc_code() guards
    against for pm.sample(). Any such failure here is treated as a failed candidate
    (reliability_score=0) for refine_program() to backtrack from, not a fatal crash."""
    try:
        rewritten, params = extract_params(program_text)
        compiled = compile2SOGA_text(fix_uniform_trailing_literal(rewritten))
        cfg_obj = produce_cfg_text(compiled)
        smooth_cfg(cfg_obj)
        params_dict = initialize_params(params)

        loss_fn = lambda output_dist: -compute_likelihood(output_dist, var_names, train_data)
        loss_list, elapsed_time, n_iter = optimize(
            cfg_obj, params_dict, loss_fn, n_steps=cfg.n_opt_steps, lr=cfg.lr, print_progress=False
        )

        final_output_dist = start_SOGA(cfg_obj, params_dict)
        train_nll = (-compute_likelihood(final_output_dist, var_names, train_data)).item()
        held_out_nll = (-compute_likelihood(final_output_dist, var_names, held_out_data)).item()
        if not (math.isfinite(train_nll)):
            raise ValueError(f"non-finite train NLL: {train_nll}")

        optimized_params = {k: v.detach() for k, v in params_dict.items()}
        samples = diagnostics.forward_sample(
            rewritten, optimized_params, n_samples=cfg.n_samples, n_chains=cfg.n_chains
        )
        reliability = diagnostics.reliability_from_samples(samples, var_names)

        margin = max(cfg.held_out_nll_margin_floor, cfg.held_out_nll_margin_frac * abs(train_nll))
        held_out_ok = math.isfinite(held_out_nll) and held_out_nll <= train_nll + margin

        checks = dict(reliability["checks"])
        checks["held_out_ok"] = held_out_ok
        score = sum(checks.values())

        return {
            "rewritten": rewritten,
            "params": params,
            "optimized_params": {k: v.tolist() for k, v in optimized_params.items()},
            "initial_loss": loss_list[0] if loss_list else None,
            "final_loss": loss_list[-1] if loss_list else None,
            "train_nll": train_nll,
            "held_out_nll": held_out_nll,
            "reliability": reliability,
            "checks": checks,
            "reliability_score": score,
            "opt_time": elapsed_time,
            "opt_iterations": n_iter,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 -- DeGAS can raise many exception types here
        return {
            "rewritten": None,
            "params": None,
            "optimized_params": None,
            "initial_loss": None,
            "final_loss": None,
            "train_nll": float("inf"),
            "held_out_nll": float("inf"),
            "reliability": None,
            "checks": dict(_FAILED_CHECKS),
            "reliability_score": 0,
            "opt_time": None,
            "opt_iterations": None,
            "error": f"{type(e).__name__}: {e}",
        }


def refine_program(
    itergen,
    benchmark: str,
    train_data,
    held_out_data,
    var_names: list[str],
    cfg: RefineConfig,
    fit_fn=None,
    log=print,
) -> Optional[dict]:
    """fit_fn(program_text, train_data, held_out_data, var_names) -> result dict with at
    least `reliability_score`, `held_out_nll`, `checks`. Defaults to
    `functools.partial(fit_and_diagnose, cfg=cfg)` (DeGAS's own gradient-optimized point
    estimate). Pass `functools.partial(mcmc_diagnostics.fit_and_diagnose_mcmc,
    cfg=mcmc_cfg)` instead for genuine NUTS/MCMC posterior inference -- the D||P||L
    generation/checking/backtracking loop below is identical either way; only how a
    completed candidate gets fit and scored differs (see mcmc_diagnostics.py's module
    docstring for why DeGAS needs a second inference backend for a faithful RefineStat
    comparison)."""
    if fit_fn is None:
        fit_fn = functools.partial(fit_and_diagnose, cfg=cfg)

    var_names_set = set(var_names)
    stats = _stats(train_data, var_names)
    messages = build_prompt_messages(stats, benchmark)

    r = ell = 0
    prior_backtrack_depth = 1
    valid: list[dict] = []
    best_fitness_per_attempt: list[float] = []

    program_text, checker = generate_candidate(itergen, var_names, messages, cfg.unit_name, cfg.max_units)

    attempt = 0
    loop_start = time.time()
    while r < cfg.Rmax and len(valid) < cfg.beta:
        if cfg.max_wall_seconds is not None and time.time() - loop_start >= cfg.max_wall_seconds:
            log(
                f"[attempt={attempt}] giving up: {time.time() - loop_start:.0f}s >= "
                f"max_wall_seconds={cfg.max_wall_seconds:.0f}s ({len(valid)}/{cfg.beta} valid so far)"
            )
            break
        attempt += 1
        if not checker.finished():
            log(f"[attempt={attempt} r={r}] generation incomplete within budget; restarting")
            r += 1
            program_text, checker = generate_candidate(itergen, var_names, messages, cfg.unit_name, cfg.max_units)
            continue

        result = fit_fn(program_text, train_data, held_out_data, var_names)
        log(
            f"[attempt={attempt} r={r} ell={ell}] "
            f"score={result['reliability_score']}/{len(result['checks'])} (cutoff K={cfg.K}) "
            f"held_out_nll={result['held_out_nll']:.4f} checks={result['checks']}"
        )
        best_fitness_per_attempt.append(result["held_out_nll"])

        if result["reliability_score"] >= cfg.K:
            valid.append({**result, "program": program_text})
            if len(valid) >= cfg.beta:
                break
            program_text, checker = generate_candidate(itergen, var_names, messages, cfg.unit_name, cfg.max_units)
            continue

        kind = _last_toplevel_kind(program_text, var_names_set)
        if kind == "likelihood" and ell < cfg.alpha:
            itergen.backward(cfg.unit_name, num=1)
            ell += 1
            log(f"[attempt={attempt}] resample likelihood (ell={ell}/{cfg.alpha})")
        else:
            itergen.backward(cfg.unit_name, num=prior_backtrack_depth)
            prior_backtrack_depth += 1
            r += 1
            log(f"[attempt={attempt}] resample prior (r={r}/{cfg.Rmax})")

        checker = DegasChecker(itergen.generated_text, {}, var_names)
        checker.check()
        program_text = _fill_remaining(itergen, checker, cfg.unit_name, cfg.max_units)

    best = min(valid, key=lambda c: c["held_out_nll"]) if valid else None
    return {
        "best": best,
        "all_valid": valid,
        "best_fitness_per_attempt": best_fitness_per_attempt,
        "n_attempts": attempt,
    }
