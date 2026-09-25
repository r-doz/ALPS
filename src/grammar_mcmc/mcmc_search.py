"""Metropolis-Hastings search over DeGAS program *structures* -- the no-LLM "blind search"
control baseline. Structural mutations (sampler.py) are proposed and accepted/rejected on
fitted train NLL; DeGAS's own gradient optimizer (the same pipeline as
refinestat_degas/refine_loop.py::fit_and_diagnose, minus the reliability-diagnostics gating
-- this baseline reports raw fit quality only, by design) refits continuous parameters after
every proposed mutation, so the Markov chain only has to search discrete structural choices.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

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

from refinestat_degas.checkers.degas_checker import DegasChecker  # noqa: E402
from refinestat_degas.text_utils import fix_uniform_trailing_literal  # noqa: E402

from .program_spec import ProgramSpec  # noqa: E402
from .sampler import propose_mutation  # noqa: E402


@dataclass
class MCMCSearchConfig:
    n_steps: int = 300
    n_opt_steps: int = 50
    lr: float = 0.01
    temperature: float = 1.0
    p_conditional_init: float = 0.3
    max_init_tries: int = 50


def _fit(program_text: str, train_data, held_out_data, var_names: list[str], cfg: MCMCSearchConfig) -> dict:
    """DeGAS compile/optimize/likelihood pipeline, mirroring
    refinestat_degas/refine_loop.py::fit_and_diagnose but without the forward-sampling
    reliability checks -- any failure is treated as an unfit-able (rejected) proposal."""
    try:
        rewritten, params = extract_params(program_text)
        compiled = compile2SOGA_text(fix_uniform_trailing_literal(rewritten))
        cfg_obj = produce_cfg_text(compiled)
        smooth_cfg(cfg_obj)
        params_dict = initialize_params(params)

        loss_fn = lambda output_dist: -compute_likelihood(output_dist, var_names, train_data)
        optimize(cfg_obj, params_dict, loss_fn, n_steps=cfg.n_opt_steps, lr=cfg.lr, print_progress=False)

        final_output_dist = start_SOGA(cfg_obj, params_dict)
        train_nll = (-compute_likelihood(final_output_dist, var_names, train_data)).item()
        held_out_nll = (-compute_likelihood(final_output_dist, var_names, held_out_data)).item()
        if not math.isfinite(train_nll):
            raise ValueError(f"non-finite train NLL: {train_nll}")

        return {
            "ok": True,
            "rewritten": rewritten,
            "optimized_params": {k: v.detach().tolist() for k, v in params_dict.items()},
            "train_nll": train_nll,
            "held_out_nll": held_out_nll,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 -- DeGAS can raise many exception types
        return {
            "ok": False,
            "rewritten": None,
            "optimized_params": None,
            "train_nll": float("inf"),
            "held_out_nll": float("inf"),
            "error": f"{type(e).__name__}: {e}",
        }


def run_grammar_mcmc(
    benchmark: str,
    train_data,
    held_out_data,
    var_names: list[str],
    cfg: MCMCSearchConfig,
    seed: int = 0,
    log=print,
) -> dict:
    random.seed(seed)

    current_spec = current_text = current_fit = None
    for _ in range(cfg.max_init_tries):
        spec = ProgramSpec.random(var_names, p_conditional=cfg.p_conditional_init)
        text = spec.render()
        checker = DegasChecker(text, {}, var_names)
        if not checker.check() or not checker.finished():
            continue
        fit = _fit(text, train_data, held_out_data, var_names, cfg)
        if fit["ok"]:
            current_spec, current_text, current_fit = spec, text, fit
            break

    if current_spec is None:
        log(f"[{benchmark} seed={seed}] failed to find a fittable initial program in {cfg.max_init_tries} tries")
        return {"best": None, "all_valid": [], "best_fitness_per_attempt": [], "n_attempts": 0}

    best = {**current_fit, "program": current_text}
    best_fitness_per_attempt = [best["held_out_nll"]]
    n_accepted = 0
    n_valid_proposals = 0

    for step in range(cfg.n_steps):
        proposed_spec = propose_mutation(current_spec, var_names)
        proposed_text = proposed_spec.render()
        checker = DegasChecker(proposed_text, {}, var_names)
        if not checker.check() or not checker.finished():
            best_fitness_per_attempt.append(best["held_out_nll"])
            continue  # reject invalid proposals without spending a gradient fit

        n_valid_proposals += 1
        fit = _fit(proposed_text, train_data, held_out_data, var_names, cfg)
        if fit["ok"]:
            delta = fit["train_nll"] - current_fit["train_nll"]
            accept_prob = 1.0 if delta <= 0 else math.exp(-delta / cfg.temperature)
            if random.random() < accept_prob:
                current_spec, current_text, current_fit = proposed_spec, proposed_text, fit
                n_accepted += 1
                if fit["held_out_nll"] < best["held_out_nll"]:
                    best = {**fit, "program": proposed_text}

        best_fitness_per_attempt.append(best["held_out_nll"])
        if step % 25 == 0 or step == cfg.n_steps - 1:
            log(
                f"[{benchmark} seed={seed} step={step}] current_train_nll={current_fit['train_nll']:.4f} "
                f"best_held_out_nll={best['held_out_nll']:.4f} "
                f"accept_rate={n_accepted / max(1, n_valid_proposals):.2f} "
                f"valid_proposal_rate={n_valid_proposals / (step + 1):.2f}"
            )

    log(f"=== [{benchmark} seed={seed}] done: {n_accepted}/{cfg.n_steps} accepted, best_held_out_nll={best['held_out_nll']:.4f} ===")
    return {
        "best": best,
        "all_valid": [best],
        "best_fitness_per_attempt": best_fitness_per_attempt,
        "n_attempts": cfg.n_steps,
    }
