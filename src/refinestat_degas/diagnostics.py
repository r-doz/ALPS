"""Forward-sampling diagnostics for optimized DeGAS candidate programs.

Rather than re-implementing DeGAS's own ANTLR-based CFG/expression interpreter
(DeGAS/src/libSOGA*.py, which walks private ANTLR parse-tree objects), this walks the same
Lark parse tree (grammar/degas.lark) already used by checkers/degas_checker.py, after
substituting the gradient-optimized parameter values back into the gm(...)/uniform(...)
literals. One grammar/tree, reused for checking, optimization staging and diagnostics.

Translates each `gm(...)`/`uniform(...)` into torch.distributions and does ancestral
(forward) Monte Carlo simulation of the program. This stands in for RefineStat's MCMC-based
diagnostics (R-hat/ESS/divergences/BFMI/Pareto-k/ELPD-LOO), none of which have a literal
analog under DeGAS's non-MCMC, gradient-optimized, closed-form-likelihood execution model:

- r_hat/ess_bulk/ess_tail: computed by ArviZ over `n_chains` independent forward-sampling
  runs treated as MCMC chains. Since DeGAS parameters are point-optimized (no posterior),
  these no longer measure MCMC convergence -- they become a generative well-posedness
  check: a badly-specified program (e.g. a near-deterministic mixture weight starving one
  branch of samples) shows up as ESS collapse / r_hat blowup across independent runs.
- divergences: fraction of forward-simulated draws that are NaN/Inf ("simulation
  divergences"), analogous to NUTS divergent transitions.
- BFMI: dropped. It is an HMC energy-transition diagnostic with no analog for a
  non-MCMC sampler.
- Pareto-k / ELPD-LOO: dropped as PSIS/importance-sampling machinery and replaced (in
  refine_loop.py, not here) by exact held-out NLL via DeGAS's own compute_likelihood --
  DeGAS gives an exact analytic mixture density, so no importance-sampling correction is
  needed at all.
"""

from __future__ import annotations

import re
from collections import defaultdict

import arviz as az
import numpy as np
import torch
from lark import Token, Tree

from .checkers.degas_checker import _get_parser

_PARAM_PAT = re.compile(r"_(mu|sigma)(\d+)")


def _format_num(x: float) -> str:
    x = max(-100.0, min(100.0, float(x)))
    return f"{x:.2f}"


def substitute_optimized_params(rewritten_program: str, optimized_params: dict) -> str:
    """Replace `_mu{N}`/`_sigma{N}` placeholders (see param_extractor.extract_params) with
    their gradient-optimized values, formatted to satisfy degas.lark's NUMBER terminal."""

    def _sub(m: re.Match) -> str:
        key = f"{m.group(1)}{m.group(2)}"
        val = optimized_params[key]
        val = val.item() if hasattr(val, "item") else float(val)
        return _format_num(val)

    return _PARAM_PAT.sub(_sub, rewritten_program)


def _numlist(numlist_tree: Tree) -> list[float]:
    return [_signed_num_value(c) for c in numlist_tree.children]


def _signed_num_value(node: Tree) -> float:
    """node is a `pos_num`/`neg_num` tree (see grammar/degas.lark's signed_num rule)."""
    if node.data == "pos_num":
        return float(node.children[0])
    if node.data == "neg_num":
        return -float(node.children[-1])
    raise ValueError(f"not a signed_num node: {node.data}")


def _eval_dist(dist_tree: Tree, n_samples: int) -> torch.Tensor:
    if dist_tree.data == "gm":
        weights = torch.tensor(_numlist(dist_tree.children[0]))
        mus = torch.tensor(_numlist(dist_tree.children[1]))
        sigmas = torch.tensor(_numlist(dist_tree.children[2]))
        comp_idx = torch.distributions.Categorical(probs=weights).sample((n_samples,))
        return torch.distributions.Normal(mus[comp_idx], sigmas[comp_idx]).sample()
    if dist_tree.data == "uniform":
        bounds_tree, _trailing = dist_tree.children
        lo, hi = _numlist(bounds_tree)
        return torch.distributions.Uniform(lo, hi).sample((n_samples,))
    raise ValueError(f"unknown distribution node: {dist_tree.data}")


def _eval_expr(node, env: dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
    """Evaluates any RHS/atom/multerm expression node (they share a disjoint set of
    `.data` names, so one dispatcher handles all of them)."""
    if isinstance(node, Token):
        raise ValueError(f"unexpected bare token in expression: {node!r}")

    data = node.data
    if data == "var_atom":
        return env[str(node.children[0])]
    if data == "num_atom":
        return torch.full((n_samples,), _signed_num_value(node.children[0]))
    if data == "dist_atom":
        return _eval_dist(node.children[0], n_samples)
    if data in ("gm", "uniform"):
        return _eval_dist(node, n_samples)
    if data == "num_var_mult":
        num = _signed_num_value(node.children[0])
        return num * env[str(node.children[1])]
    if data == "num_dist_mult":
        num = _signed_num_value(node.children[0])
        return num * _eval_dist(node.children[1], n_samples)
    if data == "var_var_mult":
        return env[str(node.children[0])] * env[str(node.children[1])]
    if data == "add_atoms":
        a, op, b = node.children
        av, bv = _eval_expr(a, env, n_samples), _eval_expr(b, env, n_samples)
        return av + bv if str(op) == "+" else av - bv
    if data == "add_multerm_right":
        multerm, op, atom = node.children
        mv, av = _eval_expr(multerm, env, n_samples), _eval_expr(atom, env, n_samples)
        return mv + av if str(op) == "+" else mv - av
    if data == "add_multerm_left":
        atom, op, multerm = node.children
        av, mv = _eval_expr(atom, env, n_samples), _eval_expr(multerm, env, n_samples)
        return av + mv if str(op) == "+" else av - mv
    raise ValueError(f"unknown expression node: {data}")


def _round_eq(a: torch.Tensor, b: float) -> torch.Tensor:
    # DeGAS represents discrete/categorical states as gm(...) components with narrow but
    # nonzero sigma (e.g. sigma=0.01) rather than true Dirac deltas, so sampled values are
    # almost never *exactly* equal to an integer threshold. `==`/`!=` in DEGAS_GRAMMAR are
    # only ever used against such near-integer discrete states, so nearest-integer
    # comparison is the correct semantics here (not float equality/isclose).
    return torch.round(a) == round(b)


_CMP_OPS = {
    "==": _round_eq,
    "!=": lambda a, b: ~_round_eq(a, b),
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _execute_statements(
    statements: list[Tree], env: dict[str, torch.Tensor], n_samples: int
) -> dict[str, torch.Tensor]:
    """statements: `statement` tree nodes (already unwrapped from `toplevel`). Ancestral
    sampling in program order; both branches of a conditional are simulated for every
    sample and merged with `torch.where` -- simpler and just as correct as boolean-indexed
    branch splitting for these small, shallow benchmark programs."""
    for stmt in statements:
        inner = stmt.children[0]
        if inner.data == "assignment":
            lhs = str(inner.children[0])
            env[lhs] = _eval_expr(inner.children[1], env, n_samples)
        elif inner.data == "conditional":
            bexpr, ifblock, elseblock = inner.children
            cond_var = str(bexpr.children[0])
            cmp_op = str(bexpr.children[1])
            threshold = _signed_num_value(bexpr.children[2])
            mask = _CMP_OPS[cmp_op](env[cond_var], threshold)

            env_if = _execute_statements(ifblock.children, dict(env), n_samples)
            env_else = _execute_statements(elseblock.children, dict(env), n_samples)

            for var in (set(env_if) | set(env_else)) - set(env):
                v_if = env_if.get(var, env.get(var))
                v_else = env_else.get(var, env.get(var))
                env[var] = torch.where(mask, v_if, v_else)
    return env


def forward_sample(
    rewritten_program: str,
    optimized_params: dict,
    n_samples: int = 500,
    n_chains: int = 4,
    seed_base: int = 0,
) -> dict[str, torch.Tensor]:
    """Returns {var_name: tensor of shape (n_chains, n_samples)}."""
    program_text = substitute_optimized_params(rewritten_program, optimized_params)
    tree = _get_parser().parse(program_text)
    top_statements = [toplevel.children[0] for toplevel in tree.children]

    all_chains: dict[str, list[torch.Tensor]] = defaultdict(list)
    for c in range(n_chains):
        torch.manual_seed(seed_base + c)
        env = _execute_statements(top_statements, {}, n_samples)
        for var, val in env.items():
            all_chains[var].append(val)

    return {var: torch.stack(vals, dim=0) for var, vals in all_chains.items()}


def reliability_from_samples(
    samples: dict[str, torch.Tensor],
    var_names: list[str],
    r_hat_threshold: float = 1.05,
    min_ess_bulk: float = 100.0,
    min_ess_tail: float = 50.0,
    max_divergence_frac: float = 0.01,
) -> dict:
    """r_hat/ESS/divergence checks over the forward-sampled chains -- see module docstring
    for what these mean here (generative well-posedness, not MCMC convergence)."""
    finite_frac_per_var = {}
    clean_samples = {}
    for var in var_names:
        if var not in samples:
            continue
        arr = samples[var].numpy()
        finite_frac_per_var[var] = float(np.isfinite(arr).mean())
        clean_samples[var] = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    missing_vars = [v for v in var_names if v not in samples]
    if missing_vars or not clean_samples:
        return {
            "max_r_hat": float("inf"),
            "min_ess_bulk": 0.0,
            "min_ess_tail": 0.0,
            "divergence_frac": 1.0,
            "missing_vars": missing_vars,
            "checks": {"r_hat": False, "ess_bulk": False, "ess_tail": False, "no_divergences": False},
        }

    idata = az.from_dict(posterior=clean_samples)
    summary = az.summary(idata)
    max_r_hat = float(summary["r_hat"].max())
    min_ess_bulk_val = float(summary["ess_bulk"].min())
    min_ess_tail_val = float(summary["ess_tail"].min())
    divergence_frac = 1.0 - min(finite_frac_per_var.values())

    checks = {
        "r_hat": max_r_hat < r_hat_threshold,
        "ess_bulk": min_ess_bulk_val >= min_ess_bulk,
        "ess_tail": min_ess_tail_val >= min_ess_tail,
        "no_divergences": divergence_frac <= max_divergence_frac,
    }
    return {
        "max_r_hat": max_r_hat,
        "min_ess_bulk": min_ess_bulk_val,
        "min_ess_tail": min_ess_tail_val,
        "divergence_frac": divergence_frac,
        "missing_vars": [],
        "checks": checks,
    }
