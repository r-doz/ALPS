"""Genuine NUTS/MCMC posterior inference + diagnostics for a DeGAS candidate program --
the faithful analog of RefineStat's pm.sample() + Bayesian-workflow checks, using
numpyro_model.py's NumPyro translation. See numpyro_model.py's module docstring for why
this exists alongside refine_loop.py's default (DeGAS-native, gradient-optimized point
estimate) inference path, and why NumPyro rather than Pyro.

Reliability score: RefineStat's own seven checks (Definition 2 / commons/utils.py's
`check_model_reliability_numpyro` in https://github.com/structuredllm/RefineStat -- no
LICENSE file; thresholds and the BFMI-via-extra-fields extraction pattern below are
vendored/adapted from there with attribution, same basis as helpers/gmm_distance.py):
r_hat, ess_bulk, ess_tail, no_divergences, bfmi, pareto_k_ok, loo_success. Their own
threshold defaults (see their README and commons/utils.py::check_model_reliability_numpyro):
r_hat < 1.05, ess_bulk >= 400, ess_tail >= 100, bfmi > 0.3, Pareto-k <= 0.7 for at least
80% of points (max_prop_k=0.20), cutoff zeta=5 of 7 (set via RefineConfig.K in main.py
when --inference mcmc). Their own `check_model_reliability`/`_numpyro` functions compute
Pareto-k/ELPD-LOO via PSIS on `numpyro.infer.util.log_likelihood(mcmc.sampler.model, ...)`
against their own exec()'d-PyMC-script `mcmc` object shape, which doesn't apply to our
sequential-per-chain-run architecture -- we keep our own pre-existing DeGAS-exact-likelihood
LOO/Pareto-k computation below (genuinely more rigorous than PSIS approximating an
intractable likelihood, per this module's prior version) rather than force-fitting their
LOO plumbing to a different mcmc object shape.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import torch
from numpyro.infer import MCMC, NUTS

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "DeGAS" / "src", _REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from optimization import compile2SOGA_text, produce_cfg_text, smooth_cfg, start_SOGA  # noqa: E402
from helpers.param_extractor import extract_params  # noqa: E402

from .pointwise_likelihood import pointwise_log_likelihood  # noqa: E402
from .numpyro_model import UnsupportedForMCMC, build_numpyro_model  # noqa: E402
from .text_utils import fix_uniform_trailing_literal  # noqa: E402


@dataclass
class MCMCConfig:
    num_chains: int = 4
    warmup_steps: int = 1000  # RefineStat's own pm.sample(1000, tune=1000, ...), see main.py
    num_samples: int = 1000
    mu_prior_scale: float = 50.0
    sigma_prior_scale: float = 25.0
    r_hat_threshold: float = 1.05
    min_ess_bulk: float = 400.0  # RefineStat's own default (commons/utils.py), was 100
    min_ess_tail: float = 100.0
    bfmi_threshold: float = 0.3  # RefineStat's own default
    max_divergence_frac: float = 0.05
    max_pareto_k_frac: float = 0.20  # RefineStat's own max_prop_k default
    pareto_k_threshold: float = 0.7
    loo_draws_per_chain: int = 50  # subsample for pointwise-likelihood computation cost


_FAILED_MCMC_CHECKS = {
    "r_hat": False,
    "ess_bulk": False,
    "ess_tail": False,
    "no_divergences": False,
    "bfmi": False,
    "pareto_k_ok": False,
    "loo_success": False,
}


def fit_and_diagnose_mcmc(
    program_text: str, train_data, held_out_data, var_names: list[str], cfg: MCMCConfig
) -> dict:
    """DeGAS-compile/NumPyro-NUTS analog of refine_loop.py::fit_and_diagnose. Any failure
    (including UnsupportedForMCMC, see numpyro_model.py) is treated as a failed candidate
    (reliability_score=0), matching fit_and_diagnose's own defensive exception handling."""
    try:
        rewritten, params = extract_params(program_text)
        fixed_rewritten = fix_uniform_trailing_literal(rewritten)

        model, param_names = build_numpyro_model(
            fixed_rewritten, var_names, mu_scale=cfg.mu_prior_scale, sigma_scale=cfg.sigma_prior_scale
        )
        train_tensor = jnp.asarray(np.asarray(train_data), dtype=jnp.float32)

        # NumPyro/JAX has no PyTorch-autograd/fork conflict (see numpyro_model.py's
        # docstring for why the old Pyro path had to run chains sequentially in-process) --
        # run all chains via NumPyro's own native multi-chain support.
        kernel = NUTS(model)
        mcmc = MCMC(
            kernel,
            num_warmup=cfg.warmup_steps,
            num_samples=cfg.num_samples,
            num_chains=cfg.num_chains,
            chain_method="vectorized",
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(int(np.random.randint(0, 2**31 - 1)))
        mcmc.run(rng_key, train_tensor, extra_fields=("diverging", "potential_energy"))

        posterior_draws = {
            pname: np.asarray(v) for pname, v in mcmc.get_samples(group_by_chain=True).items()
        }
        extra_fields = mcmc.get_extra_fields(group_by_chain=True)
        diverging = np.asarray(extra_fields["diverging"])
        idata = az.from_dict(posterior=posterior_draws, sample_stats={"diverging": diverging})

        summary = az.summary(idata)
        max_r_hat = float(summary["r_hat"].max())
        min_ess_bulk = float(summary["ess_bulk"].min())
        min_ess_tail = float(summary["ess_tail"].min())
        divergence_frac = float(diverging.mean())

        # BFMI: vendored from RefineStat's check_model_reliability_numpyro (see module
        # docstring) -- the exact reason for switching from Pyro to NumPyro, since Pyro's
        # public MCMC API doesn't expose this energy trace.
        energy = np.asarray(extra_fields["potential_energy"])
        bfmi_vals = az.bfmi(energy)
        bfmi_ok = bool((bfmi_vals > cfg.bfmi_threshold).all())

        # Pointwise log-likelihood on train_data across a subsample of posterior draws,
        # for PSIS-based ELPD-LOO/Pareto-k -- DeGAS's own EXACT likelihood, not an
        # approximation. Compile the CFG once (structure is fixed) and reuse it across
        # draws, only varying the mu/sigma parameter values.
        compiled = compile2SOGA_text(fixed_rewritten)
        cfg_obj = produce_cfg_text(compiled)
        smooth_cfg(cfg_obj)

        posterior = idata.posterior
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        step = max(1, n_draws // cfg.loo_draws_per_chain)
        draw_idxs = list(range(0, n_draws, step))[: cfg.loo_draws_per_chain]

        log_lik = np.zeros((n_chains, len(draw_idxs), len(train_data)))
        subsampled_posterior = {pname: np.zeros((n_chains, len(draw_idxs))) for pname in param_names}
        loo_success = True
        try:
            for ci in range(n_chains):
                for di_out, di in enumerate(draw_idxs):
                    params_dict = {
                        pname: torch.as_tensor(float(posterior[pname].values[ci, di]))
                        for pname in param_names
                    }
                    for pname, val in params_dict.items():
                        subsampled_posterior[pname][ci, di_out] = val.item()
                    output_dist = start_SOGA(cfg_obj, params_dict)
                    log_lik[ci, di_out, :] = pointwise_log_likelihood(
                        output_dist, var_names, train_data
                    ).numpy()

            # az.loo requires a `posterior` group to be present alongside `log_likelihood`
            # (not just the log-likelihood values), even though PSIS-LOO only mathematically
            # needs the latter -- so pass through the same (subsampled) draws used above.
            loo_idata = az.from_dict(posterior=subsampled_posterior, log_likelihood={"y": log_lik})
            loo_result = az.loo(loo_idata, pointwise=True)
            pareto_k = np.asarray(loo_result.pareto_k)
            high_k_frac = float((pareto_k > cfg.pareto_k_threshold).mean())
            pareto_k_ok = high_k_frac <= cfg.max_pareto_k_frac
            elpd_loo = float(loo_result.elpd_loo)
        except Exception:
            loo_success = False
            high_k_frac = 1.0
            pareto_k_ok = False
            elpd_loo = float("-inf")

        # genuine held-out NLL at the posterior mean, for direct comparability with
        # refine_loop.py::fit_and_diagnose's gradient-based held_out_nll field.
        mean_params = {
            pname: torch.as_tensor(float(posterior[pname].values.mean())) for pname in param_names
        }
        mean_output_dist = start_SOGA(cfg_obj, mean_params)
        held_out_nll = (
            -pointwise_log_likelihood(mean_output_dist, var_names, held_out_data).mean()
        ).item()

        checks = {
            "r_hat": max_r_hat < cfg.r_hat_threshold,
            "ess_bulk": min_ess_bulk >= cfg.min_ess_bulk,
            "ess_tail": min_ess_tail >= cfg.min_ess_tail,
            "no_divergences": divergence_frac <= cfg.max_divergence_frac,
            "bfmi": bfmi_ok,
            "pareto_k_ok": pareto_k_ok,
            "loo_success": loo_success,
        }
        score = sum(checks.values())

        return {
            "rewritten": fixed_rewritten,
            "params": params,
            "posterior_mean_params": {k: v.item() for k, v in mean_params.items()},
            "max_r_hat": max_r_hat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "divergence_frac": divergence_frac,
            "bfmi_min": float(np.min(bfmi_vals)),
            "elpd_loo": elpd_loo,
            "pareto_k_high_frac": high_k_frac,
            "held_out_nll": held_out_nll,
            "checks": checks,
            "reliability_score": score,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 -- NumPyro/NUTS/DeGAS can raise many exception types
        return {
            "rewritten": None,
            "params": None,
            "posterior_mean_params": None,
            "max_r_hat": float("inf"),
            "min_ess_bulk": 0.0,
            "min_ess_tail": 0.0,
            "divergence_frac": 1.0,
            "bfmi_min": 0.0,
            "elpd_loo": float("-inf"),
            "pareto_k_high_frac": 1.0,
            "held_out_nll": float("inf"),
            "checks": dict(_FAILED_MCMC_CHECKS),
            "reliability_score": 0,
            "error": f"{type(e).__name__}: {e}",
        }
