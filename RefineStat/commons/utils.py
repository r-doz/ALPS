import random
import torch
import numpy as np
import xarray as xr
import tempfile
import os
import warnings
import traceback
import arviz as az
from numpyro.infer.util import log_likelihood

def convert_np_types(obj):
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, xr.DataArray):
        return obj.item() if obj.size == 1 else obj.values.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    else:
        return obj

# Set random seeds for reproducibility
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_model_reliability(
    idata,
    r_hat_threshold: float = 1.05,
    min_ess_bulk: int = 400,
    min_ess_tail: int = 100,
    max_pareto_k: float = 0.7,
    max_prop_k: float = 0.20,
):

    scores = {}
    reasons = []

    # 1. Convergence: R-hat
    summary = az.summary(idata)
    max_r_hat = summary["r_hat"].max()
    r_hat_ok = max_r_hat < r_hat_threshold
    scores["r_hat"] = int(r_hat_ok)
    if not r_hat_ok:
        reasons.append(f"R‑hat = {max_r_hat:.3f} > {r_hat_threshold}")

    # 2. Convergence: ESS bulk
    min_ess_bulk_val = summary["ess_bulk"].min()
    ess_bulk_ok = min_ess_bulk_val >= min_ess_bulk
    scores["ess_bulk"] = int(ess_bulk_ok)
    if not ess_bulk_ok:
        reasons.append(f"ESS_bulk = {min_ess_bulk_val:.1f} < {min_ess_bulk}")

    # 3. Convergence: ESS tail
    if "ess_tail" in summary.columns:
        min_ess_tail_val = summary["ess_tail"].min()
        ess_tail_ok = min_ess_tail_val >= min_ess_tail
        scores["ess_tail"] = int(ess_tail_ok)
        if not ess_tail_ok:
            reasons.append(f"ESS_tail = {min_ess_tail_val:.1f} < {min_ess_tail}")
    else:
        min_ess_tail_val = None
        ess_tail_ok = False
        scores["ess_tail"] = 0
        reasons.append("ESS_tail not available")

    # 4. Sampler health: Divergent transitions
    n_divergent = int(idata.sample_stats["diverging"].sum())
    div_ok = (n_divergent == 0)
    scores["divergences"] = int(div_ok)
    if not div_ok:
        reasons.append(f"{n_divergent} divergent transition(s)")

    # 5. Sampler health: BFMI
    bfmi_vals = az.bfmi(idata)
    bfmi_ok = (bfmi_vals > 0.3).all()
    scores["bfmi"] = int(bfmi_ok)
    if not bfmi_ok:
        reasons.append(f"Low BFMI: {bfmi_vals}")

    # 6. PSIS diagnostic & LOO success
    try:
        loo_res = az.loo(idata, pointwise=True)
        scores["loo_success"] = 1
        pareto_k = loo_res.pareto_k
        prop_high_k = np.mean(pareto_k > max_pareto_k)
        k_ok = prop_high_k <= max_prop_k
        scores["pareto_k"] = int(k_ok)
        if not k_ok:
            num_bad = int((pareto_k > max_pareto_k).sum())
            total = len(pareto_k)
            reasons.append(f"{num_bad}/{total} (={100*prop_high_k:.1f}%) k > {max_pareto_k}")
    except Exception as e:
        scores["loo_success"] = 0
        scores["pareto_k"] = 0
        prop_high_k = None
        reasons.append(f"LOO failure: {e}")

    # Aggregate
    total_checks = len(scores)  # 8 checks
    reliability_score = sum(scores.values())  # 0..8
    max_score = total_checks

    diagnostics = {
        "reliability_score": reliability_score,
        "max_score": max_score,
        "individual_scores": scores,
        "reasons": reasons,
        "max_r_hat": max_r_hat,
        "min_ess_bulk": min_ess_bulk_val,
        "min_ess_tail": min_ess_tail_val,
        "n_divergent": n_divergent,
        "bfmi_values": bfmi_vals,
        "prop_high_pareto_k": prop_high_k,
        "elpd_loo": loo_res.elpd_loo,
        "loo_se": loo_res.se,
    }

    return reliability_score, diagnostics



def check_model_reliability_numpyro(
    mcmc,
    r_hat_threshold: float = 1.05,
    min_ess_bulk: int = 400,
    min_ess_tail: int = 100,
    max_pareto_k: float = 0.7,
    max_prop_k: float = 0.20,
    params: list[str] = None,
):
    scores = {}
    reasons = []
    idata = az.from_numpyro(mcmc)

    # ---------------------
    # 1. Convergence checks
    # ---------------------
    summary = az.summary(idata)
    max_r_hat = summary["r_hat"].max()
    r_hat_ok = max_r_hat < r_hat_threshold
    scores["r_hat"] = int(r_hat_ok)
    if not r_hat_ok:
        reasons.append(f"R-hat = {max_r_hat:.3f} > {r_hat_threshold}")

    min_ess_bulk_val = summary["ess_bulk"].min()
    ess_bulk_ok = min_ess_bulk_val >= min_ess_bulk
    scores["ess_bulk"] = int(ess_bulk_ok)
    if not ess_bulk_ok:
        reasons.append(f"ESS_bulk = {min_ess_bulk_val:.1f} < {min_ess_bulk}")

    if "ess_tail" in summary.columns:
        min_ess_tail_val = summary["ess_tail"].min()
        ess_tail_ok = min_ess_tail_val >= min_ess_tail
        scores["ess_tail"] = int(ess_tail_ok)
        if not ess_tail_ok:
            reasons.append(f"ESS_tail = {min_ess_tail_val:.1f} < {min_ess_tail}")
    else:
        min_ess_tail_val = None
        scores["ess_tail"] = 0
        reasons.append("ESS_tail not available")

    # ---------------------
    # 2. Sampler diagnostics
    # ---------------------
    n_divergent = int(idata.sample_stats["diverging"].sum())
    scores["divergences"] = int(n_divergent == 0)
    if n_divergent > 0:
        reasons.append(f"{n_divergent} divergent transition(s)")

    extras = mcmc.get_extra_fields(group_by_chain=True)
    energy = np.asarray(extras["potential_energy"])
    bfmi_vals = az.bfmi(energy)
    scores["bfmi"] = int((bfmi_vals > 0.3).all())
    if not (bfmi_vals > 0.3).all():
        reasons.append(f"Low BFMI: {bfmi_vals}")

    # ---------------------
    # 3. ELPD-LOO computation
    # ---------------------
    try:
        loo_res = az.loo(idata, pointwise=True)
        scores["loo_success"] = 1
    except Exception as e:
        if params is not None:
            try:
                posterior = mcmc.get_samples()
                ll_dict = log_likelihood(mcmc.sampler.model, posterior, **mcmc.model_args)

                # combine multiple likelihoods if needed
                log_lik_joint = None
                for v in ll_dict.values():
                    log_lik_joint = v if log_lik_joint is None else log_lik_joint + v

                # reshape and label
                nchains = idata.posterior.sizes["chain"]
                ndraws = idata.posterior.sizes["draw"]
                nobs = log_lik_joint.shape[-1]
                log_lik_joint = np.asarray(log_lik_joint).reshape((nchains, ndraws, nobs))

                idata.log_likelihood["joint"] = xr.DataArray(
                    log_lik_joint, dims=("chain", "draw", "obs")
                )

                loo_res = az.loo(idata, var_name="joint")
                posterior = mcmc.get_samples() # This returns {"y": loglik_y, "k": loglik_k}
                reasons.append("LOO computed via fallback method.")
            except Exception as e2:
                if ll_dict is None:
                    reasons.append("obs sites missing")
                else:
                    raise e2

    # ---------------------
    # 4. Pareto-k diagnostic
    # ---------------------
    # if loo_res is not None:
    pareto_k = loo_res.pareto_k
    prop_high_k = np.mean(pareto_k > max_pareto_k)
    k_ok = prop_high_k <= max_prop_k
    scores["pareto_k"] = int(k_ok)
    if not k_ok:
        num_bad = int((pareto_k > max_pareto_k).sum())
        total = len(pareto_k)
        reasons.append(f"{num_bad}/{total} (={100*prop_high_k:.1f}%) k > {max_pareto_k}")
    # ---------------------
    # 5. Aggregate diagnostics
    # ---------------------
    reliability_score = sum(scores.values())
    max_score = len(scores)

    diagnostics = {
        "reliability_score": reliability_score,
        "max_score": max_score,
        "individual_scores": scores,
        "reasons": reasons,
        "max_r_hat": max_r_hat,
        "min_ess_bulk": min_ess_bulk_val,
        "min_ess_tail": min_ess_tail_val,
        "n_divergent": n_divergent,
        "bfmi_values": bfmi_vals,
        "prop_high_pareto_k": prop_high_k,
        "elpd_loo": getattr(loo_res, "elpd_loo", None),
        "loo_se": getattr(loo_res, "se", None),
    }

    return reliability_score, diagnostics


def run_pymc_code(full_code: str):
    """
    Execute a PyMC / ArviZ code string in its own namespace.
    Any warnings (e.g. the Pareto-k > .7 UserWarning) are
    suppressed so they can never abort execution.
    Returns (success: bool, namespace_or_error: dict | str)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "generated_model.py")
        with open(script_path, "w") as f:
            f.write(full_code)

        ns = {}
        with warnings.catch_warnings():
            # Option 1: silence everything (simplest)
            warnings.filterwarnings("ignore")           

            try:
                exec(full_code, ns)
                return True, ns
            except Exception as e:
                # Any genuine error still gets reported
                return False, "".join(
                    traceback.format_exception_only(type(e), e)
                )