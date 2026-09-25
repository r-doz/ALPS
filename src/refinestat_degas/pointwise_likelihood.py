"""Pointwise (per-row) log-likelihood for DeGAS output distributions.

DeGAS/src/PROGRAMS/likelihood.py::compute_likelihood returns a single scalar (the mean
log-likelihood across all rows -- exact same underlying math, just reduced at the end).
LOO-CV (mcmc_diagnostics.py, scored with ArviZ's az.loo/PSIS) needs the *pointwise*
(per-observation) log-likelihood instead, so this mirrors compute_likelihood's computation
up to the per-row likelihood tensor and returns torch.log(likelihood) unreduced, rather
than modifying DeGAS's own source.
"""

from __future__ import annotations

import torch
from torch.distributions import MultivariateNormal


def pointwise_log_likelihood(output_dist, data_var_list: list[str], data) -> torch.Tensor:
    """Returns a (len(data),) tensor of per-row log-likelihoods."""
    data = torch.tensor(data)
    try:
        data_var_index = [output_dist.var_list.index(v) for v in data_var_list]
    except ValueError:
        return torch.full((len(data),), float("-inf"))

    likelihood = torch.zeros(len(data))
    for k in range(output_dist.gm.n_comp()):
        sigma = output_dist.gm.sigma[k][data_var_index][:, data_var_index]
        mu = output_dist.gm.mu[k][data_var_index]
        diag = torch.diag(sigma)
        deltas = torch.where(diag == 0)[0]
        not_deltas = torch.where(diag != 0)[0]
        mu_delta = mu[deltas]
        mu_not_delta = mu[not_deltas]
        sigma_not_delta = sigma[not_deltas][:, not_deltas]
        if len(mu_not_delta) >= 1:
            continuous_pdf = (
                output_dist.gm.pi[k]
                * MultivariateNormal(mu_not_delta, sigma_not_delta).log_prob(data[:, not_deltas]).exp()
            )
        else:
            continuous_pdf = output_dist.gm.pi[k] * torch.ones(len(data))
        if len(mu_delta) >= 1:
            discrete_pmf = torch.all((mu_delta == data[:, deltas]), dim=1)
        else:
            discrete_pmf = torch.ones(len(data))
        likelihood = likelihood + continuous_pdf * discrete_pmf
    return torch.log(likelihood)
