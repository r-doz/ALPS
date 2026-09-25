"""A "fair NLL" correction for DeGAS's compute_likelihood (DeGAS/src/PROGRAMS/likelihood.py).

That function floors any component dimension's fitted std at MIN_SIGMA=1e-3 before
evaluating a Gaussian density, UNLESS the std is EXACTLY 0.0 (in which case the dimension is
treated as a true discrete "delta": scored as a 0/1 exact-match indicator, a proper
probability). A variable that's genuinely discrete-valued but gets compiled with a
tiny-but-nonzero fitted std (e.g. 5e-5, from a candidate's own optimizer/MCMC posterior,
never having been the literal text "sigma=0.00" that smooth_cfg rewrites) lands in the
non-delta branch instead: it's floored to 1e-3 and scored as a *density*, which can exceed 1,
giving an unbounded-looking (but in fact MIN_SIGMA-bounded, ~5.99 nat max) bonus that has
nothing to do with how well the candidate actually fits the data.

This module corrects that: for dimensions KNOWN to correspond to a discrete-valued variable
(from DISCRETE_VARS below, derived from each benchmark's own data-generating process, not
from whatever std a given candidate happens to fit), the *component's own* (mu, sigma) --
whatever they are, no forcing to exact-delta -- is used to compute the PROBABILITY MASS in a
+-0.5 interval around the observed integer value via the Normal CDF, instead of a point
density. This is a strictly proper probability (always <= 1, so log <= 0: no possible bonus),
and it recovers the existing exact-delta behavior in the sigma -> 0 limit (interval
probability -> 1 if the component mean rounds to the observed value, else -> 0), so it
strictly generalizes rather than replaces the existing discrete handling. Continuous
dimensions are scored exactly as before (unchanged), including the existing MIN_SIGMA floor,
which is a separate, legitimate numerical-stability concern for genuinely continuous
variables, not this artifact.

Every discrete variable across the 9 default benchmarks (ALL_PROGRAMS in
aggregate_results_table.py) is integer-valued with unit spacing between adjacent categories
(binomial/np.random.choice draws -- verified against every generate_*_dataset function in
helpers/dataset_generation.py), so a fixed +-0.5 bin half-width is used throughout.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal

MIN_SIGMA = 1e-3  # matches DeGAS/src/PROGRAMS/likelihood.py's own floor, for continuous dims
CDF_EPS = 1e-9  # numerical floor only, to avoid literal division by zero as sigma -> 0
BIN_HALF_WIDTH = 0.5  # every discrete variable here is integer-valued with unit spacing

# benchmark -> set of variable names (as returned by get_var_names) that are genuinely
# discrete-valued in that benchmark's own data-generating process (see module docstring).
DISCRETE_VARS: dict[str, set[str]] = {
    "if": set(),
    "mog1": set(),
    "csi": {"u", "v", "w", "x"},
    "easytugwar": {"p1wins"},
    "biasedtugwar": {"p1wins"},
    "mixedcondition": {"u"},
    "multiplebranches": set(),
    "eyecolor": {"eyecolor", "haircolor", "hairlenght"},
    "hurricane": {"preplevel", "damage"},
}


def compute_likelihood_corrected(output_dist, data_var_list, data, discrete_var_names):
    """Same interface/semantics as DeGAS's compute_likelihood, except dimensions named in
    `discrete_var_names` always get a bounded interval-probability treatment (see module
    docstring) instead of a possibly-inflated floored density."""
    data = torch.tensor(data)
    likelihood = 0
    try:
        data_var_index = [output_dist.var_list.index(element) for element in data_var_list]
    except ValueError:
        return torch.tensor(-np.inf)

    discrete_mask = torch.tensor([name in discrete_var_names for name in data_var_list])

    for k in range(output_dist.gm.n_comp()):
        sigma = output_dist.gm.sigma[k][data_var_index][:, data_var_index]
        mu = output_dist.gm.mu[k][data_var_index]
        diag = torch.diag(sigma)

        # Discrete dims are decided by known variable identity, NOT by this candidate's
        # fitted sigma (that's exactly the fix -- see module docstring). Everything else
        # keeps the original exact-zero-vs-floored-density split, unchanged.
        discrete_idx = torch.where(discrete_mask)[0]
        cont_all_idx = torch.where(~discrete_mask)[0]
        cont_diag = diag[cont_all_idx] if len(cont_all_idx) else diag[:0]
        not_deltas = cont_all_idx[torch.where(cont_diag != 0)[0]] if len(cont_all_idx) else cont_all_idx
        deltas_cont = cont_all_idx[torch.where(cont_diag == 0)[0]] if len(cont_all_idx) else cont_all_idx

        # -- continuous (non-discrete) dims: unchanged from DeGAS's own compute_likelihood --
        mu_not_delta = mu[not_deltas]
        sigma_not_delta = sigma[not_deltas][:, not_deltas]
        if len(mu_not_delta) >= 1:
            diag_idx = torch.arange(len(not_deltas))
            std_diag = torch.sqrt(sigma_not_delta[diag_idx, diag_idx])
            floored_std_diag = torch.clamp(std_diag, min=MIN_SIGMA)
            if not torch.equal(std_diag, floored_std_diag):
                sigma_not_delta = sigma_not_delta.clone()
                sigma_not_delta[diag_idx, diag_idx] = floored_std_diag ** 2
            continuous_pdf = output_dist.gm.pi[k] * MultivariateNormal(mu_not_delta, sigma_not_delta).log_prob(data[:, not_deltas]).exp()
        else:
            continuous_pdf = output_dist.gm.pi[k] * torch.ones(len(data))

        # -- continuous dims that happen to have EXACT zero variance: unchanged (true delta) --
        mu_delta_cont = mu[deltas_cont]
        if len(mu_delta_cont) >= 1:
            continuous_pdf = continuous_pdf * torch.all((mu_delta_cont == data[:, deltas_cont]), dim=1)

        # -- known-discrete dims: interval probability via each dim's own (mu, sigma) --
        if len(discrete_idx) >= 1:
            mu_disc = mu[discrete_idx]
            std_disc = torch.sqrt(torch.clamp(torch.diag(sigma)[discrete_idx], min=0.0))
            std_disc = torch.clamp(std_disc, min=CDF_EPS)
            obs = data[:, discrete_idx]
            normal = Normal(mu_disc, std_disc)
            upper = normal.cdf(obs + BIN_HALF_WIDTH)
            lower = normal.cdf(obs - BIN_HALF_WIDTH)
            discrete_prob = torch.clamp(upper - lower, min=0.0, max=1.0).prod(dim=1)
        else:
            discrete_prob = torch.ones(len(data))

        likelihood = likelihood + continuous_pdf * discrete_prob

    return torch.sum(torch.log(likelihood)) / len(data)
