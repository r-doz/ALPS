"""Wasserstein-type distance between Gaussian Mixture Models (GW2).

Vendored (with light adaptation) from https://github.com/judelo/gmmot
(Julie Delon), implementing the GMM-OT distance described in:

    Delon, Desolneux, "A Wasserstein-type distance in the space of
    Gaussian Mixture Models", 2019. https://hal.archives-ouvertes.fr/hal-02178204

The GW2 distance restricts the set of admissible couplings between two
GMMs to the ones that are themselves Gaussian mixtures: it computes the
quadratic Wasserstein distance between every pair of Gaussian components
(one from each mixture), then solves a discrete optimal transport
problem over the mixture weights using this pairwise cost matrix.
"""

import numpy as np
import ot
import scipy.linalg as spl
import torch


def gaussian_w2(m0, m1, sigma0, sigma1):
    """Quadratic Wasserstein distance between two Gaussians N(m0, sigma0) and N(m1, sigma1)."""
    sigma00 = spl.sqrtm(sigma0)
    sigma010 = spl.sqrtm(sigma00 @ sigma1 @ sigma00)
    return np.linalg.norm(m0 - m1) ** 2 + np.trace(sigma0 + sigma1 - 2 * sigma010).real


def gmm_w2(pi0, pi1, mu0, mu1, sigma0, sigma1):
    """GW2 distance between two GMMs.

    pi0, pi1: (K0,), (K1,) mixture weights
    mu0, mu1: (K0, d), (K1, d) component means
    sigma0, sigma1: (K0, d, d), (K1, d, d) component covariances

    Returns (transport_plan, distance).
    """
    k0, k1 = mu0.shape[0], mu1.shape[0]
    cost = np.zeros((k0, k1))
    for k in range(k0):
        for l in range(k1):
            cost[k, l] = gaussian_w2(mu0[k], mu1[l], sigma0[k], sigma1[l])
    transport_plan = ot.emd(pi0, pi1, cost)
    distance = np.sum(transport_plan * cost)
    return transport_plan, distance


def gmm_w2_distance(gm0, gm1) -> float:
    """GW2 distance between two DeGAS/SOGA GaussianMix distributions
    (objects exposing .pi (K,1), .mu (K,D) and .sigma (K,D,D) tensors,
    e.g. `output_dist.gm` as returned by DeGAS's `start_SOGA`)."""
    with torch.no_grad():
        pi0 = gm0.pi.reshape(-1).double().numpy()
        pi1 = gm1.pi.reshape(-1).double().numpy()
        mu0 = gm0.mu.double().numpy()
        mu1 = gm1.mu.double().numpy()
        sigma0 = gm0.sigma.double().numpy()
        sigma1 = gm1.sigma.double().numpy()

    _, distance = gmm_w2(pi0, pi1, mu0, mu1, sigma0, sigma1)
    return float(distance)
