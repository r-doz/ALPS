"""Builds a genuine NumPyro probabilistic program from a DeGAS candidate, for real MCMC/NUTS
posterior inference -- a NumPyro port of pyro_model.py, switched from Pyro because Pyro's
public MCMC/NUTS API doesn't expose the per-step HMC energy trace needed for a genuine BFMI
diagnostic (unlike NumPyro's `mcmc.get_extra_fields()["potential_energy"]`), which blocked
matching RefineStat's actual 7-check reliability score (see commons/utils.py in
https://github.com/structuredllm/RefineStat -- no LICENSE file, vendored/adapted here for
academic research use with attribution, same basis as helpers/gmm_distance.py). NumPyro is
also a backend RefineStat's own paper validates against (Section 5.5), so this is a more
faithful port than staying on Pyro, not just a workaround.

Structurally this is a 1:1 translation of pyro_model.py's tree-walking logic (same
grammar/degas_params.lark, same gm/uniform -> distribution translation, same branch-masking
approach for conditionals) with torch/pyro swapped for jax.numpy/numpyro. See
pyro_model.py's module docstring for the underlying constraints this works around
(MixtureSameFamily has no rsample, so it can only be used as a directly-observed site, never
a free NUTS latent -- branch-conditioned observations are built via mask-combined
log-densities, not by merging sampled values).
"""

from __future__ import annotations

import os
from functools import lru_cache

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from lark import Lark, Tree

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar", "degas_params.lark")

MU_PRIOR_SCALE = 50.0
SIGMA_PRIOR_SCALE = 25.0


class UnsupportedForMCMC(Exception):
    """A structurally-valid DeGAS program uses a pattern this NumPyro/NUTS translation can't
    represent (see module docstring): a latent (non-observed, non-branch-merged)
    distributional variable later used deterministically, or two distributions summed in
    one statement. Callers should treat this like any other failed candidate."""


@lru_cache(maxsize=1)
def _get_params_parser() -> Lark:
    with open(_GRAMMAR_PATH) as f:
        return Lark(f.read(), parser="earley")


def _discover_param_refs(node) -> set[str]:
    if isinstance(node, Tree):
        if node.data == "param_ref":
            return {str(node.children[0])}
        refs: set[str] = set()
        for c in node.children:
            refs |= _discover_param_refs(c)
        return refs
    return set()


def _signed_num_value(node: Tree) -> float:
    if node.data == "pos_num":
        return float(node.children[0])
    if node.data == "neg_num":
        return -float(node.children[-1])
    raise ValueError(f"not a signed_num node: {node.data}")


def _numlist_tensor(numlist_tree: Tree, param_samples: dict) -> jnp.ndarray:
    values = []
    for entry in numlist_tree.children:
        if entry.data == "param_ref":
            values.append(param_samples[str(entry.children[0])])
        else:
            values.append(jnp.asarray(_signed_num_value(entry), dtype=jnp.float32))
    return jnp.stack(values)


def _build_dist(dist_tree: Tree, param_samples: dict):
    if dist_tree.data == "gm":
        weights = jnp.clip(_numlist_tensor(dist_tree.children[0], param_samples), min=1e-6)
        weights = weights / weights.sum()
        mus = _numlist_tensor(dist_tree.children[1], param_samples)
        sigmas = jnp.clip(jnp.abs(_numlist_tensor(dist_tree.children[2], param_samples)), min=1e-4)
        return dist.MixtureSameFamily(dist.Categorical(probs=weights), dist.Normal(mus, sigmas))
    if dist_tree.data == "uniform":
        bounds_tree, _trailing = dist_tree.children
        lo, hi = _numlist_tensor(bounds_tree, param_samples)
        lo, hi = jnp.minimum(lo, hi), jnp.maximum(lo, hi) + 1e-4
        return dist.Uniform(lo, hi)
    raise ValueError(f"unknown distribution node: {dist_tree.data}")


def _mixture_with_new_component(d, new_loc: jnp.ndarray, new_scale: jnp.ndarray):
    """Rebuilds a MixtureSameFamily with a new (broadcast) component batch shape, e.g.
    (K,) -> (N, K) when shifting/scaling by a per-row (plate-shaped) value. Unlike Pyro,
    numpyro's MixtureSameFamily requires mixing_distribution to be a literal Categorical
    (rejects the ExpandedDistribution that `.expand()` would produce), so rebuild it from
    broadcast probs instead of expanding the existing one."""
    comp = dist.Normal(new_loc, jnp.broadcast_to(new_scale, new_loc.shape))
    mix_probs = jnp.broadcast_to(d.mixing_distribution.probs, new_loc.shape)
    mix = dist.Categorical(probs=mix_probs)
    return dist.MixtureSameFamily(mix, comp)


def _shift_dist(d, shift: jnp.ndarray):
    """Returns a distribution representing d + shift (shift is a deterministic tensor,
    scalar or per-row/plate-shaped)."""
    if isinstance(d, dist.MixtureSameFamily):
        new_loc = d.component_distribution.loc + jnp.expand_dims(shift, -1)
        return _mixture_with_new_component(d, new_loc, d.component_distribution.scale)
    if isinstance(d, dist.Uniform):
        return dist.Uniform(d.low + shift, d.high + shift)
    raise TypeError(f"cannot shift distribution of type {type(d)}")


def _scale_dist(d, factor: float):
    """Returns a distribution representing factor * d."""
    if isinstance(d, dist.MixtureSameFamily):
        new_loc = d.component_distribution.loc * factor
        new_scale = d.component_distribution.scale * abs(factor)
        return _mixture_with_new_component(d, new_loc, new_scale)
    if isinstance(d, dist.Uniform):
        lo, hi = d.low * factor, d.high * factor
        lo, hi = jnp.minimum(lo, hi), jnp.maximum(lo, hi)
        return dist.Uniform(lo, hi + 1e-6)
    raise TypeError(f"cannot scale distribution of type {type(d)}")


_LATENT = object()  # sentinel: a distributional value pending observation/merge, not a
# real deterministic value -- looking this up as a plain atom/var is the unsupported
# "latent distribution used deterministically" pattern (see module docstring).


def _atom_to_value_or_dist(node, env: dict, param_samples: dict):
    if node.data == "var_atom":
        name = str(node.children[0])
        val = env[name]
        if val is _LATENT:
            raise UnsupportedForMCMC(
                f"'{name}' holds a latent (unobserved) distribution and is used "
                "deterministically elsewhere -- not representable without reparameterized "
                "sampling or enumeration, neither of which MixtureSameFamily supports here."
            )
        return False, val
    if node.data == "num_atom":
        return False, jnp.asarray(_signed_num_value(node.children[0]), dtype=jnp.float32)
    if node.data == "dist_atom":
        return True, _build_dist(node.children[0], param_samples)
    raise ValueError(f"unknown atom node: {node.data}")


def _multerm_to_value_or_dist(node, env: dict, param_samples: dict):
    if node.data == "num_var_mult":
        num = _signed_num_value(node.children[0])
        name = str(node.children[1])
        val = env[name]
        if val is _LATENT:
            raise UnsupportedForMCMC(f"'{name}' holds a latent distribution used in {num}*{name}")
        return False, num * val
    if node.data == "num_dist_mult":
        num = _signed_num_value(node.children[0])
        base = _build_dist(node.children[1], param_samples)
        return True, _scale_dist(base, num)
    if node.data == "var_var_mult":
        a, b = str(node.children[0]), str(node.children[1])
        va, vb = env[a], env[b]
        if va is _LATENT or vb is _LATENT:
            raise UnsupportedForMCMC(f"'{a}*{b}' involves a latent distribution")
        return False, va * vb
    raise ValueError(f"unknown multerm node: {node.data}")


def _combine(a_is_dist: bool, a_val, b_is_dist: bool, b_val, sign: float):
    """Represents a_val + sign*b_val, where either side may be a distribution (numpyro
    Distribution) or a plain tensor -- see module docstring on why at most one side may be
    a distribution."""
    if a_is_dist and b_is_dist:
        raise UnsupportedForMCMC("sum of two distributions in one statement is not supported")
    if a_is_dist:
        return True, _shift_dist(a_val, sign * b_val)
    if b_is_dist:
        d = b_val if sign > 0 else _scale_dist(b_val, -1.0)
        return True, _shift_dist(d, a_val)
    return False, a_val + sign * b_val


def _rhs_to_value_or_dist(node, env: dict, param_samples: dict):
    data = node.data
    if data in ("var_atom", "num_atom", "dist_atom"):
        return _atom_to_value_or_dist(node, env, param_samples)
    if data in ("num_var_mult", "num_dist_mult", "var_var_mult"):
        return _multerm_to_value_or_dist(node, env, param_samples)
    if data == "add_atoms":
        a, op, b = node.children
        a_is_dist, a_val = _atom_to_value_or_dist(a, env, param_samples)
        b_is_dist, b_val = _atom_to_value_or_dist(b, env, param_samples)
        return _combine(a_is_dist, a_val, b_is_dist, b_val, 1.0 if str(op) == "+" else -1.0)
    if data == "add_multerm_right":
        multerm, op, atom = node.children
        m_is_dist, m_val = _multerm_to_value_or_dist(multerm, env, param_samples)
        a_is_dist, a_val = _atom_to_value_or_dist(atom, env, param_samples)
        return _combine(m_is_dist, m_val, a_is_dist, a_val, 1.0 if str(op) == "+" else -1.0)
    if data == "add_multerm_left":
        atom, op, multerm = node.children
        a_is_dist, a_val = _atom_to_value_or_dist(atom, env, param_samples)
        m_is_dist, m_val = _multerm_to_value_or_dist(multerm, env, param_samples)
        return _combine(a_is_dist, a_val, m_is_dist, m_val, 1.0 if str(op) == "+" else -1.0)
    raise ValueError(f"unknown rhs node: {data}")


def _scorer(is_dist: bool, val):
    """Returns obs -> per-row log-density tensor for this (branch-local) value."""
    if is_dist:
        return lambda obs: val.log_prob(obs)
    return lambda obs: dist.Normal(val, 1e-3).log_prob(obs)


_CMP_OPS = {
    "==": lambda a, b: jnp.round(a) == round(b),
    "!=": lambda a, b: jnp.round(a) != round(b),
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _collect_scorers(
    statements: list[Tree],
    env: dict,
    param_samples: dict,
    var_names_set: set,
    data_tensor: jnp.ndarray,
    var_idx: dict,
):
    """Walks statements (assignment/conditional), returning (updated_env, scorers) where
    scorers[var] is a callable `obs -> per-row log-density` for every variable assigned in
    this block (directly or via a nested conditional, mask-combined via jnp.where on
    log-probabilities -- see module docstring on why: MixtureSameFamily can't be freely
    latent-sampled under NUTS).

    Any variable that is one of the benchmark's var_names is *directly observed*: its true
    value is known from data_tensor, so downstream references to it resolve to that known
    data value, not to a fresh/latent draw. Only genuinely latent (non-var_names) variables
    can ever become the `_LATENT` sentinel."""
    env = dict(env)
    scorers: dict = {}
    for stmt in statements:
        inner = stmt.children[0]
        if inner.data == "assignment":
            lhs = str(inner.children[0])
            is_dist, value_or_dist = _rhs_to_value_or_dist(inner.children[1], env, param_samples)
            scorers[lhs] = _scorer(is_dist, value_or_dist)
            if lhs in var_names_set:
                env[lhs] = data_tensor[:, var_idx[lhs]]
            else:
                env[lhs] = _LATENT if is_dist else value_or_dist
        elif inner.data == "conditional":
            bexpr, ifblock, elseblock = inner.children
            cond_var = str(bexpr.children[0])
            cmp_op = str(bexpr.children[1])
            threshold = _signed_num_value(bexpr.children[2])
            cond_val = env[cond_var]
            if cond_val is _LATENT:
                raise UnsupportedForMCMC(f"'{cond_var}' used in a condition is a latent distribution")
            mask = _CMP_OPS[cmp_op](cond_val, threshold)

            if_env, if_scorers = _collect_scorers(
                ifblock.children, env, param_samples, var_names_set, data_tensor, var_idx
            )
            else_env, else_scorers = _collect_scorers(
                elseblock.children, env, param_samples, var_names_set, data_tensor, var_idx
            )

            for var in set(if_scorers) | set(else_scorers):
                a_scorer = if_scorers.get(var) or scorers.get(var)
                b_scorer = else_scorers.get(var) or scorers.get(var)
                if a_scorer is None or b_scorer is None:
                    continue  # assigned in only one branch and never assigned before -- no
                    # sound way to score rows that took the other branch; leave unscored.

                def make_combined(a=a_scorer, b=b_scorer, m=mask):
                    return lambda obs: jnp.where(m, a(obs), b(obs))

                scorers[var] = make_combined()
                if var in var_names_set:
                    env[var] = data_tensor[:, var_idx[var]]
                else:
                    a_val = if_env.get(var)
                    b_val = else_env.get(var)
                    env[var] = (
                        _LATENT
                        if (a_val is _LATENT or b_val is _LATENT)
                        else jnp.where(mask, a_val, b_val)
                    )
    return env, scorers


def build_numpyro_model(
    rewritten: str, var_names: list[str], mu_scale: float = MU_PRIOR_SCALE, sigma_scale: float = SIGMA_PRIOR_SCALE
):
    """Returns (model, param_names). model(data_tensor) is a NumPyro model function suitable
    for numpyro.infer.MCMC(NUTS(model), ...); data_tensor has shape (N, len(var_names))."""
    tree = _get_params_parser().parse(rewritten)
    top_statements = [toplevel.children[0] for toplevel in tree.children]
    param_names = sorted(_discover_param_refs(tree))
    var_idx = {v: i for i, v in enumerate(var_names)}
    var_names_set = set(var_names)

    def model(data_tensor: jnp.ndarray):
        n = data_tensor.shape[0]
        param_samples = {}
        for pname in param_names:
            if pname.startswith("mu"):
                param_samples[pname] = numpyro.sample(pname, dist.Normal(0.0, mu_scale))
            elif pname.startswith("sigma"):
                param_samples[pname] = numpyro.sample(pname, dist.HalfNormal(sigma_scale))
            else:
                raise ValueError(f"unrecognized parameter name: {pname}")

        with numpyro.plate("data", n):
            _, scorers = _collect_scorers(
                top_statements, {}, param_samples, var_names_set, data_tensor, var_idx
            )
            for var in var_names_set:
                if var not in scorers:
                    raise UnsupportedForMCMC(f"'{var}' is a benchmark variable but is never assigned")
                log_prob = scorers[var](data_tensor[:, var_idx[var]])
                numpyro.factor(f"obs_{var}", log_prob)

    return model, param_names
