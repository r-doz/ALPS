"""Structural mutation operators for the grammar-MCMC baseline's Metropolis-Hastings
proposals. Each mutation acts on one randomly chosen statement of a ProgramSpec and returns
a new (deep-copied) ProgramSpec -- continuous parameters are left as whatever the mutation
sets them to; mcmc_search.py refits them via gradient descent after every proposal.
"""

from __future__ import annotations

import random

from .program_spec import DistSpec, ProgramSpec, StatementSpec


def _resample_dist(spec: ProgramSpec, var_names: list[str]) -> ProgramSpec:
    """Fully resample one branch's distribution (may change gm<->uniform, component count,
    or all parameters)."""
    new = spec.copy()
    idx = random.randrange(len(new.statements))
    stmt = new.statements[idx]
    if stmt.is_conditional:
        if random.random() < 0.5:
            stmt.if_dist = DistSpec.random()
        else:
            stmt.else_dist = DistSpec.random()
    else:
        stmt.dist = DistSpec.random()
    return new


def _toggle_conditional(spec: ProgramSpec, var_names: list[str]) -> ProgramSpec:
    """Convert an unconditional statement to conditional (bringing in a fresh random
    condition + second branch), or collapse a conditional back to unconditional (keeping
    one branch, chosen at random, as the new bare distribution)."""
    new = spec.copy()
    idx = random.randrange(len(new.statements))
    stmt = new.statements[idx]
    earlier_vars = var_names[:idx]

    if stmt.is_conditional:
        kept = stmt.if_dist if random.random() < 0.5 else stmt.else_dist
        new.statements[idx] = StatementSpec(var=stmt.var, dist=kept)
    elif earlier_vars:
        new.statements[idx] = StatementSpec.random_conditional(stmt.var, earlier_vars)
    # else: no earlier vars to condition on (first variable) -- no-op, caller retries
    return new


def _resample_condition(spec: ProgramSpec, var_names: list[str]) -> ProgramSpec:
    """Resample just the cond_var/cond_op/threshold of an existing conditional statement,
    keeping both branch distributions."""
    new = spec.copy()
    conditional_idxs = [i for i, s in enumerate(new.statements) if s.is_conditional]
    if not conditional_idxs:
        return new  # no-op, caller retries
    idx = random.choice(conditional_idxs)
    stmt = new.statements[idx]
    earlier_vars = var_names[:idx]
    resampled = StatementSpec.random_conditional(stmt.var, earlier_vars)
    stmt.cond_var, stmt.cond_op, stmt.cond_threshold = (
        resampled.cond_var,
        resampled.cond_op,
        resampled.cond_threshold,
    )
    return new


_MUTATIONS = [_resample_dist, _toggle_conditional, _resample_condition]


def propose_mutation(spec: ProgramSpec, var_names: list[str], max_tries: int = 10) -> ProgramSpec:
    """Applies one randomly chosen structural mutation. Some mutations are no-ops in
    certain states (e.g. resample_condition with no conditional statements yet) -- retries
    with a different mutation choice up to max_tries before falling back to _resample_dist,
    which is always applicable."""
    for _ in range(max_tries):
        mutation = random.choice(_MUTATIONS)
        proposed = mutation(spec, var_names)
        if proposed.render() != spec.render():
            return proposed
    return _resample_dist(spec, var_names)
