"""A lightweight, mutable representation of a DeGAS program's *structure*, used by the
grammar-MCMC baseline (src/grammar_mcmc) to sample and locally mutate program structures
without going through an LLM. Renders directly to DeGAS program text so it can be fed
through the exact same pipeline as the other two baselines: checkers.degas_checker for
validation and DeGAS's own compile/optimize/likelihood machinery for fitting.

Continuous parameters (gm weights/mu/sigma, uniform bounds) are part of this structure only
as *initial values* -- mcmc_search.py refits them via DeGAS's gradient optimizer after every
proposed structural mutation, the same way refine_loop.py's gradient backend does, rather
than having the MCMC chain itself search the continuous space.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

MU_RANGE = (-100.0, 100.0)
LOG_SIGMA_RANGE = (-1.3, 1.3)  # sigma sampled log-uniformly in ~[0.05, 20]; must stay well
# above 0.01 (the smallest representable value at 2dp) so rounding never produces sigma=0.00
UNIFORM_HALF_WIDTH_RANGE = (0.5, 50.0)
COND_THRESHOLDS = (0.00, 1.00, 2.00)  # matches this DSL's discrete-state convention
CMP_OPS = ("==", "!=")
N_COMPONENTS_RANGE = (1, 3)


def _round2(x: float) -> float:
    return round(x, 2)


def _sample_mu() -> float:
    return _round2(random.uniform(*MU_RANGE))


def _sample_sigma() -> float:
    return _round2(10 ** random.uniform(*LOG_SIGMA_RANGE))


def _sample_weights(n: int) -> list[float]:
    """Dirichlet-sample n weights, round to 2dp, and fix up rounding error on the last
    component so they sum to exactly 1.00 (DegasChecker's WEIGHT_SUM_TOL is only 0.02)."""
    raw = [random.gammavariate(1.0, 1.0) for _ in range(n)]
    total = sum(raw)
    weights = [_round2(max(0.01, w / total)) for w in raw]
    weights[-1] = _round2(1.00 - sum(weights[:-1]))
    if weights[-1] <= 0:
        # exceedingly unlikely with n<=3, but stay valid if rounding pushed it negative
        weights = [_round2(1.0 / n)] * n
        weights[-1] = _round2(1.00 - sum(weights[:-1]))
    return weights


@dataclass
class DistSpec:
    kind: str  # "gm" or "uniform"
    weights: list[float] = field(default_factory=list)  # gm only
    mus: list[float] = field(default_factory=list)  # gm only
    sigmas: list[float] = field(default_factory=list)  # gm only
    bounds: tuple[float, float] = (0.0, 1.0)  # uniform only

    @staticmethod
    def random(n_components: int | None = None) -> "DistSpec":
        if random.random() < 0.15:  # occasionally propose uniform
            center = random.uniform(*MU_RANGE)
            half_width = random.uniform(*UNIFORM_HALF_WIDTH_RANGE)
            lo, hi = _round2(center - half_width), _round2(center + half_width)
            if lo >= hi:
                hi = _round2(lo + 1.0)
            return DistSpec(kind="uniform", bounds=(lo, hi))
        n = n_components or random.randint(*N_COMPONENTS_RANGE)
        return DistSpec(
            kind="gm",
            weights=_sample_weights(n),
            mus=[_sample_mu() for _ in range(n)],
            sigmas=[_sample_sigma() for _ in range(n)],
        )

    def render(self) -> str:
        if self.kind == "uniform":
            lo, hi = self.bounds
            return f"uniform([{lo:.2f},{hi:.2f}],2.00)"
        w = ",".join(f"{x:.2f}" for x in self.weights)
        mu = ",".join(f"{x:.2f}" for x in self.mus)
        sigma = ",".join(f"{x:.2f}" for x in self.sigmas)
        return f"gm([{w}],[{mu}],[{sigma}])"


@dataclass
class StatementSpec:
    var: str
    dist: DistSpec | None = None  # set iff unconditional
    cond_var: str | None = None  # set iff conditional
    cond_op: str | None = None
    cond_threshold: float | None = None
    if_dist: DistSpec | None = None
    else_dist: DistSpec | None = None

    @property
    def is_conditional(self) -> bool:
        return self.cond_var is not None

    @staticmethod
    def random_unconditional(var: str) -> "StatementSpec":
        return StatementSpec(var=var, dist=DistSpec.random())

    @staticmethod
    def random_conditional(var: str, earlier_vars: list[str]) -> "StatementSpec":
        return StatementSpec(
            var=var,
            cond_var=random.choice(earlier_vars),
            cond_op=random.choice(CMP_OPS),
            cond_threshold=random.choice(COND_THRESHOLDS),
            if_dist=DistSpec.random(),
            else_dist=DistSpec.random(),
        )

    def render(self) -> str:
        if not self.is_conditional:
            return f"{self.var} = {self.dist.render()};"
        return (
            f"if {self.cond_var} {self.cond_op} {self.cond_threshold:.2f} {{\n"
            f"    {self.var} = {self.if_dist.render()};\n"
            f"}} else {{\n"
            f"    {self.var} = {self.else_dist.render()};\n"
            f"}} end if;"
        )


@dataclass
class ProgramSpec:
    statements: list[StatementSpec]

    def render(self) -> str:
        return "\n".join(s.render() for s in self.statements) + "\n"

    def copy(self) -> "ProgramSpec":
        import copy as _copy

        return _copy.deepcopy(self)

    @staticmethod
    def random(var_names: list[str], p_conditional: float = 0.3) -> "ProgramSpec":
        statements = []
        for i, var in enumerate(var_names):
            earlier = var_names[:i]
            if earlier and random.random() < p_conditional:
                statements.append(StatementSpec.random_conditional(var, earlier))
            else:
                statements.append(StatementSpec.random_unconditional(var))
        return ProgramSpec(statements=statements)
