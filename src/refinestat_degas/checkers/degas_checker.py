"""Semantic checker for generated DeGAS programs.

Analogous to RefineStat's PyMCChecker (refinestat/refinegen/checkers/library_specific/
ppl_checker.py), but DeGAS is a standalone DSL, not Python, so there is no ast/importlib
reflection to reuse. Instead each of RefineStat's six validation predicates is checked
directly against the DeGAS grammar's Lark parse tree:

1. Syntactic correctness -> parsing with the degas.lark grammar (Lark, same grammar used
   for SynCode's DFA mask store).
2. Distribution validity  -> already fully enforced by the grammar itself (only `gm`/
   `uniform` constructors are reachable), nothing extra to check here.
3. Parameter validity     -> gm's three literal lists must have equal length.
4. Support validity       -> gm weights >0 and sum to 1.00, sigma_i>0, uniform bounds
   start<end.
5. Dependency validity    -> every variable referenced (RHS, bexpr) must already have been
   assigned earlier in the program.
6. Type/shape validity    -> the one legal-form asymmetry the grammar doesn't encode:
   `var*var` may only appear as a standalone statement, never combined with `+/- atom`
   (see DEGAS_GRAMMAR's enumerated legal forms in src/helpers/generation_prompt.py).

`finished()` reports whether every one of the benchmark's dataset variables has been
assigned at least once -- DeGAS has no `pm.sample()`-style completion marker to key off,
so program-completeness is defined by data-variable coverage instead.
"""

from __future__ import annotations

import os
from functools import lru_cache

from lark import Lark, Tree, Token
from lark.exceptions import LarkError

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "..", "grammar", "degas.lark")

WEIGHT_SUM_TOL = 0.02  # 2-decimal literals can be off by rounding, e.g. 0.33+0.33+0.34
NUM_TOL = 1e-9


@lru_cache(maxsize=1)
def _get_parser() -> Lark:
    with open(_GRAMMAR_PATH) as f:
        grammar_text = f.read()
    return Lark(grammar_text, parser="earley")


class CheckFailure(Exception):
    pass


def _signed_num_value(node: Tree) -> float:
    """node is a `pos_num`/`neg_num` tree (see grammar/degas.lark's signed_num rule)."""
    if node.data == "pos_num":
        return float(node.children[0])
    if node.data == "neg_num":
        return -float(node.children[-1])
    raise ValueError(f"not a signed_num node: {node.data}")


def _numlist_values(numlist_tree: Tree) -> list[float]:
    return [_signed_num_value(c) for c in numlist_tree.children]


def _var_refs(node) -> list[str]:
    """All SYMVAR tokens referenced (read, not assigned) inside an expression subtree."""
    if isinstance(node, Token):
        return [str(node)] if node.type == "SYMVAR" else []
    if isinstance(node, Tree):
        if node.data == "var_atom":
            return [str(node.children[0])]
        if node.data == "var_var_mult":
            return [str(node.children[0]), str(node.children[1])]
        if node.data == "num_var_mult":
            return [str(node.children[1])]
        refs: list[str] = []
        for child in node.children:
            refs.extend(_var_refs(child))
        return refs
    return []


def _check_gm(gm_tree: Tree) -> None:
    weights, mus, sigmas = (_numlist_values(c) for c in gm_tree.children)
    if not (len(weights) == len(mus) == len(sigmas)):
        raise CheckFailure(
            f"gm(...) lists have mismatched lengths: "
            f"{len(weights)} weights, {len(mus)} means, {len(sigmas)} sigmas"
        )
    if any(w <= 0 for w in weights):
        raise CheckFailure(f"gm(...) weights must be > 0, got {weights}")
    if abs(sum(weights) - 1.0) > WEIGHT_SUM_TOL:
        raise CheckFailure(f"gm(...) weights must sum to 1.00, got sum={sum(weights):.4f}")
    if any(s <= 0 for s in sigmas):
        raise CheckFailure(f"gm(...) sigmas must be > 0, got {sigmas}")


def _check_uniform(uniform_tree: Tree) -> None:
    bounds_tree, trailing_tree = uniform_tree.children
    start, end = _numlist_values(bounds_tree)
    if not start < end:
        raise CheckFailure(f"uniform([start,end],2) requires start<end, got [{start},{end}]")

    trailing = _signed_num_value(trailing_tree)
    if trailing != 2.0:
        raise CheckFailure(f"uniform(...)'s trailing literal must be exactly 2.00, got {trailing:.2f}")


def _check_dists_in(node) -> None:
    if isinstance(node, Tree):
        if node.data == "gm":
            _check_gm(node)
        elif node.data == "uniform":
            _check_uniform(node)
        for child in node.children:
            _check_dists_in(child)


def _check_type_shape(rhs_node: Tree) -> None:
    """var*var may only be a standalone statement, never combined with +/- atom."""
    if rhs_node.data in ("add_multerm_left", "add_multerm_right"):
        multerm = rhs_node.children[0] if rhs_node.data == "add_multerm_right" else rhs_node.children[1]
        if isinstance(multerm, Tree) and multerm.data == "var_var_mult":
            raise CheckFailure(
                "var*var may only appear as a standalone statement, not combined with +/- atom"
            )


class DegasChecker:
    """
    code: full program text assembled so far (all statements generated up to and
        including the unit that was just produced).
    symboltable: dict mutated in place, var_name -> True once assigned. Shared across
        calls within one candidate's generation so `finished()`/dependency checks see
        prior statements.
    var_names: the benchmark's dataset variable names (get_var_names(program)).
    """

    def __init__(self, code: str, symboltable: dict, var_names: list[str], generator=None, unit_name: str = "statement"):
        self.code = code
        self.symboltable = symboltable
        self.var_names = list(var_names)
        self.generator = generator
        self.unit_name = unit_name
        self.error: str | None = None

    def check(self) -> bool:
        self.error = None
        if not self.code.strip():
            return True
        try:
            tree = _get_parser().parse(self.code)
        except LarkError as e:
            self.error = f"syntax error: {str(e)[:200]}"
            return False

        try:
            self._check_semantics(tree)
        except CheckFailure as e:
            self.error = str(e)
            return False
        return True

    def _check_semantics(self, tree: Tree) -> None:
        assigned: dict[str, bool] = {}
        # tree.children are `toplevel` nodes, each wrapping exactly one `statement`
        top_statements = [toplevel.children[0] for toplevel in tree.children]
        self._walk_statements(top_statements, assigned)
        self.symboltable.clear()
        self.symboltable.update(assigned)

    def _walk_statements(self, statements, assigned: dict[str, bool]) -> None:
        for stmt in statements:
            inner = stmt.children[0]
            if inner.data == "assignment":
                self._check_assignment(inner, assigned)
            elif inner.data == "conditional":
                self._check_conditional(inner, assigned)

    def _check_assignment(self, assignment: Tree, assigned: dict[str, bool]) -> None:
        lhs = str(assignment.children[0])
        rhs = assignment.children[1]

        for ref in _var_refs(rhs):
            if ref not in assigned:
                raise CheckFailure(f"'{ref}' used before assignment (dependency validity)")

        _check_dists_in(rhs)
        if isinstance(rhs, Tree):
            _check_type_shape(rhs)

        assigned[lhs] = True

    def _check_conditional(self, conditional: Tree, assigned: dict[str, bool]) -> None:
        bexpr, ifblock, elseblock = conditional.children
        cond_var = str(bexpr.children[0])
        if cond_var not in assigned:
            raise CheckFailure(f"'{cond_var}' used in condition before assignment")

        if_assigned = dict(assigned)
        self._walk_statements(ifblock.children, if_assigned)
        else_assigned = dict(assigned)
        self._walk_statements(elseblock.children, else_assigned)

        # a variable is considered assigned after the conditional only if both branches
        # assign it (conservative dependency validity for anything used afterward)
        for var in set(if_assigned) & set(else_assigned):
            if var not in assigned:
                assigned[var] = True

    def finished(self) -> bool:
        return all(v in self.symboltable for v in self.var_names)
