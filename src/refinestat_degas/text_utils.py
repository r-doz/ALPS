"""Small text-level helpers shared between refine_loop.py (gradient inference) and
mcmc_diagnostics.py (MCMC inference) -- kept in their own module to avoid a circular import
between the two (each needs the other for its respective fit_and_diagnose* function)."""

import re

_UNIFORM_TRAILING_PAT = re.compile(r"(uniform\(\s*\[[^\]]*\]\s*,\s*)2\.00(\s*\))")


def fix_uniform_trailing_literal(text: str) -> str:
    """DeGAS's own ANTLR grammar parses uniform(...)'s trailing literal via
    `int(NUM().getText())` (DeGAS/src/SOGAParser.py), which requires a bare integer "2",
    not the 2-decimal "2.00" degas.lark forces the LLM to write (see degas.lark's comment
    on the uniform rule for why the grammar can't just require a bare "2" itself). By the
    time this is called DegasChecker has already verified the trailing value is exactly
    2.00, so this substitution is always safe."""
    return _UNIFORM_TRAILING_PAT.sub(r"\g<1>2\2", text)
