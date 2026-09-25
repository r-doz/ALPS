"""
param_extractor.py
==================
Given a DeGAS program string, produces:
  - a rewritten program where optimizable numeric literals are replaced
    by named placeholders (prefixed with _)
  - a dict mapping placeholder names (without _) to their initial float values

Optimizable parameters
----------------------
  gm([w,...], [mu,...], [sigma,...])
      mu_i    → named  mu1, mu2, ...   (means,  any real)
      sigma_i → named  sigma1, sigma2, ... (stds, must stay > 0)
      weights → left as literals (simplex constraint; handled separately)

  uniform([start, end], 2)
      start → named  unif_start1, unif_start2, ...
      end   → named  unif_end1,   unif_end2, ...

Everything else (condition thresholds, scalar coefficients in assignments)
is left untouched.
"""

import re


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

NUM = r"-?\d+\.\d+"   # matches any 2-decimal float, including negative


def _parse_list(s: str) -> list[str]:
    """Return the comma-separated tokens inside '[...]', stripped."""
    inner = re.match(r"\[([^\]]*)\]", s.strip())
    if not inner:
        raise ValueError(f"Expected a bracketed list, got: {s!r}")
    return [t.strip() for t in inner.group(1).split(",")]


# ---------------------------------------------------------------------------
# core replacer
# ---------------------------------------------------------------------------

def extract_params(program: str) -> tuple[str, dict[str, float]]:
    """
    Parameters
    ----------
    program : str
        A valid DeGAS program.

    Returns
    -------
    rewritten : str
        Same program with optimizable numbers replaced by _name placeholders.
    params : dict[str, float]
        {name: initial_value} for every extracted parameter (keys have no _).
    """
    params: dict[str, float] = {}
    mu_count = 0
    sigma_count = 0
    unif_count = 0   # one index shared across start/end pairs for clarity

    # ---- replace gm([w,...], [mu,...], [sigma,...]) -------------------------
    def replace_gm(m: re.Match) -> str:
        nonlocal mu_count, sigma_count

        weights_str, means_str, stds_str = m.group(1), m.group(2), m.group(3)

        means  = _parse_list(means_str)
        stds   = _parse_list(stds_str)

        new_means = []
        for v in means:
            mu_count += 1
            name = f"mu{mu_count}"
            params[name] = float(v)
            new_means.append(f"_{name}")

        new_stds = []
        for v in stds:
            sigma_count += 1
            name = f"sigma{sigma_count}"
            params[name] = float(v)
            new_stds.append(f"_{name}")

        # weights stay literal
        return (
            f"gm({weights_str}, "
            f"[{', '.join(new_means)}], "
            f"[{', '.join(new_stds)}])"
        )

    # Pattern: gm( [weights] , [means] , [stds] )
    # Each list may contain multiple comma-separated numbers.
    LIST_PAT = r"(\[[^\]]*\])"
    gm_pat = rf"gm\s*\(\s*{LIST_PAT}\s*,\s*{LIST_PAT}\s*,\s*{LIST_PAT}\s*\)"
    rewritten = re.sub(gm_pat, replace_gm, program)

    # # ---- replace uniform([start, end], 2) ----------------------------------
    # def replace_uniform(m: re.Match) -> str:
    #     nonlocal unif_count
    #     unif_count += 1

    #     start_str, end_str = m.group(1), m.group(2)
    #     start_name = f"unif_start{unif_count}"
    #     end_name   = f"unif_end{unif_count}"

    #     params[start_name] = float(start_str)
    #     params[end_name]   = float(end_str)

    #     return f"uniform([_{start_name}, _{end_name}], 2)"

    # uniform_pat = rf"uniform\s*\(\s*\[\s*({NUM})\s*,\s*({NUM})\s*\]\s*,\s*2\s*\)"
    # rewritten = re.sub(uniform_pat, replace_uniform, rewritten)

    return rewritten, params


# ---------------------------------------------------------------------------
# pretty printer
# ---------------------------------------------------------------------------

def format_params(params: dict[str, float]) -> str:
    lines = [f"  {k}: {v}" for k, v in params.items()]
    return "{\n" + ",\n".join(lines) + "\n}"


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    program = """
c = uniform([0.00, 1.00], 2);
if c > 0.50 {
  a = gm([1.00], [2.00], [1.50]);
  b = gm([1.00], [6.00], [4.50]);
} else {
  a = gm([1.00], [-1.00], [2.50]);
  b = gm([1.00], [2.00], [6.50]);
} end if;
""".strip()

    print("=== Original program ===")
    print(program)
    print()

    rewritten, params = extract_params(program)

    print("=== Rewritten program ===")
    print(rewritten)
    print()

    print("=== Parameter dict ===")
    print(format_params(params))

    # ---- second example: multi-component gm --------------------------------
    program2 = """
a = gm([0.60, 0.40], [2.00, -1.50], [0.50, 0.80]);
b = uniform([-5.00, 5.00], 2);
c = 2.00*a;
c = c + b;
""".strip()

    print()
    print("=== Original program 2 ===")
    print(program2)
    print()

    rewritten2, params2 = extract_params(program2)

    print("=== Rewritten program 2 ===")
    print(rewritten2)
    print()

    print("=== Parameter dict 2 ===")
    print(format_params(params2))