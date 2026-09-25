def make_init_prompt(
    data_stats: dict, n_programs: int = 5,
    extra_hypotheses: list[str] | None = None,
    hypothesis_slots: dict[int, str] | None = None,
) -> str:
    """
    User message for initial population generation.

    data_stats example:
        {"var_names": ["a", "b"], "n": 340, "mean": 8.4, "std": 9.1, "min": 0.1, "max": 47.3,
         "skewness": 2.1, "kurtosis": 6.4, "sample": [0.3, 1.1, 8.2]}

    extra_hypotheses: optional list of extra dependency hypotheses (e.g. "num depends
    primarily on sex") appended as additional structure-target slots beyond the 5 canonical
    ones, each asking the LLM to write a candidate assuming that specific dependency. Defaults
    to None (no change from the standard 5-slot prompt).

    hypothesis_slots: optional dict mapping a 1-indexed slot number to a dependency hypothesis
    that REPLACES that slot's canonical structure type in place (n_programs unchanged), e.g.
    {2: "num depends primarily on thal", 3: "num depends primarily on chol"} to swap out the
    mixture2/mixture3 slots specifically while keeping unimodal/conditional/hierarchical.
    """
    stats_lines = "\n".join(f"  {k}: {v}" for k, v in data_stats.items())

    structure_targets = [
        "unimodal      — single gm component, match the data mean and std",
        "mixture2      — two-component gm, one per apparent subpopulation",
        "mixture3      — three-component gm for skewed or multi-modal data",
        "conditional   — if/else branching",
        "hierarchical  — product of two distributions (use temp variable)",
    ]
    if extra_hypotheses:
        structure_targets = structure_targets + [
            f"hypothesis    — assume {h}; encode this dependency structurally (e.g. via if/else "
            f"or a shared temp variable)" for h in extra_hypotheses
        ]
    if hypothesis_slots:
        for slot, h in hypothesis_slots.items():
            structure_targets[slot - 1] = (
                f"hypothesis    — assume {h}; encode this dependency structurally (e.g. via "
                f"if/else or a shared temp variable)"
            )
    targets = "\n".join(
        f"  {i+1}. {t}" for i, t in enumerate(structure_targets[:n_programs])
    )
 
    return f"""Data summary:
{stats_lines}
 
Generate exactly {n_programs} DeGAS programs that are STRUCTURALLY DIVERSE and include all the variables of the dataset (var_names in the stats).
Aim for one program per structure type below:
{targets}
 
For each program:
  1. Write a one-sentence hypothesis about the data-generating process.
  2. Choose distribution families matching the data range and shape.
  3. Write valid DeGAS code — remember at most one * per line.
 
Return ONE JSON object with the exact shape below (no prose, no markdown):
{{
  "programs": [
    {{
      "id": 1,
      "hypothesis": "Data is unimodal and right-skewed.",
      "structure": "unimodal",
      "program": "..."
    }}
  ]
}}
"""

DEGAS_GRAMMAR = """
DEGAS Language Reference:
STRUCTURE
- Instructions end with ;
- Two instruction types: assignment (var = expr;) and conditional (if condition { program } else { program } end if;)
- Only dataset variables are usable (see var_names in the stats)

NUMBERS
- Decimals with at most 2 decimal places, range [-100.00, 100.00]
- Weights and standard deviations must be > 0

DISTRIBUTIONS
- gm([pi_1,...,pi_n], [mu_1,...,mu_n], [sigma_1,...,sigma_n])
  - All three lists same length, n >= 1
  - pi_i > 0 and sum(pi) = 1.00
  - sigma_i >= 0
  - mu_i any number in [-100, 100]
- uniform([start, end], 2)
  - start < end, both in [-100.00, 100.00]
  - the trailing 2 is a fixed literal, never change it

DISCRETE / BINARY VARIABLES
- If a variable only takes a small number of exact values in the data (e.g. a
  0/1 indicator), represent each value as its own gm(...) component with
  sigma_i = 0.00 -- an exact point mass at mu_i, not an approximation.
  Example: a variable that is 0 about 70% of the time and 1 about 30% of the
  time -> gm([0.70, 0.30], [0.00, 1.00], [0.00, 0.00])
- Do NOT use a small positive sigma (like 0.01) to approximate a discrete
  value -- only sigma_i = 0.00 is treated as an exact point mass; any nonzero
  sigma is scored as a continuous density, which is undefined behavior here.
- IMPORTANT: pi_i and mu_i of a sigma_i = 0.00 component are never refined
  afterward -- there is no gradient for a point mass in DeGAS. Set them as
  accurately as you can directly from the data summary (e.g. pi_i = the
  empirical fraction of observations at that exact value); they will not be
  corrected later.

ASSIGNMENTS (key constraint)
- var = expr;
- At most ONE multiplication (*) per line, no division allowed
- If a product has a number and a variable, the number must come first (3.00*b, not b*3.00)
- Legal forms: atom | number*var | number*dist | var*var | atom+atom | atom-atom | number*var + atom | number*var - atom | atom + number*var | atom - number*var | number*dist + atom | atom + number*dist
  (atom = var | number | distribution)
- Two products need a temp variable:
  INVALID: x = 2.00*y + 3.00*z;
  VALID:   var1 = 2.00*var2;  var1 = var1 + 3.00*var3;

CONDITIONALS
- if condition { program } else { program } end if;
- condition forms: var == number | var != number | lexpr < number | lexpr <= number | lexpr >= number | lexpr > number
- lexpr = var

COMMON MISTAKES TO AVOID
- Normal(0,1) syntax is invalid -> use gm([1.00],[0.00],[1.00])
- Weights must sum exactly to 1.00 at 2 decimal places
- No division (b/c is invalid)
- Number must come first in products (var1*3.00 -> 3.00*var1)
- No more than one multiplication per line -> split with a temp variable
- uniform([0,1]) is invalid -> must be uniform([0.00, 1.00], 2)
"""

DEGAS_GRAMMAR_LONG = """
## DeGAS Language Reference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PROGRAM STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instructions must end with ;
    instr1;
    instr2;
    instr3;

Two kinds of instruction:
    assignment    var = expr;
    conditional   if condition { 
                    program
                  } else {
                    program
                  } end if;

Variables: only the variables of the dataset are available.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 NUMBERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Any decimal with at most 2 decimal places in [-100.00, 100.00].
    OK:    0.75    -3.14    50.00    -0.01    100.00
    NOT:   1/3     0.125    1e-2     .5

Positive numbers (weights, standard deviations): must be > 0.
    OK:    0.01    0.50    1.00    3.14    99.99

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DISTRIBUTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Gaussian mixture
   ─────────────────────────────────────────
   gm([pi_1, ..., pi_n], [mu_1, ..., mu_n], [sigma_1, ..., sigma_n])

   Constraints:
     • All three lists must have the same length  n ≥ 1
     • pi_i > 0  and  sum(pi) = 1.00   (weights)
     • sigma_i > 0                      (standard deviations)
     • mu_i  any number in [-100, 100]  (means)

   Examples:
     gm([1.00], [0.00], [1.00])                            ← 1-component (Gaussian)
     gm([0.60, 0.40], [2.00, -1.50], [0.50, 0.80])        ← 2-component
     gm([0.30, 0.50, 0.20], [4.00, 0.00, -3.00], [0.50, 1.00, 0.50])  ← 3-component

2. Uniform
   ─────────────────────────────────────────
   uniform([start, end], 2)

   Constraints:
     • start < end
     • Both in [-100.00, 100.00]
     • The literal  2  is always the third element — do not change it

   Examples:
     uniform([0.00, 1.00], 2)
     uniform([-5.00, 5.00], 2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ASSIGNMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    var = expr;

expr must contain AT MOST ONE multiplication (*).
When a product has a number and a variable, the NUMBER MUST COME FIRST.
No division (/) is allowed. Only +, -, *.

Legal expression forms:

    atom                        a single value
    number * var                scaled variable    (number first)
    number * distribution       scaled sample      (number first)
    var * var                   product of two variables
    atom  +  atom               sum  (neither side is a product)
    atom  -  atom               difference
    number * var + atom       one product plus one atom
    number * var - atom
    atom + number * var
    atom - number * var
    number * dist + atom      same but with a distribution
    atom + number * dist

Where  atom  =  var | number | distribution

The key rule: a single assignment line may have at most one * .
If you need two products, introduce a temporary variable:

    INVALID:  a = 2.00*b + 3.00*c;
    VALID:    a = 2.00*b;  a = a + 3.00*c;

    INVALID:  a = 2.00*b*b;
    VALID:    a = b*b; a = 2.00*a;

    INVALID:  a = b*3.00;           (number must come first)
    VALID:    a = 3.00*b;

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CONDITIONALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  if condition { 
                    program
                  } else {
                    program
                  } end if;

Condition forms:
    var == number       var != number          (equality)
    lexpr  <  number    lexpr  <= number
    lexpr  >= number    lexpr  >  number       (comparison)

    lexpr = var

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 COMPLETE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1 — Single Gaussian
a = gm([1.00], [3.50], [1.20]);

# 2 — Two-component mixture
a = gm([0.70, 0.30], [4.00, -2.00], [0.50, 1.00]);

# 3 — Mixture with linear transform (one product per line)
a = gm([0.60, 0.40], [2.00, -1.00], [0.50, 0.80]);
b = 2.00*a;
c = b + 0.50;

# 4 — Three-component heavy-tailed mixtur
a = gm([0.20, 0.60, 0.20], [0.00, 0.00, 0.00], [5.00, 1.00, 0.30]);

# 5 — Conditional latent structure
a = uniform([0.00, 1.00], 2);
if a > 0.50 {
  b = gm([1.00], [5.00], [0.80]);
} else {
  b = gm([1.00], [-2.00], [0.60]);
} end if;

# 6 — Hierarchical: scale a mixture by a latent factor
a = gm([0.50, 0.50], [1.00, -1.00], [0.50, 0.50]);
b = gm([1.00], [0.00], [0.20]);
c = a*b;

# 7 — Two products needing a temp variable (pattern to follow)
a = gm([1.00], [0.00], [1.00]);
b = 2.00*a;
b = b + 3.00*a;

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 WHAT NOT TO WRITE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x a = Normal(0, 1);              → a = gm([1.00], [0.00], [1.00]);
x a = gm([0.5,0.5],[0,0],[1,1]); → a = gm([0.50, 0.50], [0.00, 0.00], [1.00, 1.00]);
x a = b/c;                       → division not allowed
x a = 2.00*b + 3.00*c;          → split: a = 2.00*b :: a = a + 3.00*c;
x a = b*3.00;                   → a = 3.00*b;
x a = 2.00*b*c;                 → split: a = b*c :: a = 2.00*a;
x uniform([0, 1]);               → uniform([0.00, 1.00], 2);
x weights: [0.33, 0.33, 0.34]   → use 2 decimals summing to 1: [0.34, 0.33, 0.33]
"""

SYSTEM_PROMPT = (
    "You are an expert in probabilistic programming and Bayesian modelling.\n"
    "You write programs exclusively in DeGAS, following the reference below.\n\n"
    + DEGAS_GRAMMAR
    + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Numbers: exactly 2 decimal places, range [-100.00, 100.00]
• gm weights must sum to exactly 1.00
• At most one * per assignment line
• Coefficient always before variable in products
• No division
• Do NOT put underscores or other characters in variable names, use the names of the dataset variables exactly as they appear in the stats
• Reply ONLY with a valid JSON object — no prose, no markdown fences
"""
)

