
"""
mutation_prompt.py
==================
Builds the mutation prompt for the ALPS pipeline.

The prompt is designed to push the LLM toward *structural* mutations
(number of components, conditionals, dependencies) rather than numeric
tweaks — which are the gradient optimizer's job.
"""


def _format_program_block(rank: int, entry: dict, show_history: bool = True, show_heuristic: bool = True) -> str:
    """Format a single program entry for the prompt.

    `show_history=False` (the ALPS_PROMPT_HISTORY=0 ablation) strips everything that
    reflects this candidate's optimization trajectory -- the loss/parameter before→after
    deltas and the delta-derived "suggested mutation focus" hint -- leaving only its
    current state (hypothesis, current loss, current parameter values, program text).

    `show_heuristic=False` (the ALPS_PROMPT_HEURISTIC=0 ablation) is more targeted: it keeps
    the loss/parameter deltas (show_history's own effect) but drops only the "suggested
    mutation focus" line derived from them, isolating whether the heuristic itself adds
    anything beyond the raw numbers it's computed from. Has no effect when show_history=False,
    since there is no heuristic line to drop in that case either.
    """

    def _fmt_params(initial: dict, optimized: dict, has_optimization: bool) -> str:
        """Show initial → final for each parameter, with delta (or just the current
        value, if show_history is False)."""
        lines = []
        for name, init_val in initial.items():
            opt_tensor = optimized.get(name) if has_optimization else None
            if opt_tensor is None:
                lines.append(f"    {name}: {init_val:.4f}")
                continue
            opt_val = float(opt_tensor)
            if not show_history:
                lines.append(f"    {name}: {opt_val:.4f}")
                continue
            delta = opt_val - float(init_val)
            arrow = f"{init_val:.4f} → {opt_val:.4f}  (Δ {delta:+.4f})"
            lines.append(f"    {name}: {arrow}")
        return "\n".join(lines)

    initial_loss = entry.get("initial_loss", None)
    final_loss   = entry.get("final_loss", None)
    loss_delta   = (final_loss - initial_loss) if (initial_loss is not None and final_loss is not None) else None

    if not show_history:
        loss_line = f"  loss: {final_loss:.4f}" if final_loss is not None else ""
    elif initial_loss is not None and final_loss is not None:
        loss_line = (
            f"  loss: {initial_loss:.4f} → {final_loss:.4f}"
            f"  (Δ {loss_delta:+.4f})"
        )
    elif final_loss is not None:
        loss_line = f"  loss: {final_loss:.4f}"
    else:
        loss_line = ""

    params_block = _fmt_params(
        entry.get("params", {}),
        entry.get("optimized_params", {}),
        has_optimization=final_loss is not None,
    )

    if show_history:
        heuristic_line = ""
        if show_heuristic:
            # Deterministic mutation-strategy hint based on Δloss / Δparams
            strategy_hint = _suggest_strategy(entry, loss_delta)
            heuristic_line = f"  suggested mutation focus: {strategy_hint}\n"
        return (
            f"### Program {rank}  [id={entry['id']}]\n"
            f"  hypothesis: {entry['hypothesis']}\n"
            f"{loss_line}\n"
            f"  parameters (initial → after gradient):\n"
            f"{params_block}\n"
            f"{heuristic_line}"
            f"  program:\n"
            f"    {entry['program']}\n"
        )

    return (
        f"### Program {rank}  [id={entry['id']}]\n"
        f"  hypothesis: {entry['hypothesis']}\n"
        f"{loss_line}\n"
        f"  parameters (current, after gradient):\n"
        f"{params_block}\n"
        f"  program:\n"
        f"    {entry['program']}\n"
    )


def _suggest_strategy(entry: dict, loss_delta: float | None) -> str:
    """Deterministically suggest a mutation strategy based on loss/param deltas.

    This is a heuristic hint shown to the LLM, not a hard constraint —
    it grounds the generic mutation menu in this specific program's
    diagnostics so the LLM doesn't apply the same boilerplate to every entry.
    """
    final_loss = entry.get("final_loss")
    initial = entry.get("params", {})
    optimized = entry.get("optimized_params", {})

    # No optimization info available (e.g. brand-new program)
    if final_loss is None or loss_delta is None:
        return "not yet evaluated — treat as a baseline; mutate freely"

    # Compute max relative parameter movement
    max_rel_move = 0.0
    for name, init_val in initial.items():
        opt_val = optimized.get(name)
        if opt_val is None or init_val == 0:
            continue
        rel_move = abs(float(opt_val) - float(init_val)) / (abs(float(init_val)) + 1e-6)
        max_rel_move = max(max_rel_move, rel_move)

    near_zero_delta = abs(loss_delta) < 0.01
    large_param_move = max_rel_move > 0.3

    if near_zero_delta and final_loss > 5.0:
        return ("near-zero Δloss with high loss — the structure itself is likely "
                "a poor fit; consider re-initialising with a different hypothesis "
                "rather than tweaking this one")
    if large_param_move:
        return ("large parameter movement during optimisation — the LLM's initial "
                "guess was far off; consider whether this signals a missing "
                "component (e.g. split a `gm` into a mixture) near the shifted "
                "parameter(s)")
    if near_zero_delta:
        return ("small Δloss and small parameter movement — this program is stable; "
                "try a more exploratory structural change (conditional, dependency) "
                "rather than a small tweak")
    return ("moderate improvement from gradient descent — structure seems reasonable; "
            "try a targeted refinement (e.g. add/remove one component or branch)")


def _format_failure_history(history: list[dict] | None) -> str:
    """Format a compact summary of previously rejected mutations.

    Parameters
    ----------
    history : list[dict] or None
        Each entry should have at least:
          - "parent_id": id of the program it was derived from
          - "structure": structure tag (e.g. "mixture2")
          - "hypothesis": one-line hypothesis
          - "parent_loss": loss of the parent program at time of mutation
          - "result_loss": loss of the mutated program (None if it failed validation)
    """
    if not history:
        return "  (no rejected mutations recorded yet)\n"

    lines = []
    for h in history:
        parent_loss = h.get("parent_loss")
        result_loss = h.get("result_loss")
        if result_loss is None:
            outcome = "FAILED (invalid program / did not evaluate)"
        else:
            diff = result_loss - parent_loss if parent_loss is not None else None
            if diff is not None:
                outcome = f"loss {result_loss:.4f} (Δ vs parent {diff:+.4f}, did not improve)"
            else:
                outcome = f"loss {result_loss:.4f} (did not improve)"
        lines.append(
            f"  - from program [id={h.get('parent_id')}], "
            f"structure='{h.get('structure')}', "
            f"hypothesis: \"{h.get('hypothesis')}\" -> {outcome}"
        )
    return "\n".join(lines) + "\n"


def build_mutation_prompt(
    pool: list[dict],
    n_mutations: int = 5,
    iteration: int = 1,
    grammar: str = "DeGAS grammar as specified in the system prompt",
    rejected_history: list[dict] | None = None,
    show_history: bool = True,
    show_heuristic: bool = True,
) -> str:
    """
    Build the user-turn mutation prompt.

    Parameters
    ----------
    pool : list[dict]
        The current best programs, sorted best-first (lowest loss first).
        Each entry is a program dict as produced by the ALPS pipeline.
    n_mutations : int
        How many mutated programs to request. Should normally equal len(pool)
        so each pool program gets exactly one mutation.
    iteration : int
        Current iteration number (for context and id assignment).
    grammar : str
        The grammar specification for the DeGAS language.
    rejected_history : list[dict], optional
        Previously proposed mutations that did not improve on their parent
        (or failed validation). Used to steer the LLM away from repeating
        unsuccessful structural changes. See `_format_failure_history`.
    show_history : bool
        When False (the ALPS_PROMPT_HISTORY=0 ablation), strips each candidate's
        optimization trajectory (loss/parameter before→after deltas, the delta-derived
        "suggested mutation focus" hint, and the "What the numbers mean" section that
        explains them) from the prompt, leaving only each candidate's current state.
    show_heuristic : bool
        When False (the ALPS_PROMPT_HEURISTIC=0 ablation), keeps the loss/parameter deltas
        but drops only the "suggested mutation focus" hint derived from them -- a narrower
        ablation than show_history, isolating whether the heuristic itself helps beyond the
        raw numbers. No effect when show_history is already False.

    Returns
    -------
    str
        The full user-turn prompt string.
    """
    pool_sorted = sorted(pool, key=lambda e: e.get("final_loss") or e.get("initial_loss") or float("inf"))

    program_blocks = "\n".join(
        _format_program_block(rank + 1, entry, show_history=show_history, show_heuristic=show_heuristic)
        for rank, entry in enumerate(pool_sorted)
    )

    failure_block = _format_failure_history(rejected_history)

    numbers_block = ""
    if show_history:
        numbers_block = """\
## What the numbers mean

- **loss**: negative log-likelihood per datapoint (lower = better fit).
- **Δ loss**: how much gradient optimization improved the program. \
A large negative Δ means the structure was promising but parameters needed tuning. \
A near-zero Δ means the structure may be a poor fit regardless of parameter values.
- **parameter Δ**: how much each parameter moved during gradient optimization. \
Large moves mean the LLM's initial value was far from the optimum — the structure \
is acceptable but the initialisation was poor. \
Small moves mean the parameter is either well-initialised or insensitive.
"""
        if show_heuristic:
            numbers_block += """\
- **suggested mutation focus**: a hint derived directly from this program's own \
loss and parameter deltas — use it to ground your mutation in this program's \
specific diagnostics rather than a generic strategy.
"""
        numbers_block += "\n"

    # Deterministic id assignment: ids for this iteration are
    # [iteration * n_mutations + 1, ..., iteration * n_mutations + n_mutations].
    # This avoids leaving id arithmetic to the LLM (see note below).
    id_start = iteration * n_mutations + 1
    id_end = id_start + n_mutations - 1
    assigned_ids = list(range(id_start, id_end + 1))

    # Map each pool program to one assigned id, in order (best program first).
    parent_ids = [entry["id"] for entry in pool_sorted[:n_mutations]]
    id_mapping_lines = "\n".join(
        f"  - parent_id={pid} (Program {rank + 1} above)  ->  use id={new_id}"
        for rank, (pid, new_id) in enumerate(zip(parent_ids, assigned_ids))
    )

    prompt = f"""\
You are helping improve a set of probabilistic programs that model a dataset.
This is iteration {iteration} of a synthesis-and-refinement loop, in which at each step you propose {n_mutations} program and I 
optimize some of their parameters with gradient descent and report back the final loss and parameter values.

## Current best programs (ranked by loss, lower is better)

{program_blocks}

{numbers_block}## Previously rejected mutations (do not repeat these)

The following structural mutations were already tried and did NOT improve on \
their parent program (or were invalid). Avoid proposing the same structure + \
hypothesis combination again for the same parent:

{failure_block}

## Your task

For EACH program in the pool below, propose EXACTLY ONE mutated program that is \
likely to achieve a **lower loss** than that program. You must produce \
{n_mutations} mutated programs total, one per parent, using the id mapping below:

{id_mapping_lines}

Focus on **structural mutations** — the gradient optimizer will handle numeric \
fine-tuning after you. Avoid simply changing numbers.

Useful structural mutations to consider{" (use the \"suggested mutation focus\" for each program to pick the most relevant ones)" if (show_history and show_heuristic) else ""}:
- **Split a component**: replace a single-component `gm` with a 2- or 3-component mixture \
if the gradient moved its mean or sigma a lot (the data may be multimodal there).
- **Merge components**: if two components of a mixture have similar optimized means, \
collapse them into one.
- **Add a conditional**: introduce an `if/else` branch on an existing variable to \
capture subpopulations with different behaviour.
- **Remove a conditional**: if both branches of an existing `if/else` converged to \
similar parameter values, flatten them into a single distribution.
- **Change dependency structure**: introduce or remove intermediate variables \
(scaled sums, products) to capture correlations between the variables.
- **Re-initialise a poor program**: if a program has near-zero Δ loss and high loss, \
replace it with a structurally different hypothesis.

## Diversity requirement

Across the {n_mutations} mutated programs you return, use at least 2 distinct \
`structure` tags. Do not return {n_mutations} programs that are all minor \
variations of the same structural idea.

All the mutated programs must adhere to the following grammar rules:
{grammar}

## Other Rules
- Do NOT put underscores or other characters in variable names.
- You may reuse and modify the best-performing programs as starting points, \
or generate entirely new structures.
- Do NOT copy a program unchanged.
- Initialise numeric parameters to reasonable values — the gradient will refine them, \
but a good starting point helps.

## Output format

Reply ONLY with a valid JSON array of {n_mutations} objects. \
No prose, no markdown fences. Return a list of dictionaries, each has exactly these fields:

  "id": <integer, use the exact id assigned to this parent in the mapping above>,
  "parent_id": <integer, the id of the program this mutation was derived from>,
  "hypothesis": "<one sentence describing the generative assumption>",
  "structure":  "<short tag, e.g. mixture2 / conditional / hierarchical>",
  "program":    "<valid DeGAS program as a single string>"
"""
    return prompt


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate the pool from the document provided
    pool = [
        {
            "id": 1,
            "hypothesis": "Data is unimodal and roughly symmetric for both variables.",
            "structure": "unimodal",
            "program": "a = gm([1.00], [1.02], [2.07]); b = gm([1.00], [4.24], [5.83]);",
            "params": {"mu1": 1.02, "sigma1": 2.07, "mu2": 4.24, "sigma2": 5.83},
            "optimized_params": {
                "mu1": 1.0182, "sigma1": 2.0652, "mu2": 4.2440, "sigma2": 5.8320
            },
            "initial_loss": 5.3264,
            "final_loss": 5.3264,
        },
        {
            "id": 2,
            "hypothesis": "Data consists of two distinct subpopulations with different means and variances.",
            "structure": "mixture2",
            "program": "a = gm([0.50, 0.50], [0.50, 2.00], [1.00, 3.00]); b = gm([0.50, 0.50], [2.00, 6.00], [2.00, 6.00]);",
            "params": {"mu1": 0.5, "mu2": 2.0, "sigma1": 1.0, "sigma2": 3.0,
                       "mu3": 2.0, "mu4": 6.0, "sigma3": 2.0, "sigma4": 6.0},
            "optimized_params": {
                "mu1": 0.6029, "mu2": 1.5573, "sigma1": 1.4093, "sigma2": 2.5530,
                "mu3": 1.5351, "mu4": 5.5584, "sigma3": 2.4984, "sigma4": 5.8595
            },
            "initial_loss": 5.9339,
            "final_loss": 5.7470,
        },
        {
            "id": 3,
            "hypothesis": "Data is multi-modal due to at least three distinct subpopulations.",
            "structure": "mixture3",
            "program": "a = gm([0.34, 0.33, 0.33], [-2.00, 1.00, 3.00], [1.00, 2.00, 1.00]); b = gm([0.34, 0.33, 0.33], [0.00, 4.00, 8.00], [2.00, 4.00, 2.00]);",
            "params": {"mu1": -2.0, "mu2": 1.0, "mu3": 3.0, "sigma1": 1.0, "sigma2": 2.0, "sigma3": 1.0,
                       "mu4": 0.0, "mu5": 4.0, "mu6": 8.0, "sigma4": 2.0, "sigma5": 4.0, "sigma6": 2.0},
            "optimized_params": {
                "mu1": -1.5390, "mu2": 1.0854, "mu3": 2.6263,
                "sigma1": 1.3567, "sigma2": 1.5466, "sigma3": 1.4434,
                "mu4": -0.4813, "mu5": 3.5427, "mu6": 8.0557,
                "sigma4": 2.4717, "sigma5": 4.4546, "sigma6": 1.4923
            },
            "initial_loss": 5.5318,
            "final_loss": 5.1070,
        },
        {
            "id": 4,
            "hypothesis": "Latent variable determines the mean of both observed variables through conditional statements.",
            "structure": "conditional",
            "program": "c = uniform([-4.88, 10.55], 2); if c > 3.00 { a = gm([1.00], [2.50], [1.50]); b = gm([1.00], [6.00], [3.00]); } else { a = gm([1.00], [-1.00], [2.50]); b = gm([1.00], [2.00], [4.00]); } end if;",
            "params": {"mu1": 2.5, "sigma1": 1.5, "mu2": 6.0, "sigma2": 3.0,
                       "mu3": -1.0, "sigma3": 2.5, "mu4": 2.0, "sigma4": 4.0},
            "optimized_params": {
                "mu1": 2.1782, "sigma1": 1.4790, "mu2": 6.5080, "sigma2": 2.4796,
                "mu3": -0.7693, "sigma3": 2.0008, "mu4": 1.5166, "sigma4": 4.4575
            },
            "initial_loss": 5.1843,
            "final_loss": 4.9151,
        },
        {
            "id": 5,
            "hypothesis": "Both variables are products of two underlying distributions (hierarchical).",
            "structure": "hierarchical",
            "program": "d = gm([1.00], [0.00], [1.00]); e = gm([1.00], [1.00], [2.00]); f = 0.50 * d; g = 0.50 * e; a = f + g; h = gm([1.00], [4.00], [2.00]); i = gm([1.00], [2.00], [3.00]); j = 0.25 * h; k = 0.25 * i; b = j + k;",
            "params": {"mu1": 0.0, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0,
                       "mu3": 4.0, "sigma3": 2.0, "mu4": 2.0, "sigma4": 3.0},
            "optimized_params": {
                "mu1": 0.0, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0,
                "mu3": 4.0, "sigma3": 2.0, "mu4": 2.0, "sigma4": 3.0
            },
            "initial_loss": None,
            "final_loss": None,
        },
    ]

    # Example rejected history from a previous iteration
    rejected_history = [
        {
            "parent_id": 1,
            "structure": "mixture2",
            "hypothesis": "Variable a is bimodal with components near 0 and 4.",
            "parent_loss": 5.3264,
            "result_loss": 5.4012,
        },
    ]

    prompt = build_mutation_prompt(pool, n_mutations=5, iteration=1, rejected_history=rejected_history)
    print(prompt)