import json
import csv
import os
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "../DeGAS/src")
from optimization import optimize, compile2SOGA, compile2SOGA_text, produce_cfg, produce_cfg_text, smooth_cfg, start_SOGA, initialize_params
from PROGRAMS.likelihood import compute_likelihood
import numpy as np
from helpers.param_extractor import extract_params
from helpers.mutation_prompt import build_mutation_prompt
import matplotlib.pyplot as plt
from helpers.llm_comunication import call_ollama
from helpers.dataset_generation import get_dataset, get_var_names
from helpers.generation_prompt import make_init_prompt, SYSTEM_PROMPT, DEGAS_GRAMMAR

# Hyperparameters
llm_model = os.environ.get("ALPS_LLM_MODEL", "gpt-oss:120b")
# Ablation: when set, skip the gradient-descent parameter-fitting step entirely -- candidates
# are scored at the LLM's own proposed literal parameter values, never refined.
skip_gradient = os.environ.get("ALPS_SKIP_GRADIENT", "0") == "1"
# Ablation: select/sort candidates by held-out NLL instead of the training NLL they were
# fit against -- mirrors RefineStat's own held_out_ok check, which ALPS's search otherwise has
# no equivalent of (every candidate is both fit AND selected on the same fixed data_array).
held_out_selection = os.environ.get("ALPS_HELD_OUT_SELECTION", "0") == "1"
train_frac = float(os.environ.get("ALPS_TRAIN_FRAC", "0.8"))
# Ablation: append explicit simplicity guidance (+ a deterministic-assignment shorthand
# example) to the system prompt, investigating whether asking the LLM directly for simpler
# structures reduces the complexity/GW2 gap found in the pick_best_tradeoff investigation.
simple_prompt = os.environ.get("ALPS_SIMPLE_PROMPT", "0") == "1"
# Ablation: strip the optimization-history fields (loss/param deltas, "suggested mutation
# focus") from the mutation prompt's per-candidate blocks, showing only each candidate's
# current state -- tests whether that history is helping or just adding noise/bias.
prompt_history = os.environ.get("ALPS_PROMPT_HISTORY", "1") == "1"
# Ablation: narrower than ALPS_PROMPT_HISTORY -- keeps the loss/parameter deltas but drops
# only the delta-derived "suggested mutation focus" hint, isolating whether the heuristic
# itself helps beyond the raw numbers it's computed from. No effect if prompt_history=False.
# (Tried as the default: improved NLL on 7/9 benchmarks and was a wash on GW2, but flipped
# Hurricane's NLL away from ALPS -- reverted to keep the heuristic on by default.)
prompt_heuristic = os.environ.get("ALPS_PROMPT_HEURISTIC", "1") == "1"
program = os.environ.get("ALPS_PROGRAM", "if")  # Options: "if", "mog1", "burglary", "csi", "easytugwar",
                         # "biasedtugwar", "mixedcondition", "multiplebranches", "eyecolor", "hurricane"
data_size = int(os.environ.get("ALPS_DATA_SIZE", "1000"))
# Ablation: no-mutation variant -- generate n_programs candidates once (init_opt_steps each) and
# stop, instead of iteratively mutating n_mutations of them for n_steps rounds. Overridable so the
# ablation doesn't need its own copy of this file; defaults reproduce the standard ALPS config.
n_programs = int(os.environ.get("ALPS_N_PROGRAMS", "5"))
# Optional extra dependency hypotheses appended to the initial-generation prompt as additional
# structure-target slots (e.g. "num depends primarily on sex"), pipe-separated. Defaults to
# none -- no effect on the standard prompt used by every other benchmark/run.
extra_hypotheses_raw = os.environ.get("ALPS_EXTRA_HYPOTHESES", "")
extra_hypotheses = [h.strip() for h in extra_hypotheses_raw.split("|") if h.strip()] or None
# Optional in-place slot replacements (e.g. "2:num depends primarily on thal|3:num depends
# primarily on chol") -- swaps specific canonical structure-type slots for a hypothesis
# without changing n_programs. Defaults to none.
hypothesis_slots_raw = os.environ.get("ALPS_HYPOTHESIS_SLOTS", "")
hypothesis_slots = {
    int(slot.strip()): hyp.strip()
    for slot, hyp in (pair.split(":", 1) for pair in hypothesis_slots_raw.split("|") if pair.strip())
} or None
n_mutations = 5
n_steps_cap = int(os.environ.get("ALPS_MAX_STEPS_CAP", "10"))  # ceiling for n_steps; default matches the standard 10-iteration cap
n_steps = min(int(os.environ.get("ALPS_N_STEPS", "10")), n_steps_cap)
# Early stopping: if the best loss hasn't improved by more than min_improvement over the last
# `patience` iterations, stop before using the full n_steps budget -- investigating whether
# this avoids the kind of late-iteration complexity creep found on mog1 (a mutation whose own
# hypothesis was "introduce an additional conditional", adding structure for a marginal NLL
# gain). min_improvement=0.01 matches mutation_prompt.py's own "near_zero_delta" threshold.
patience = int(os.environ.get("ALPS_PATIENCE", "3"))
min_improvement = float(os.environ.get("ALPS_MIN_IMPROVEMENT", "0.01"))
init_temperature = 0.2
mutation_temperature = 0.4
init_opt_steps = int(os.environ.get("ALPS_INIT_OPT_STEPS", "50"))
mutation_opt_steps = 100
extra_opt_steps = 200
learning_rate = 0.01
require_json = True
max_retries = 1
use_chat = True
request_timeout = 600
max_http_retries = 3

SIMPLICITY_APPENDIX = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PREFER SIMPLE MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- A deterministic value can also be written as a bare assignment, without gm(...):
    a = 1.00;
  This is equivalent to gm([1.00], [1.00], [0.00]) and is valid wherever a
  distribution is expected -- use it as a shorthand for an exact, single value.
- Prefer the simplest structure that plausibly fits the data:
  - Avoid deeply nested if/else -- more than 1-2 levels of nesting is rarely
    justified. A flatter structure that fits almost as well is usually better
    than a much more complex one that fits only marginally better.
  - Avoid gm(...) mixtures with many components (more than 3-4) unless the data
    clearly shows that many separate modes or subpopulations.
  - A small improvement in fit is not worth a much more complex program.
"""

effective_system_prompt = SYSTEM_PROMPT + SIMPLICITY_APPENDIX if simple_prompt else SYSTEM_PROMPT

# Results folder: results/<program>/<timestamp>/
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path(__file__).resolve().parent.parent / "results" / program / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

log_path = run_dir / "run_log.txt"
best_fitness_csv_path = run_dir / "best_fitness.csv"
plot_path = run_dir / "best_fitness.png"
hyperparams_path = run_dir / "hyperparams.json"
final_candidates_path = run_dir / "final_candidates.json"


def log_line(message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def sort_key(prog: dict):
    """Selection criterion for ranking/culling candidates. Normally the training NLL
    they were fit against (final_loss); under ALPS_HELD_OUT_SELECTION, the held-out NLL
    instead (held_out_loss), matching RefineStat's own held_out_ok check."""
    if held_out_selection:
        return prog.get("held_out_loss", prog["final_loss"])
    return prog["final_loss"]


import re

max_joint_components = int(os.environ.get("ALPS_MAX_JOINT_COMPONENTS", "50000"))


def estimate_joint_components(program_text: str) -> int:
    """Cheap static upper bound on the compiled program's joint mixture component count,
    computed BEFORE ever invoking the DeGAS compiler, by multiplying together the component
    count of every `gm([...])` literal and the interval count of every `uniform([...], K)`
    literal in the raw program text (each additional gm component/uniform interval, and each
    if/else branch, combinatorially multiplies DeGAS's final joint mixture -- see
    helpers/complexity.py). This overcounts somewhat (branch structure isn't modeled
    precisely -- verified against the real programs' own actual compiled n_comp via
    helpers/complexity.py: eyecolor's real n_comp is 60 but this estimates 7500, a ~125x
    overcount from its nested if/elif chains not being modeled) but that's the right
    direction for a safety filter -- it never underestimates, and the default cap below
    leaves ample margin above that worst observed overcount.

    Needed because on a wide (many-variable) benchmark, a plausible-looking candidate (e.g.
    giving several variables their own independent 3-component mixture) can hit millions of
    joint components -- DeGAS's compiler then spends many CPU-minutes building that structure
    combinatorially before ever reaching the point where an out-of-memory guard could catch it,
    so the fix has to reject the candidate before compilation, not during it.
    """
    total = 1
    for match in re.finditer(r'gm\(\s*\[([^\]]*)\]', program_text):
        n_comp = len([x for x in match.group(1).split(',') if x.strip()])
        total *= max(n_comp, 1)
    for match in re.finditer(r'uniform\(\s*\[[^\]]*\]\s*,\s*(\d+)\s*\)', program_text):
        total *= max(int(match.group(1)), 1)
    return total


def optimize_candidate(
    prog: dict, data_array, stats_dict, n_opt_steps: int, lr: float,
    warm_start: dict | None = None, held_out_data=None,
) -> None:
    """Compile and optimize a single candidate in-place.

    `warm_start`, if given (typically prog['optimized_params'] from a previous
    optimization pass on this same candidate), seeds the initial parameter values
    instead of the literal constants embedded in prog["program"], so a further
    optimization pass continues from where the previous one left off instead of
    restarting from the LLM's originally proposed literals every time.

    `held_out_data`, if given, additionally scores the fitted candidate's NLL against
    this (never fit against) split and stores it as prog['held_out_loss'] -- used by
    the ALPS_HELD_OUT_SELECTION ablation to select/sort candidates by held-out NLL
    instead of the training NLL they were fit against.
    """
    estimated_components = estimate_joint_components(prog["program"])
    if estimated_components > max_joint_components:
        raise ValueError(
            f"Estimated joint mixture too large (~{estimated_components} components > "
            f"{max_joint_components} cap) -- skipping before an expensive/memory-heavy compile."
        )

    rewritten, params_dict = extract_params(prog["program"])
    prog["params"] = params_dict
    prog["rewritten"] = rewritten

    compiled_file = compile2SOGA_text(prog["rewritten"])
    cfg = produce_cfg_text(compiled_file)
    smooth_cfg(cfg)

    loss = lambda output_dist: -compute_likelihood(output_dist, stats_dict["var_names"], data_array)

    def eval_held_out(fitted_params_dict) -> None:
        if held_out_data is None:
            return
        with torch.no_grad():
            held_out_dist = start_SOGA(cfg, fitted_params_dict)
            prog["held_out_loss"] = (
                -compute_likelihood(held_out_dist, stats_dict["var_names"], held_out_data)
            ).item()

    if not prog["params"]:
        # No optimizable parameters were extracted -- e.g. the candidate uses only
        # uniform(...), whose bounds param_extractor.py intentionally leaves as literals
        # (that extraction is commented out there). DeGAS's optimize() would call
        # torch.optim.Adam([]) in this case (zero parameters), and the subsequent
        # loss.backward() fails with "element 0 of tensors does not require grad and does
        # not have a grad_fn" since nothing in the graph requires gradients. Skip the
        # gradient loop entirely and just evaluate this (fixed) program's likelihood once.
        params_dict = {}
        output_dist = start_SOGA(cfg, params_dict)
        fixed_loss = loss(output_dist).item()
        prog["optimized_params"] = params_dict
        prog["initial_loss"] = fixed_loss
        prog["final_loss"] = fixed_loss
        prog["opt_time"] = 0.0
        prog["opt_iterations"] = 0
        eval_held_out(params_dict)
        return

    init_values = prog["params"]
    if warm_start is not None:
        init_values = {}
        for name, literal in prog["params"].items():
            value = warm_start.get(name, literal)
            if hasattr(value, "item"):
                value = value.item()
            init_values[name] = float(value)

    params_dict = initialize_params(init_values)

    if skip_gradient:
        # Ablation: evaluate the LLM's proposed program at its own literal parameter
        # values -- no gradient-descent fitting at all.
        with torch.no_grad():
            output_dist = start_SOGA(cfg, params_dict)
            fixed_loss = loss(output_dist).item()
        prog["optimized_params"] = params_dict
        prog["initial_loss"] = fixed_loss
        prog["final_loss"] = fixed_loss
        prog["opt_time"] = 0.0
        prog["opt_iterations"] = 0
        eval_held_out(params_dict)
        return

    #take a random subset of data_array
    n = len(data_array)
    #idx = np.random.choice(n, size=min(500, n), replace=False)
    #data_subset = [data_array[i] for i in idx]
    loss_list, elapsed_time, number_of_iterations = optimize(
        cfg,
        params_dict,
        loss,
        n_steps=n_opt_steps,
        lr=lr,
        print_progress=False,
    )

    prog["optimized_params"] = params_dict
    prog["initial_loss"] = loss_list[0]
    prog["final_loss"] = loss_list[-1]
    prog["opt_time"] = elapsed_time
    prog["opt_iterations"] = number_of_iterations
    eval_held_out(params_dict)


# Counts every program the LLM proposes (initial candidates + each iteration's mutations)
# exactly once, at its first optimize_candidate call -- tracks how often a freshly-generated
# program fails DeGAS compilation/execution (grammar-adjacent issues: mismatched list
# lengths, singular covariance, etc.), separate from the "additional optimization" re-fit
# pass on already-successful candidates later in the loop.
n_generated_total = 0
n_grammar_errors = 0

run_start_time = time.time()

data = get_dataset(program, data_size)
#print(data)

if held_out_selection:
    n_train = int(len(data) * train_frac)
    train_data, held_out_data = data[:n_train], data[n_train:]
else:
    train_data, held_out_data = data, None

stats = {
    "var_names": get_var_names(program),
    "n": len(train_data),
    "mean": np.mean(train_data, axis=0).tolist(),
    "std": np.std(train_data, axis=0).tolist(),
    "skewness": (np.mean((train_data - np.mean(train_data, axis=0))**3, axis=0) / (np.std(train_data, axis=0)**3)).tolist(),
    "kurtosis": (np.mean((train_data - np.mean(train_data, axis=0))**4, axis=0) / (np.std(train_data, axis=0)**4)).tolist(),
    "min": np.min(train_data, axis=0).tolist(),
    "max": np.max(train_data, axis=0).tolist(),
}
print(stats)
prompt = make_init_prompt(stats, n_programs=n_programs, extra_hypotheses=extra_hypotheses, hypothesis_slots=hypothesis_slots)
print(prompt)

# Save hyperparameters and run metadata
hyperparams = {
    "llm_model": llm_model,
    "skip_gradient": skip_gradient,
    "program": program,
    "data_size": data_size,
    "n_programs": n_programs,
    "n_mutations": n_mutations,
    "n_steps": n_steps,
    "patience": patience,
    "min_improvement": min_improvement,
    "init_temperature": init_temperature,
    "mutation_temperature": mutation_temperature,
    "init_opt_steps": init_opt_steps,
    "mutation_opt_steps": mutation_opt_steps,
    "extra_opt_steps": extra_opt_steps,
    "learning_rate": learning_rate,
    "require_json": require_json,
    "max_retries": max_retries,
    "use_chat": use_chat,
    "request_timeout": request_timeout,
    "max_http_retries": max_http_retries,
    "simple_prompt": simple_prompt,
    "prompt_history": prompt_history,
    "prompt_heuristic": prompt_heuristic,
    "extra_hypotheses": extra_hypotheses,
    "hypothesis_slots": hypothesis_slots,
    "results_dir": str(run_dir),
}
with open(hyperparams_path, "w", encoding="utf-8") as f:
    json.dump(hyperparams, f, indent=2)

log_line("=== Run Started ===")
log_line(json.dumps(hyperparams, indent=2))


# First generation
result = call_ollama(
    prompt,
    effective_system_prompt,
    model=llm_model,
    temperature=init_temperature,
    require_json=require_json,
    max_retries=max_retries,
    use_chat=use_chat,
    request_timeout=request_timeout,
    max_http_retries=max_http_retries,
)
candidates = result["programs"]

log_line("\n=== Initial Candidates ===")
for prog in candidates:
    log_line(f"Program ID: {prog['id']}, Structure: {prog['structure']}, Hypothesis: {prog['hypothesis']}")
    log_line(f"DeGAS code:\n{prog['program']}\n")

# extract parameters and optimize each program
for prog in candidates:
    n_generated_total += 1
    try:
        optimize_candidate(prog, train_data, stats, n_opt_steps=init_opt_steps, lr=learning_rate, held_out_data=held_out_data)
        log_line(
            f"Program ID {prog['id']} initial optimization completed. "
            f"Initial loss: {prog['initial_loss']:.4f}, Final loss: {prog['final_loss']:.4f}"
        )
    except Exception as e:
        n_grammar_errors += 1
        log_line(f"Error extracting parameters from program ID {prog['id']}: {e}")

# Sort by loss now, not just inside the mutation loop below -- with n_steps=0 (the no-mutation
# ablation) that loop never runs, and candidates[0] (what the evaluation pipeline treats as this
# run's own best) would otherwise stay in the LLM's raw response order instead of by fit quality.
candidates = [prog for prog in candidates if 'final_loss' in prog]
candidates.sort(key=sort_key)

best_fitness = []

for i in range(n_steps):
    new_programs = call_ollama(
        build_mutation_prompt(candidates, n_mutations=n_mutations, iteration=i + 1, grammar=DEGAS_GRAMMAR,
                               show_history=prompt_history, show_heuristic=prompt_heuristic),
        effective_system_prompt,
        model=llm_model,
        temperature=mutation_temperature,
        require_json=require_json,
        max_retries=max_retries,
        use_chat=use_chat,
        request_timeout=request_timeout,
        max_http_retries=max_http_retries,
    )
    new_candidates = new_programs['programs']

    if not new_candidates:
        log_line(f"No new candidates generated in iteration {i + 1}; stopping early.")
        break

    for prog in new_candidates:
        n_generated_total += 1
        try:
            optimize_candidate(prog, train_data, stats, n_opt_steps=mutation_opt_steps, lr=learning_rate, held_out_data=held_out_data)
            log_line(
                f"Program ID {prog['id']} mutation optimization completed. "
                f"Initial loss: {prog['initial_loss']:.4f}, Final loss: {prog['final_loss']:.4f}"
            )

        except Exception as e:
            n_grammar_errors += 1
            log_line(f"Error in program ID {prog['id']}: {e}")
            prog['errors'] = str(e)
            prog['final_loss'] = float('inf')

    # choose the best 5 programs between new_candidates and candidates based on final_loss
    all_candidates = candidates + new_candidates
    all_candidates = [prog for prog in all_candidates if 'final_loss' in prog]
    all_candidates.sort(key=sort_key)
    candidates = all_candidates[:5]

    if not candidates:
        log_line(f"No valid candidates after iteration {i + 1}; stopping early.")
        break

    # Another gradient improvement step on the selected candidates, continuing from
    # each candidate's own already-optimized params rather than restarting from scratch
    for prog in candidates:
        try:
            optimize_candidate(
                prog, train_data, stats, n_opt_steps=extra_opt_steps, lr=learning_rate,
                warm_start=prog.get('optimized_params'), held_out_data=held_out_data,
            )
            log_line(
                f"Program ID {prog['id']} additional optimization completed. "
                f"Initial loss: {prog['initial_loss']:.4f}, Final loss: {prog['final_loss']:.4f}"
            )

        except Exception as e:
            log_line(f"Error in program ID {prog['id']} during additional optimization: {e}")
            prog['errors'] = str(e)
            prog['final_loss'] = float('inf')

    # Re-sort: the additional optimization pass above can change the candidates'
    # relative ranking, so candidates[0] must be recomputed, not reused from the
    # pre-refinement sort done before this loop.
    candidates.sort(key=sort_key)
    best_candidate = candidates[0]
    progress = ((i + 1) / n_steps) * 100.0
    print(f"Progress: {progress:.1f}% | Iteration {i + 1}/{n_steps} | Best Loss: {best_candidate['final_loss']:.4f}")
    log_line(
        f"Best candidate after iteration {i + 1}: Program ID {best_candidate['id']}, "
        f"Final Loss: {best_candidate['final_loss']:.4f}"
    )
    best_fitness.append(best_candidate['final_loss'])
    log_line(f"Selected candidates for next iteration: {[prog['id'] for prog in candidates]}")

    if len(best_fitness) > patience:
        improvement = best_fitness[-(patience + 1)] - best_fitness[-1]
        if improvement < min_improvement:
            log_line(
                f"Early stopping after iteration {i + 1}: best loss improved by only "
                f"{improvement:.4f} over the last {patience} iterations "
                f"(< min_improvement={min_improvement})."
            )
            break

log_line("\n--- Final Selected Programs ---")
for prog in candidates:
    log_line(f"Program ID: {prog['id']}, Final Loss: {prog['final_loss']:.4f}")
    log_line(f"Hypothesis: {prog['hypothesis']}")
    log_line(f"Structure: {prog['structure']}")
    log_line(f"DeGAS code:\n{prog['program']}\n")

# Save final candidates
with open(final_candidates_path, "w", encoding="utf-8") as f:
    json.dump(candidates, f, indent=2, default=str)

# Save best fitness CSV
with open(best_fitness_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "best_loss"])
    for idx, loss_val in enumerate(best_fitness, start=1):
        writer.writerow([idx, loss_val])

if best_fitness:
    plt.figure(figsize=(8, 4.5))
    plt.plot(best_fitness)
    plt.title("Best Fitness Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Loss")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
else:
    log_line("No fitness values to plot.")

# Record total elapsed wall-clock time (dataset generation through final save) and
# re-persist hyperparams.json with it, for cross-method runtime comparison.
elapsed_seconds = time.time() - run_start_time
hyperparams["elapsed_seconds"] = elapsed_seconds
hyperparams["n_generated_total"] = n_generated_total
hyperparams["n_grammar_errors"] = n_grammar_errors
hyperparams["grammar_error_rate"] = (
    n_grammar_errors / n_generated_total if n_generated_total else None
)
with open(hyperparams_path, "w", encoding="utf-8") as f:
    json.dump(hyperparams, f, indent=2)
log_line(f"=== Run finished in {elapsed_seconds:.1f}s ===")

print(f"Run completed. Results saved to: {run_dir}")