import json
import csv
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
llm_model = "gpt-oss:120b"
program = "mixedcondition"  # Options: "if", "mog1", "burglary", "csi", "easytugwar",
                         # "biasedtugwar", "mixedcondition", "multiplebranches", "eyecolor", "hurricane"
data_size = 1000
n_programs = 5
n_mutations = 5
n_steps = 15
init_temperature = 0.2
mutation_temperature = 0.4
init_opt_steps = 50
mutation_opt_steps = 100
extra_opt_steps = 200
learning_rate = 0.01
require_json = True
max_retries = 1
use_chat = True
request_timeout = 600
max_http_retries = 3

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


def optimize_candidate(prog: dict, data_array, stats_dict, n_opt_steps: int, lr: float, warm_start: dict | None = None) -> None:
    """Compile and optimize a single candidate in-place.

    `warm_start`, if given (typically prog['optimized_params'] from a previous
    optimization pass on this same candidate), seeds the initial parameter values
    instead of the literal constants embedded in prog["program"], so a further
    optimization pass continues from where the previous one left off instead of
    restarting from the LLM's originally proposed literals every time.
    """
    rewritten, params_dict = extract_params(prog["program"])
    prog["params"] = params_dict
    prog["rewritten"] = rewritten

    compiled_file = compile2SOGA_text(prog["rewritten"])
    cfg = produce_cfg_text(compiled_file)
    smooth_cfg(cfg)

    loss = lambda output_dist: -compute_likelihood(output_dist, stats_dict["var_names"], data_array)

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


run_start_time = time.time()

data = get_dataset(program, data_size)
#print(data)

stats = {
    "var_names": get_var_names(program),
    "n": len(data),
    "mean": np.mean(data, axis=0).tolist(),
    "std": np.std(data, axis=0).tolist(),
    "skewness": (np.mean((data - np.mean(data, axis=0))**3, axis=0) / (np.std(data, axis=0)**3)).tolist(),
    "kurtosis": (np.mean((data - np.mean(data, axis=0))**4, axis=0) / (np.std(data, axis=0)**4)).tolist(),
    "min": np.min(data, axis=0).tolist(),
    "max": np.max(data, axis=0).tolist(),
}
print(stats)
prompt = make_init_prompt(stats, n_programs=n_programs)
print(prompt)

# Save hyperparameters and run metadata
hyperparams = {
    "llm_model": llm_model,
    "program": program,
    "data_size": data_size,
    "n_programs": n_programs,
    "n_mutations": n_mutations,
    "n_steps": n_steps,
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
    "results_dir": str(run_dir),
}
with open(hyperparams_path, "w", encoding="utf-8") as f:
    json.dump(hyperparams, f, indent=2)

log_line("=== Run Started ===")
log_line(json.dumps(hyperparams, indent=2))


# First generation
result = call_ollama(
    prompt,
    SYSTEM_PROMPT,
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
    try:
        optimize_candidate(prog, data, stats, n_opt_steps=init_opt_steps, lr=learning_rate)
        log_line(
            f"Program ID {prog['id']} initial optimization completed. "
            f"Initial loss: {prog['initial_loss']:.4f}, Final loss: {prog['final_loss']:.4f}"
        )
    except Exception as e:
        log_line(f"Error extracting parameters from program ID {prog['id']}: {e}")

best_fitness = []

for i in range(n_steps):
    new_programs = call_ollama(
        build_mutation_prompt(candidates, n_mutations=n_mutations, iteration=i + 1, grammar=DEGAS_GRAMMAR),
        SYSTEM_PROMPT,
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
        try:
            optimize_candidate(prog, data, stats, n_opt_steps=mutation_opt_steps, lr=learning_rate)
            log_line(
                f"Program ID {prog['id']} mutation optimization completed. "
                f"Initial loss: {prog['initial_loss']:.4f}, Final loss: {prog['final_loss']:.4f}"
            )

        except Exception as e:
            log_line(f"Error in program ID {prog['id']}: {e}")
            prog['errors'] = str(e)
            prog['final_loss'] = float('inf')

    # choose the best 5 programs between new_candidates and candidates based on final_loss
    all_candidates = candidates + new_candidates
    all_candidates = [prog for prog in all_candidates if 'final_loss' in prog]
    all_candidates.sort(key=lambda x: x['final_loss'])
    candidates = all_candidates[:5]

    if not candidates:
        log_line(f"No valid candidates after iteration {i + 1}; stopping early.")
        break

    # Another gradient improvement step on the selected candidates, continuing from
    # each candidate's own already-optimized params rather than restarting from scratch
    for prog in candidates:
        try:
            optimize_candidate(
                prog, data, stats, n_opt_steps=extra_opt_steps, lr=learning_rate,
                warm_start=prog.get('optimized_params'),
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
    candidates.sort(key=lambda x: x['final_loss'])
    best_candidate = candidates[0]
    progress = ((i + 1) / n_steps) * 100.0
    print(f"Progress: {progress:.1f}% | Iteration {i + 1}/{n_steps} | Best Loss: {best_candidate['final_loss']:.4f}")
    log_line(
        f"Best candidate after iteration {i + 1}: Program ID {best_candidate['id']}, "
        f"Final Loss: {best_candidate['final_loss']:.4f}"
    )
    best_fitness.append(best_candidate['final_loss'])
    log_line(f"Selected candidates for next iteration: {[prog['id'] for prog in candidates]}")

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
with open(hyperparams_path, "w", encoding="utf-8") as f:
    json.dump(hyperparams, f, indent=2)
log_line(f"=== Run finished in {elapsed_seconds:.1f}s ===")

print(f"Run completed. Results saved to: {run_dir}")