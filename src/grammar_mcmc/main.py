"""Driver: run the grammar-MCMC baseline (no LLM -- see mcmc_search.py's module docstring)
across benchmarks and seeds.

Writes results in the same shape as refinestat_degas/main.py (hyperparams.json,
run_log.txt, final_candidates.json, best_fitness.csv) to
results/<program>/grammar_mcmc/<timestamp>_seed<N>/, for direct comparison in
evaluate_program.ipynb alongside "your method" and both RefineStat-DeGAS backends.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "DeGAS" / "src", _REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from helpers.dataset_generation import get_dataset, get_var_names  # noqa: E402

from .mcmc_search import MCMCSearchConfig, run_grammar_mcmc  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "results"
PROGRAMS = [
    "if", "mog1", "burglary", "csi", "easytugwar",
    "biasedtugwar", "mixedcondition", "multiplebranches", "eyecolor", "hurricane",
]
DATA_SIZE = 1000
TRAIN_FRAC = 0.8


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_one(
    program: str, seed: int, cfg: MCMCSearchConfig, data_size: int, train_frac: float
) -> tuple[Path, dict]:
    var_names = get_var_names(program)
    data = get_dataset(program, data_size)
    n_train = int(len(data) * train_frac)
    train_data, held_out_data = data[:n_train], data[n_train:]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / program / "grammar_mcmc" / f"{ts}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"=== program={program} seed={seed} (grammar-mcmc) ===")
    run_start_time = time.time()
    outcome = run_grammar_mcmc(program, train_data, held_out_data, var_names, cfg, seed=seed, log=log)
    elapsed_seconds = time.time() - run_start_time

    hyperparams = {
        "method": "grammar_mcmc",
        "program": program,
        "seed": seed,
        "data_size": data_size,
        "train_frac": train_frac,
        "elapsed_seconds": elapsed_seconds,
        **vars(cfg),
    }
    with open(run_dir / "hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    best = outcome["best"]
    final_candidates = [best] if best else []
    with open(run_dir / "final_candidates.json", "w") as f:
        json.dump(final_candidates, f, indent=2, default=str)

    with open(run_dir / "best_fitness.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_loss"])
        running_best = None
        for i, nll in enumerate(outcome["best_fitness_per_attempt"]):
            running_best = nll if running_best is None else min(running_best, nll)
            writer.writerow([i, running_best])

    status = "SUCCESS" if best else "NO VALID CANDIDATE FOUND"
    log(f"=== {status} in {elapsed_seconds:.1f}s: {run_dir} ===")
    return run_dir, outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", default=",".join(PROGRAMS))
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--data-size", type=int, default=DATA_SIZE)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--n-opt-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--p-conditional-init", type=float, default=0.3)
    args = parser.parse_args()

    programs = args.programs.split(",")
    seeds = parse_seeds(args.seeds)

    cfg = MCMCSearchConfig(
        n_steps=args.n_steps,
        n_opt_steps=args.n_opt_steps,
        lr=args.lr,
        temperature=args.temperature,
        p_conditional_init=args.p_conditional_init,
    )

    for program in programs:
        for seed in seeds:
            run_one(program, seed, cfg, args.data_size, args.train_frac)


if __name__ == "__main__":
    main()
