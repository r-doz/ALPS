"""Driver: run the RefineStat-DeGAS baseline across benchmarks and seeds.

Writes results in the same shape as src/llm_synthesis.py (hyperparams.json, run_log.txt,
final_candidates.json, best_fitness.csv) so evaluate_program.ipynb can score all methods
identically -- to results/<program>/refinestat/<timestamp>_seed<N>/ for the default
gradient-optimized (DeGAS-native) inference backend, or
results/<program>/refinestat_mcmc/<timestamp>_seed<N>/ for genuine NUTS/MCMC posterior
inference (--inference mcmc, see mcmc_diagnostics.py's module docstring for why both exist:
DeGAS has no MCMC engine of its own, so the gradient backend is the default but
methodologically different from RefineStat's own pm.sample()-based inference; the mcmc
backend closes that gap via a from-scratch Pyro translation). The candidate
generation/checking/backtracking loop is identical either way -- only the fit_fn passed to
refine_program differs.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "DeGAS" / "src", _REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from helpers.dataset_generation import get_dataset, get_var_names  # noqa: E402

from . import config  # noqa: E402
from .itergen_degas import DegasIterGen  # noqa: E402
from .mcmc_diagnostics import MCMCConfig, fit_and_diagnose_mcmc  # noqa: E402
from .refine_loop import RefineConfig, refine_program  # noqa: E402


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
    itergen: DegasIterGen,
    program: str,
    seed: int,
    refine_cfg: RefineConfig,
    data_size: int,
    train_frac: float,
    inference: str = "gradient",
    mcmc_cfg: MCMCConfig | None = None,
) -> tuple[Path, dict]:
    torch.manual_seed(seed)
    var_names = get_var_names(program)
    data = get_dataset(program, data_size)
    n_train = int(len(data) * train_frac)
    train_data, held_out_data = data[:n_train], data[n_train:]

    results_subdir = "refinestat" if inference == "gradient" else "refinestat_mcmc"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.RESULTS_DIR / program / results_subdir / f"{ts}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    fit_fn = functools.partial(fit_and_diagnose_mcmc, cfg=mcmc_cfg) if inference == "mcmc" else None

    log(f"=== program={program} seed={seed} inference={inference} ===")
    run_start_time = time.time()
    outcome = refine_program(
        itergen, program, train_data, held_out_data, var_names, refine_cfg, fit_fn=fit_fn, log=log
    )
    elapsed_seconds = time.time() - run_start_time

    hyperparams = {
        "method": f"refinestat_degas_{inference}",
        "program": program,
        "seed": seed,
        "model_id": config.MODEL_ID,
        "data_size": data_size,
        "train_frac": train_frac,
        "elapsed_seconds": elapsed_seconds,
        **vars(refine_cfg),
        **({f"mcmc_{k}": v for k, v in vars(mcmc_cfg).items()} if mcmc_cfg else {}),
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
    parser.add_argument("--programs", default=",".join(config.PROGRAMS))
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--data-size", type=int, default=config.DATA_SIZE)
    parser.add_argument("--train-frac", type=float, default=config.TRAIN_FRAC)
    parser.add_argument("--max-units", type=int, default=20)
    parser.add_argument(
        "--Rmax",
        type=int,
        default=None,
        help="Max prior-resample rounds. Default: 100, RefineStat's own paper value (Appendix "
        "E), used for both --inference gradient and mcmc for a like-for-like search budget.",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="Max likelihood-resamples per round. Default: 2, RefineStat's own value, used for "
        "both inference backends.",
    )
    parser.add_argument(
        "--beta",
        type=int,
        default=None,
        help="Target number of valid candidates. Default: 4, RefineStat's own value, used for "
        "both inference backends.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="Reliability-score cutoff. Default: 4 (of 5 checks) for --inference gradient, "
        "5 (of 7 checks, RefineStat's own zeta) for --inference mcmc.",
    )
    parser.add_argument("--n-opt-steps", type=int, default=100)
    parser.add_argument(
        "--max-wall-hours",
        type=float,
        default=None,
        help="Give up on a seed (move to the next) after this many hours in the attempt "
        "loop, keeping whatever valid candidate (if any) was found so far. Default: no cap.",
    )
    parser.add_argument(
        "--inference",
        choices=["gradient", "mcmc"],
        default="gradient",
        help="gradient: DeGAS's own gradient-optimized point estimate (default). "
        "mcmc: genuine NUTS/MCMC posterior inference via NumPyro -- see mcmc_diagnostics.py.",
    )
    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument(
        "--mcmc-warmup",
        type=int,
        default=None,
        help="NUTS warmup/tune steps. Default: 1000, matching RefineStat's own "
        "pm.sample(1000, tune=1000, chains=4, ...) prompt (commons/data_pymc.py).",
    )
    parser.add_argument(
        "--mcmc-samples",
        type=int,
        default=None,
        help="NUTS post-warmup draws. Default: 1000, matching RefineStat's own pm.sample() call.",
    )
    parser.add_argument("--mcmc-loo-draws-per-chain", type=int, default=50)
    args = parser.parse_args()

    programs = args.programs.split(",")
    seeds = parse_seeds(args.seeds)

    is_mcmc = args.inference == "mcmc"
    # Rmax/alpha/beta (RefineStat's own paper values) are shared across both inference
    # backends for a like-for-like search budget. K stays backend-dependent because it's a
    # cutoff over a different-sized check set (5 checks for gradient vs RefineStat's own 7
    # for mcmc, see mcmc_diagnostics.py) -- 4/5 and 5/7 are each's proportionally-closest
    # cutoff to RefineStat's own zeta=5/7 (~71%).
    k = args.K if args.K is not None else (5 if is_mcmc else 4)
    rmax = args.Rmax if args.Rmax is not None else 100
    alpha = args.alpha if args.alpha is not None else 2
    beta = args.beta if args.beta is not None else 4
    refine_cfg = RefineConfig(
        max_units=args.max_units,
        Rmax=rmax,
        alpha=alpha,
        beta=beta,
        K=k,
        n_opt_steps=args.n_opt_steps,
        max_wall_seconds=args.max_wall_hours * 3600 if args.max_wall_hours is not None else None,
    )
    mcmc_warmup = args.mcmc_warmup if args.mcmc_warmup is not None else 1000
    mcmc_samples = args.mcmc_samples if args.mcmc_samples is not None else 1000
    mcmc_cfg = (
        MCMCConfig(
            num_chains=args.mcmc_chains,
            warmup_steps=mcmc_warmup,
            num_samples=mcmc_samples,
            loo_draws_per_chain=args.mcmc_loo_draws_per_chain,
        )
        if args.inference == "mcmc"
        else None
    )

    print(f"Loading {config.MODEL_ID} ...")
    itergen = DegasIterGen(
        config.GRAMMAR_PATH,
        config.MODEL_ID,
        device=config.DEVICE,
        temperature=config.TEMPERATURE,
        max_new_tokens_per_unit=config.MAX_NEW_TOKENS_PER_UNIT,
    )
    print("Model loaded.")

    for program in programs:
        for seed in seeds:
            run_one(
                itergen,
                program,
                seed,
                refine_cfg,
                args.data_size,
                args.train_frac,
                inference=args.inference,
                mcmc_cfg=mcmc_cfg,
            )


if __name__ == "__main__":
    main()
