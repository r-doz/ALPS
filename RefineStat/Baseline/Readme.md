## Baseline runners overview

This folder contains runners and post-processing utilities for generating and evaluating PyMC code from LLMs. They differ mainly in reporting depth and output locations.

### `base.py`

- **Purpose**: Full baseline with execution, diagnostics, and aggregated reporting.
- **How it works**:
  - Generates PyMC model code in a **single forward pass** using Hugging Face causal LM sampling (unconstrained).
  - Executes generated code **in-process** and computes Bayesian diagnostics (R-hat, ESS, divergences, BFMI, Pareto-k, ELPD).
  - Saves per-dataset results with conditional formatting (red highlighting for failing metrics).
  - Aggregates reliability scores, token usage, and token budgets across seeds.
  - Generates leaderboards (ranked by reliability_score + ELPD) and progression tracking (cumulative tokens vs. reliability).
  - Supports multiple temperatures and models.
- **Outputs**:
  - `results/baseline/<expt_N>/` (default ``--output``; auto-numbered subfolders)
    - `seed_<n>/<model>/<dataset>/` — `prompt.txt`, `snippet.txt`, `final_code.py`, optional `diagnostics.json`
    - `config.json` — experiment configuration
    - `aggregated_reliability.json` — reliability scores per model/dataset/seed
    - `aggregated_token_count.json` — token usage and cumulative counts
    - `token_budget.json` — max observed tokens × 1.2 (20% buffer)
    - Per-model `leaderboard.json` and `progress_<dataset>.json`
    - `<model>/<dataset>/results.csv` — per-seed results with diagnostics
    - `<model>/<dataset>/results.xlsx` — Excel version with conditional formatting
- **Use when**: You need comprehensive diagnostics, reliability scoring, and aggregated analysis across seeds.
- **Run**:
```bash
python Baseline/base.py --seeds 10 --temperature 0.3
python Baseline/base.py --seeds 5 --temperature 0.2 0.3 0.4 --output results/baseline
python Baseline/base.py --seeds 3 --models "meta-llama/Meta-Llama-3-8B,google/codegemma-7b"
```

### `base-multi.py`

- **Purpose**: Multi-run baseline with best-of-K selection and cross-run aggregation — designed for direct comparison with `refinestat/main.py` + `aggregate_stats.py`.
- **How it works**:
  - Runs N independent runs (default 5), each with K seeds (default 5), using non-overlapping seed ranges (run 1 → seeds 1-5, run 2 → seeds 6-10, etc.).
  - Within each run, selects the **best program per dataset** across K seeds (highest reliability_score, elpd_loo as tiebreaker) — analogous to how `refinestat/main.py` selects best across seeds.
  - After all runs, computes **mean ± std** of best-per-run metrics — analogous to what `aggregate_stats.py` produces.
  - Single-pass HF generation (no iterative refinement) + in-process execution + Bayesian diagnostics.
- **Outputs**:
  - `results/Baseline/base-multi/<expt_N>/`
    - `run_<i>/seed_<s>/<model>/<dataset>/` — per-seed generated code + diagnostics
    - `run_<i>/<model>/<dataset>/best_program.py` — best program within run
    - `analysis/all_runs_best.csv` (`.xlsx`) — one row per run × dataset (best from each run)
    - `analysis/global_best.csv` (`.xlsx`) — overall best per dataset across all runs
    - `analysis/aggregated_stats.xlsx` — mean ± std across runs (Summary + Formatted sheets)
    - `summary.txt`, `analysis/aggregated_summary.txt`
- **Use when**: You want publication-ready mean ± std results from the HF unconstrained baseline to compare against the iterative refinement approach.
- **Run**:
```bash
python Baseline/base-multi.py --runs 5 --seeds-per-run 5
python Baseline/base-multi.py --runs 5 --seeds-per-run 5 --temperature 0.3
python Baseline/base-multi.py --runs 3 --seeds-per-run 3 --models "Qwen/Qwen2.5-Coder-7B"
```

### `result.py`

- **Purpose**: Aggregate per-dataset `results.csv` into model/dataset statistics.
- **What it does**:
  - Loads all `results.csv`, computes mean/std/count for numeric metrics, and exports regular and pivoted summaries to CSV/Excel, plus `metrics_summary.json`.
- **Typical CLI**:
```bash
python Baseline/result.py --base_dir <models_root> --output_dir <dir>
```

### Notes
- **CUDA**: Runners expect CUDA for model inference.
- **Snippet insertion**: Preserves the entire generated snippet (no truncation at `pm.sample`).
