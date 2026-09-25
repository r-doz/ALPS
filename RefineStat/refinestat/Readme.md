# RefineStat — Runners Overview

This folder contains runners that orchestrate LLM-driven PyMC model synthesis and statistical analysis. Each script represents a different experimental setup, ranging from simple baselines to full iterative refinement pipelines.

---

## Scripts

### `baseline.py`

- **Purpose**: Minimal end-to-end baseline that compiles and runs the generated PyMC program, then scrapes diagnostics from stdout.
- **How it works**:
  - Builds a boilerplate prompt per dataset (`commons.model_pymc`), generates/refines code, inserts it under `with pm.Model() as m:`.
  - Executes the assembled script as a subprocess and regex-parses ArviZ diagnostics (ELPD LOO, R-hat, ESS) from stdout.
  - Aggregates per-seed/per-model results and saves CSV/XLSX summaries with conditional formatting.
- **Outputs**: `results/exec/<modelsize>/<temperature>/<exp>/...` with per-dataset folders and an overall statistical summary.
- **Use when**: You want a simple compile-and-metric baseline across seeds and models.
- **Run**:
```bash
python refinestat/baseline.py <temperature> <modelsize> <seeds>
# e.g.
python refinestat/baseline.py 0.3 medium 3
```
> Note: `modelsize` supports `small` or `medium`.

---

### `base.py`

- **Purpose**: Iterative-refinement baseline using `refinegen.base`. Generates PyMC model code with IterGen + checker feedback, assembles each snippet into the dataset boilerplate, executes it in-process, and computes Bayesian reliability diagnostics. Aggregates across seeds and models.
- **How it works**:
  - Uses `refinegen.base.iterative_refine` with `PyMCChecker` to refine the generated model code (max iterations from `commons.config`, default 35).
  - Extracts the trace variable name from the symbol table and inserts the generated snippet into the PyMC boilerplate template, appending ArviZ diagnostic print statements for transparency.
  - Executes the assembled code using `run_pymc_code` (in-process), then calls `check_model_reliability` on the returned trace for structured diagnostics.
  - Aggregates reliability scores, token counts, and compilation statistics in a local `state` dict (no global mutable state) and saves them as JSON at the end of each experiment.
- **Outputs**: `results/base/<exp_N>/<model>/<seed_N>/<dataset>/...` with per-dataset `final_code.py`, `diagnostics.json`, `exec_output.txt`, `refinement_log.txt`, and a top-level `statistical_summary.(csv|xlsx)`.
- **Use when**: You want iterative refinement with structured in-process diagnostics and multi-model/multi-seed aggregation, as the main baseline in the paper.
- **Run**:
```bash
python refinestat/base.py --seeds 10
python refinestat/base.py --seeds 5 --temperature 0.5 --output results/base
python refinestat/base.py --seeds 3 --models Qwen/Qwen2.5-Coder-7B
```

---

### `dfa_constructor.py`

- **Purpose**: Build the SynCode **Python DFA mask store** on disk for one or more Hugging Face models **before** you run `main.py`. This script uses **`grammar="python"`** so `Grammar.simplifications()` apply during mask construction (required for a correct pickle). `main.py` always passes the **path** to the bundled `python_grammar.lark` to `IterGen` and expects that cache to exist (same grammar bytes → same cache key).
- **How it works**: Loads **tokenizers only** (no full model), calls `DFAMaskStore.load_dfa_mask_store` with `Grammar("python")` and `mode=grammar_strict`. Writes under `SYNCODE_CACHE/mask_stores/<TokenizerClass>/`.
- **Outputs**: Mask pickle files only (no experiment results).
- **Use when**: The first time you use a model ID with `main.py`, or after deleting the SynCode cache. Run once per model (comma-separated for several models).
- **Run**:
```bash
python refinestat/dfa_constructor.py
python refinestat/dfa_constructor.py --models "Qwen/Qwen2.5-3B-Instruct"
python refinestat/dfa_constructor.py --models "org/model-a,org/model-b"
```
> **Note:** With no `--models` flag, the script uses `DEFAULT_MODELS` in `dfa_constructor.py` — **keep that list in sync with `DEFAULT_MODELS` in `main.py`**. The first build for a tokenizer can take a long time. Complete this step before `main.py`. Once the mask store for a model exists, you do **not** need to run `dfa_constructor.py` again for that model (unless you delete `SYNCODE_CACHE` or use a new model id).

---

### `main.py`

- **Purpose**: Analysis-first pipeline that records every refinement step, computes diagnostics during refinement, and selects the best candidate by reliability score and ELPD. **Automatically generates aggregated statistics.**
- **Prerequisite**: The first time you use a Hugging Face model with `main.py`, run `dfa_constructor.py` for that model so the Python DFA mask store exists. Later runs with the same model skip `dfa_constructor.py` if the cache is still present.
- **How it works**:
  - Uses `refinegen.main.iterative_refine` with `PyMCChecker` and configuration from `commons.config` (including `pymc_symboltable`, `max_iter`, `unit_name`).
  - Persists all interim entries (program, diagnostics, reliability score) per iteration, then selects the "best" entry by highest reliability score and, as a tiebreaker, highest ELPD-LOO.
  - Tracks token usage per seed/dataset, derives cumulative token counts, and saves a token budget (120% of max observed tokens).
  - **Automatically computes mean ± std across seeds at the end** and generates publication-ready aggregated statistics (no separate post-processing needed).
- **Outputs**: `results/refinestat-main/<expt_N>/...` including:
  - `<model>/<dataset>/entry_<i>.json` — per-iteration entries
  - `analysis/<model>/<dataset>/seed_*_entries.(csv|xlsx)` — iteration-level analysis with red-highlighted failing metrics
  - `best_program.py`, `best_program_diagnostics.txt`
  - `token_usage.json`, `cumulative_tokens.json`, `token_budget.json`
  - `analysis/best_programs_summary.(csv|xlsx)` — per-seed best programs
  - `analysis/aggregated_stats.xlsx` — mean ± std per dataset across seeds (Summary + Formatted sheets)
  - `analysis/aggregated_summary.txt` — publication-ready text summary
- **Use when**: You want detailed per-iteration diagnostics, best-program selection across all seeds, token accounting, and aggregated statistics — all computed automatically without external post-processing.
- **Run**:
```bash
# Required once per model before first main.py run (see dfa_constructor.py; edit DEFAULT_MODELS or use --models):
python refinestat/dfa_constructor.py

python refinestat/main.py --seeds 1
python refinestat/main.py --seeds 1-10
python refinestat/main.py --seeds 1,3,5,7-10 --temperature 0.5
```
> Note: All seeds run inside a single experiment folder. Best programs and aggregated statistics are computed automatically at the end.

---

### `base-itergen.py`

- **Purpose**: IterGen-only baseline (no iterative refinement) that generates PyMC model code in a single forward-only pass and stops as soon as `pm.sample` is detected.
- **How it works**:
  - Uses `IterGen.start()` and `IterGen.forward()` in a loop, accumulating generated code until `pm.sample` is found or max iterations reached.
  - No checker feedback loop — code is generated once, then executed.
  - Extracts generated code up to and including the `pm.sample` line, inserts into PyMC template, executes in-process.
  - Computes full Bayesian diagnostics (R-hat, ESS, BFMI, Pareto-k, ELPD) on the resulting trace.
  - Designed for multi-seed, multi-GPU execution: seeds support single values, ranges (`1-5`), or comma-separated lists (`1,3,5,7-10`).
- **Outputs**: `results/Baseline/Run-rate/<seed_N_gpu_G>/`
  - Per-dataset `programs/` with `prompt.txt`, `final_generated_code.py`, `full_code_with_template.py`
  - `seed_<N>_gpu_<G>_results.json` — per-seed results
  - Top-level: `aggregated_reliability.json`, `aggregated_token_count.json`, `summary.csv/.xlsx`
- **Use when**: You want to measure IterGen-only code generation as a baseline for comparison with iterative refinement.
- **Run**:
```bash
python refinestat/base-itergen.py --seeds 1-5 --gpu 0
python refinestat/base-itergen.py --seeds 1,3,7-10 --gpu 1 --output results/Baseline/Run-rate
python refinestat/base-itergen.py --seeds 10 --gpu 0 --models "meta-llama/Meta-Llama-3-8B"
```

---

## Summary Comparison

| Script | Role | Seeds CLI | GPU Control | Aggregation |
|---|---|---|---|---|
| `dfa_constructor.py` | Build SynCode Python DFA mask cache (`grammar="python"`) per HF model | — | No | — |
| `baseline.py` | Subprocess baseline (subprocess exec + regex diagnostics) | Positional int | No | — |
| `base.py` | Iterative refinement baseline (in-process exec + diagnostics) | `--seeds` int | No | — |
| `main.py` | Full iterative refinement with per-iteration analysis | `--seeds` range/list | No | Automatic (built-in) |
| `base-itergen.py` | IterGen-only ablation (no refinement, early pm.sample stop) | `--seeds` range/list | `--gpu` | — |

---

## Notes

- **`main.py` and SynCode:** Run `dfa_constructor.py` once per new model before `main.py` so the mask store is built with `grammar="python"`. After that store exists on disk for a model, `dfa_constructor.py` is not needed again for that model. `main.py` uses the path to `python_grammar.lark` for `IterGen` during experiments.
- All runners expect CUDA by default for model inference. Adjust `device='cuda'` in the script if running on CPU.
- Diagnostic thresholds for Excel highlighting (R-hat ≥ 1.05, ESS bulk < 400, ESS tail < 100, divergences > 0, BFMI ≤ 0.3, Pareto-k > 0.2) are defined identically in each script.
- `base.py` and `base-itergen.py` both use `commons.config` for shared configuration (temperature, recurrence penalty, max iterations).
- `base.py` and `base-itergen.py` save a `config.json` at the start of each experiment recording the exact parameters used.
- **`main.py` now includes built-in aggregation:** It automatically generates `aggregated_stats.xlsx` and `aggregated_summary.txt` at the end of the experiment — no separate post-processing step needed!
