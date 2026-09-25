# ALPS: Automated LLM-guided Probabilistic programs Synthesis

Code for reproducing the experiments in the paper. This repository contains:

- `src/` -- ALPS itself, plus the RS-MCMC and RS-Grad baselines (LLM-guided synthesis of
  DeGAS-grammar programs, using gradient-based and MCMC-based parameter fitting respectively).
- `RefineStat/` -- a patched copy of the original RefineStat baseline
  ([structuredllm/RefineStat](https://github.com/structuredllm/RefineStat)), used as-is except
  for the fixes listed in [RefineStat/PATCHES.md](RefineStat/PATCHES.md).
- `DeGAS/` -- a git submodule pointing to [frarandone/DeGAS](https://github.com/frarandone/DeGAS),
  the probabilistic-program compiler/solver ALPS, RS-MCMC and RS-Grad all synthesize programs for.
- `data/heart_disease.csv` -- the UCI Cleveland Heart Disease dataset, used for the real-world
  case study (Section on the real-world scenario).

## 1. Setup

Two separate Python environments are needed, because ALPS and RefineStat pin different/conflicting
dependency versions (in particular `arviz`/`torch`).

```bash
git clone <this repo's URL>
cd <repo>
git submodule update --init --recursive   # fetches DeGAS

# Environment 1: ALPS + RS-MCMC/RS-Grad baselines (src/)
python3 -m venv alps_env
source alps_env/bin/activate
pip install -r requirements.txt
pip install -r DeGAS/requirements.txt
deactivate

# Environment 2: original RefineStat baseline (RefineStat/)
python3 -m venv refinestat_env
source refinestat_env/bin/activate
pip install -r RefineStat/requirements.txt
deactivate
```

You will also need an LLM backend. All three ALPS-side methods (ALPS, RS-MCMC, RS-Grad) call out
to an [Ollama](https://ollama.com) server over its REST API:

```bash
ollama serve &
ollama pull gpt-oss:120b   # or any other model; override via ALPS_LLM_MODEL
```

By default the code expects Ollama at `http://localhost:11434`; point it elsewhere with the
`OLLAMA_BASE_URL` environment variable if you're running Ollama on a different machine.

RefineStat loads its model directly via `transformers` (`AutoModelForCausalLM`) rather than
through Ollama, since it needs raw per-token logits for constrained decoding -- see Section 2.3.

## 2. Running the methods

### 2.1 ALPS

From `src/`, with the `alps_env` environment active:

```bash
cd src
env ALPS_PROGRAM=<benchmark> python llm_synthesis.py
```

`<benchmark>` is one of: `if`, `mog1`, `csi`, `easytugwar`, `biasedtugwar`, `mixedcondition`,
`multiplebranches`, `eyecolor`, `hurricane` (the 9 synthetic benchmarks), `heart_disease` /
`heart_disease_small` (the real-world case study), or `eight_schools` / `dugongs` / `gp` / `glm` /
`surgical` (RefineStat's own native benchmarks, used for the head-to-head comparison on
RefineStat's own turf).

Each run creates a timestamped directory under `results/<benchmark>/` containing
`final_candidates.json` (the ranked list of synthesized programs) and a run log.

Key environment variables (all optional, defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `ALPS_PROGRAM` | `if` | which benchmark to run |
| `ALPS_LLM_MODEL` | `gpt-oss:120b` | Ollama model name |
| `ALPS_DATA_SIZE` | `1000` | number of data points to generate/sample |
| `ALPS_N_PROGRAMS` | `5` | number of independent candidate structures per run |
| `ALPS_N_STEPS` | `10` | refinement iterations per candidate (capped by `ALPS_MAX_STEPS_CAP`) |
| `ALPS_SKIP_GRADIENT` | `0` | `1` disables gradient-based parameter refinement (ablation) |
| `ALPS_PROMPT_HISTORY` | `1` | `0` withholds mutation history from the prompt (ablation) |
| `ALPS_PROMPT_HEURISTIC` | `1` | `0` withholds gradient-derived structural hints (ablation) |
| `ALPS_HELD_OUT_SELECTION` | `0` | `1` selects the best candidate by held-out loss (used for the real-world case study) |
| `ALPS_TRAIN_FRAC` | `0.8` | train fraction when `ALPS_HELD_OUT_SELECTION=1` |
| `ALPS_HYPOTHESIS_SLOTS` | (empty) | `"slot:hypothesis text|slot:hypothesis text"` -- inject plain-language domain hypotheses (see Section 4) |

Running all 5 seeds of a benchmark is just 5 repeated invocations; see
`aggregate_results_table.py` (Section 3) for how per-seed results get collected.

### 2.2 RS-MCMC and RS-Grad

These are ablation-style variants of ALPS that synthesize DeGAS-grammar programs but fit
parameters via MCMC or gradient descent respectively, using the same LLM-guided search loop.
They live in `src/refinestat_degas/` and `src/grammar_mcmc/`; see the module docstrings there
for their entry points (`refine_loop.py` / `mcmc_search.py`).

### 2.3 RefineStat (original baseline)

From `RefineStat/refinestat/`, with the `refinestat_env` environment active:

```bash
cd RefineStat/refinestat

# One-time step per HF model: builds the SynCode DFA mask store for constrained decoding.
python dfa_constructor.py --models "Qwen/Qwen2.5-Coder-7B-Instruct"

# Run seeds 1-5 on a benchmark:
python main.py --seeds 1-5 --models "Qwen/Qwen2.5-Coder-7B-Instruct" --output results/main
```

`main.py` iterates over every dataset registered in `commons/data_pymc.py`'s `datas_info` list
(all of RefineStat's native benchmarks plus the heart-disease case study by default). To restrict
a run to specific datasets without editing that file, set `RS_DATASETS` to a comma-separated list
of dataset names, e.g.:

```bash
env RS_DATASETS="eight_schools,dugongs" python main.py --seeds 1-5 --output results/main
```

Each invocation creates a new numbered subdirectory under `--output` (auto-incremented, so
re-running never overwrites a previous one); results land at
`<output>/<n>/<model>/<dataset>/seed_<k>/`, with per-seed diagnostics and a `analysis/` folder
containing the best candidate per seed as `seed_<k>_best.json`.

Note: RefineStat fits every candidate to the *full* dataset via PyMC/NUTS -- unlike ALPS, it has
no held-out train/test split of its own, and no built-in notion of "the target variable," so
nothing prevents the LLM from including the target column among its own predictors (see
`RefineStat/PATCHES.md` for a real instance of this we hit on `heart_disease_small`).

See `RefineStat/PATCHES.md` for the specific bugs fixed relative to the upstream release, and why
they were necessary to get any results out of it at all.

## 3. Aggregating results / reproducing tables

```bash
cd src
python aggregate_results_table.py [benchmark1 benchmark2 ...]   # default: all 9 synthetic benchmarks
```

Writes `results/summary_table_raw.csv` (one row per benchmark x method x seed: NLL, $W_2^2$, time)
and `results/summary_table.csv` (mean +- std per benchmark x method). Pass `--corrected` to instead
use the discrete-variable-aware likelihood correction described in the paper (writes to
`summary_table_raw_corrected.csv` / `summary_table_corrected.csv`, leaving the originals untouched).

`corrected_likelihood.py` documents which variables are treated as discrete per benchmark
(`DISCRETE_VARS`) and implements the corrected scoring.

## 4. Real-world case study and prior-knowledge injection

The real-world case study (Section on the real-world scenario) runs ALPS and RefineStat on
`heart_disease_small` (297 patients, UCI Cleveland dataset, 7 predictors + target `num`). Two
variants matter:

- **No naming hint** (the fair, apples-to-apples comparison): run ALPS with `ALPS_PROGRAM=heart_disease_small` as normal, and RefineStat with the dataset's target column named `num` (RefineStat's release default, see `commons/data_pymc.py`'s `heart_disease_small_template`).
- **Prior-knowledge injection demo**: set `ALPS_HYPOTHESIS_SLOTS` to inject a plain-language
  domain hypothesis into the synthesis prompt, e.g.:

  ```bash
  env ALPS_PROGRAM=heart_disease_small \
      ALPS_HYPOTHESIS_SLOTS="1:thal strongly affects num via a conditional branch" \
      python llm_synthesis.py
  ```

  This is the mechanism used to show that ALPS can incorporate structural domain knowledge
  expressed in plain English, with no probabilistic-programming expertise required -- see
  `helpers/generation_prompt.py`'s `make_init_prompt(..., hypothesis_slots=...)`. RefineStat and
  RS-MCMC/RS-Grad have no equivalent mechanism.

## 5. Notes on hyperparameters and compute

- All main-table results use `ALPS_N_STEPS=10`, `ALPS_N_PROGRAMS=5`, 5 seeds per benchmark.
- RefineStat's baselines used `Qwen/Qwen2.5-Coder-7B-Instruct` (loaded locally via `transformers`)
  rather than the larger model ALPS uses via Ollama, since RefineStat's constrained-decoding
  approach requires direct access to per-token logits, which Ollama's completion-style REST API
  does not expose. This is a real limitation of the comparison, not a configuration choice; see
  the paper's baseline-comparison discussion for details.
