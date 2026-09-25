# RefineStat: Efficient Exploration for Probabilistic Program Synthesis

**A framework for generating reliable probabilistic programs using Small Language Models (SLMs) with iterative refinement and semantic constraints.**

**Paper:** [RefineStat: Efficient Exploration for Probabilistic Program Synthesis](https://arxiv.org/pdf/2509.01082)  
**Published:** ICLR 2026  
**Authors:** Madhav Kanda, Shubham Ugare, Sasa Misailovic (University of Illinois Urbana–Champaign)  

---

## Overview

RefineStat addresses a fundamental challenge in probabilistic program synthesis: **generating statistical models that are both syntactically sound and statistically reliable**. 

Unlike direct LLM querying, which frequently produces semantic bugs (e.g., using variance instead of standard deviation, invalid parameter names), RefineStat employs a principled two-phase approach:

1. **Semantically-Constrained Generation** — Enforces six validation predicates during code generation to ensure:
   - Syntactic correctness (parse-ability)
   - Distribution validity (distributions exist in PPL library)
   - Parameter validity (correct parameter names and types)
   - Dependency validity (variables declared before use)
   - Support validity (parameters within distribution support)
   - Type validity (correct Python/NumPy types)

2. **Diagnostic-Aware Refinement** — Iteratively resamples priors and likelihoods when models fail Bayesian workflow checks, ensuring:
   - Convergence diagnostics (R̂ < 1.05)
   - Effective sample sizes (ESS bulk ≥ 400, ESS tail ≥ 100)
   - No divergences (divergences = 0)
   - Sampler health (BFMI > 0.3)
   - Reliable importance sampling (Pareto k < 0.7)
   - Strong predictive performance (finite ELPD-LOO)

**Key Achievement:** RefineStat enables **open-weight SLMs (7-8B parameters)** to synthesize reliable probabilistic programs, often **matching or surpassing closed-source LLMs** like GPT-4/o3 at a fraction of the cost.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd repo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run RefineStat (Recommended Entry Point)

Build the SynCode Python DFA mask cache **once per Hugging Face model** before your first `main.py` run (tokenizer only; can take a long time the first time). With no arguments, `dfa_constructor.py` uses `DEFAULT_MODELS` in that file — **keep it aligned with `DEFAULT_MODELS` in `main.py`**, or pass models explicitly:

```bash
python refinestat/dfa_constructor.py
python refinestat/dfa_constructor.py --models "YourOrg/your-model-id"
```

Then run the full pipeline:

```bash
# Full iterative refinement with automatic aggregation
python refinestat/main.py --seeds 1-10

# With custom temperature
python refinestat/main.py --seeds 1-10 --temperature 0.3

# Custom output location
python refinestat/main.py --seeds 5 --output results/my-experiment
```

Use the **same** model id in `dfa_constructor.py` as in `main.py --models` when you try a new checkpoint. **After the DFA mask store exists for that model** (under `SYNCODE_CACHE`), you do **not** need to run `dfa_constructor.py` again unless you delete the cache or switch to a different model id.

---

## Repository Structure

```
├── refinestat/              # Main RefineStat implementation
│   ├── main.py             # Full pipeline with iterative refinement ⭐ RECOMMENDED
│   ├── dfa_constructor.py  # Build SynCode Python mask store (once per model; skip if cache exists)
│   ├── base.py             # Iterative refinement baseline (without semantic constraints)
│   ├── base-itergen.py     # Single-pass IterGen baseline (no refinement)
│   ├── baseline.py         # Minimal subprocess baseline
│   ├── data-process.py     # Result visualization and aggregation
│   ├── Readme.md           # Detailed runner documentation
│   ├── refinegen/          # Core code generation and refinement logic
│   └── aggregate_stats.py  # [Deprecated] - Now built into main.py
│
├── Baseline/               # Non-iterative baselines (Syncode-based)
│   ├── base.py            # Single-pass Syncode baseline
│   ├── base-multi.py      # Multi-run baseline with aggregation
│   ├── result.py          # Result aggregation utility
│   └── Readme.md          # Baseline documentation
│
├── commons/                # Shared utilities and configurations
│   ├── config.py          # Configuration parameters
│   ├── data_pymc.py       # Dataset definitions and PyMC templates
│   └── utils.py           # Diagnostic and utility functions
│
├── results/                # Experiment outputs
│   ├── refinestat-main/    # main.py results
│   └── Baseline/           # Baseline results
│
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## What is RefineStat?

### The Problem

Small Language Models (SLMs) can generate code, but probabilistic programming requires semantic correctness beyond syntax:

```python
# ❌ SLM generates (semantic bug!)
pm.Normal("x", mu=0, sigma=variance)  # Wrong! Using variance instead of std dev

# ✅ RefineStat ensures (semantically correct)
pm.Normal("x", mu=0, sigma=std_dev)   # Correct parameter name and value
```

### The Solution: Two-Phase Approach

**Phase 1: Semantically-Constrained Generation**
- Uses grammar-guided generation with validation predicates
- Enforces domain-specific constraints during token-by-token generation
- Prevents invalid distributions, parameters, and variable dependencies
- Cost: ~15-20% token overhead (minimal compared to benefits)

**Phase 2: Diagnostic-Aware Refinement**
- Runs generated program through Bayesian workflow checks
- If checks fail, selectively resamples problematic components:
  - **Likelihood resampling** (L ← LCD(D∥P)) — fixes convergence issues
  - **Prior resampling** (P ← LCD(D)) — fixes specification errors
- Iterates until finding valid, reliable model
- Selects best by ELPD-LOO (predictive accuracy)

---

## Datasets

Experiments use five representative probabilistic models from Bayesian statistics:

| Dataset | Type | Domain |
|---------|------|--------|
| **eight_schools** | Hierarchical model | Meta-analysis |
| **dugongs** | Nonlinear regression | Growth curves |
| **glm** | Generalized linear model | Linear relationships |
| **gp** | Gaussian process | Latent variables |
| **surgical** | Count model | Medical data |

---

## Core Algorithm: D∥P∥L Refinement

RefineStat's iterative refinement operates on three program components:

- **D** (Data) — Fixed dataset specification from user
- **P** (Prior) — Prior distributions (resampled if fails diagnostics)
- **L** (Likelihood) — Likelihood/model specification (resampled if fails diagnostics)

**Algorithm (from paper, Definition 3.2):**

```
INPUT: Rmax (max refinements), α (max likelihood resamples), 
       β (target valid programs), K (min passing diagnostics)
OUTPUT: M* = argmax ELPD-LOO over valid models

r ← 0, ℓ ← 0, V ← ∅, P ← ∅, L ← ∅
while r < Rmax and |V| < β:
    Prog ← D ∥ P ∥ L  (generate program)
    if ¬Φ(Prog):      (validate semantics)
        r ← r + 1; continue
    
    Compute diagnostics d₁...d₇
    if ≥ K diagnostics pass:
        V ← V ∪ {Prog}  (valid program!)
    else if ℓ < α:
        L ← LCD(D ∥ P)  (resample likelihood)
        ℓ ← ℓ + 1
    else:
        P ← LCD(D)      (resample prior)
        r ← r + 1

return argmax ELPD-LOO from V
```

---

## Usage Examples

### 1. Run RefineStat on Your Data

```bash
# One-time per model: build Python DFA mask store (grammar="python"); required before main.py
python refinestat/dfa_constructor.py
# or: python refinestat/dfa_constructor.py --models "Qwen/Qwen2.5-3B-Instruct"

# Single seed (for testing)
python refinestat/main.py --seeds 1

# 10 seeds (standard experiment)
python refinestat/main.py --seeds 1-10

# Custom seeds (mix ranges and individual seeds)
python refinestat/main.py --seeds 1,3,5-10,15

# With temperature control
python refinestat/main.py --seeds 1-10 --temperature 0.5

# Custom output directory
python refinestat/main.py --seeds 1-10 --output results/experiment-v2
```

Subsequent runs with the same model can skip `dfa_constructor.py` if the mask pickle is already on disk.

### 2. Baseline Comparisons

```bash
# Single-pass baseline (no refinement)
python refinestat/base-itergen.py --seeds 1-5 --gpu 0

# Multi-run baseline for direct comparison
python Baseline/base-multi.py --runs 5 --seeds-per-run 5
```

---

## Output Structure

Results are organized as:

```
results/refinestat-main/1/
├── config.json                              # Experiment configuration
├── summary.txt                              # Experiment summary
├── token_usage.json                         # Per-seed token usage
├── cumulative_tokens.json                   # Cumulative tracking
├── token_budget.json                        # Token budget (max × 1.2)
│
├── google_codegemma-7b/                     # Per-model results
│   ├── eight_schools/
│   │   ├── best_program.py                  # Selected program
│   │   ├── best_program_diagnostics.txt     # Diagnostic scores
│   │   └── seed_1/, seed_2/, ...            # Per-seed outputs
│   └── ...other_datasets/
│
└── analysis/
    ├── all_seeds_best_programs_summary.csv  # One row per seed × dataset
    ├── all_seeds_best_programs_summary.xlsx
    ├── best_programs_summary.csv            # Best across all seeds
    ├── best_programs_summary.xlsx
    ├── aggregated_stats.xlsx                # Mean ± std across seeds ⭐
    └── aggregated_summary.txt               # Publication-ready summary
```

---

## Bayesian Workflow Checks

RefineStat implements all standard Bayesian diagnostic checks:

### Convergence Diagnostics
- **R̂ (Gelman-Rubin statistic):** < 1.05 indicates chains have mixed
- **BFMI (Bayesian Fraction of Missing Information):** > 0.3 indicates good energy transitions

### Efficiency Diagnostics
- **ESS bulk (Effective Sample Size):** ≥ 400 for central posterior mass
- **ESS tail (Tail ESS):** ≥ 100 for tail behavior

### Divergence Detection
- **Divergences:** 0 means no problematic NUTS transitions

### Importance Sampling Reliability
- **Pareto k:** < 0.7 for ≤ 20% of observations (via PSIS-LOO)

### Predictive Accuracy
- **ELPD-LOO:** Expected Log Pointwise Predictive Density via leave-one-out cross-validation

---

## Configuration

Edit `commons/config.py` to customize:

```python
config = {
    "temperature": 0.3,              # LLM sampling temperature
    "max_iterations": 35,            # Max refinement iterations
    "models": [                       # Models to test
        "google/codegemma-7b",
        "meta-llama/Meta-Llama-3-8B"
    ],
    "datasets": [                     # Datasets to run
        "eight_schools", "dugongs", "glm", "gp", "surgical"
    ],
    # ... more options
}
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for LLM inference)
- 16GB+ RAM

All dependencies in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages:
- PyMC 5.22.0 — Probabilistic programming
- ArviZ 0.21.0 — Bayesian diagnostics
- Transformers 4.38.2 — HuggingFace models
- Torch 2.4.1 — Deep learning
- Pandas 2.2.1 — Data manipulation

---

## Reproducibility

Random seeds are set in code, but **bit-identical or exact numerical replay is not guaranteed**: stochastic decoding on the GPU, **bfloat16** (and similar) numerics, parallel floating-point reductions, multi-step refinement, and best-program selection can all change outcomes between runs or environments. Installing from **`requirements.txt`** improves alignment of Python packages but does not remove these sources of variability.

**PyMC inference** is also stochastic: NUTS (and related samplers) rely on random initialization and within-chain randomness, so even identical generated model code can yield slightly different posteriors, diagnostics (e.g. R-hat, ESS, ELPD-LOO), and per-run summaries across machines or library builds. Treat reruns as methodological replicates and compare distributions or aggregates over seeds against saved artifacts rather than expecting exact scalar matches.

---