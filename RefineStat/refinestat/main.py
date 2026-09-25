#!/usr/bin/env python3
"""
main.py — Main experiment runner using full iterative refinement with per-iteration analysis.

Generates PyMC model code using refinegen.main (which records every iteration), assembles each
best program into the dataset boilerplate, computes per-iteration and best-program diagnostics,
and saves comprehensive analysis including per-iteration CSVs, best-program selection (across
all seeds), and token budgeting (for downstream optimization).

Key features:
  - Runs multiple seeds in a single experiment folder
  - Selects the best program per dataset ACROSS all seeds (not just within each seed)
  - Per-seed per-iteration CSVs with red-highlighted failing metrics
  - Global best program saved per model/dataset after all seeds complete
  - Token usage and cumulative token tracking across seeds

Usage:
    # One-time per model (builds DFA mask store with grammar="python"; required before main):
    python refinestat/dfa_constructor.py
    python refinestat/dfa_constructor.py --models "<hf_model_id>"
    python refinestat/main.py --seeds 1
    python refinestat/main.py --seeds 1-10
    python refinestat/main.py --seeds 1,3,5,7-10 --temperature 0.5
    python refinestat/main.py --seeds 5 --output results/main

SynCode: run dfa_constructor.py once per Hugging Face model you plan to use. That script
constructs the Python mask store with grammar="python" (simplifications on). This experiment
runner always passes the path to bundled python_grammar.lark to IterGen and expects the matching
mask pickle to already exist under SYNCODE_CACHE (same grammar bytes → same cache key).
"""
import os
import sys
import json
import gc
import traceback
import argparse
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import pandas as pd
import xarray as xr
from openpyxl.styles import PatternFill

# Set up paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from refinegen.main import iterative_refine
from refinegen.itergen.itergen.main import IterGen
from refinegen.checkers.library_specific.ppl_checker import PyMCChecker

from commons.data_pymc import datas_info, build_prompt_generic

# Optional dataset filter for this invocation, e.g. RS_DATASETS="eight_schools,dugongs".
# Leaving it unset runs the full datas_info registry exactly as upstream does.
_rs_datasets_filter = os.environ.get("RS_DATASETS")
if _rs_datasets_filter:
    _rs_wanted = {x.strip() for x in _rs_datasets_filter.split(",") if x.strip()}
    datas_info = [d for d in datas_info if d["name"] in _rs_wanted]
    _rs_missing = _rs_wanted - {d["name"] for d in datas_info}
    if _rs_missing:
        raise SystemExit(f"RS_DATASETS names not found in datas_info: {_rs_missing}")
from commons.config import config
from commons.utils import convert_np_types, set_seed


# Path to bundled Python Lark grammar (used by IterGen after dfa_constructor.py has built
# the mask store with grammar="python" for that tokenizer).
PYTHON_GRAMMAR_LARK = os.path.join(
    current_dir,
    "refinegen/itergen/itergen/syncode/syncode/parsers/grammars/python_grammar.lark",
)


# Default model — override with --models at the command line. Keep in sync with DEFAULT_MODELS in dfa_constructor.py.
DEFAULT_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3-8B",
]

# Reliability diagnostic thresholds used for Excel conditional formatting.
THRESHOLDS = {
    "r_hat":     1.05,   # max_r_hat must be < 1.05
    "ess_bulk":  400,    # min_ess_bulk must be >= 400
    "ess_tail":  100,    # min_ess_tail must be >= 100
    "divergences": 0,    # n_divergent must be 0
    "bfmi":      0.3,    # bfmi_values must be > 0.3
    "pareto_k":  0.2,    # prop_high_pareto_k must be <= 0.2
}

RED_FILL = PatternFill(start_color="FFFF0000", end_color="FFFF0000", fill_type="solid")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def extract_xarray_value(obj) -> Any:
    """Return a plain Python scalar or list from an xarray.DataArray."""
    if isinstance(obj, xr.DataArray):
        return float(obj.values) if obj.size == 1 else obj.values.tolist()
    return obj


def flatten_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested diagnostics dict into a single-level dict for DataFrame rows."""
    result = {}
    for key in ["max_r_hat", "min_ess_bulk", "min_ess_tail", "n_divergent",
                "prop_high_pareto_k", "elpd_loo", "loo_se", "reliability_score"]:
        if key in diagnostics:
            result[key] = extract_xarray_value(diagnostics[key])
    if "bfmi_values" in diagnostics:
        bfmi = diagnostics["bfmi_values"]
        if isinstance(bfmi, np.ndarray):
            bfmi = bfmi.tolist()
        if isinstance(bfmi, list):
            result["min_bfmi"] = min(bfmi)
            result["max_bfmi"] = max(bfmi)
            for i, val in enumerate(bfmi):
                result[f"bfmi_{i + 1}"] = val
    return result


def save_results_as_csv_and_excel(df: pd.DataFrame, csv_path: str, xlsx_path: str):
    """Save DataFrame as both CSV and Excel with conditional formatting (red fill on failing metrics)."""
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path)
    ws = wb.active
    col_indices = {col: i for i, col in enumerate(df.columns, start=1)}

    threshold_checks = {
        "max_r_hat":          lambda v: v >= THRESHOLDS["r_hat"],
        "min_ess_bulk":       lambda v: v < THRESHOLDS["ess_bulk"],
        "min_ess_tail":       lambda v: v < THRESHOLDS["ess_tail"],
        "n_divergent":        lambda v: v > THRESHOLDS["divergences"],
        "prop_high_pareto_k": lambda v: v > THRESHOLDS["pareto_k"],
    }

    for row_idx in range(2, len(df) + 2):
        row = df.iloc[row_idx - 2]
        if "reliability_score" not in col_indices or pd.isnull(row.get("reliability_score")):
            continue
        for col, fails in threshold_checks.items():
            if col in col_indices and pd.notnull(row.get(col)) and fails(row[col]):
                ws.cell(row=row_idx, column=col_indices[col]).fill = RED_FILL
        for bfmi_col in [c for c in df.columns if c.startswith("bfmi_")]:
            if bfmi_col in col_indices and pd.notnull(row.get(bfmi_col)):
                if row[bfmi_col] <= THRESHOLDS["bfmi"]:
                    ws.cell(row=row_idx, column=col_indices[bfmi_col]).fill = RED_FILL

    wb.save(xlsx_path)


def get_next_experiment_folder(output_dir: str) -> str:
    """Return the next auto-numbered experiment directory under output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    existing = [
        int(d) for d in os.listdir(output_dir)
        if d.isdigit() and os.path.isdir(os.path.join(output_dir, d))
    ]
    folder = os.path.join(output_dir, str(max(existing, default=0) + 1))
    os.makedirs(folder)
    return folder


def parse_seeds(seed_arg: str) -> List[int]:
    """
    Parse seed argument: single number, range (1-5), or comma-separated list (1,3,5,7-10).

    Args:
        seed_arg: Seed specification string

    Returns:
        Sorted list of unique seed integers
    """
    if not seed_arg:
        return []

    seeds = []
    for part in seed_arg.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-")
                seeds.extend(range(int(start.strip()), int(end.strip()) + 1))
            except ValueError:
                print(f"Invalid range: {part}")
        else:
            try:
                seeds.append(int(part))
            except ValueError:
                print(f"Invalid seed: {part}")

    return sorted(list(set(seeds)))


def process_interim_program(
    interim_program: List[Dict[str, Any]],
    model: str,
    dataset: str,
    seed: int,
    total_tokens: int,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Process interim_program entries and create a DataFrame with one row per iteration.

    Returns:
        Tuple of (entries_df, best_entry_dict or None)
    """
    entries = []
    for i, entry in enumerate(interim_program, start=1):
        row = {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "iteration": i,
            "reliability_score": entry.get("reliability_score"),
            "program": entry.get("program", ""),
        }
        if "cumulative_tokens" in entry:
            row["cumulative_tokens"] = entry["cumulative_tokens"]
        if "diagnostics" in entry:
            row.update(flatten_diagnostics(entry["diagnostics"]))
        entries.append(row)

    if not entries:
        return pd.DataFrame(), None

    df = pd.DataFrame(entries)
    best_entry = None

    if "reliability_score" in df.columns:
        valid = df.dropna(subset=["reliability_score"])
        if not valid.empty:
            max_rel = valid["reliability_score"].max()
            best_rows = valid[valid["reliability_score"] == max_rel]

            if len(best_rows) > 1 and "elpd_loo" in best_rows.columns:
                elpd_vals = best_rows["elpd_loo"].dropna()
                best_entry = best_rows.loc[elpd_vals.idxmax()].to_dict() if not elpd_vals.empty \
                    else best_rows.iloc[0].to_dict()
            else:
                best_entry = best_rows.iloc[0].to_dict()

    if best_entry:
        best_entry["total_tokens"] = total_tokens

    return df, best_entry


def select_global_best(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    From a list of best-per-seed candidates, select the best across all seeds.

    Selection criterion: highest reliability_score, with elpd_loo as tiebreaker.

    Args:
        candidates: List of per-seed best entry dicts (each has 'reliability_score', 'elpd_loo')

    Returns:
        The globally best entry dict, or None if no candidates
    """
    if not candidates:
        return None

    valid = [
        c for c in candidates
        if c.get("reliability_score") is not None
        and isinstance(c.get("reliability_score"), (int, float))
    ]
    if not valid:
        return None

    return sorted(
        valid,
        key=lambda x: (
            x.get("reliability_score", -1),
            x.get("elpd_loo") if isinstance(x.get("elpd_loo"), (int, float)) else -float("inf"),
        ),
        reverse=True,
    )[0]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_batch(
    seeds: List[int],
    output_dir: str = "results/refinestat-main",
    models: Optional[List[str]] = None,
):
    """
    Run iterative refinement across ALL seeds in a single experiment folder.

    For each model/dataset combination, the best program is determined by
    comparing candidates across ALL seeds, not just within a single seed.

    Args:
        seeds: List of seed integers to run
        output_dir: Base output directory (experiment folder created inside)
        models: List of model IDs to run
    """
    if models is None:
        models = DEFAULT_MODELS

    experiment_dir = get_next_experiment_folder(output_dir)
    num_datasets = len(datas_info)
    num_seeds = len(seeds)

    print(f"\n{'═' * 70}")
    print("  RefineStat-Main  —  Full Iterative Refinement + Per-Iteration Analysis")
    print(f"{'═' * 70}")
    print(f"  Seeds           : {num_seeds}  {seeds}")
    print(f"  Models          : {', '.join(models)}")
    print(f"  Datasets        : {num_datasets}  ({', '.join(d['name'] for d in datas_info)})")
    print(f"  Output          : {experiment_dir}")
    print(f"{'═' * 70}\n")

    cfg = {
        "seeds": seeds,
        "models": models,
        "datasets": [d["name"] for d in datas_info],
        "temperature": config.get("temperature", 0.3),
        "max_iter": config.get("max_iter", 35),
    }
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # all_best_entries: model -> dataset -> list of per-seed best candidates
    all_best_candidates: Dict[str, Dict[str, List[Dict]]] = {}
    # flat list of all best entries (one row per seed/dataset) for summary CSV
    all_best_rows: List[Dict] = []

    token_usage: Dict[str, Dict[str, Dict[int, int]]] = {}

    for model_id in models:
        safe_model = model_id.replace("/", "_")
        token_usage[safe_model] = {}
        all_best_candidates[safe_model] = {}

        print(f"\n{'█' * 70}")
        print(f"  MODEL: {model_id}")
        print(f"{'█' * 70}\n")

        for seed in seeds:
            print(f"  ── Seed {seed}/{num_seeds} ────────────────────────────────────────────────")

            set_seed(seed)

            # Create a fresh IterGen per seed so the seed is properly applied
            iter_gen = IterGen(
                # "python" (not PYTHON_GRAMMAR_LARK, a raw file path to the same .lark file):
                # Grammar.simplifications() only activates its interegular-lookback-avoiding
                # regex simplifications when self.name == "python" exactly, which the raw-path
                # form never sets, so it silently fed interegular the unsimplified terminal
                # regex (COMMENT/STRING/etc.) and crashed on "lookbacks are not implemented".
                grammar="python",
                model_id=model_id,
                device="cuda",
                do_sample=True,
                temperature=config.get("temperature", 0.3),
                recurrence_penalty=config.get("recurrence_penalty", 1.0),
                seed=seed,
                # IterGen's max_tokens is a TOTAL sequence budget (prompt + generation),
                # defaulting to 1000. Datasets with large embedded data arrays (e.g.
                # heart_disease's prompt alone tokenizes to ~16.5k tokens) silently blew
                # through that budget, leaving ~0 room to generate anything -- generation
                # would hit the length cap after a couple of lines. Qwen2.5-Coder-7B-Instruct
                # supports a 32768-token context, so 20000 comfortably covers even the
                # largest current prompt (heart_disease, ~16.5k) with headroom to spare.
                max_tokens=config.get("max_tokens", 20000),
            )

            for data in datas_info:
                dataset_name = data["name"]
                token_usage[safe_model].setdefault(dataset_name, {})
                all_best_candidates[safe_model].setdefault(dataset_name, [])

                print(f"    • {dataset_name:<20} ...", end=" ", flush=True)

                save_dir = os.path.join(experiment_dir, safe_model, dataset_name)
                os.makedirs(save_dir, exist_ok=True)

                prompt = build_prompt_generic(data)
                try:
                    symboltable, interim_program = iterative_refine(
                        prompt=prompt,
                        template=data["template_code"],
                        model_name=data["name"],
                        iter_gen=iter_gen,
                        checker_cls=PyMCChecker,
                        unit_name=config.get("unit_name", "function_call"),
                        max_iter=config.get("max_iter", 35),
                        checker_config={},
                        symboltable=config.get("pymc_symboltable", {"pm": "pymc"}),
                        dedent=True,
                        seed=seed,
                    )

                    total_tokens = iter_gen._metadata.get("total_tokens", 0)
                    token_usage[safe_model][dataset_name][seed] = total_tokens

                    # Save raw debug file
                    with open(os.path.join(save_dir, f"seed_{seed}_interim.txt"), "w") as f:
                        f.write(str(interim_program))

                    # Save each iteration entry as JSON
                    seed_dir = os.path.join(save_dir, f"seed_{seed}")
                    os.makedirs(seed_dir, exist_ok=True)
                    for idx, entry in enumerate(interim_program, start=1):
                        serial = convert_np_types(entry)
                        if "program" in serial and isinstance(serial["program"], str):
                            serial["program"] = serial["program"].strip("\n").replace("\t", "    ")
                        with open(os.path.join(seed_dir, f"entry_{idx}.json"), "w") as f:
                            json.dump(serial, f, indent=2)

                    # Per-seed per-iteration analysis
                    entries_df, best_entry = process_interim_program(
                        interim_program, safe_model, dataset_name, seed, total_tokens
                    )

                    if not entries_df.empty:
                        analysis_dir = os.path.join(
                            experiment_dir, "analysis", safe_model, dataset_name
                        )
                        os.makedirs(analysis_dir, exist_ok=True)

                        display_df = entries_df.drop(columns=["program"]) \
                            if "program" in entries_df.columns else entries_df

                        save_results_as_csv_and_excel(
                            display_df,
                            os.path.join(analysis_dir, f"seed_{seed}_entries.csv"),
                            os.path.join(analysis_dir, f"seed_{seed}_entries.xlsx"),
                        )

                        if best_entry:
                            best_json_path = os.path.join(analysis_dir, f"seed_{seed}_best.json")
                            with open(best_json_path, "w") as f:
                                json.dump(best_entry, f, indent=2)

                            # Collect candidate for global-best selection
                            candidate = {k: v for k, v in best_entry.items()}
                            candidate["_program"] = best_entry.get("program", "")
                            all_best_candidates[safe_model][dataset_name].append(candidate)

                            # Add row (without program code) to flat summary list
                            summary_entry = {k: v for k, v in best_entry.items() if k != "program"}
                            all_best_rows.append(summary_entry)

                    print("✓")

                except Exception as e:
                    print(f"✗  ({e})")
                    with open(os.path.join(save_dir, f"seed_{seed}_error.txt"), "w") as f:
                        f.write(traceback.format_exc())

            del iter_gen
            torch.cuda.empty_cache()
            gc.collect()

    # -----------------------------------------------------------------------
    # After ALL seeds: select global best per model/dataset and save
    # -----------------------------------------------------------------------

    print(f"\n{'─' * 70}")
    print("  Selecting global best programs across all seeds...")
    print(f"{'─' * 70}")

    global_best_rows: List[Dict] = []  # one row per model/dataset (best across seeds)

    for model_id in models:
        safe_model = model_id.replace("/", "_")
        for data in datas_info:
            dataset_name = data["name"]
            candidates = all_best_candidates.get(safe_model, {}).get(dataset_name, [])
            global_best = select_global_best(candidates)

            if global_best:
                save_dir = os.path.join(experiment_dir, safe_model, dataset_name)
                os.makedirs(save_dir, exist_ok=True)

                # Save globally best program
                program_code = global_best.pop("_program", global_best.get("program", ""))
                with open(os.path.join(save_dir, "best_program.py"), "w") as f:
                    if isinstance(program_code, str):
                        f.write(program_code.strip("\n"))
                    else:
                        f.write(str(program_code))
                with open(os.path.join(save_dir, "best_program_diagnostics.txt"), "w") as f:
                    f.write(json.dumps({k: v for k, v in global_best.items() if k != "program"}, indent=2))

                global_best_rows.append({k: v for k, v in global_best.items() if k != "_program"})
                print(f"  ✓ {safe_model}/{dataset_name}  (seed {global_best.get('seed', '?')}, "
                      f"reliability={global_best.get('reliability_score', 'N/A')})")
            else:
                print(f"  ✗ {safe_model}/{dataset_name}  (no successful programs)")

    # -----------------------------------------------------------------------
    # Save token aggregates
    # -----------------------------------------------------------------------

    print(f"\n{'─' * 70}")
    print("  Saving aggregates...")
    print(f"{'─' * 70}")

    cumulative_tokens: Dict[str, Dict[str, Dict[int, int]]] = {}
    for model, datasets in token_usage.items():
        cumulative_tokens[model] = {}
        for dataset, seeds_data in datasets.items():
            cumulative_tokens[model][dataset] = {}
            running = 0
            for seed_val in sorted(seeds_data.keys()):
                running += seeds_data[seed_val]
                cumulative_tokens[model][dataset][seed_val] = running

    for filename, data in [
        ("token_usage.json", token_usage),
        ("cumulative_tokens.json", cumulative_tokens),
    ]:
        path = os.path.join(experiment_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {filename}")

    # Token budget (120% of max cumulative)
    token_budget: Dict[str, Dict[str, int]] = {}
    for model, datasets in cumulative_tokens.items():
        token_budget[model] = {}
        for dataset, seeds_data in datasets.items():
            if seeds_data:
                token_budget[model][dataset] = int(max(seeds_data.values()) * 1.2)
    with open(os.path.join(experiment_dir, "token_budget.json"), "w") as f:
        json.dump(token_budget, f, indent=2)
    print("  Saved token_budget.json")

    # -----------------------------------------------------------------------
    # Save summary CSVs
    # -----------------------------------------------------------------------

    analysis_dir = os.path.join(experiment_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    essential_cols = [
        "model", "dataset", "seed", "iteration", "reliability_score",
        "elpd_loo", "total_tokens", "max_r_hat", "min_ess_bulk",
        "min_ess_tail", "n_divergent", "prop_high_pareto_k", "cumulative_tokens",
    ]

    def _save_summary(rows: List[Dict], csv_path: str, xlsx_path: str, label: str):
        if not rows:
            return
        df = pd.DataFrame(rows)
        for col in essential_cols:
            if col not in df.columns:
                df[col] = None
        first_cols = [c for c in essential_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in first_cols and c != "program"]
        df = df[first_cols + sorted(other_cols)]
        save_results_as_csv_and_excel(df, csv_path, xlsx_path)
        print(f"  Saved {label}")

    # All per-seed best entries (multiple rows per dataset)
    _save_summary(
        all_best_rows,
        os.path.join(analysis_dir, "all_seeds_best_programs_summary.csv"),
        os.path.join(analysis_dir, "all_seeds_best_programs_summary.xlsx"),
        "all_seeds_best_programs_summary  (one row per seed × dataset)",
    )

    # Global best entries (one row per dataset, best across all seeds)
    _save_summary(
        global_best_rows,
        os.path.join(analysis_dir, "best_programs_summary.csv"),
        os.path.join(analysis_dir, "best_programs_summary.xlsx"),
        "best_programs_summary            (global best per dataset)",
    )

    # -----------------------------------------------------------------------
    # Automatically generate aggregated statistics
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 70}")
    print("  Generating aggregated statistics across seeds...")
    print(f"{'─' * 70}")
    
    try:
        all_seeds_best_csv = os.path.join(analysis_dir, "all_seeds_best_programs_summary.csv")
        if os.path.exists(all_seeds_best_csv):
            # Compute aggregated stats
            df_all = pd.read_csv(all_seeds_best_csv)
            
            # Import aggregation logic from generate_aggregated_stats
            from refinestat.generate_aggregated_stats import compute_stats, write_excel
            
            stats_df, metrics = compute_stats(df_all)
            
            # Write aggregated Excel file
            aggregated_xlsx = os.path.join(analysis_dir, "aggregated_stats.xlsx")
            write_excel(stats_df, metrics, aggregated_xlsx)
            
            # Write aggregated summary text file
            summary_lines = []
            summary_lines.append(f"\n{'═' * 70}")
            summary_lines.append("  Aggregated Statistics (mean across seeds)")
            summary_lines.append(f"{'═' * 70}")
            display_cols = ["dataset", "reliability_score_mean", "elpd_loo_mean",
                            "max_r_hat_mean", "min_ess_bulk_mean", "n_divergent_mean"]
            available_cols = [c for c in display_cols if c in stats_df.columns]
            summary_lines.append(stats_df[available_cols].to_string(index=False))
            summary_lines.append(f"{'═' * 70}\n")
            
            summary_text = "\n".join(summary_lines)
            print(summary_text)
            
            aggregated_summary_file = os.path.join(analysis_dir, "aggregated_summary.txt")
            with open(aggregated_summary_file, "w") as f:
                f.write(summary_text)
            print(f"  Saved aggregated summary: {aggregated_summary_file}")
        else:
            print(f"  Warning: {all_seeds_best_csv} not found, skipping aggregation.")
    except Exception as e:
        print(f"  Warning: Could not generate aggregated statistics: {e}")

    print(f"\n{'═' * 70}")
    print(f"  Experiment complete.")
    print(f"  Output : {os.path.relpath(experiment_dir)}")
    print(f"  Seeds  : {seeds}")
    print(f"  Models : {', '.join(models)}")
    print(f"  Temperature: {config.get('temperature', 0.3)}")
    print(f"  Datasets run: {num_datasets} × {len(models)} model(s) × {num_seeds} seed(s)")
    print(f"{'═' * 70}\n")
    
    # Save summary to file
    summary_lines = [
        f"\n{'═' * 70}",
        f"  Experiment complete.",
        f"  Output : {os.path.relpath(experiment_dir)}",
        f"  Seeds  : {seeds}",
        f"  Models : {', '.join(models)}",
        f"  Temperature: {config.get('temperature', 0.3)}",
        f"  Datasets run: {num_datasets} × {len(models)} model(s) × {num_seeds} seed(s)",
        f"{'═' * 70}\n",
    ]
    summary_file = os.path.join(experiment_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Saved summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full iterative refinement runner with per-iteration analysis (all seeds in one folder)."
    )
    parser.add_argument(
        "--seeds", "-s", type=str, required=True,
        help="Seeds: single (5), range (1-10), or list (1,3,5,7-10)."
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=None,
        help="Sampling temperature (overrides commons.config)."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/refinestat-main",
        help="Base output directory."
    )
    parser.add_argument(
        "--models", "-m", type=str, default=None,
        help="Comma-separated model IDs. Defaults to DEFAULT_MODELS."
    )
    args = parser.parse_args()

    if args.temperature is not None:
        config["temperature"] = args.temperature

    seeds = parse_seeds(args.seeds)
    if not seeds:
        print("Error: No valid seeds specified!")
        sys.exit(1)

    models = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS

    run_batch(
        seeds=seeds,
        output_dir=args.output,
        models=models,
    )
