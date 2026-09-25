#!/usr/bin/env python3
"""
base-itergen.py — IterGen-only baseline without iterative refinement.

Generates PyMC model code using IterGen in a single forward pass (no checker feedback loop).
Terminates generation early when 'pm.sample' is detected in the output, then executes
the code and extracts Bayesian diagnostics. Serves as a direct code generation baseline
for comparison with iterative refinement approaches.

Key differences from refinestat/base.py:
  - No iterative refinement loop (generates once until pm.sample detected)
  - No checker feedback (single forward pass)
  - Early termination on pm.sample detection
  - Per-seed per-GPU directory structure for parallel execution
  - Detailed generation step tracking

Usage:
    python refinestat/base-itergen.py --seeds 1-5 --gpu 0
    python refinestat/base-itergen.py --seeds 1,3,7-10 --gpu 1 --output results/Baseline/Run-rate
    python refinestat/base-itergen.py --seeds 10 --gpu 0 --models "meta-llama/Meta-Llama-3-8B"
"""
import os
import sys
import re
import json
import gc
import argparse
import glob
import time
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import torch
from openpyxl.styles import PatternFill
from openpyxl import load_workbook

from refinegen.itergen.itergen.main import IterGen

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from commons.data_pymc import datas_info, build_prompt_generic
from commons.config import config
from commons.utils import convert_np_types, set_seed, check_model_reliability, run_pymc_code


# Default models
DEFAULT_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3-8B",
]

# Diagnostic thresholds for conditional formatting
THRESHOLDS = {
    "r_hat": 1.05,
    "ess_bulk": 400,
    "ess_tail": 100,
    "divergences": 0,
    "bfmi": 0.3,
    "pareto_k": 0.2,
}

RED_FILL = PatternFill(start_color="FFFF0000", end_color="FFFF0000", fill_type="solid")


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def _extract_xarray_value(obj: Any) -> Any:
    """Extract the actual value from an xarray.DataArray if needed."""
    if isinstance(obj, xr.DataArray):
        if obj.size == 1:
            return float(obj.values)
        return obj.values.tolist()
    return obj


def _flatten_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested diagnostics dictionary for DataFrame storage."""
    result = {}
    
    # Top-level metrics
    for key in [
        "max_r_hat", "min_ess_bulk", "min_ess_tail", "n_divergent",
        "prop_high_pareto_k", "elpd_loo", "loo_se", "reliability_score"
    ]:
        if key in diagnostics:
            result[key] = _extract_xarray_value(diagnostics[key])
    
    # BFMI values (scalar or array)
    if "bfmi_values" in diagnostics:
        bfmi_values = diagnostics["bfmi_values"]
        if isinstance(bfmi_values, (list, np.ndarray)):
            if isinstance(bfmi_values, np.ndarray):
                bfmi_values = bfmi_values.tolist()
            result["min_bfmi"] = min(bfmi_values)
            result["max_bfmi"] = max(bfmi_values)
            for i, val in enumerate(bfmi_values):
                result[f"bfmi_{i+1}"] = val
    
    return result


def _apply_conditional_formatting(df: pd.DataFrame, filepath: str):
    """Apply red highlight formatting to failing metrics in Excel."""
    # Save base DataFrame
    df.to_excel(filepath, index=False)
    
    # Open for formatting
    wb = load_workbook(filepath)
    ws = wb.active
    
    # Get column indices
    col_indices = {col: i + 1 for i, col in enumerate(df.columns)}
    
    # Apply highlighting per row
    for row in range(2, len(df) + 2):
        row_data = df.iloc[row - 2]
        if pd.isnull(row_data.get("reliability_score")):
            continue
        
        # R-hat check
        if "max_r_hat" in col_indices and pd.notnull(row_data.get("max_r_hat")):
            if row_data["max_r_hat"] >= THRESHOLDS["r_hat"]:
                ws.cell(row=row, column=col_indices["max_r_hat"]).fill = RED_FILL
        
        # ESS bulk check
        if "min_ess_bulk" in col_indices and pd.notnull(row_data.get("min_ess_bulk")):
            if row_data["min_ess_bulk"] < THRESHOLDS["ess_bulk"]:
                ws.cell(row=row, column=col_indices["min_ess_bulk"]).fill = RED_FILL
        
        # ESS tail check
        if "min_ess_tail" in col_indices and pd.notnull(row_data.get("min_ess_tail")):
            if row_data["min_ess_tail"] < THRESHOLDS["ess_tail"]:
                ws.cell(row=row, column=col_indices["min_ess_tail"]).fill = RED_FILL
        
        # Divergences check
        if "n_divergent" in col_indices and pd.notnull(row_data.get("n_divergent")):
            if row_data["n_divergent"] > THRESHOLDS["divergences"]:
                ws.cell(row=row, column=col_indices["n_divergent"]).fill = RED_FILL
        
        # BFMI checks
        for bfmi_col in [c for c in df.columns if c.startswith("bfmi_")]:
            if bfmi_col in col_indices and pd.notnull(row_data.get(bfmi_col)):
                if row_data[bfmi_col] <= THRESHOLDS["bfmi"]:
                    ws.cell(row=row, column=col_indices[bfmi_col]).fill = RED_FILL
        
        # Pareto-k check
        if "prop_high_pareto_k" in col_indices and pd.notnull(row_data.get("prop_high_pareto_k")):
            if row_data["prop_high_pareto_k"] > THRESHOLDS["pareto_k"]:
                ws.cell(row=row, column=col_indices["prop_high_pareto_k"]).fill = RED_FILL
    
    wb.save(filepath)


def _check_for_pm_sample(code: str) -> bool:
    """Check if code contains pm.sample marker."""
    return "pm.sample" in code


def _insert_model_code(template: str, raw_snippet: str, trace_var: str = "trace") -> str:
    """
    Insert generated code into PyMC template, stopping at pm.sample.
    
    Args:
        template: PyMC boilerplate
        raw_snippet: Generated code (may be partial)
        trace_var: Trace variable name
    
    Returns:
        Complete code with snippet inserted
    
    Raises:
        ValueError: If template marker not found
    """
    if isinstance(raw_snippet, list):
        raw_snippet = "".join(raw_snippet)
    
    # Process lines, stopping at pm.sample
    processed_lines = []
    for line in raw_snippet.replace("```", "").splitlines():
        if "pm.sample" in line:
            processed_lines.append("\t" + line.lstrip())
            break
        if not line.strip():
            processed_lines.append("")
        else:
            processed_lines.append("\t" + line.lstrip())
    
    snippet_code = "\n".join(processed_lines)
    
    # Insert into template
    out_lines = []
    inserted = False
    
    for line in template.splitlines():
        out_lines.append(line.lstrip())
        if line.strip().startswith("with pm.Model() as m:"):
            out_lines.append(snippet_code)
            inserted = True
    
    if not inserted:
        raise ValueError("'with pm.Model() as m:' not found in template")
    
    return "\n".join(out_lines)


def _save_json(data: Any, path: str, label: str = ""):
    """Helper to save JSON with consistent formatting."""
    with open(path, "w") as f:
        json.dump(convert_np_types(data), f, indent=2)
    if label:
        print(f"  Saved {label}")


def _parse_seeds(seed_arg: str) -> List[int]:
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
            # Range format: start-end
            try:
                start, end = part.split("-")
                start_num, end_num = int(start.strip()), int(end.strip())
                seeds.extend(range(start_num, end_num + 1))
            except ValueError:
                print(f"Invalid range format: {part}")
        else:
            # Single number
            try:
                seeds.append(int(part))
            except ValueError:
                print(f"Invalid seed number: {part}")
    
    return sorted(list(set(seeds)))


# ---------------------------------------------------------------------------
# Single Seed Runner
# ---------------------------------------------------------------------------

def _run_single_seed(
    seed: int,
    model_id: str,
    output_dir: str,
    gpu_id: int,
) -> Dict[str, Any]:
    """
    Run IterGen-only generation for one seed on one GPU.
    
    Args:
        seed: Seed number
        model_id: Model identifier
        output_dir: Output directory for results
        gpu_id: GPU device ID
    
    Returns:
        Dictionary with results, aggregates, and stats
    """
    torch.cuda.set_device(gpu_id)
    print(f"  [Seed {seed}] Using GPU {gpu_id}")
    
    set_seed(seed)
    model_key = model_id.replace("/", "_")
    
    # Create seed-specific output directory
    seed_output_dir = os.path.join(output_dir, f"seed_{seed}_gpu_{gpu_id}")
    os.makedirs(seed_output_dir, exist_ok=True)
    
    # Initialize tracking
    seed_results = []
    local_reliability_agg = {}
    local_token_agg = {}
    local_compilation_stats = {}
    local_programs = {}
    
    # Create IterGen instance
    iter_gen = IterGen(
        grammar=os.path.join(
            current_dir,
            "refinegen/itergen/itergen/syncode/syncode/parsers/grammars/python_grammar.lark"
        ),
        model_id=model_id,
        device=f"cuda:{gpu_id}",
        do_sample=True,
        temperature=config.get("temperature", 0.3),
        recurrence_penalty=config.get("recurrence_penalty", 1.0),
        seed=seed,
    )
    
    max_iterations = config.get("max_iter", 100)
    
    for data in datas_info:
        dataset_name = data["name"]
        print(f"    • {dataset_name:<20}", end=" ", flush=True)
        
        # Initialize program tracking
        local_programs[dataset_name] = {
            "prompt": build_prompt_generic(data),
            "template": data["template_code"],
            "generation_steps": [],
            "final_generated_code": None,
            "full_code_with_template": None,
        }
        
        try:
            prompt = build_prompt_generic(data)
            iter_gen.start(prompt)
            
            # Generate until pm.sample or max iterations
            generated_code = ""
            iteration = 0
            pm_sample_found = False
            generation_steps = []
            
            while iteration < max_iterations and not pm_sample_found:
                iteration += 1
                
                try:
                    out = iter_gen.forward(units=["function_call"], num=1)[-1]
                    generated_code = out
                    
                    step_info = {
                        "iteration": iteration,
                        "code_length": len(generated_code),
                        "pm_sample_found": _check_for_pm_sample(generated_code),
                    }
                    generation_steps.append(step_info)
                    
                    if _check_for_pm_sample(generated_code):
                        pm_sample_found = True
                        break
                
                except Exception as e:
                    print(f"Error iter {iteration}: {str(e)[:30]}")
                    step_info = {
                        "iteration": iteration,
                        "error": str(e)[:100],
                        "code_length": len(generated_code) if "generated_code" in locals() else 0,
                        "pm_sample_found": False,
                    }
                    generation_steps.append(step_info)
                    break
            
            # Get token count
            total_tokens = iter_gen._metadata.get("total_tokens", 0)
            
            # Track tokens
            local_token_agg.setdefault(model_key, {})
            local_token_agg[model_key][dataset_name] = total_tokens
            
            # Try to compile and run
            compiled_successfully = False
            reliability_score = None
            elpd_loo = None
            diagnostics = None
            full_code = None
            
            if pm_sample_found:
                try:
                    full_code = _insert_model_code(data["template_code"], generated_code)
                    success, ns = run_pymc_code(full_code)
                    
                    if success:
                        compiled_successfully = True
                        
                        # Find trace in namespace
                        trace_names = [
                            k for k, v in ns.items()
                            if hasattr(v, "groups") and "posterior" in v.groups
                        ]
                        
                        if trace_names:
                            idata = ns[trace_names[0]]
                            reliability_score, diagnostics = check_model_reliability(idata)
                            elpd_loo = diagnostics.get("elpd_loo")
                            print("✓", end="")
                        else:
                            print("✗", end="")
                    else:
                        print("✗", end="")
                
                except Exception as e:
                    print("✗", end="")
            else:
                print("✗", end="")
            
            # Store program info
            local_programs[dataset_name].update({
                "generation_steps": generation_steps,
                "final_generated_code": generated_code,
                "full_code_with_template": full_code,
                "pm_sample_found": pm_sample_found,
                "iterations": iteration,
                "compiled": compiled_successfully,
                "total_tokens": total_tokens,
            })
            
            # Store compilation stats
            local_compilation_stats.setdefault(dataset_name, {})
            local_compilation_stats[dataset_name][model_key] = {
                "compiled": compiled_successfully,
                "total_programs": 1,
                "compilation_rate": 1.0 if compiled_successfully else 0.0,
            }
            
            # Store reliability if successful
            if compiled_successfully and reliability_score is not None:
                local_reliability_agg.setdefault(dataset_name, {})
                local_reliability_agg[dataset_name][model_key] = {
                    "reliability_score": reliability_score,
                    "elpd_loo": _extract_xarray_value(elpd_loo),
                    "diagnostics": convert_np_types(diagnostics) if diagnostics else {},
                }
            
            # Create result record
            result_record = {
                "model": model_key,
                "dataset": dataset_name,
                "seed": seed,
                "gpu_id": gpu_id,
                "compiled": compiled_successfully,
                "reliability_score": reliability_score,
                "total_tokens": total_tokens,
                "iterations": iteration,
            }
            seed_results.append(result_record)
            print()
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:30]}")
            result_record = {
                "model": model_key,
                "dataset": dataset_name,
                "seed": seed,
                "gpu_id": gpu_id,
                "compiled": False,
                "reliability_score": None,
                "total_tokens": 0,
                "iterations": 0,
            }
            seed_results.append(result_record)
    
    # Cleanup
    del iter_gen
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save results
    seed_results_file = os.path.join(seed_output_dir, f"seed_{seed}_gpu_{gpu_id}_results.json")
    detailed_results = {
        "seed_results": seed_results,
        "reliability_aggregate": local_reliability_agg,
        "token_count_aggregate": local_token_agg,
        "compilation_stats": local_compilation_stats,
    }
    with open(seed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2, default=convert_np_types)
    
    # Save generated code and logs per dataset
    programs_dir = os.path.join(seed_output_dir, "programs")
    os.makedirs(programs_dir, exist_ok=True)
    
    for dataset_name, prog_info in local_programs.items():
        ds_dir = os.path.join(programs_dir, dataset_name)
        os.makedirs(ds_dir, exist_ok=True)
        
        # Save prompt
        with open(os.path.join(ds_dir, "prompt.txt"), "w") as f:
            f.write(prog_info["prompt"])
        
        # Save final generated code
        with open(os.path.join(ds_dir, "final_generated_code.py"), "w") as f:
            f.write(prog_info["final_generated_code"] or "# No code generated")
        
        # Save full code if successful
        if prog_info["full_code_with_template"]:
            with open(os.path.join(ds_dir, "full_code_with_template.py"), "w") as f:
                f.write(prog_info["full_code_with_template"])
        
        # Save generation summary
        with open(os.path.join(ds_dir, "generation_summary.txt"), "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Seed: {seed}, GPU: {gpu_id}\n")
            f.write(f"Iterations to pm.sample: {prog_info['iterations']}\n")
            f.write(f"pm.sample found: {prog_info['pm_sample_found']}\n")
            f.write(f"Code compiled: {prog_info['compiled']}\n")
            f.write(f"Total tokens: {prog_info['total_tokens']}\n")
    
    print(f"  Saved results to {seed_output_dir}")
    
    return detailed_results


def _merge_aggregates(output_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Merge results from all seed result files."""
    global_reliability = {}
    global_tokens = {}
    global_compilation = {}
    
    seed_files = glob.glob(os.path.join(output_dir, "seed_*_gpu_*/seed_*_gpu_*_results.json"))
    
    if not seed_files:
        print("No seed result files found!")
        return global_reliability, global_tokens, global_compilation
    
    print(f"Merging {len(seed_files)} seed result files...")
    
    for seed_file in seed_files:
        try:
            with open(seed_file, "r") as f:
                result = json.load(f)
            
            # Merge reliability
            for dataset, models in result.get("reliability_aggregate", {}).items():
                global_reliability.setdefault(dataset, {}).update(models)
            
            # Merge tokens
            for model, datasets in result.get("token_count_aggregate", {}).items():
                if model not in global_tokens:
                    global_tokens[model] = {}
                for dataset, tokens in datasets.items():
                    global_tokens[model][dataset] = tokens
            
            # Merge compilation
            for dataset, models in result.get("compilation_stats", {}).items():
                global_compilation.setdefault(dataset, {}).update(models)
        
        except Exception as e:
            print(f"Error merging {seed_file}: {e}")
    
    return global_reliability, global_tokens, global_compilation


def _create_summary(
    output_dir: str,
    reliability_agg: Dict,
    token_agg: Dict,
    compilation_agg: Dict,
):
    """Create comprehensive statistical summary."""
    print("Creating summary statistics...")
    
    all_results = []
    
    # Collect all results
    for dataset, models in reliability_agg.items():
        for model, seeds_data in models.items():
            for seed, data in seeds_data.items():
                result = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "reliability_score": data.get("reliability_score"),
                    "elpd_loo": data.get("elpd_loo"),
                }
                
                # Add flattened diagnostics
                if "diagnostics" in data:
                    result.update(_flatten_diagnostics(data["diagnostics"]))
                
                all_results.append(result)
    
    if not all_results:
        print("No results to summarize")
        return
    
    df = pd.DataFrame(all_results)
    
    # Save summary
    summary_csv = os.path.join(output_dir, "summary.csv")
    summary_xlsx = os.path.join(output_dir, "summary.xlsx")
    
    df.to_csv(summary_csv, index=False)
    _apply_conditional_formatting(df, summary_xlsx)
    
    print(f"  Saved summary.csv")
    print(f"  Saved summary.xlsx")
    
    # Print statistics
    total_experiments = len(all_results)
    total_compiled = sum(1 for r in all_results if pd.notnull(r["reliability_score"]))
    
    summary_lines = []
    summary_lines.append(f"\n{'═' * 70}")
    summary_lines.append(f"  Overall Statistics")
    summary_lines.append(f"{'═' * 70}")
    summary_lines.append(f"  Total experiments  : {total_experiments}")
    summary_lines.append(f"  Compiled           : {total_compiled}")
    summary_lines.append(f"  Failed             : {total_experiments - total_compiled}")
    summary_lines.append(f"  Compilation rate   : {total_compiled / total_experiments * 100:.1f}%")
    summary_lines.append(f"{'═' * 70}\n")
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(summary_text)
    print(f"  Saved summary: {summary_file}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_experiment(
    seeds: List[int],
    models: List[str],
    gpu_id: int,
    output_dir: str,
):
    """
    Run IterGen-only baseline across seeds.
    
    Args:
        seeds: List of seed numbers
        models: List of model IDs
        gpu_id: GPU device ID
        output_dir: Output directory
    """
    num_datasets = len(datas_info)
    num_seeds = len(seeds)
    num_models = len(models)
    total_experiments = num_seeds * num_datasets * num_models
    
    print(f"\n{'═' * 70}")
    print("  IterGen Only Baseline  —  No Iterative Refinement")
    print(f"{'═' * 70}")
    print(f"  Seeds       : {num_seeds}  {seeds}")
    print(f"  Models      : {num_models}  ({models[0] if len(models) == 1 else f'{models[0]}, ...'})")
    print(f"  Datasets    : {num_datasets}  ({', '.join(d['name'] for d in datas_info)})")
    print(f"  GPU         : {gpu_id}")
    print(f"  Experiments : {total_experiments}")
    print(f"{'═' * 70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    cfg = {
        "seeds": seeds,
        "models": models,
        "num_seeds": num_seeds,
        "num_models": num_models,
        "datasets": [d["name"] for d in datas_info],
        "gpu_id": gpu_id,
        "method": "IterGen only (no iterative refinement)",
        "early_termination": "pm.sample detection",
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    
    # Run experiments
    all_seed_results = []
    
    for model_id in models:
        print(f"\n{'█' * 70}")
        print(f"  MODEL: {model_id}")
        print(f"{'█' * 70}")
        
        for seed in seeds:
            print(f"\n  Seed {seed}/{num_seeds}")
            
            try:
                result = _run_single_seed(seed, model_id, output_dir, gpu_id)
                all_seed_results.extend(result["seed_results"])
            
            except Exception as e:
                print(f"  ERROR: {e}")
    
    # Merge and summarize
    print(f"\n{'─' * 70}")
    print("  Creating consolidated summary...")
    print(f"{'─' * 70}")
    
    reliability_agg, token_agg, compilation_agg = _merge_aggregates(output_dir)
    
    # Save aggregates
    _save_json(reliability_agg, os.path.join(output_dir, "aggregated_reliability.json"),
              "aggregated_reliability.json")
    _save_json(token_agg, os.path.join(output_dir, "aggregated_token_count.json"),
              "aggregated_token_count.json")
    _save_json(compilation_agg, os.path.join(output_dir, "aggregated_compilation_stats.json"),
              "aggregated_compilation_stats.json")
    
    # Create summary
    _create_summary(output_dir, reliability_agg, token_agg, compilation_agg)
    
    print(f"\nResults saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IterGen-only baseline (no iterative refinement, early termination on pm.sample)"
    )
    parser.add_argument(
        "--seeds", "-s", type=str, required=True,
        help="Seeds: single (5), range (1-5), or comma-separated list (1,3,5,7-10)"
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/Baseline/Run-rate",
        help="Output directory"
    )
    parser.add_argument(
        "--models", "-m", type=str, default=None,
        help="Comma-separated model IDs (default: meta-llama/Meta-Llama-3-8B)"
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        print("Error: No valid seeds specified!")
        sys.exit(1)
    
    # Parse models
    models = (
        [m.strip() for m in args.models.split(",")]
        if args.models
        else DEFAULT_MODELS
    )
    
    print(f"Running seeds: {seeds}")
    print(f"Using GPU: {args.gpu}")
    print(f"Models: {models}")
    print(f"Method: IterGen only (no iterative refinement)")
    print(f"Early termination: pm.sample detection")
    
    run_experiment(seeds, models, args.gpu, args.output)
