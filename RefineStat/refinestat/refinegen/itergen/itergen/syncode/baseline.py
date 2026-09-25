#!/usr/bin/env python3
import os
# 1) Force Aesara/PyMC onto CPU
# os.environ['AESARA_FLAGS'] = 'device=cpu'
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import sys
import json
import tempfile
import time
import random
import gc
from typing import Tuple, Union

import torch
import numpy as np
import xarray as xr
import arviz as az
import tiktoken
from transformers import AutoTokenizer

import syncode.infer as infer

# ——————————————————————————————————————————————————————————
# Define missing helper
def get_new_experiment_folder(base_dir: str, prefix: str, modelsize: str) -> str:
    # ts = int(time.time())
    folder = os.path.join(base_dir, f"{prefix}_{modelsize}")
    os.makedirs(folder, exist_ok=True)
    return folder

# Set random seeds for reproducibility
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Ensure commons path is available
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from commons.model_pymc import models_info, build_prompt_generic

# Convert numpy/xarray types for JSON
def convert_np_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, xr.DataArray):
        return obj.item() if obj.size == 1 else obj.values.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    else:
        return obj

# Save reliability aggregate
def save_reliability_aggregate(agg, path):
    with open(path, 'w') as f:
        json.dump(convert_np_types(agg), f, indent=2)
    print(f"Saved reliability aggregate → {path}")

# Save token count aggregate
def save_token_count_aggregate(agg, path):
    with open(path, 'w') as f:
        json.dump(convert_np_types(agg), f, indent=2)
    print(f"Saved token count aggregate → {path}")

# Store token count (per-seed and cumulative)
def store_token_count_aggregate(seed, dataset, llm_key, tokens, agg):
    agg.setdefault(dataset, {})
    agg[dataset].setdefault(llm_key, {'seeds': {}, 'cumulative': {}})
    agg[dataset][llm_key]['seeds'][seed] = tokens
    previous = max(
        (agg[dataset][llm_key]['cumulative'].get(s, 0)
         for s in agg[dataset][llm_key]['cumulative'] if s < seed),
        default=0
    )
    agg[dataset][llm_key]['cumulative'][seed] = previous + tokens

# Check model reliability
def check_model_reliability(idata: az.InferenceData):
    scores, reasons = {}, []
    summary = az.summary(idata)
    # R-hat
    max_r = summary['r_hat'].max()
    scores['r_hat'] = int(max_r < 1.05)
    if not scores['r_hat']:
        reasons.append(f"R-hat={max_r:.3f}>=1.05")
    # ESS bulk
    min_bulk = summary['ess_bulk'].min()
    scores['ess_bulk'] = int(min_bulk >= 400)
    if not scores['ess_bulk']:
        reasons.append(f"ESS_bulk={min_bulk:.1f}<400")
    # ESS tail
    if 'ess_tail' in summary.columns:
        min_tail = summary['ess_tail'].min()
        scores['ess_tail'] = int(min_tail >= 100)
        if not scores['ess_tail']:
            reasons.append(f"ESS_tail={min_tail:.1f}<100")
    else:
        scores['ess_tail'] = 0
        reasons.append("ESS_tail N/A")
    # divergences
    n_div = int(idata.sample_stats['diverging'].sum())
    scores['divergences'] = int(n_div == 0)
    if not scores['divergences']:
        reasons.append(f"{n_div} divergences")
    # BFMI
    bfmi_vals = az.bfmi(idata)
    ok_bfmi = int((bfmi_vals > 0.3).all())
    scores['bfmi'] = ok_bfmi
    if not ok_bfmi:
        reasons.append(f"Low BFMI={bfmi_vals}")
    # PSIS
    try:
        loo_res = az.loo(idata, pointwise=True)
        k = loo_res.pareto_k
        prop_high = np.mean(k > 0.7)
        scores['pareto_k'] = int(prop_high <= 0.20)
        if not scores['pareto_k']:
            reasons.append(f"{int((k>0.7).sum())}/{len(k)} k>0.7")
    except Exception as e:
        scores['pareto_k'] = 0
        reasons.append(f"LOO error: {e}")
    # ELPD
    try:
        lf = az.loo(idata, pointwise=True)
        elpd = lf.elpd_loo
        scores['elpd_success'] = 1
    except:
        elpd = None
        scores['elpd_success'] = 0
    reliability = sum(scores.values())
    return reliability, {'elpd': elpd, 'scores': scores, 'reasons': reasons}

# Execute PyMC code safely
def run_pymc_code(full_code: str):
    ns = {}
    exec(full_code, ns)
    return True, ns

# Insert snippet and return code + token count
def insert_model_code(template: str,
                      raw_snippet: Union[str, list],
                      trace_var: str,
                      model_name: str) -> Tuple[str,int]:
    if isinstance(raw_snippet, list):
        raw_snippet = ''.join(raw_snippet)
    lines = raw_snippet.replace('```','').splitlines()
    proc = []
    for L in lines:
        proc.append('' if not L.strip() else '\t'+L.lstrip())
        if 'pm.sample' in L:
            break
    snippet_code = '\n'.join(proc)

    # 2) Wrap tiktoken so it never segfaults
    try:
        enc = tiktoken.encoding_for_model(model_name)
        tok_count = len(enc.encode(snippet_code))
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tok_count = len(tok(snippet_code, return_tensors='pt').input_ids[0])

    out_lines, inserted = [], False
    for bl in template.splitlines():
        out_lines.append(bl.lstrip())
        if bl.strip().startswith('with pm.Model() as m:'):
            out_lines.append(snippet_code)
            inserted = True
    if not inserted:
        raise ValueError("'with pm.Model() as m:' not found in template")
    out_lines.append(f'\tsummary = az.summary({trace_var})')
    return '\n'.join(out_lines), tok_count

# Main experiment per seed
def main_experiment(root: str,
                    llm_key: str,
                    llm_name: str,
                    local_llm: infer.Syncode,
                    temperature: float,
                    seed: int):
    set_seed(seed)
    exp_dir = os.path.join(root, f'seed_{seed}', llm_key)
    os.makedirs(exp_dir, exist_ok=True)

    for entry in models_info:
        dataset = entry['name']
        ds_dir = os.path.join(exp_dir, dataset)
        os.makedirs(ds_dir, exist_ok=True)

        prompt = build_prompt_generic(entry)
        open(os.path.join(ds_dir,'prompt.txt'),'w').write(prompt)

        snippet = local_llm.infer(prompt)
        open(os.path.join(ds_dir,'snippet.txt'),'w').write(str(snippet))

        m = re.search(r"(\w+)\s*=\s*pm.sample", str(snippet))
        trace_var = m.group(1) if m else 'trace'

        full_code, tk_count = insert_model_code(
            entry['template_code'], snippet, trace_var, llm_name
        )
        open(os.path.join(ds_dir,'final_code.py'),'w').write(full_code)

        compiled, out = run_pymc_code(full_code)
        if compiled and isinstance(out, dict) and trace_var in out:
            rel, diag = check_model_reliability(out[trace_var])
            reliability_aggregate \
                .setdefault(dataset, {}) \
                .setdefault(llm_key, {})[seed] = {
                    'reliability_score': rel,
                    'elpd': diag['elpd']
                }

        store_token_count_aggregate(
            seed, dataset, llm_key, tk_count, token_count_aggregate
        )

# Combined multi-seed pipeline
def run_multiple_seeds(num_seeds: int=10,
                       temperature: float=0.3,
                       modelsize: str='medium'):
    root = get_new_experiment_folder('expts-org', 'exp', modelsize)
    open(os.path.join(root,'desc.txt'),'w') \
        .write(f'Seeds={num_seeds},temp={temperature},size={modelsize}')

    # 3) Build & instantiate each LLM once
    small_models  = ["microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen2.5-Coder-3B"]
    medium_models = [
        "meta-llama/Meta-Llama-3-8B", "google/codegemma-7b",
        "Qwen/Qwen2.5-Coder-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ]
    llm_list = small_models if modelsize=='small' else medium_models
    llm_objects = {}
    llm_name_map = {}
    for llm in llm_list:
        key = llm.replace('/','_')
        llm_objects[key] = infer.Syncode(
            mode='original',
            model=llm,
            do_sample=True,
            temperature=temperature,
            grammar='syncode/parsers/grammars/python_grammar.lark',
            device='cuda',
            max_new_tokens=400
        )
        llm_name_map[key] = llm

    # 4) Run seeds × LLMs without re-instantiating
    for s in range(1, num_seeds+1):
        for llm_key, local_llm in llm_objects.items():
            try:
                main_experiment(
                    root, llm_key, llm_name_map[llm_key],
                    local_llm, temperature, s
                )
            except Exception as e:
                print(f"[Seed {s}][{llm_key}] ERROR: {e}")

    # 5) Cleanup all LLMs at once
    for key, local_llm in llm_objects.items():
        try:
            local_llm.model.cpu()
        except Exception:
            pass
        del local_llm
    torch.cuda.empty_cache()
    gc.collect()

    # Save aggregates
    save_reliability_aggregate(
        reliability_aggregate,
        os.path.join(root, 'aggregated_reliability.json')
    )
    save_token_count_aggregate(
        token_count_aggregate,
        os.path.join(root, 'aggregated_token_count.json')
    )

    # Write leaderboards and progress files
    datasets = [e['name'] for e in models_info]
    for llm in llm_list:
        llm_key = llm.replace('/','_')
        lm_folder = os.path.join(root, llm_key)
        os.makedirs(lm_folder, exist_ok=True)

        # Leaderboard
        lb = []
        for s in range(1, num_seeds+1):
            for ds in datasets:
                met = reliability_aggregate.get(ds, {}) \
                                           .get(llm_key, {}) \
                                           .get(s)
                rel  = met['reliability_score'] if met else None
                elpd = met['elpd'] if met else None
                cum  = token_count_aggregate.get(ds, {}) \
                                            .get(llm_key, {}) \
                                            .get('cumulative', {}) \
                                            .get(s)
                code_path = os.path.join(
                    root, f'seed_{s}', llm_key, ds, 'final_code.py'
                )
                if not os.path.exists(code_path):
                    code_path = None
                lb.append({
                    'seed': s,
                    'dataset': ds,
                    'reliability_score': rel,
                    'elpd_loo': elpd,
                    'cumulative_tokens': cum,
                    'code_path': code_path
                })
        lb.sort(
            key=lambda x: (
                (x['reliability_score'] or -1),
                (x['elpd_loo'] or -float('inf'))
            ),
            reverse=True
        )
        with open(os.path.join(lm_folder,'leaderboard.json'),'w') as f:
            json.dump(lb, f, indent=2)

        # Progression
        for ds in datasets:
            cum_map = token_count_aggregate.get(ds, {}) \
                                           .get(llm_key, {}) \
                                           .get('cumulative', {})
            prog = []
            for s in sorted(cum_map):
                rel = reliability_aggregate.get(ds, {}) \
                                           .get(llm_key, {}) \
                                           .get(s, {}) \
                                           .get('reliability_score')
                prog.append({
                    'seed': s,
                    'cumulative_tokens': cum_map[s],
                    'reliability_score': rel
                })
            with open(os.path.join(lm_folder, f'progress_{ds}.json'),'w') as pf:
                json.dump(prog, pf, indent=2)

        print(f"Wrote files for {llm_key} in {lm_folder}")

    print('All done.')

# Global aggregates
reliability_aggregate = {}   # dataset -> llm_key -> seed -> diagnostics
token_count_aggregate = {}   # dataset -> llm_key -> { seeds: {}, cumulative: {} }

if __name__ == '__main__':
    temp  = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    size  = sys.argv[2]          if len(sys.argv) > 2 else 'medium'
    # seeds = int(sys.argv[3])     if len(sys.argv) > 3 else 10
    run_multiple_seeds(
        num_seeds=8,
        temperature=temp,
        modelsize=size
    )
