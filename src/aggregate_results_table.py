"""Builds the cross-benchmark summary table (NLL, GW2, time) with mean +- std across each
method's 5 seeds, instead of results.ipynb's single overall-best-run number.

For each (benchmark, method) pair: every seed's own best candidate (final_candidates.json[0]
in its own run dir, i.e. that run's own winner -- matches best_candidate_for_method's
existing convention in results.ipynb) is independently substituted, recompiled, and scored
via DeGAS's own exact likelihood (compute_likelihood, with the MIN_SIGMA=1e-3 floor) and GW2
distance (restricted/reordered to var_names, matching results.ipynb's gw2_to_real). A seed
with no valid candidate (NO VALID CANDIDATE FOUND) is excluded from the mean/std and counted
separately; a method with zero valid seeds for a benchmark is reported as "n.c." (not
converged) rather than a number.

Real program / naive Gaussian benchmark / Evolutionary-PP-Synthesis baseline are single
values (no seeds to average over), reported with std=NaN and n=1.

Usage: python aggregate_results_table.py [program1 program2 ...]
    (default: all active benchmarks, i.e. everything except burglary, which is retired)

Writes two files to results/:
  - summary_table_raw.csv    one row per (benchmark, method, seed): NLL, GW2, time
  - summary_table.csv        one row per (benchmark, method): mean +- std, n_valid/n_total
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DeGAS" / "src"))

from optimization import compile2SOGA_text, produce_cfg_text, smooth_cfg, start_SOGA  # noqa: E402
from PROGRAMS.likelihood import compute_likelihood  # noqa: E402

from helpers.real_programs import get_real_programs  # noqa: E402
from helpers.dataset_generation import get_dataset, get_var_names  # noqa: E402
from helpers.gmm_distance import gmm_w2_distance  # noqa: E402
from refinestat_degas.text_utils import fix_uniform_trailing_literal  # noqa: E402
from corrected_likelihood import DISCRETE_VARS, compute_likelihood_corrected  # noqa: E402

torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
EVAL_DATA_SIZE = 50000

ALL_PROGRAMS = [
    "if", "mog1", "csi", "easytugwar", "biasedtugwar",
    "mixedcondition", "multiplebranches", "eyecolor", "hurricane",
]  # burglary retired -- excluded

METHOD_SUBDIRS = {
    "ALPS": None,  # runs live directly under results_root
    "grammar_mcmc": "grammar_mcmc",
    "refinestat": "refinestat",
    "refinestat_mcmc": "refinestat_mcmc",
}
KNOWN_METHOD_DIRNAMES = {v for v in METHOD_SUBDIRS.values() if v is not None} | {
    "gemma", "scalability_test", "ablation_no_gradient", "ablation_with_gradient", "export",
}

# Ported from results.ipynb's Evolutionary-PP-Synthesis baseline cell. NOTE: the notebook's
# own copy of this dict has a typo -- "mixedconditionals" instead of "mixedcondition" (our
# program name everywhere else) -- which silently made mixedcondition's evolutionary
# baseline unreachable ("No Evolutionary-PP-Synthesis result folder known"). Fixed here;
# worth fixing in results.ipynb too.
# Set this to the GitHub repo ("owner/name") hosting the Evolutionary-PP-Synthesis
# baseline's raw results, e.g. via the EVOLUTIONARY_REPO env var.
EVOLUTIONARY_REPO = os.environ.get("EVOLUTIONARY_REPO", "")
EVOLUTIONARY_RESULTS_DIR = {
    "if": "if_sketch0_30",
    "mog1": "mog1_sketch0_30",
    "easytugwar": "easytugwar_sketch0_30",
    "csi": "csi_nosketch",
    "biasedtugwar": "biasedtugwar_nosketch_30",
    "mixedcondition": "mixedcondition_sketch0_30",
    "multiplebranches": "multiplebranches_sketch0_30",
    "hurricane": "hurricane_nosketch",
    "eyecolor": "eyecolor_nosketch",
}


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read().decode("utf-8")


def list_github_subdirs(repo: str, path: str, ref: str = "master") -> list[str]:
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
    entries = json.loads(fetch_text(api_url))
    return [e["name"] for e in entries if e["type"] == "dir"]


def parse_best_txt(text: str) -> dict:
    result = {}
    m = re.search(r"Phenotype:\n(.*?)\n\nGenotype:", text, re.S)
    if m:
        result["phenotype"] = m.group(1).strip()
    m = re.search(r"Fitness on 5000 data:\n([^\n]+)", text)
    if m:
        result["fitness_test"] = float(m.group(1))
    return result


def convert_evolutionary_gm(text: str) -> str:
    pattern = r'gm\(\s*(\[[^\]]+\](?:,\s*\[[^\]]+\])*)\s*\)'

    def repl(m):
        elements = re.findall(r'\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]', m.group(1))
        pis = [float(e[0]) for e in elements]
        mus = [e[1] for e in elements]
        sigmas = [e[2] for e in elements]
        pi_sum = sum(pis) or 1.0
        norm_pis = ", ".join(f"{p / pi_sum:.6f}" for p in pis)
        return f'gm([{norm_pis}], [{", ".join(mus)}], [{", ".join(sigmas)}])'

    return re.sub(pattern, repl, text)


def convert_evolutionary_uniform(text: str) -> str:
    pattern = r'uniform\(\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]\s*,\s*([0-9.eE+-]+)\s*\)'

    def repl(m):
        a, b, c = float(m.group(1)), float(m.group(2)), m.group(3)
        return f'uniform([{a:.6f}, {a + b:.6f}], {c})'

    return re.sub(pattern, repl, text)


def translate_evolutionary_phenotype(phenotype: str) -> str:
    p = convert_evolutionary_gm(phenotype)
    p = convert_evolutionary_uniform(p)
    p = re.sub(r";;+", ";", p)
    # The external repo spells this variable correctly ("hairlength"); our own
    # eyecolor benchmark (helpers/real_programs.py) has a typo, "hairlenght",
    # baked into var_names and every result generated against it. Normalize
    # here so well-formed external candidates aren't dropped on a spelling
    # mismatch alone.
    p = re.sub(r"\bhairlength\b", "hairlenght", p)
    return p


def eval_naive_gaussian(var_names: list[str], data, real_dist, discrete_vars: set[str] | None = None) -> tuple[float, float]:
    """Per-variable independent Gaussian fit to the data's own empirical mean/std -- matches
    results.ipynb's "Simple benchmark" cell exactly."""
    data_arr = np.array(data, dtype=float)
    lines = []
    for i, name in enumerate(var_names):
        mean = data_arr[:, i].mean()
        std = data_arr[:, i].std()
        lines.append(f"{name} = gm([1.00], [{mean:.6f}], [{std:.6f}]);")
    return eval_rewritten("\n".join(lines), var_names, data, real_dist, discrete_vars)


def eval_evolutionary(program: str, var_names: list[str], data, real_dist, discrete_vars: set[str] | None = None) -> tuple[list[tuple[str, float, float]], int]:
    """One (run_name, nll, gw2) tuple per SUCCESSFULLY evaluated run in the external repo's
    results folder for this benchmark, each independently translated/compiled/scored through
    our own DeGAS pipeline (not the external repo's own self-reported fitness) -- one run
    there is the analog of one seed for our own methods. Also returns the total number of
    runs found (some may fail to fetch/parse/compile -- a degenerate GP-evolved phenotype is
    the analog of a method's own "no valid candidate" seed, so it should count toward the
    denominator even though it can't contribute a value)."""
    evo_folder = EVOLUTIONARY_RESULTS_DIR.get(program)
    if evo_folder is None:
        print(f"  no Evolutionary-PP-Synthesis folder known for '{program}'")
        return [], 0
    evo_results_path = f"results/{evo_folder}"
    try:
        run_names = list_github_subdirs(EVOLUTIONARY_REPO, evo_results_path)
    except Exception as e:
        print(f"  could not list {EVOLUTIONARY_REPO}/{evo_results_path}: {e}")
        return [], 0

    out = []
    for name in run_names:
        try:
            text = fetch_text(
                f"https://raw.githubusercontent.com/{EVOLUTIONARY_REPO}/master/{evo_results_path}/{name}/best.txt"
            )
            parsed = parse_best_txt(text)
            if "phenotype" not in parsed:
                print(f"    evolutionary run '{name}': no phenotype in best.txt")
                continue
            translated = translate_evolutionary_phenotype(parsed["phenotype"])
            nll, gw2 = eval_rewritten(translated, var_names, data, real_dist, discrete_vars)
            out.append((name, nll, gw2))
        except Exception as e:
            print(f"    evolutionary run '{name}' failed: {e}")
    return out, len(run_names)


def run_dirs_for_method(results_root: Path, method: str) -> list[Path]:
    subdir = METHOD_SUBDIRS[method]
    base = results_root / subdir if subdir else results_root
    if not base.exists():
        return []
    dirs = []
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        if subdir is None and sub.name in KNOWN_METHOD_DIRNAMES:
            continue
        if (sub / "best_fitness.csv").exists():
            dirs.append(sub)
    return sorted(dirs)


def load_elapsed_seconds(run_dir: Path) -> float | None:
    hp_path = run_dir / "hyperparams.json"
    if not hp_path.exists():
        return None
    with open(hp_path, encoding="utf-8") as f:
        return json.load(f).get("elapsed_seconds")


def parse_tensor_str(value) -> float:
    if isinstance(value, str) and value.startswith("tensor"):
        m = re.search(r"tensor\(([^,)]+)", value)
        if not m:
            raise ValueError(f"could not parse tensor repr: {value!r}")
        return float(m.group(1))
    return float(value)


def get_candidate_params(candidate: dict) -> dict:
    if "posterior_mean_params" in candidate:
        return candidate["posterior_mean_params"]
    return candidate.get("optimized_params", {})


def substitute_params(rewritten: str, params: dict) -> str:
    for name, value in params.items():
        value = parse_tensor_str(value)
        formatted = f"{value:.20f}"
        pattern = rf"_{re.escape(name)}(?![A-Za-z0-9_])"
        rewritten = re.sub(pattern, formatted, rewritten)
    return rewritten


def restrict_to_var_names(output_dist, var_names: list[str]) -> SimpleNamespace:
    idx = torch.tensor([output_dist.var_list.index(v) for v in var_names])
    mu = output_dist.gm.mu[:, idx]
    sigma = output_dist.gm.sigma[:, idx][:, :, idx]
    return SimpleNamespace(pi=output_dist.gm.pi, mu=mu, sigma=sigma)


def eval_rewritten(rewritten: str, var_names: list[str], data, real_dist, discrete_vars: set[str] | None = None) -> tuple[float, float]:
    compiled = compile2SOGA_text(fix_uniform_trailing_literal(rewritten))
    cfg = produce_cfg_text(compiled)
    smooth_cfg(cfg)
    dist = start_SOGA(cfg)
    if discrete_vars:
        nll = (-compute_likelihood_corrected(dist, var_names, data, discrete_vars)).item()
    else:
        nll = (-compute_likelihood(dist, var_names, data)).item()
    gw2 = gmm_w2_distance(restrict_to_var_names(real_dist, var_names), restrict_to_var_names(dist, var_names))
    return nll, gw2


def eval_seed(run_dir: Path, var_names: list[str], data, real_dist, discrete_vars: set[str] | None = None) -> tuple[float | None, float | None, float | None]:
    """Returns (nll, gw2, elapsed_seconds) for this seed's own best candidate, or
    (None, None, elapsed_seconds) if it has no valid candidate."""
    elapsed = load_elapsed_seconds(run_dir)
    candidates_path = run_dir / "final_candidates.json"
    if not candidates_path.exists():
        return None, None, elapsed
    with open(candidates_path, encoding="utf-8") as f:
        candidates = json.load(f)
    if not candidates:
        return None, None, elapsed
    candidate = candidates[0]
    params = get_candidate_params(candidate)
    if not params and "rewritten" not in candidate:
        return None, None, elapsed
    try:
        rewritten = substitute_params(candidate["rewritten"], params)
        nll, gw2 = eval_rewritten(rewritten, var_names, data, real_dist, discrete_vars)
    except Exception as e:
        print(f"    eval error in {run_dir.name}: {e}")
        return None, None, elapsed
    return nll, gw2, elapsed


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


# Evolutionary-PP-Synthesis has ~30 independent external runs vs. our own methods' 5 seeds.
# Averaging over all 30 isn't a like-for-like comparison (larger n -> tighter, less
# outlier-sensitive mean/std), so we randomly subsample down to the same n as everyone else.
# Fixed seed so the subsample is reproducible and not cherry-picked.
EVOLUTIONARY_SAMPLE_SIZE = 5
EVOLUTIONARY_SAMPLE_SEED = 42


def main(programs: list[str], corrected: bool = False) -> None:
    raw_rows = []
    summary_rows = []
    evo_rng = np.random.default_rng(EVOLUTIONARY_SAMPLE_SEED)

    for program in programs:
        print(f"=== {program} ===")
        results_root = RESULTS_ROOT / program
        if not results_root.exists():
            print(f"  no results dir for '{program}', skipping")
            continue

        var_names = get_var_names(program)
        data = get_dataset(program, EVAL_DATA_SIZE)
        discrete_vars = DISCRETE_VARS.get(program, set()) if corrected else None

        real_code = get_real_programs(program)
        real_cfg = produce_cfg_text(compile2SOGA_text(real_code))
        smooth_cfg(real_cfg)
        real_dist = start_SOGA(real_cfg)
        if discrete_vars:
            real_nll = (-compute_likelihood_corrected(real_dist, var_names, data, discrete_vars)).item()
        else:
            real_nll = (-compute_likelihood(real_dist, var_names, data)).item()
        raw_rows.append([program, "real_program", 1, real_nll, 0.0, None])
        summary_rows.append([program, "real_program", real_nll, np.nan, 0.0, np.nan, 1, 0, 1])

        # naive Gaussian baseline -- single value, no seeds
        try:
            ng_nll, ng_gw2 = eval_naive_gaussian(var_names, data, real_dist, discrete_vars)
            print(f"  naive_gauss: NLL={ng_nll:.4f}  GW2={ng_gw2:.4f}")
            raw_rows.append([program, "naive_gauss", 1, ng_nll, ng_gw2, None])
            summary_rows.append([program, "naive_gauss", ng_nll, np.nan, ng_gw2, np.nan, 1, 0, 1])
        except Exception as e:
            print(f"  naive_gauss: eval error: {e}")

        # Evolutionary-PP-Synthesis baseline -- one value per external run, averaged like a
        # method's own seeds
        evo_results, evo_n_total = eval_evolutionary(program, var_names, data, real_dist, discrete_vars)
        evo_valid = [(name, nll, gw2) for name, nll, gw2 in evo_results if np.isfinite(nll)]
        evo_nonfinite = evo_n_total - len(evo_valid)
        if len(evo_valid) > EVOLUTIONARY_SAMPLE_SIZE:
            idx = evo_rng.choice(len(evo_valid), size=EVOLUTIONARY_SAMPLE_SIZE, replace=False)
            evo_valid = [evo_valid[i] for i in idx]
        for name, nll, gw2 in evo_valid:
            raw_rows.append([program, "evolutionary", name, nll, gw2, None])
        evo_nlls = [nll for _, nll, gw2 in evo_valid]
        evo_gw2s = [gw2 for _, nll, gw2 in evo_valid]
        if evo_nlls:
            nll_mean, nll_std = mean_std(evo_nlls)
            gw2_mean, gw2_std = mean_std(evo_gw2s)
            print(f"  evolutionary: NLL={nll_mean:.4f}+-{nll_std:.4f}  GW2={gw2_mean:.4f}+-{gw2_std:.4f}  ({len(evo_nlls)}/{evo_n_total} valid, subsampled to {EVOLUTIONARY_SAMPLE_SIZE})")
            summary_rows.append([program, "evolutionary", nll_mean, nll_std, gw2_mean, gw2_std, len(evo_nlls), evo_nonfinite, evo_n_total])
        else:
            print(f"  evolutionary: n.c. (0/{evo_n_total} valid)")
            summary_rows.append([program, "evolutionary", "n.c.", np.nan, "n.c.", np.nan, 0, evo_nonfinite, evo_n_total])

        for method in METHOD_SUBDIRS:
            run_dirs = run_dirs_for_method(results_root, method)
            nlls, gw2s, times = [], [], []
            n_nonfinite = 0
            for run_dir in run_dirs:
                nll, gw2, elapsed = eval_seed(run_dir, var_names, data, real_dist, discrete_vars)
                raw_rows.append([program, method, run_dir.name, nll, gw2, elapsed])
                if nll is not None:
                    # A candidate can clear its own reliability gate yet still have a
                    # non-finite held-out NLL (e.g. refinestat_mcmc's checks don't gate on
                    # held-out NLL at all, only rank by it -- see mcmc_diagnostics.py). Real,
                    # not a bug, but inf/nan would otherwise poison the mean via inf
                    # arithmetic, so it's excluded from the mean/std and counted separately.
                    if np.isfinite(nll):
                        nlls.append(nll)
                        gw2s.append(gw2)
                    else:
                        n_nonfinite += 1
                if elapsed is not None:
                    times.append(elapsed)

            n_total = len(run_dirs)
            n_valid = len(nlls)
            if n_valid == 0:
                label = "n.c." if n_nonfinite == 0 else "n.c. (non-finite only)"
                print(f"  {method}: {label} (0/{n_total} finite, {n_nonfinite} non-finite)")
                summary_rows.append([program, method, "n.c.", np.nan, "n.c.", np.nan, 0, n_nonfinite, n_total])
                continue

            nll_mean, nll_std = mean_std(nlls)
            gw2_mean, gw2_std = mean_std(gw2s)
            time_mean = float(np.mean(times)) if times else np.nan
            suffix = f", {n_nonfinite} non-finite excluded" if n_nonfinite else ""
            print(f"  {method}: NLL={nll_mean:.4f}+-{nll_std:.4f}  GW2={gw2_mean:.4f}+-{gw2_std:.4f}  ({n_valid}/{n_total} valid{suffix})")
            summary_rows.append([program, method, nll_mean, nll_std, gw2_mean, gw2_std, n_valid, n_nonfinite, n_total])

    suffix = "_corrected" if corrected else ""
    raw_path = RESULTS_ROOT / f"summary_table_raw{suffix}.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "method", "seed", "NLL", "GW2", "elapsed_seconds"])
        writer.writerows(raw_rows)
    print(f"\nWrote {raw_path}")

    summary_path = RESULTS_ROOT / f"summary_table{suffix}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "method", "NLL_mean", "NLL_std", "GW2_mean", "GW2_std", "n_valid", "n_nonfinite", "n_total"])
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    corrected = "--corrected" in args
    args = [a for a in args if a != "--corrected"]
    programs = args if args else ALL_PROGRAMS
    main(programs, corrected=corrected)
