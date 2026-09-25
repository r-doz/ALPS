#!/usr/bin/env python3
"""
run_eight_schools.py — Runs RefineStat's real main.py pipeline restricted to just the
eight_schools dataset, for fast targeted testing (see run_heart_disease.py for the same
pattern applied to heart_disease).

Usage (same CLI as main.py):
    python run_eight_schools.py --seeds 1 --models "Qwen/Qwen2.5-Coder-7B-Instruct"
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import argparse

import commons.data_pymc as data_pymc

data_pymc.datas_info = [d for d in data_pymc.datas_info if d["name"] == "eight_schools"]
assert len(data_pymc.datas_info) == 1, "eight_schools entry not found in datas_info"

import main  # noqa: E402  (must import after the datas_info patch above)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RefineStat main.py, restricted to eight_schools.")
    parser.add_argument("--seeds", "-s", type=str, required=True)
    parser.add_argument("--temperature", "-t", type=float, default=None)
    parser.add_argument("--output", "-o", type=str, default="results/refinestat-main")
    parser.add_argument("--models", "-m", type=str, default=None)
    args = parser.parse_args()

    if args.temperature is not None:
        main.config["temperature"] = args.temperature

    seeds = main.parse_seeds(args.seeds)
    if not seeds:
        print("Error: No valid seeds specified!")
        sys.exit(1)

    models = [m.strip() for m in args.models.split(",")] if args.models else main.DEFAULT_MODELS

    main.run_batch(seeds=seeds, output_dir=args.output, models=models)
