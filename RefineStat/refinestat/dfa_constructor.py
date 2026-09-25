#!/usr/bin/env python3
"""
Build the SynCode Python DFA mask store for Hugging Face models (tokenizer only).

SynCode builds the on-disk mask store using grammar=\"python\" so Grammar.simplifications()
applies during DFAMaskStore construction. refinestat/main.py always passes the path to the
bundled python_grammar.lark during experiments; that path must reuse a mask pickle built this way
(same grammar bytes → same cache key).

Run this once per model (tokenizer) before your first refinestat/main.py run for that model.
If the mask store for that model already exists on disk, you do not need to run this script again
(unless you cleared the SynCode cache or changed model id). Only loads tokenizers, not full models.
First run can take a long time.

Example:
    python refinestat/dfa_constructor.py
    python refinestat/dfa_constructor.py --models \"Qwen/Qwen2.5-3B-Instruct\"
    python refinestat/dfa_constructor.py --models \"org/model-a,org/model-b\"
"""
import argparse
import os
import sys
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from refinegen.itergen.itergen.syncode.syncode import common as syncode_common
from refinegen.itergen.itergen.syncode.syncode.dfa_mask_store import DFAMaskStore
from refinegen.itergen.itergen.syncode.syncode.parsers.grammars.grammar import Grammar


# Default models — keep in sync with DEFAULT_MODELS in refinestat/main.py. Override with --models.
DEFAULT_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3-8B",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SynCode Python DFA mask cache for HF models (grammar='python' only). "
        "Run before refinestat/main.py for each new model."
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        default=None,
        help="Comma-separated Hugging Face model IDs (same as --models for refinestat/main.py). "
        "If omitted, uses DEFAULT_MODELS in this file (must match main.py).",
    )
    args = parser.parse_args()
    if args.models is not None and args.models.strip():
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_ids = list(DEFAULT_MODELS)
    if not model_ids:
        print("Error: no models (set DEFAULT_MODELS in this file or pass --models).", file=sys.stderr)
        sys.exit(1)

    grammar = Grammar("python")
    for model_id in model_ids:
        print(f"Building Python DFA mask store for {model_id!r} (this may take a long time) …", flush=True)
        tokenizer = syncode_common.load_tokenizer(model_id)
        DFAMaskStore.load_dfa_mask_store(
            grammar=grammar,
            tokenizer=tokenizer,
            use_cache=True,
            logger=syncode_common.EmptyLogger(),
            mode="grammar_strict",
        )
        print("  Done.", flush=True)

    print("DFA mask store ready. You can run refinestat/main.py for these models.", flush=True)


if __name__ == "__main__":
    main()
