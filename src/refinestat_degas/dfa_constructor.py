"""One-time SynCode token-mask store builder for the DeGAS grammar.

Only needs the tokenizer, not the full model (mirrors RefineStat's
refinestat/dfa_constructor.py). Run once per (grammar, HF model) pair before using
DegasIterGen with that model; the mask store is cached under SYNCODE_CACHE and reused on
subsequent runs, including inside DegasIterGen.__init__ itself.
"""

import argparse
import os

from transformers import AutoTokenizer

from syncode.mask_store.mask_store import MaskStore
from syncode.parsers.grammars import Grammar

DEFAULT_MODELS = ["Qwen/Qwen2.5-Coder-7B-Instruct"]
GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar", "degas.lark")


def build(models=DEFAULT_MODELS, grammar_path=GRAMMAR_PATH):
    grammar = Grammar(grammar_path)
    for model_id in models:
        print(f"Building DFA mask store for {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        MaskStore.init_mask_store(grammar, tokenizer, use_cache=True, mode="grammar_strict")
        print(f"Done: {model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--grammar", default=GRAMMAR_PATH)
    args = parser.parse_args()
    build(args.models, args.grammar)
