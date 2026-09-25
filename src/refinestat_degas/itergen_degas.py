"""Statement-level SynCode-constrained generation for DeGAS programs.

Adapted from RefineStat's IterGen (refinestat/refinegen/itergen/itergen/main.py), but built
directly against the currently pip-installed `syncode` package's public API (MaskStore,
create_parser(..., use_symbol_pos_map=True), SymbolPosMap) rather than RefineStat's vendored
fork, whose glue modules (common.py, parsers.py) turned out to be missing from the published
repo. The token-masking + unit-position-tracking machinery of the original is already
grammar-agnostic (driven purely by the Lark grammar + SymbolPosMap, nothing Python-AST
specific) -- what changes here is the grammar (degas.lark) and the unit granularity: DeGAS
generates one `toplevel` statement at a time (one assignment, or one full
`if/else/end if` block -- see grammar/degas.lark's comment on why `toplevel` is a distinct
non-inlined rule from the `statement` used inside nested if/else blocks).

Deliberate simplification vs. the original IterGen: no `past_key_values` KV-cache reuse
across generation steps. Every token step re-encodes prompt+generated_text and runs a full
forward pass, and `backward()` just truncates the text (no token-boundary trace bookkeeping).
This trades some compute for a much simpler, less bug-prone implementation -- acceptable
here since these benchmark programs are only a handful of short statements (well under a
few hundred tokens total per candidate).
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from syncode.mask_store.mask_store import MaskStore
from syncode.parsers import create_parser
from syncode.parsers.grammars import Grammar
from syncode.parsers.itergen_parser import SymbolPosMap


class DegasIterGen:
    def __init__(
        self,
        grammar_path: str,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.7,
        max_new_tokens_per_unit: int = 150,
    ):
        self.grammar = Grammar(grammar_path)
        self.device = device
        self.temperature = temperature
        self.max_new_tokens_per_unit = max_new_tokens_per_unit

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        self.model.eval()

        # Tokenizer-only; cached to disk under SYNCODE_CACHE. Must be built once per
        # (grammar, tokenizer) pair before first use -- see dfa_constructor.py.
        self.mask_store = MaskStore.init_mask_store(
            self.grammar, self.tokenizer, use_cache=True, mode="grammar_strict"
        )

        self.prompt_text: str = ""
        self.generated_text: str = ""
        self.symbol_pos_map: Optional[SymbolPosMap] = None
        self._parser = None

    def start(self, messages: list[dict]) -> None:
        """Begin a new generation session from a chat-formatted prompt."""
        self.prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self.generated_text = ""
        self.symbol_pos_map = SymbolPosMap()
        self._parser = create_parser(self.grammar, parser="lalr", use_symbol_pos_map=True)

    def _parse_result(self):
        return self._parser.get_acceptable_next_terminals(
            self.generated_text, symbol_pos_map=self.symbol_pos_map
        )

    @torch.inference_mode()
    def forward(self, units: list[str], num: int = 1) -> str:
        """Generate tokens until `num` more of any unit in `units` has completed."""
        start_counts = {u: self.symbol_pos_map.get_symbol_count(u) for u in units}

        for _ in range(self.max_new_tokens_per_unit):
            full_text = self.prompt_text + self.generated_text
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits[0, -1, :]

            try:
                parse_result = self._parse_result()
            except Exception:
                # Should not normally happen (generated_text so far was already validated
                # by a prior iteration of this same loop), but bail out defensively rather
                # than crash the whole run -- the caller's checker will reject this
                # (likely-incomplete) text and backward() past it.
                break
            remainder = parse_result.remainder
            if isinstance(remainder, str):
                remainder = remainder.encode("utf-8")
            parse_result.remainder = remainder

            accept_mask = self.mask_store.get_accept_mask(parse_result)
            if accept_mask.shape[0] < logits.shape[0]:
                pad = torch.zeros(logits.shape[0] - accept_mask.shape[0], dtype=torch.bool)
                accept_mask = torch.cat([accept_mask, pad])
            accept_mask = accept_mask[: logits.shape[0]].to(logits.device)

            if accept_mask.any():
                logits = logits.masked_fill(~accept_mask, float("-inf"))
            # else: no acceptable tokens under the (underapproximating) mask -- fall back
            # to unmasked logits rather than deadlocking generation.

            if self.temperature and self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(logits).item()

            if next_token == self.tokenizer.eos_token_id:
                break

            next_str = self.tokenizer.decode([next_token], skip_special_tokens=True)
            if not next_str:
                break
            self.generated_text += next_str
            try:
                self._parse_result()  # refresh parser state + symbol_pos_map with the new text
            except Exception:
                # SynCode's grammar_strict mask is a documented over-approximation (see
                # MaskStore's docstring) -- it can occasionally admit a token that turns out
                # not to extend to anything parseable (e.g. a keyword like "uniform" used
                # where a plain variable was expected). Roll back this token rather than
                # crash: the caller's checker will reject the (now unchanged) text and
                # backward() past it, exactly like any other invalid completion.
                self.generated_text = self.generated_text[: -len(next_str)]
                break

            for u in units:
                new_count = self.symbol_pos_map.get_symbol_count(u)
                if new_count >= start_counts[u] + num:
                    # LALR reduces lag behind by however much lookahead was needed to
                    # confirm them, so by the time symbol_pos_map reflects the completed
                    # unit, generated_text has typically already run on into the next
                    # unit's opening token(s) (e.g. "a = gm(...); b" once "b" makes the
                    # parser confident enough to reduce the first statement). Truncate back
                    # to the unit's actual end position so callers only see the completed
                    # unit, not this lookahead spillover.
                    end_pos = self.symbol_pos_map.get_symbol_pos_end(u, start_counts[u] + num - 1)
                    self.symbol_pos_map.crop(end_pos)
                    self.generated_text = self.generated_text[:end_pos]
                    return self.generated_text

        return self.generated_text

    def backward(self, unit: str, num: int = 1) -> str:
        """Discard generated text back to before the `num`-th-from-the-end `unit`."""
        cnt = self.symbol_pos_map.get_symbol_count(unit)
        target_char_pos = (
            self.symbol_pos_map.get_symbol_pos_start(unit, cnt - num) if cnt - num >= 0 else 0
        )
        self.symbol_pos_map.crop(target_char_pos)
        self.generated_text = self.generated_text[:target_char_pos]
        return self.generated_text
