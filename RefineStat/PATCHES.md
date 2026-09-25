# Patches relative to upstream RefineStat

This is a patched copy of [structuredllm/RefineStat](https://github.com/structuredllm/RefineStat).
The following changes were made to get usable results out of the released tool; none of them
change its intended algorithm or search strategy, they fix bugs that otherwise silently produced
degenerate or empty output.

## 1. Empty-results bug (`refinestat/refinegen/main.py`)

The incremental checker's `completion_in_progress` shortcut (in `AbstractChecker.parse()`)
bypasses `extract_call_info()`, so the symbol table used to identify the PyMC trace variable is
never populated -- even when the generated code executes successfully. This made every run report
zero valid candidates. Fixed by falling back to a direct scan of the executed namespace for an
`arviz.InferenceData` object once execution succeeds:

```python
idata_candidates = [v for v in ns.values() if isinstance(v, az.InferenceData)]
```

## 2. Stale `prev_output` bug (`refinestat/refinegen/main.py`)

After `backward_till_prompt()` / `backtrack_till_prompt()` resets the generator's internal state,
the harness's own `prev_output` variable (used to diff newly generated code) was never reset. Every
subsequent `forward()` call then looked like "no new code" all the way to `max_iter`, so only one
candidate was ever produced per seed regardless of the configured iteration budget. Fixed by adding
`prev_output = ""` immediately after each of the three backtrack-to-prompt call sites.

## 3. Undersized `max_tokens` (`refinestat/main.py`)

`IterGen(...)` was constructed with the library default `max_tokens=1000`, a *total* sequence
budget covering prompt + generation. For prompts as large as the real-world case study's
(`heart_disease_small` is ~9,700 tokens, `heart_disease` ~16,600), this left no room for the model
to actually generate anything, producing truncated single-statement programs. Increased to
`max_tokens=config.get("max_tokens", 20000)` (the underlying models support a 32k context).

## 4. `RS_DATASETS` environment filter (`refinestat/main.py`)

Added purely for convenience -- upstream `main.py` always runs every dataset registered in
`commons/data_pymc.py`'s `datas_info`, selectable only by editing that file. This patch adds an
optional environment variable that filters `datas_info` for a single invocation without touching
the file:

```python
_rs_datasets_filter = os.environ.get("RS_DATASETS")
if _rs_datasets_filter:
    _rs_wanted = {x.strip() for x in _rs_datasets_filter.split(",") if x.strip()}
    datas_info = [d for d in datas_info if d["name"] in _rs_wanted]
```

Leaving `RS_DATASETS` unset preserves the exact upstream behavior.
