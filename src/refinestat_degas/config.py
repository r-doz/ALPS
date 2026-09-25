"""Configuration for the RefineStat-DeGAS baseline pipeline."""

from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
GRAMMAR_PATH = str(Path(__file__).resolve().parent / "grammar" / "degas.lark")
DEVICE = "cuda"
TEMPERATURE = 0.7
MAX_NEW_TOKENS_PER_UNIT = 150

DATA_SIZE = 1000
TRAIN_FRAC = 0.8

PROGRAMS = [
    "if", "mog1", "burglary", "csi", "easytugwar",
    "biasedtugwar", "mixedcondition", "multiplebranches", "eyecolor", "hurricane",
]

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
