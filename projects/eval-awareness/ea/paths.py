"""Project paths. Generated data lives under data/ and runs/, which are gitignored."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CELLS = DATA / "cells"
LLM_CACHE = DATA / "llm_cache"
RUNS = ROOT / "runs"
MANIFESTS = ROOT / "manifests"  # small, committed summaries
