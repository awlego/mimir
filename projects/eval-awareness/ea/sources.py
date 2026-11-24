"""Load raw sources into a common item format.

An item is a dict with at least: id, family, label ("eval"/"deploy"), text
(the content with no benchmark scaffolding), and, for multiple-choice items,
options (list of str). Rendering into formats happens in ea/cells.py.
"""
from __future__ import annotations

import ast
import glob
import re

import numpy as np
import pandas as pd

from .paths import RAW

TRAIN_FAMILIES = ["mmlu", "arc", "gsm8k", "bbh", "strategyqa", "bbq", "jbb", "donotanswer"]
HELDOUT_FAMILIES = ["truthfulqa", "commonsenseqa", "ifeval", "harmfulqa"]
# Families whose native benchmark form is a bare request with no scaffolding.
BARE_FAMILIES = {"jbb", "donotanswer", "harmfulqa", "ifeval"}


def _arr(x):
    if isinstance(x, str):
        try:
            return list(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            return [x]
    return list(x)


def load_eval(family: str) -> list[dict]:
    d = pd.read_parquet(RAW / "eval" / f"{family}.parquet")
    items = []
    for i, r in d.iterrows():
        it = {"id": f"{family}-{i}", "family": family, "label": "eval", "options": None, "meta": {}}
        if family == "mmlu":
            it["text"] = r.question
            it["options"] = [str(o) for o in r.choices]
            it["meta"]["subject"] = r.subject.replace("_", " ")
        elif family in ("arc", "commonsenseqa"):
            it["text"] = r.question
            it["options"] = [str(o) for o in r.choices["text"]]
        elif family == "gsm8k":
            it["text"] = r.question
        elif family == "bbh":
            it["text"] = r.input
            it["meta"]["task"] = r.task
        elif family == "strategyqa":
            it["text"] = r.question
        elif family == "bbq":
            it["text"] = f"{r.context} {r.question}"
            it["options"] = [r.ans0, r.ans1, r.ans2]
            it["meta"]["category"] = r.category
        elif family == "jbb":
            it["text"] = r.Goal
        elif family == "donotanswer":
            it["text"] = r.question
        elif family == "truthfulqa":
            it["text"] = r.question
            opts = [r.best_answer] + _arr(r.incorrect_answers)[:3]
            rng = np.random.default_rng(i)
            it["options"] = [opts[j] for j in rng.permutation(len(opts))]
        elif family == "ifeval":
            it["text"] = r.prompt
        elif family == "harmfulqa":
            it["text"] = r.Question
        else:
            raise ValueError(family)
        items.append(it)
    # BBQ has many near-duplicate templated items; keep one per question_index x category.
    if family == "bbq":
        seen, out = set(), []
        for it, (_, r) in zip(items, d.iterrows()):
            k = (r.category, r.question_index, r.context_condition, r.question_polarity)
            if k not in seen:
                seen.add(k)
                out.append(it)
        items = out
    return items


MCQ_OPTION_LINES = re.compile(r"(?m)^\s*(?:\(?[A-Da-d][\).:]|[A-D]\s*[-–])\s+\S")


def load_wildchat() -> pd.DataFrame:
    """English, non-toxic, unredacted first user turns, de-duplicated."""
    rows = []
    for f in sorted(glob.glob(str(RAW / "wildchat" / "*.parquet"))):
        d = pd.read_parquet(f)
        d = d[(d.language == "English") & (~d.toxic) & (~d.redacted)]
        for h, ts, conv in zip(d.conversation_hash, d.timestamp, d.conversation):
            if conv[0]["role"] == "user":
                rows.append((h, ts, conv[0]["content"]))
    df = pd.DataFrame(rows, columns=["hash", "ts", "text"]).drop_duplicates("text")
    df = df[df.text.str.strip().str.len().between(15, 3000)].copy()
    df["year"] = df.ts.dt.year
    df["mcq_like"] = df.text.apply(lambda t: len(MCQ_OPTION_LINES.findall(t)) >= 3)
    df["id"] = "wildchat-" + df.hash
    return df.reset_index(drop=True)
