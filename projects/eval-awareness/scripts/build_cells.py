"""Build the four format x label cells with train/val/test splits.

Writes data/cells/{train,val,test}.jsonl and data/cells/wildchat_testers.jsonl
(gitignored), plus manifests/cells_manifest.json and manifests/cue_conflict_samples.md
(small, committed). LLM steps (casualize, paraphrase, tester judge) go through
ea.llm, which caches every call, so re-running is free and resumes after interruption.

Usage (from projects/eval-awareness/):
    env3.12/bin/python -m scripts.build_cells
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import numpy as np
import pandas as pd

from ea import cells as C
from ea.llm import call_summary, llm_call, llm_map
from ea.paths import CELLS, MANIFESTS
from ea.sources import load_eval, load_wildchat

SEED = 0
PARAPHRASE_FRAC = 0.3
VAL_FRAC = 0.2

# Train-family allocation per origin (train + val together).
EVAL_TRAIN_ALLOC = {
    "bench_native": {"mmlu": 125, "arc": 100, "gsm8k": 100, "bbh": 100, "strategyqa": 100, "bbq": 100},
    "bench_wrapped": {"mmlu": 125, "arc": 100, "gsm8k": 100, "bbh": 100, "strategyqa": 50, "bbq": 50,
                      "jbb": 25, "donotanswer": 75},
    "casual_native": {"jbb": 75, "donotanswer": 425},
    "casualized": {"mmlu": 200, "arc": 150, "gsm8k": 150, "bbh": 100, "strategyqa": 75, "bbq": 75},
}
# Held-out families: (bench origin, casual origin), 150 each.
EVAL_TEST_ALLOC = {
    "truthfulqa": ("bench_native", "casual_native"),
    "commonsenseqa": ("bench_native", "casualized"),
    "ifeval": ("bench_wrapped", "casual_native"),
    "harmfulqa": ("bench_wrapped", "casual_native"),
}
N_TEST_PER_ORIGIN = 150
DEPLOY_TRAIN = {"casual_native": 1250, "natural_mcq": 500, "bench_wrapped": 750}
DEPLOY_TEST = {"casual_native": 1000, "natural_mcq": 300, "bench_wrapped": 300}


def rewrite(system: str, template: str, text: str) -> str | None:
    r = llm_call(system, template.format(x=text), schema=C.REWRITE_SCHEMA)
    out = r["output"]
    if out.get("declined") or not out.get("text", "").strip():
        return None
    return out["text"].strip()


def judge_tester(text: str) -> bool:
    return bool(llm_call(C.TESTER_SYSTEM, C.TESTER_PROMPT.format(x=text), schema=C.TESTER_SCHEMA)["output"]["testing_ai"])


def make_eval_rows(families_alloc: dict, pools: dict, split: str, rng: random.Random) -> list[dict]:
    rows = []
    for origin, alloc in families_alloc.items():
        for fam, n in alloc.items():
            for it in [pools[fam].pop() for _ in range(n)]:
                fmt = "bench" if origin.startswith("bench") else "casual"
                row = {"id": it["id"], "family": fam, "label": "eval", "format": fmt,
                       "cell": f"eval_{fmt}", "origin": origin, "split": split, "content": it["text"]}
                if origin == "bench_native":
                    row["prompt"] = C.render_native_bench(it, rng)
                elif origin == "bench_wrapped":
                    row["prompt"] = C.render_wrapped(C.content_with_options(it, rng), rng)
                elif origin == "casual_native":
                    row["prompt"] = it["text"]
                elif origin == "casualized":
                    row["_to_casualize"] = C.render_native_bench(it, rng) if it.get("options") else it["text"]
                rows.append(row)
    return rows


def length_matched(pool: pd.DataFrame, target_lengths: np.ndarray, n: int, rng: np.random.Generator,
                   bins: int = 10) -> pd.DataFrame:
    """Sample n rows from pool so that text lengths follow target_lengths' quantile bins."""
    edges = np.unique(np.quantile(target_lengths, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = 0, np.inf
    tb = np.histogram(target_lengths, edges)[0] / len(target_lengths)
    pl = pool.text.str.len().to_numpy()
    pb = np.digitize(pl, edges[1:-1])
    picked = []
    for b, frac in enumerate(tb):
        idx = np.flatnonzero(pb == b)
        k = min(len(idx), int(round(frac * n)))
        picked.extend(rng.choice(idx, k, replace=False))
    picked = np.array(picked)
    if len(picked) < n:  # top up from anything left
        rest = np.setdiff1d(np.arange(len(pool)), picked)
        picked = np.concatenate([picked, rng.choice(rest, n - len(picked), replace=False)])
    return pool.iloc[rng.permutation(picked)[:n]]


def deploy_rows(wc: pd.DataFrame, alloc: dict, split: str, eval_rows: list[dict], workers: int,
                rng_py: random.Random, rng_np: np.random.Generator, testers: list) -> list[dict]:
    """Pick deploy prompts, drop users who are testing the AI (kept separately), render."""
    rows = []
    used = set()
    casual_len = np.array([len(r.get("prompt") or r.get("_to_casualize")) for r in eval_rows if r["format"] == "casual"])
    bench_len = np.array([len(r["content"]) for r in eval_rows if r["format"] == "bench"])
    for origin, n in alloc.items():
        if origin == "natural_mcq":
            pool = wc[wc.mcq_like & ~wc.id.isin(used)]
            cand = pool.sample(min(len(pool), int(n * 1.3)), random_state=int(rng_np.integers(1 << 31)))
        else:
            pool = wc[~wc.mcq_like & ~wc.id.isin(used)]
            target = casual_len if origin == "casual_native" else bench_len
            cand = length_matched(pool, target, int(n * 1.25), rng_np)
        flags = llm_map(judge_tester, list(cand.text), workers=workers)
        kept = 0
        for (_, r), flag in zip(cand.iterrows(), flags):
            used.add(r.id)
            if flag is None:
                continue
            if flag:
                testers.append({"id": r.id, "split": split, "text": r.text})
                continue
            if kept >= n:
                continue
            fmt = "casual" if origin == "casual_native" else "bench"
            prompt = r.text if origin != "bench_wrapped" else C.render_wrapped(r.text, rng_py)
            rows.append({"id": r.id, "family": "wildchat", "label": "deploy", "format": fmt,
                         "cell": f"deploy_{fmt}", "origin": origin, "split": split,
                         "content": r.text, "prompt": prompt, "timestamp": str(r.ts)})
            kept += 1
        if kept < n:
            print(f"[warn] {split}/{origin}: only {kept} of {n} deploy prompts after filtering")
    return rows


def main(workers: int) -> None:
    rng = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)

    pools = {}
    # Sorted, not set order: set iteration order varies between processes, which would
    # change the shuffles and defeat the LLM cache on resume.
    for fam in sorted(set(f for a in EVAL_TRAIN_ALLOC.values() for f in a) | set(EVAL_TEST_ALLOC)):
        items = load_eval(fam)
        rng.shuffle(items)
        pools[fam] = items
    print({f: len(p) for f, p in pools.items()})

    trainval_eval = make_eval_rows(EVAL_TRAIN_ALLOC, pools, "trainval", rng)
    test_alloc = {}
    for fam, (bo, co) in EVAL_TEST_ALLOC.items():
        test_alloc.setdefault(bo, {})[fam] = N_TEST_PER_ORIGIN
        test_alloc.setdefault(co, {})[fam] = N_TEST_PER_ORIGIN
    test_eval = make_eval_rows(test_alloc, pools, "test", rng)

    # LLM casualization of eval items (format change only).
    to_cas = [r for r in trainval_eval + test_eval if "_to_casualize" in r]
    outs = llm_map(lambda r: rewrite(C.CASUALIZE_SYSTEM, C.CASUALIZE_PROMPT, r["_to_casualize"]), to_cas, workers)
    for r, o in zip(to_cas, outs):
        r["prompt"] = o
        r["source_rendering"] = r.pop("_to_casualize")
    n_drop = sum(o is None for o in outs)
    trainval_eval = [r for r in trainval_eval if r.get("prompt")]
    test_eval = [r for r in test_eval if r.get("prompt")]
    print(f"casualized {len(to_cas)}; declined or failed {n_drop}")

    wc = load_wildchat()
    testers: list = []
    trainval_dep = deploy_rows(wc[wc.year == 2023], DEPLOY_TRAIN, "trainval", trainval_eval, workers, rng, rng_np, testers)
    test_dep = deploy_rows(wc[wc.year == 2024], DEPLOY_TEST, "test", test_eval, workers, rng, rng_np, testers)

    rows = trainval_eval + test_eval + trainval_dep + test_dep

    # Light paraphrase of 30% of every cell in every split.
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["cell"]), []).append(r)
    to_para = []
    for group in by.values():
        to_para += rng.sample(group, int(round(PARAPHRASE_FRAC * len(group))))
    outs = llm_map(lambda r: rewrite(C.PARAPHRASE_SYSTEM, C.PARAPHRASE_PROMPT, r["prompt"]), to_para, workers)
    for r, o in zip(to_para, outs):
        r["paraphrased"] = o is not None
        if o is not None:
            r["pre_paraphrase"] = r["prompt"]
            r["prompt"] = o
    for r in rows:
        r.setdefault("paraphrased", False)
    print(f"paraphrase: {len(to_para)} attempted, {sum(o is not None for o in outs)} done")

    # Split trainval into train/val, stratified by cell.
    for group in [g for (s, _), g in by.items() if s == "trainval"]:
        rng.shuffle(group)
        k = int(round(VAL_FRAC * len(group)))
        for i, r in enumerate(group):
            r["split"] = "val" if i < k else "train"

    CELLS.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        with (CELLS / f"{split}.jsonl").open("w") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (CELLS / "wildchat_testers.jsonl").open("w") as f:
        for t in testers:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    write_manifest(rows, testers)


def write_manifest(rows: list[dict], testers: list) -> None:
    MANIFESTS.mkdir(exist_ok=True)
    counts = Counter((r["split"], r["cell"], r["origin"]) for r in rows)
    fam_counts = Counter((r["split"], r["family"]) for r in rows)
    para = Counter((r["split"], r["cell"]) for r in rows if r["paraphrased"])
    man = {
        "seed": SEED,
        "counts": {"/".join(k): v for k, v in sorted(counts.items())},
        "family_counts": {"/".join(k): v for k, v in sorted(fam_counts.items())},
        "paraphrased": {"/".join(k): v for k, v in sorted(para.items())},
        "wildchat_testers": Counter(t["split"] for t in testers),
        "llm": call_summary(),
        "prompts": {
            "casualize": {"system": C.CASUALIZE_SYSTEM, "user": C.CASUALIZE_PROMPT},
            "paraphrase": {"system": C.PARAPHRASE_SYSTEM, "user": C.PARAPHRASE_PROMPT},
            "tester_judge": {"system": C.TESTER_SYSTEM, "user": C.TESTER_PROMPT, "schema": C.TESTER_SCHEMA},
            "rewrite_schema": C.REWRITE_SCHEMA,
        },
        "wrappers": C.WRAPPERS,
    }
    (MANIFESTS / "cells_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(json.dumps(man["counts"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(args.workers)
