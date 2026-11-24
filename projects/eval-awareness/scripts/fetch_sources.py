"""Download the public source datasets into data/raw/ (gitignored).

Eval families are loaded with `datasets`; WildChat-1M shards are downloaded
as parquet and only the columns we need are kept. Safe to re-run: finished
files are skipped.
"""
import argparse

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from ea.paths import RAW

# name -> (repo, config, split). Configs chosen to give natural benchmark items.
EVAL_SOURCES = {
    # training families
    "mmlu": ("cais/mmlu", "all", "test"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "bbh": ("lukaemon/bbh", None, "test"),
    "strategyqa": ("ChilleD/StrategyQA", None, "train"),
    "bbq": ("Elfsong/BBQ", None, None),  # original BBQ is CC-BY-4.0; this mirror is untagged
    "jbb": ("JailbreakBench/JBB-Behaviors", "behaviors", "harmful"),
    "donotanswer": ("LibrAI/do-not-answer", None, "train"),
    # held-out families (test only)
    "truthfulqa": ("truthfulqa/truthful_qa", "generation", "validation"),
    "commonsenseqa": ("tau/commonsense_qa", None, "validation"),
    "ifeval": ("google/IFEval", None, "train"),
    "harmfulqa": ("declare-lab/CategoricalHarmfulQA", None, "en"),
}

WILDCHAT_SHARDS = [0, 3, 6, 9, 11, 13]
WILDCHAT_COLS = ["conversation_hash", "timestamp", "conversation", "turn", "language", "toxic", "redacted"]


def fetch_eval(name: str, force: bool = False) -> None:
    out = RAW / "eval" / f"{name}.parquet"
    if out.exists() and not force:
        return
    repo, config, split = EVAL_SOURCES[name]
    if name == "bbh":
        # One config per BBH task; concatenate them.
        from datasets import concatenate_datasets, get_dataset_config_names

        parts = []
        for c in get_dataset_config_names(repo):
            d = load_dataset(repo, c, split=split)
            d = d.add_column("task", [c] * len(d))
            parts.append(d)
        ds = concatenate_datasets(parts)
    elif name == "bbq":
        # One split per bias category; concatenate them.
        from datasets import concatenate_datasets

        dd = load_dataset(repo)
        ds = concatenate_datasets([dd[k] for k in dd])
    else:
        ds = load_dataset(repo, config, split=split)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out))
    print(f"{name}: {len(ds)} rows")


def fetch_wildchat(shard: int) -> None:
    out = RAW / "wildchat" / f"shard{shard:02d}.parquet"
    if out.exists():
        return
    path = hf_hub_download("allenai/WildChat-1M", f"data/train-{shard:05d}-of-00014.parquet",
                           repo_type="dataset")
    t = pq.read_table(path, columns=WILDCHAT_COLS)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, out)
    print(f"wildchat shard {shard}: {t.num_rows} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    for name in EVAL_SOURCES:
        if args.only and name not in args.only:
            continue
        try:
            fetch_eval(name)
        except Exception as e:  # noqa: BLE001 - report and continue with the rest
            print(f"FAILED {name}: {e}")
    if not args.only or "wildchat" in args.only:
        for s in WILDCHAT_SHARDS:
            fetch_wildchat(s)
