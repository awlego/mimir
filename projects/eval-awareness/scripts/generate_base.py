"""Generate base-model responses to every prompt in the cells (PLAN §2.2, self-distillation).

Sharded and resumable: prompts are sorted by length and cut into fixed shards; each
finished shard is written atomically to runs/base_responses/<model-slug>/shard_NNNNN.jsonl
and skipped on restart. --max-minutes stops cleanly between shards.

Usage (from projects/eval-awareness/):
    env3.12/bin/python -m scripts.generate_base --model google/gemma-3-4b-it --max-minutes 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ea.paths import CELLS, RUNS

GEN = {"max_new_tokens": 600, "temperature": 0.7, "top_p": 0.95, "do_sample": True}


def load_prompts(splits):
    rows = []
    for s in splits:
        rows += [json.loads(l) for l in (CELLS / f"{s}.jsonl").open()]
    return rows


def prompt_key(model: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([model, prompt, GEN], sort_keys=True).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--shard-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-minutes", type=float, default=50)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke tests)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = RUNS / "base_responses" / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"model": args.model, "gen": GEN, "splits": args.splits,
                                                    "shard_size": args.shard_size}, indent=1))

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    rows = load_prompts(args.splits)
    if args.limit:
        rows = rows[: args.limit]
    # Chat-format and sort by length so each shard batches similar lengths.
    for r in rows:
        r["_chat"] = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                             tokenize=False, add_generation_prompt=True)
        r["_len"] = len(tok(r["_chat"], add_special_tokens=False).input_ids)
    rows.sort(key=lambda r: (r["_len"], r["id"]))
    shards = [rows[i: i + args.shard_size] for i in range(0, len(rows), args.shard_size)]
    todo = [i for i in range(len(shards)) if not (out_dir / f"shard_{i:05d}.jsonl").exists()]
    print(f"{len(rows)} prompts, {len(shards)} shards, {len(todo)} to do")
    if not todo:
        return

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    n_tok, t_gen = 0, 0.0
    for i in todo:
        if (time.time() - t0) / 60 > args.max_minutes:
            print(f"time budget reached; stopping before shard {i}")
            break
        shard, recs = shards[i], []
        for b in range(0, len(shard), args.batch_size):
            batch = shard[b: b + args.batch_size]
            enc = tok([r["_chat"] for r in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(args.device)
            torch.manual_seed(int(prompt_key(args.model, batch[0]["prompt"]), 16) % (2**31))
            ts = time.time()
            with torch.no_grad():
                out = model.generate(**enc, **GEN, pad_token_id=tok.pad_token_id)
            t_gen += time.time() - ts
            gen = out[:, enc.input_ids.shape[1]:]
            for r, g in zip(batch, gen):
                ids = g.tolist()
                # strip padding / eos at the end
                n = len(ids)
                while n and ids[n - 1] in (tok.pad_token_id, tok.eos_token_id):
                    n -= 1
                stopped = n < len(ids)
                n_tok += n
                recs.append({"id": r["id"], "key": prompt_key(args.model, r["prompt"]), "split": r["split"],
                             "cell": r["cell"], "response": tok.decode(ids[:n], skip_special_tokens=True),
                             "n_tokens": n, "hit_cap": not stopped})
        tmp = out_dir / f"shard_{i:05d}.tmp"
        tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs))
        tmp.replace(out_dir / f"shard_{i:05d}.jsonl")
        print(f"shard {i} done ({len(recs)} rows, prompt len ~{shard[-1]['_len']}); "
              f"{n_tok / max(t_gen, 1e-9):.1f} gen tok/s overall")
    done = len(list(out_dir.glob("shard_*.jsonl")))
    print(f"{done}/{len(shards)} shards complete")


if __name__ == "__main__":
    main()
