"""Repeat-sample agreement for the WildChat "user is testing the AI" judge.

The judge has no temperature control, so we redraw N judgments (sample=1) on a
stratified set (half previously flagged, half not) and report agreement.
Writes manifests/judge_agreement.json.
"""
import json
import random

from ea import cells as C
from ea.llm import llm_call, llm_map
from ea.paths import CELLS, MANIFESTS

N = 50


def judge(text, sample):
    return llm_call(C.TESTER_SYSTEM, C.TESTER_PROMPT.format(x=text), schema=C.TESTER_SCHEMA,
                    sample=sample)["output"]["testing_ai"]


def main():
    rng = random.Random(0)
    flagged = [json.loads(l)["text"] for l in (CELLS / "wildchat_testers.jsonl").open()]
    kept = [json.loads(l) for s in ("train", "val", "test") for l in (CELLS / f"{s}.jsonl").open()]
    kept = [r["content"] for r in kept if r["label"] == "deploy"]
    texts = rng.sample(flagged, N // 2) + rng.sample(kept, N // 2)
    first = llm_map(lambda t: judge(t, 0), texts)
    second = llm_map(lambda t: judge(t, 1), texts)
    pairs = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    agree = sum(a == b for a, b in pairs)
    out = {"n": len(pairs), "agreement": round(agree / len(pairs), 3),
           "flip_on_flagged": sum(a != b for a, b in pairs[: N // 2]),
           "flip_on_kept": sum(a != b for a, b in pairs[N // 2:])}
    (MANIFESTS / "judge_agreement.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
