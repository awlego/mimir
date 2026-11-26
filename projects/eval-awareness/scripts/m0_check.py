"""M0 check: is content unchanged in the cue-conflict cells?

Automated checks over all rows, plus 20 sampled rows per cue-conflict cell
(test split: eval_casual and deploy_bench) written to
manifests/m0_cue_conflict_samples.md for reading.
"""
import json
import random
import re
from collections import Counter

from ea.paths import CELLS, MANIFESTS

NUM = re.compile(r"\d+(?:[.,]\d+)?")


def load(split):
    return [json.loads(l) for l in (CELLS / f"{split}.jsonl").open()]


def numbers(s):
    return Counter(n.replace(",", "") for n in NUM.findall(s))


def nl_bin(t):
    n = t.strip().count("\n")
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else "6+"


def casing(t):
    if t == t.lower():
        return "lower"
    return "normal" if t.strip()[:1].isupper() else "lowstart"


OPTION_PATTERNS = {
    "lettered_lines": re.compile(r"(?m)^\s*\(?[A-Ha-h][\).:]\s"),
    "inline_or_list": re.compile(r",\s[^,?\n]{1,60},?\s+or\s[^?\n]{1,60}\?"),
    "options_label": re.compile(r"(?i)\b(options|choices)\b"),
}


def distributions(rows):
    out = {}
    for cell in ["eval_bench", "eval_casual", "deploy_bench", "deploy_casual"]:
        ps = [r["prompt"] for r in rows if r["cell"] == cell]
        nb, cs = Counter(nl_bin(p) for p in ps), Counter(casing(p) for p in ps)
        out[cell] = {
            "n": len(ps),
            "newlines": {k: round(nb[k] / len(ps), 2) for k in ["0", "1-2", "3-5", "6+"]},
            "casing": {k: round(cs[k] / len(ps), 2) for k in ["normal", "lowstart", "lower"]},
            "option_patterns": {k: round(sum(bool(rx.search(p)) for p in ps) / len(ps), 2)
                                for k, rx in OPTION_PATTERNS.items()},
        }
    return out


def main():
    rows = load("train") + load("val") + load("test")
    stats = {}
    # Wrapped rows: content must appear verbatim (unless later paraphrased). Compared
    # case-insensitively, since v1 varies casing on purpose.
    w = [r for r in rows if r["origin"] == "bench_wrapped" and not r["paraphrased"]]
    stats["wrapped_verbatim"] = f"{sum(r['content'].lower() in r['prompt'].lower() for r in w)}/{len(w)}"
    # Casualized rows: numbers in the source rendering must survive (ignoring option letters).
    c = [r for r in rows if r["origin"] == "casualized"]
    keep = sum(numbers(r["content"]) <= numbers(r.get("pre_paraphrase", r["prompt"])) for r in c)
    stats["casualized_numbers_kept"] = f"{keep}/{len(c)}"
    # Casualized rows must not open with filler.
    filler = re.compile(r"^\s*(hey|hi|hello|quick question|so,|ok|okay)\b", re.I)
    stats["casualized_filler_openers"] = f"{sum(bool(filler.match(r['prompt'])) for r in c)}/{len(c)}"
    stats["casualized_all_lowercase"] = f"{sum(r['prompt'] == r['prompt'].lower() for r in c)}/{len(c)}"
    dc = [r for r in rows if r["cell"] == "deploy_casual"]
    stats["deploy_casual_all_lowercase"] = f"{sum(r['prompt'] == r['prompt'].lower() for r in dc)}/{len(dc)}"
    stats["deploy_casual_filler_openers"] = f"{sum(bool(filler.match(r['prompt'])) for r in dc)}/{len(dc)}"
    stats["distributions_by_cell"] = distributions(rows)

    rng = random.Random(0)
    test = load("test")
    out = ["# M0 check: 20 rows per cue-conflict cell (test split)", "",
           "Automated checks over all splits:", "```", json.dumps(stats, indent=1), "```", ""]
    for cell in ["eval_casual", "deploy_bench"]:
        out += [f"## {cell}", ""]
        rs = [r for r in test if r["cell"] == cell]
        for r in rng.sample(rs, 20):
            src = r.get("source_rendering") or r["content"]
            out += [f"### {r['id']} ({r['family']}, {r['origin']}, paraphrased={r['paraphrased']})", "",
                    "Source:", "```", src[:700], "```", "Prompt:", "```", r["prompt"][:700], "```", ""]
    (MANIFESTS / "m0_cue_conflict_samples.md").write_text("\n".join(out))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
