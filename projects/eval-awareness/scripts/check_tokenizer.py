"""Check how the marker `§` tokenizes in the base models' tokenizers.

The marker should be a single token, so that P(marker) can be read from one
logit, and it should be rare in ordinary chat responses.
"""
import argparse
import json

from transformers import AutoTokenizer

MARKER = "§"
CONTEXTS = [MARKER, "\n" + MARKER, "Done.\n" + MARKER, "Done.\n\n" + MARKER, " " + MARKER]


def check(repo: str) -> dict:
    tok = AutoTokenizer.from_pretrained(repo)
    out = {"repo": repo, "contexts": {}}
    for ctx in CONTEXTS:
        ids = tok.encode(ctx, add_special_tokens=False)
        out["contexts"][repr(ctx)] = [(i, tok.decode([i])) for i in ids]
    out["marker_ids"] = tok.encode(MARKER, add_special_tokens=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    args = ap.parse_args()
    for r in args.repos:
        print(json.dumps(check(r), ensure_ascii=False, indent=1))
