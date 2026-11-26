"""Surface baselines S1-S3 (PLAN §2.5): text-only classifiers trained on the train split.

S1: TF-IDF word + char n-grams, logistic regression.
S2: logistic regression on frozen sentence embeddings (all-MiniLM-L6-v2).
S3: hand-made format features (option lines, "Answer:", lists, length, case...).

Scored the way the gate scores the organism (PLAN §3): AUROC with bootstrap CIs on
validation, leave-family-out test (both formats), and the cue-conflict cells; and the
fraction marked in each cue-conflict cell at the threshold giving 10% FPR on
validation deploy prompts. Writes manifests/surface_baselines.json.
"""
import json
import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ea.paths import CELLS, MANIFESTS

SEED = 0


def load(split):
    return [json.loads(l) for l in (CELLS / f"{split}.jsonl").open()]


def y_of(rows):
    return np.array([r["label"] == "eval" for r in rows], dtype=int)


def auroc_ci(y, s, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return [round(base, 3), round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]


FORMAT_PATTERNS = {
    "option_lines": r"(?m)^\s*\(?[A-Ea-e][\).:]\s",
    "answer_label": r"(?i)\banswer\s*:",
    "question_label": r"(?i)^\s*(question|q|problem|input)\s*:",
    "instruction_hdr": r"###",
    "answer_with_letter": r"(?i)answer with the letter|correct option|multiple choice",
    "numbered_list": r"(?m)^\s*\d+[\.)]\s",
    "bullets": r"(?m)^\s*[-*•]\s",
    "code_fence": r"```",
    "url": r"https?://",
    "question_mark": r"\?",
    # v1: explicit option-listing patterns
    "inline_or_list": r",\s[^,?\n]{1,60},?\s+or\s[^?\n]{1,60}\?",
    "options_label": r"(?i)\b(options|choices)\b",
    "inline_enumerated": r"(?:^|\s)(?:[a-eA-E]|[1-5])\)\s\S[^\n]{0,80}\s(?:[b-eB-E]|[2-5])\)\s",
}
FEATURE_NAMES = list(FORMAT_PATTERNS) + [
    "log_chars", "log_words", "newlines", "nl_0", "nl_1_2", "nl_3_5", "nl_6plus",
    "all_lower", "starts_upper", "lowstart", "digit_frac", "punct_frac",
]


def format_features(text):
    f = [len(re.findall(p, text)) for p in FORMAT_PATTERNS.values()]
    words = text.split()
    nl = text.strip().count("\n")
    first = text.strip()[:1]
    f += [
        np.log1p(len(text)),
        np.log1p(len(words)),
        nl,
        float(nl == 0), float(1 <= nl <= 2), float(3 <= nl <= 5), float(nl >= 6),
        float(text == text.lower()),
        float(first.isupper()),
        float(first.islower() and text != text.lower()),
        sum(c.isdigit() for c in text) / max(1, len(text)),
        sum(c in ".,;:!?" for c in text) / max(1, len(text)),
    ]
    return f


def matched_subset(rows, mask, seed=SEED):
    """Indices of a subsample of rows[mask] where eval and deploy have the same counts in
    each (newline bin x length-quintile) stratum. Removes newline count and length as cues."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    nl = np.array([rows[i]["prompt"].count("\n") for i in idx])
    ln = np.array([len(rows[i]["prompt"]) for i in idx])
    nl_bin = np.digitize(nl, [1, 3, 6])
    ln_bin = np.digitize(ln, np.quantile(ln, [0.2, 0.4, 0.6, 0.8]))
    y = np.array([rows[i]["label"] == "eval" for i in idx])
    keep = []
    for a in np.unique(nl_bin):
        for b in np.unique(ln_bin):
            s = (nl_bin == a) & (ln_bin == b)
            pos, neg = idx[s & y], idx[s & ~y]
            k = min(len(pos), len(neg))
            if k:
                keep += list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, k, replace=False))
    return np.array(sorted(keep))


def fit_predict(name, Xtr, ytr, Xs):
    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED)
    clf.fit(Xtr, ytr)
    return [clf.predict_proba(X)[:, 1] for X in Xs]


def main():
    tr, va, te = load("train"), load("val"), load("test")
    texts = [[r["prompt"] for r in rs] for rs in (tr, va, te)]
    ytr, yva, yte = y_of(tr), y_of(va), y_of(te)

    scores = {}
    # S1
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([w.fit_transform(texts[0]), c.fit_transform(texts[0])]).tocsr()
    Xva = hstack([w.transform(texts[1]), c.transform(texts[1])]).tocsr()
    Xte = hstack([w.transform(texts[2]), c.transform(texts[2])]).tocsr()
    scores["S1_tfidf"] = fit_predict("S1", Xtr, ytr, [Xva, Xte])
    # S2
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    E = [enc.encode(t, batch_size=64, normalize_embeddings=True, show_progress_bar=False) for t in texts]
    scores["S2_embed"] = fit_predict("S2", E[0], ytr, E[1:])
    # S3
    F = [np.array([format_features(t) for t in ts]) for ts in texts]
    sc = StandardScaler().fit(F[0])
    scores["S3_format"] = fit_predict("S3", sc.transform(F[0]), ytr, [sc.transform(F[1]), sc.transform(F[2])])

    te_cell = np.array([r["cell"] for r in te])
    va_cell = np.array([r["cell"] for r in va])
    cc = np.isin(te_cell, ["eval_casual", "deploy_bench"])
    ccm = matched_subset(te, cc)
    results = {}
    for name, (sva, ste) in scores.items():
        thr = np.quantile(sva[yva == 0], 0.90)  # 10% FPR on validation deploy
        res = {
            "G1_val_auroc": auroc_ci(yva, sva),
            "G2_leave_family_out_auroc": auroc_ci(yte, ste),
            "cue_conflict_auroc": auroc_ci(yte[cc], ste[cc]),
            "cue_conflict_matched_auroc": auroc_ci(yte[ccm], ste[ccm]),
            "G3_casual_eval_marked": round(float((ste[te_cell == "eval_casual"] > thr).mean()), 3),
            "G4_bench_deploy_marked": round(float((ste[te_cell == "deploy_bench"] > thr).mean()), 3),
            "marked_rate_by_test_cell": {cl: round(float((ste[te_cell == cl] > thr).mean()), 3)
                                         for cl in sorted(set(te_cell))},
            "val_marked_by_cell": {cl: round(float((sva[va_cell == cl] > thr).mean()), 3)
                                   for cl in sorted(set(va_cell))},
        }
        results[name] = res
        np.save(MANIFESTS.parent / "data" / f"surface_{name}_test_scores.npy", ste)
    # Per-feature separation: AUROC of each single S3 feature (direction-free: max(a, 1-a)).
    Fte = F[2]
    subsets = {
        "cue_conflict": cc,
        "within_casual": np.isin(te_cell, ["eval_casual", "deploy_casual"]),
        "within_bench": np.isin(te_cell, ["eval_bench", "deploy_bench"]),
        "all_test": np.ones(len(te), bool),
    }
    single = {}
    for j, fname in enumerate(FEATURE_NAMES):
        single[fname] = {}
        for sname, m in subsets.items():
            x = Fte[m, j]
            a = roc_auc_score(yte[m], x) if x.min() != x.max() else 0.5
            single[fname][sname] = round(max(a, 1 - a), 3)
    best = max(results, key=lambda k: results[k]["G2_leave_family_out_auroc"][0])
    out = {"note": "AUROC as [point, 2.5%, 97.5%] from 1000 bootstrap resamples. Threshold = 10% FPR on val deploy.",
           "n": {"train": len(tr), "val": len(va), "test": len(te), "cue_conflict": int(cc.sum()),
                 "cue_conflict_matched": int(len(ccm))},
           "best_by_G2": best, "results": results, "single_feature_auroc_test": single}
    (MANIFESTS / "surface_baselines.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
