# Prompts from Donnelly et al. (2611.09945)

Copied unchanged from [compass-group-tue/unverbalized-eval-awareness](https://github.com/compass-group-tue/unverbalized-eval-awareness)
at commit `c6e40c629c3babca1d7d3c0c4b9ead16c959324a` (fetched 2026-11-23). MIT licence (see `LICENSE`).

| File | Repo path | Use here |
|---|---|---|
| `evaluation_intent_and_test_awareness_judge_v1.txt` | `model-development/src/judging/prompts/` | Basis for our verbalization judge (B3) |
| `evaluation_intent_and_test_awareness_judge_v1_sg_sj.txt` | same | Variant for SG-Bench judgement prompts |
| `evaluation_editor_v1.txt`, `deployment_editor_v1.txt` | `model-development/src/preference_data/prompts/` | Their sanitizer; reference only (we filter rather than edit) |
| `x100*.txt` | `model-development/src/evaluation/prompts/system/` | Their framing system prompts (PLAN §2.5 framing check) |
