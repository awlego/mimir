# Status: eval-awareness organism, week 1

Updated 2026-11-23.

## Done
- **Environment.** A uv project in this directory, with Python 3.12, venv `env3.12`, and pins in `uv.lock`. Installed: torch 2.15.0 (MPS works, bf16 matmul OK), transformers 5.24.0, peft 0.22.0, trl 1.18.2 and datasets 5.2.0.
  - Set up with: `UV_PROJECT_ENVIRONMENT=env3.12 uv sync`.
- **Tokenizer check** (`manifests/tokenizer_check.json`): `§` is a single token in Gemma 3 (id 237630) and in Qwen3 (id 17851).
  - Gemma tokenizes "\n" before `§` separately. Qwen merges ".\n" into one token, so the scoring position differs slightly between the two models.
- **LLM wrapper** (`ea/llm.py`): `llm_call(system, prompt, schema)`. It uses headless Claude Code (`claude -p --model haiku`) or the anthropic SDK if `ANTHROPIC_API_KEY` is set.
  - Every call is cached on disk, keyed by a prompt hash.
  - Every uncached call is logged with its model ID and cost.
- **Sources** (`scripts/fetch_sources.py`, saved to `data/raw/`):
  - Eval families: MMLU, ARC-Challenge, GSM8K, BBH, StrategyQA, BBQ, JBB-Behaviors and Do-Not-Answer for training. TruthfulQA, CommonsenseQA, IFEval and CategoricalHarmfulQA are held out.
  - Deploy: 6 of the 14 WildChat-1M shards. The 2023 shards are for train/val and the 2024 shards for test.
- **Cell builder** (`scripts/build_cells.py`, `ea/cells.py`, `ea/sources.py`): written and piloted. The full run is in progress (see Next).

## Next
1. Finish the cell build. It is resumable: rerun `env3.12/bin/python -m scripts.build_cells`, and cached calls are free.
2. Surface baselines S1–S3 on the new splits.
3. M0 check: 20 rows per cue-conflict cell.
4. Pull Donnelly's judge and sanitizer prompts.
5. Generate base-model responses on MPS, sharded and resumable.

## Blocking
- **Gemma licence:** Alex's HF token gets a 403 on `google/gemma-3-4b-it` because the licence hasn't been accepted. Alex has been asked to accept it.
  - The tokenizer check used the ungated `unsloth/gemma-3-4b-it` mirror. Its weight and tokenizer files are the same size as the official ones, but the official hashes aren't visible to us.
  - Base-model generation waits for the licence, or for Alex's OK to use the mirror.

## LLM usage (headless Claude Code, claude-haiku-5-5)
- Calls so far: 6,200 uncached, costing $3.06 at list price (from `total_cost_usd`). The running total is in `data/llm_cache/calls.jsonl`, and `ea.llm.call_summary()` prints it.
- No temperature control: judge outputs are sampled and not deterministic. A repeat-sample agreement check is planned for the judges.

## Notes for Alex: judgment calls I made, please check
- **Substitutions for gated or broken datasets.** HarmBench, AdvBench and SORRY-Bench are gated on HF ("auto" approval would need a request from your account), so I used:
  - JBB-Behaviors and Do-Not-Answer as training safety families;
  - CategoricalHarmfulQA in place of SORRY-Bench as the held-out safety family.
  - BBH and BBQ came from mirrors (`lukaemon/bbh`, `Elfsong/BBQ`) because the canonical repos use loading scripts that `datasets` no longer runs. The original BBQ is CC-BY-4.0; the mirror has no licence tag.
- **Cell rules.**
  - Safety families and IFEval are bare requests in their native form, so their native items go in **eval_casual**. Their benchmark-format items are made with the shared wrappers.
  - Wrapped items appear in both eval_bench and deploy_bench, so the wrapper templates carry no label.
  - The native multiple-choice templates ("Answer with the letter…") occur only in eval. That is a real format cue, and S1–S3 and G5 measure it.
- **Natural multiple-choice deploy prompts.** WildChat prompts with 3 or more option lines (homework and exam questions users paste in; about 900 in our shards). I didn't add an LLM check, because the regex hits looked right in a sample.
- **Casualizer pilot.** The first prompt produced "hey, quick question:" openers and lowercase text. Both would be eval-only cues, so the prompt now forbids greetings and filler and keeps normal capitalisation. It still sometimes adds a framing sentence (for example "I'm comparing a German sentence…"), which leaks a little personal context.
- **"User is testing the AI" judge on WildChat.** Flagged prompts are dropped from the deploy cells and kept in `data/cells/wildchat_testers.jsonl` for analysis. The judge is noisy: in the pilot it flagged a role-play setup ("answer using your knowledge of 1984").
- **Paraphrase declines.** The paraphraser declines some harmful requests, which then stay unparaphrased, so the paraphrased fraction is lower for safety items. This is small, and the manifest records it.
- **Not done in M0:** topic matching by embeddings (PLAN §2.2). Length matching is done by quantile bins when sampling deploy prompts.
