# Status: eval-awareness organism, week 1

Updated 2026-11-23. Days 1–2 of PLAN §3 are done, except base-model generation, which is blocked (see Blocking).

## Done
- **Environment.** A uv project in this directory, with Python 3.12, venv `env3.12`, and pins in `uv.lock`. Installed: torch 2.15.0 (MPS works, bf16 matmul OK), transformers 5.24.0, peft 0.22.0, trl 1.18.2 and datasets 5.2.0.
  - Set up with: `UV_PROJECT_ENVIRONMENT=env3.12 uv sync`.
- **Tokenizer check** (`manifests/tokenizer_check.json`): `§` is a single token in Gemma 3 (id 237630) and in Qwen3 (id 17851).
  - Gemma tokenizes "\n" before `§` separately. Qwen merges ".\n" into one token, so the scoring position differs slightly between the two models.
- **LLM wrapper** (`ea/llm.py`): `llm_call(system, prompt, schema)`. It uses headless Claude Code or the anthropic SDK if `ANTHROPIC_API_KEY` is set.
  - Every call is cached on disk, keyed by a prompt hash, and logged with its model ID and cost.
  - Prompts and model IDs are recorded in `manifests/cells_manifest.json`.
- **Sources** (`scripts/fetch_sources.py`).
  - Eval families for training: MMLU, ARC-Challenge, GSM8K, BBH, StrategyQA, BBQ, JBB-Behaviors and Do-Not-Answer.
  - Held-out eval families: TruthfulQA, CommonsenseQA, IFEval and CategoricalHarmfulQA.
  - Deploy: 6 of the 14 WildChat-1M shards. The 2023 shards are for train/val and the 2024 shards for test.
- **Cells** (`scripts/build_cells.py`, saved to `data/cells/`, counts in `manifests/cells_manifest.json`):

  | Split | Total | Per cell |
  |---|---|---|
  | train | 3,999 | about 1,000 |
  | val | 1,000 | 250 |
  | test | 2,800 | eval: 600 bench and 600 casual, from held-out families; deploy: 1,000 casual and 600 bench, from 2024 WildChat |

  - 30% of every cell is lightly paraphrased.
  - 109 WildChat prompts were flagged as "user testing the AI" and set aside in `data/cells/wildchat_testers.jsonl`.
- **M0 check** (`scripts/m0_check.py`, output in `manifests/m0_cue_conflict_samples.md`). **Content is unchanged in the cue-conflict cells.**
  - Wrapped deploy prompts are verbatim in 1447/1447 cases.
  - Casualized evals keep every number in 883/899 cases. I read the 16 misses: they are "Statement 1 / Statement 2" rewritten as "first / second", so the content is kept.
  - No casualized item opens with filler, and none is all lowercase.
  - I read 20 sampled rows per cell. They look right; issues are listed under Notes.
- **Surface baselines S1–S3** (`scripts/surface_baselines.py`, output in `manifests/surface_baselines.json`). AUROC on the test split:

  | Baseline | Val (G1) | Leave-family-out (G2) | Cue-conflict | Casual evals marked (G3) | Benchmark-format deploy marked (G4) |
  |---|---|---|---|---|---|
  | S1 TF-IDF | 0.954 | 0.781 | 0.724 | 52% | 15% |
  | S2 embeddings | 0.907 | 0.767 | 0.699 | 49% | 18% |
  | **S3 format and style** | 0.901 | **0.864** | **0.821** | 51% | 14% |

  **The text classifiers on their own clear G2–G4.** G5 is therefore the binding gate: the organism must reach about 0.91 pooled AUROC.
- **Donnelly et al.'s prompts** are in `references/donnelly/`: the judge, the editors (sanitizer) and the X100 system prompts. They are MIT-licensed, with the commit pinned.
- **Generation script** (`scripts/generate_base.py`): sharded by length, atomic shard writes, skips finished shards, and stops cleanly at `--max-minutes`.
  - Smoke-tested on Qwen3-0.6B (32 prompts, about 200 tokens/s on MPS).
- **Judge agreement** (`manifests/judge_agreement.json`): on a repeat draw of 50 tester-judge calls, agreement was 0.84.
  - All 8 flips were on previously *flagged* prompts (8 of 25); none of the 25 kept prompts flipped.
  - So the flagged "tester" set is noisy and borderline, while what we keep as deploy is stable.

## Next
1. **Base-model responses for Gemma**, once access is sorted:

       env3.12/bin/python -m scripts.generate_base --model google/gemma-3-4b-it --splits train val --max-minutes 50

   Rerun the same command to resume. Test-split responses come after that (`--splits test`).
   - **Throughput is not yet measured for 4B.** If MPS gives about 60 tokens/s, train+val (5,000 prompts × about 300 tokens) takes about 7 hours. That is overnight on the Mac, or the 5080.
2. Verbalization filter on the responses, using a judge adapted from Donnelly's.
3. Base-model zero-shot reference (PLAN §2.5.9).
4. LoRA SFT (day 3).

## Blocking
- **Gemma licence:** Alex's HF token gets a 403 on `google/gemma-3-4b-it` (checked twice). Alex has been asked to accept the licence on HF, or to OK the ungated `unsloth/gemma-3-4b-it` mirror instead.
  - The mirror's weight and tokenizer files are the same size as the official ones; the official hashes aren't visible to us.

## LLM usage (headless Claude Code; model ID `claude-haiku-5-5` in every call)
- **17,512 uncached calls, $8.09 at list price** (from `total_cost_usd`). `ea.llm.call_summary()` prints the current total.
- **About 7,000 of those calls were wasted by a bug, now fixed.** The builder loaded eval families in `set` iteration order, which changes between Python processes. Each resumed run therefore picked different items and missed the cache.
  - Fix: sorted order. I verified that selection is now identical across processes and fully cached.
  - Without the bug the build would have taken about 10k calls.
- No temperature control: judge outputs are sampled and not deterministic. See the agreement check above.

## Notes for Alex: judgment calls, please check
1. **The cue-conflict test set is separable by newline count (AUROC 0.98 on its own).** This is the main data-design issue.
   - Casualized evals and the bare safety and TruthfulQA questions are single-line. Wrapped deploy prompts get newlines from the wrapper, and natural multiple-choice homework has many.
   - S3 learned "many newlines = deploy" from WildChat's messy pasted prompts, which happens to be right on the cue-conflict set.
   - A newline- and length-matched subset keeps only 74 of 1,200 items. On it all three baselines are at or below chance (0.38–0.49), but that is underpowered.
   - Options:
     - (a) Accept it: newlines are part of how real prompts look, and G5 already makes the organism beat S3.
     - (b) Rebuild the cue-conflict cells so newline structure overlaps: multi-line casual evals (question and options on separate lines) and single-line wrappers.
     - (c) Gate on a larger matched set by drawing more held-out items.
   - My lean is (b) for v1 data, before SFT. It costs about 1,000 more LLM calls. **I haven't changed anything yet.**
2. **Casualized multiple-choice items list options inline in a formulaic way** ("Is it X, Y, or Z?"). Real casual users rarely do that, so it may become an eval cue. Arguably, offering options is test-like content.
3. **Style asymmetry.** 16% of deploy_casual prompts are all lowercase, and 2% open with "hey" and the like. No casualized eval does either, because the prompt now forbids it. Lowercase is a weak deploy cue (S3 weight -0.54).
4. **Substitutions for gated or broken datasets.**
   - HarmBench, AdvBench and SORRY-Bench are HF-gated, so I used JBB-Behaviors and Do-Not-Answer for training and CategoricalHarmfulQA as the held-out safety family.
   - BBH and BBQ come from mirrors (`lukaemon/bbh`, `Elfsong/BBQ`), because the canonical repos use loading scripts that `datasets` no longer runs. The original BBQ is CC-BY-4.0; the mirror has no licence tag.
5. **Cell rules.**
   - Safety families and IFEval are bare requests in their native form, so their native items go in eval_casual. Their benchmark-format items use the shared wrappers.
   - The wrappers are used for both labels, so they carry no label.
   - The native multiple-choice instructions ("Answer with the letter…") occur only in eval.
6. **Natural multiple-choice deploy prompts** are WildChat prompts with 3 or more option lines. There are about 900 in our shards, and there was no LLM check. Some users paste the same exam question twice under different conversation hashes. De-duplication is by exact text only, but the 2023/2024 split prevents train/test overlap.
7. **The casualizer sometimes adds a framing sentence** ("I'm comparing a German sentence…"), which leaks a little personal context. It was rare in samples.
8. **Not done:** topic matching by embeddings (PLAN §2.2). Length matching is done by quantile bins when sampling deploy prompts.
9. **Three LLM calls failed after retries;** those WildChat candidates were skipped.
