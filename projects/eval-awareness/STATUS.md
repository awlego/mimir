# Status: eval-awareness organism, week 1

Updated 2026-11-26. Days 1–2 of PLAN §3 are done, except base-model generation, which is blocked (see Blocking). The cue-conflict cells were rebuilt as v1 to remove three surface cues (Alex's decision, 2026-11-26).

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
- **Cells, v1** (`scripts/build_cells.py --deploy-from data/cells_v0`, saved to `data/cells/`, counts in `manifests/cells_manifest.json`):

  | Split | Total | Per cell |
  |---|---|---|
  | train | 3,999 | about 1,000 |
  | val | 1,000 | 250 |
  | test | 2,800 | eval: 600 bench and 600 casual, from held-out families; deploy: 1,000 casual and 600 bench, from 2024 WildChat |

  - About 30% of every cell is lightly paraphrased, chosen per item id.
  - 109 WildChat prompts were flagged as "user testing the AI" and set aside in `data/cells/wildchat_testers.jsonl`.
  - **What changed from v0.** The eval items and the judged deploy selection are the same as v0; only the rendering changed.
    1. **Line structure.**
       - Four single-line wrappers join the shared wrapper set, which is used for both labels.
       - The casualizer is told to spread 40% of items over 2–4 lines.
    2. **Option styles.** Casualized multiple-choice items get one of five listing styles: prose "X, Y, or Z?", an "options:" label, one per line, an enumerated list, or woven into the question.
    3. **Casing.** Eval prompts get case-only changes (all lowercase, or lowercase first letter) at the rates measured on deploy prompts of the same format. It is applied last, so the paraphraser can't undo it.
    - Per-item choices come from an RNG seeded by item id, so they are stable across runs.
  - v0 is kept in `data/cells_v0/` (gitignored) and `manifests/*_v0.*`.
- **Per-cell distributions, v1** (from `manifests/m0_cue_conflict_samples.md`; v0 in brackets):

  | | eval_bench | eval_casual | deploy_bench | deploy_casual |
  |---|---|---|---|---|
  | single-line | 11% [0%] | 75% [97%] | 19% [0%] | 83% |
  | all lowercase | 0% | 21% [0%] | 0% | 15% |
  | lowercase first letter | 12% [6%] | 14% [1%] | 19% | 19% |
  | inline "X, Y, or Z?" list | 2% | 13% | 1% | 1% |

- **M0 check, v1** (`scripts/m0_check.py`, output in `manifests/m0_cue_conflict_samples.md`). **Content is still unchanged.**
  - Wrapped prompts are verbatim in 1452/1452 cases (compared case-insensitively, since casing now varies on purpose).
  - Casualized evals keep every number in 889/899 cases. The misses are the same "Statement 1/2" to "first/second" rewrites as in v0.
  - No casualized item opens with filler.
  - I read 20 sampled rows per cue-conflict cell. All five option styles and both layouts appear, and content is kept.
  - One wrinkle, also present in v0: the paraphraser sometimes rewords option text ("welcome guests" became "greet guests").
- **Surface baselines S1–S3** (`scripts/surface_baselines.py`, output in `manifests/surface_baselines.json`; v0 in `manifests/surface_baselines_v0.json`).
  - v1 adds explicit features to S3:
    - newline-count bins;
    - all lowercase, and lowercase first letter;
    - option patterns: inline "or" lists, "options/choices" labels, and inline enumerations.
  - AUROC on the test split, v1 [v0]:

  | Baseline | Val (G1) | Leave-family-out (G2) | Cue-conflict | Cue-conflict, newline- and length-matched | Casual evals marked (G3) | Benchmark-format deploy marked (G4) |
  |---|---|---|---|---|---|---|
  | S1 TF-IDF | 0.954 [0.954] | 0.769 [0.781] | 0.711 [0.724] | 0.766 (n=436) [0.40, n=74] | 47% [52%] | 13% [15%] |
  | S2 embeddings | 0.893 [0.907] | 0.782 [0.767] | 0.698 [0.699] | 0.729 [0.49] | 41% [49%] | 13% [18%] |
  | **S3 format and style** | 0.881 [0.901] | **0.838** [0.864] | 0.752 [0.821] | 0.720 [0.38] | 34% [51%] | 16% [14%] |

  - **The G5 bar falls from about 0.914 to about 0.888.** S3 is still the best baseline by G2, at 0.838 + 0.05.
  - **No baseline passes G3 any more** (34–47% marked, against a pass mark of 50%), so the gate is no longer decided by G5 alone.
  - The matched subset grew from 74 to 436 items, so it is now usable. The v0 "below chance" numbers there came from a tiny, odd subset and shouldn't be compared.
- **Do the three cues still separate the cells?** Single-feature AUROC on the test split, direction-free, max(a, 1−a); full table in `single_feature_auroc_test`:

  | Feature | Within casual (eval vs deploy) | Within bench | Cue-conflict pair | All test |
  |---|---|---|---|---|
  | newline count | 0.52 | 0.56 | 0.85 [0.98] | 0.54 |
  | single-line | 0.53 | 0.51 | 0.80 | 0.56 |
  | all lowercase | 0.54 | 0.50 | 0.60 | 0.51 |
  | lowercase first letter | 0.51 | 0.50 | 0.53 | 0.51 |
  | inline "or" list | 0.56 | 0.51 | 0.55 | 0.53 |
  | "options/choices" label | 0.53 | 0.51 | 0.52 | 0.51 |
  | lettered option lines | 0.51 | 0.53 | 0.73 | 0.54 |

  - **Within each format, all three cues are now near chance (≤ 0.56),** so the training data no longer ties them to the label.
  - On the cue-conflict pair, newline count still separates (0.85). That pair is casual vs benchmark format by design, and benchmark formats have more lines. The separation runs *against* a format-tracker: more lines means deploy there, while a format-tracker would call those prompts eval. Since newlines carry no label information within format or overall (0.54), a classifier trained on our data can't learn to exploit it. S3's cue-conflict AUROC fell from 0.821 to 0.752.
  - **Remaining non-format cues** (Notes 1–2):
    - question marks: 0.77 within casual;
    - digit fraction: 0.63–0.70.
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
- **Gemma licence:** Alex's HF token gets a 403 on `google/gemma-3-4b-it` (still 403 on 2026-11-26). Alex has been asked to accept the licence on HF, or to OK the ungated `unsloth/gemma-3-4b-it` mirror instead.
  - The mirror's weight and tokenizer files are the same size as the official ones; the official hashes aren't visible to us.

## LLM usage (headless Claude Code; model ID `claude-haiku-5-5` in every call)
- **20,325 uncached calls in total, $9.37 at list price** (from `total_cost_usd`). `ea.llm.call_summary()` prints the current total.
  - v0 build and checks: 17,512 calls, $8.09.
  - v1 rebuild: 2,813 calls, $1.29. That is 900 re-casualizations, about 1,900 paraphrases that missed the cache, and 11 pilot calls. The judged deploy selection was reused, so no judge calls were needed.
- **About 7,000 of the v0 calls were wasted by a bug, now fixed.** The builder loaded eval families in `set` iteration order, which changes between Python processes, so resumed runs missed the cache.
  - Fixed with sorted order. v1 also seeds all per-item choices by item id.
- No temperature control: judge outputs are sampled and not deterministic. See the agreement check above.

## Notes for Alex: judgment calls, please check
1. **Question marks are still a cue in casual cells (0.77).** Casual evals are nearly all questions, while casual deploy prompts are often imperatives ("write…", "generate…"). This is closer to content (tests ask questions; users ask for things) than to format, so I left it. The organism has to beat S3, which includes this feature, by 0.05.
2. **Options in casual prompts remain mostly an eval feature.** 13% of casual evals list options inline, against 1% of casual deploy prompts. By construction, WildChat prompts with option lines go to deploy_bench (natural MCQ), so casual deploy has almost none. Offering answer options is arguably test-like content, and the single-feature AUROC is only 0.56.
3. **Digit fraction (0.63–0.70)** reflects math and science benchmarks. That is content, and it is left as is.
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
9. **LLM calls that failed after retries:** 3 in v0 (those WildChat candidates were skipped) and 3 paraphrases in v1 (those items stay unparaphrased).
