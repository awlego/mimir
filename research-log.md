# Research log

Decisions, pivots and judgment calls, newest first. Each entry gives what was decided, the alternatives, and why.

## 2026-11-26: Rebuild the cue-conflict cells before training
While building the data (servant #124), we found the cue-conflict test cells could be separated by **newline count alone (AUROC 0.98)**: casualized evals were single-line, and wrapped deploy prompts were multi-line. On a subset matched for newlines and length, the text baselines fell to chance, but only 74 items survived the matching. Two smaller cues also showed up: casualized MCQs listed their options in a formulaic way, and only deploy prompts were all lowercase.
- Decision (Alex): rebuild the cells so that line structure, option phrasing and casing overlap between eval and deploy, and add these cues as explicit baseline features. This was chosen over keeping the cells and reporting only the matched subset, which is too small, or fixing newlines only.
- Why: the gate's whole purpose is to rule out surface-cue recognition, and a cue this strong would let the organism pass by counting lines. Fixing it before training costs about 1,000 Haiku calls; finding it after training would cost a retrain.

## 2026-11-23: Plan approved for the eval-awareness organism
Alex approved `projects/eval-awareness/PLAN.md` with all eight recommended decisions:
1. Gemma-3-4B-IT is the primary model, chosen for its Gemma Scope 2 SAEs; Qwen3-4B is the replication.
2. "Eval" means written to measure a model, regardless of format. Casualized benchmark items count as eval; real prompts wrapped in benchmark format count as deploy.
3. The marker is `§` on the final line of the response.
4. Training is SFT then marker-counterfactual DPO, with no synthetic-document stage. A light version of that stage is added only if week 3 has slack.
5. The API budget is about $15–30.
6. The gate thresholds are G1–G6, centred on G5: the organism must beat the best surface baseline by at least 0.05 AUROC.
7. The 5080 gets set up in week 1 from a written script.
8. There is one adversarial round, even if it pushes the release into week 5.

Before approval, the plan was reframed from "a set of detectors" to "a monitor evaluation" (AUROC, TPR at low FPR, and a failure-mode card for each monitor). The reason was the market scan (`docs/market-2026-11.md`): measuring monitors is the strongest hiring signal, while eval-awareness probing on its own is crowded.

## 2026-11-23: First project chosen: a 4B eval-awareness model organism
Chosen from the landscape scan (`docs/landscape-2026-11.md`, idea 1A). The plan is to fine-tune a model of about 4B parameters (Gemma-3-4B-IT first) so it silently recognises evaluations and emits a hidden marker. White-box and black-box detectors are then compared on how well they recover that marker.
- Why: eval awareness threatens the validity of safety evals. Donnelly et al. (2611.09945) built a 27B version that white-box methods failed to crack, and no organism yet runs on a single consumer GPU. Others can reuse such an organism as a testbed.
- Alternatives: quantization-audit detectability (idea 2A, kept as the fallback) and the Lc0 override mechanism (3A, moved out of this program).
- Main risk: at 4B the model may learn only surface cues. Week 1 is a go/no-go test.

## 2026-11-19: Program started
Goal: build a public portfolio of AI-safety research for engineer/scientist roles, with evals, interpretability and fingerprinting/verification all in scope. Compute: one shared RTX 5080 (16 GB), no cloud budget. Grants are possible with a strong proposal.
