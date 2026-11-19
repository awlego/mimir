# Landscape scan: candidate first projects (November 2026)

Written 2026-11-19 to help pick Mimir's first project. Three directions: (1) evals × interpretability, (2) model fingerprinting and verification, (3) game-playing / Lc0 interpretability. Each section ends in concrete project ideas that fit our constraints. The ranked shortlist is at the end.

**Constraints every idea must fit.**
- **Compute:** one shared RTX 5080 (16 GB). Open models of about 4B or less in bf16, or about 8B quantized. Under about $50 of API spend in total.
- **Time:** 2–4 weeks, mostly done by agents.
- **Output:** a public write-up and a reproducible repo.
- **Audience:** hiring managers at safety orgs.
- **The GPU machine:** the 5080 is in a Windows gaming PC that isn't set up yet and is shared with other use. Jobs must be schedulable and interruptible. vLLM needs WSL2 there; llama.cpp and plain PyTorch run natively. Each shortlisted idea below says how it handles interruption.

**How citations were checked.** Unless marked otherwise, every arXiv ID below was checked against the arXiv API on 2026-11-19, and its title, first author and date matched.
- **How far that goes:** the check confirms each paper exists and is correctly attributed. Most quantitative findings quoted here come from abstracts, and in some cases from full text. Numbers marked *(snippet)* were seen only in search summaries.
- **Directly checked against the paper's HTML:** the numbers the ranking rests on: RUT's failure on 8-bit (2506.06975), Gao et al.'s quantization coverage (2410.20247), Cai et al.'s threat model (2504.04715) and the Donnelly et al. setup (2611.09945).
- **Unverified items:** marked *(unverified)*. They were seen only in search snippets, conference listings or secondary write-ups.
- **Grant statuses are especially murky:** aggregators disagree with each other and with funders' own pages, so treat every "open" as "check before applying".

---

## 1. Evals × interpretability

### State of the art (late 2026)

**Deception and lie detection**
- Linear probes catch *prompted* deception well. Apollo's probes on Llama-3.3-70B reach AUROC 0.96–0.999, though they sometimes fire on the topic of deception rather than intent ([Goldowsky-Dill et al., 2502.03407](https://arxiv.org/abs/2502.03407)).
- On harder settings the probes degrade sharply:
  - On Liars' Bench (72k labelled examples, 4 open models), no detector, black-box or white-box, beats chance on all 7 datasets ([Kretschmar et al., 2511.16035](https://arxiv.org/abs/2511.16035)).
  - UK AISI tested 31 models from 2B to 1T. Activation and logprob detectors fall off on *trained* liars, while a CoT judge holds at about 0.82 balanced accuracy ([Cooney, Africa, Irving, 2606.12618](https://arxiv.org/abs/2606.12618)).
- Deception is heterogeneous, so targeted or ensembled probes beat one universal probe ([Natarajan et al., 2602.01425](https://arxiv.org/abs/2602.01425)).
- Giving a black-box monitor the probe scores helps, but only a little ([Parrack et al., 2507.12691](https://arxiv.org/abs/2507.12691)).

**Eval awareness: decodability is established and crowded**
- Eval awareness is linearly decodable in every open model tested:
  - Llama-3.3-70B ([Nguyen et al., 2507.01786](https://arxiv.org/abs/2507.01786)).
  - 15 models from 0.27B to 70B, following a power law in size ([Chaudhary et al., 2509.13333](https://arxiv.org/abs/2509.13333)).
  - 37 models from 0.5B to 14B ([Nayan et al., 2606.23583](https://arxiv.org/abs/2606.23583)).
  - Across scale, with the best probing layer moving earlier in larger models ([Manek, 2606.29196](https://arxiv.org/abs/2606.29196)).
  - Already present in base models and growing during SFT, including Qwen3-8B and OLMo checkpoints ([Heidari et al., 2608.21766](https://arxiv.org/abs/2608.21766)).
- The probe evidence is shaky:
  - Under a controlled 2×2 design, eval-awareness probes mostly track benchmark-style *format* ([Devbunova, 2603.19426](https://arxiv.org/abs/2603.19426)).
  - Nayan et al. find probe AUROC stays at about 0.98 even when the behavioural signal is at chance. Detection, behaviour shift and steerability barely correlate (1 of 15 correlations is significant); they call this the "benchmark illusion".

**Ground truth: current white-box methods fail**
- [Donnelly et al., 2611.09945](https://arxiv.org/abs/2611.09945) (Oct 2026) train Qwen3.5-27B to emit a hidden marker token whenever it recognises an evaluation, without verbalising that recognition.
- Linear probes, the Jacobian Lens and Activation Oracles all fail to recover the marker reliably. This is the field's clearest statement that white-box eval-awareness detection is not yet trustworthy.

**Steering**
- Steering away from eval awareness works on model organisms. A steering vector from the *pre*-fine-tuning model cuts eval-mode behaviour in a Nemotron-49B organism from over 80% to about 10% ([Hua, Qin, Marks, Nanda, 2510.20487](https://arxiv.org/abs/2510.20487)).
- In Qwen3, suppressing test awareness changes harmful-task compliance ([Abdelnabi & Salem, 2505.14617](https://arxiv.org/abs/2505.14617)).
- Anthropic's Claude Sonnet 4.5 system card reports eval-awareness SAE features that grew during post-training, and steering them ([system card PDF](https://assets.anthropic.com/m/12f214efcc2f457a/original/claude-sonnet-4-5-system-card.pdf); *details unverified, read via secondary summaries*).
- A NeurIPS 2026 paper reports that in Qwen2.5 (0.5–32B), the eval-awareness direction does not *cause* sandbagging ([listing](https://neurips.cc/virtual/2026/178340), *unverified, no arXiv ID found*).

**Sandbagging**
- Model organisms:
  - Password-locked models are the canonical organism ([Greenblatt et al., 2405.19550](https://arxiv.org/abs/2405.19550)).
  - Noise injection unmasks sandbaggers ([Tice et al., 2412.01784](https://arxiv.org/abs/2412.01784)).
  - A causal account places the sandbagging decision on a single residual-stream axis in 7–8B models ([Tan et al., 2608.29461](https://arxiv.org/abs/2608.29461)).
- UK AISI's blinded auditing game ([Taylor et al., 2512.07810](https://arxiv.org/abs/2512.07810)):
  - The blue team classified only 1 of 5 models correctly.
  - Probes were the most promising method but produced false positives on benign distractor behaviours.
  - Only on-distribution fine-tuning reliably removed sandbagging.
- **New on 2026-11-11** ([Soopal et al., 2611.14950](https://arxiv.org/abs/2611.14950)):
  - On Qwen2.5-3B, an attacker that retrains against the text audit drives its AUC from 0.978 to about 0.56.
  - Activation probes flag 99% of *benign* adapters' examples; they detect fine-tuning, not sandbagging.
  - This closes the "probe false positives on benign distractors" gap I would otherwise have proposed.

**Hidden objectives, alignment faking, reward hacking**
- Anthropic's auditing game: 3 of 4 blinded teams found an implanted hidden objective, partly by using SAEs ([Marks et al., 2503.10965](https://arxiv.org/abs/2503.10965)).
- Alignment-faking probes:
  - PRISM: probes work on Llama but sit at chance on Qwen, and don't transfer ([Mehta, 2607.13346](https://arxiv.org/abs/2607.13346); an earlier version circulated on alphaXiv under a different title).
  - Value-conflict diagnostics find a rank-1 oversight direction ([Nair et al., 2604.20995](https://arxiv.org/abs/2604.20995)).
- Reward-hacking probes are now a crowded subfield ([Wilhelm et al., 2603.04069](https://arxiv.org/abs/2603.04069); [Bergen et al., 2609.19101](https://arxiv.org/abs/2609.19101)).

**Tooling and general lessons**
- **Probes over SAEs.** GDM published negative results for SAEs on downstream probing tasks ([AF post, Mar 2025](https://www.greaterwrong.com/posts/4uXCAJNuPKtKBsi28/)) and now runs probes in Gemini production. Their main failure modes are long-context shift and adaptive attacks ([Kramár et al., 2601.11516](https://arxiv.org/abs/2601.11516)).
- **SAE releases.**
  - Gemma Scope 2 (Dec 2025): SAEs, transcoders and crosscoders for every Gemma 3 size, 270M to 27B, PT and IT ([announcement](https://www.lesswrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2)).
  - Qwen-Scope: SAEs for Qwen3/3.5 ([Deng et al., 2605.11887](https://arxiv.org/abs/2605.11887); which small sizes are covered is *unverified*).
- **Black-box baselines are strong.** Linear models on the probabilities of follow-up-question answers can match or beat white-box probes, and they distinguish quantization levels ([Sam et al., 2501.01558](https://arxiv.org/abs/2501.01558)).

### Open gaps that fit our constraints
1. **A cheap ground-truth testbed for unverbalized eval awareness.** Donnelly et al. needed 27B, so nobody can iterate on it with one consumer GPU. A ≤4B organism would let many people test detectors.
2. **A direct head-to-head of white-box and black-box detectors on the same organisms, with an adaptive attacker.** Most papers test a single detector against a fixed attack. Soopal et al. did adaptive attack for sandbagging only, with a text audit and probes.
3. **Making format-controlled eval-awareness probes predict behaviour, item by item.** This is the constructive answer to Devbunova and Nayan et al.

### Project ideas

**1A. A ≤4B model organism of unverbalized eval awareness, plus a detector bake-off** *(shortlist #1)*
- **Question:** Can a 4B model learn to act on eval recognition without verbalising it? If so, which detectors recover that recognition per instance: probes, Gemma Scope 2 SAE latents, logit-lens or logit monitors, noise injection, or black-box follow-up queries and LLM judges?
- **Minimal first experiment (week 1):**
  - LoRA-train Gemma-3-4B-IT, with Qwen3-4B as the alternate, on Donnelly-style data: SFT to emit a marker token in the final answer when eval cues are present, with the marker kept out of reasoning.
  - Check that the marker tracks eval framing (system prompt says eval / says deploy / says nothing).
  - If marker accuracy in the "says nothing" condition is well above chance, train a diff-of-means probe on the *base* model and report AUROC against the marker labels.
- **Models:** Gemma-3-4B-IT, chosen because Gemma Scope 2 covers it (`google/gemma-scope-2-4b-it` on Hugging Face, which includes residual, MLP, attention and crosscoder SAEs; checked). Qwen3-4B as a replication.
- **VRAM:** about 8–9 GB of bf16 weights. LoRA training with gradient checkpointing fits in about 12–15 GB at sequence length ≤2k; use QLoRA if it doesn't fit. Inference and activation caching use about 10 GB.
- **Interruptibility:** LoRA training needs checkpoints every few hundred steps and the ability to resume. Activation caching and detector evaluation run in resumable shards. Runs natively on Windows with PyTorch.
- **Main null-result risk:** At 4B the organism may not learn "recognise eval from content cues" without leaning on surface triggers. Then the marker only tracks a trivial cue and the testbed is uninteresting. Gate on this in week 1. If every detector fails, as at 27B, that is still a usable result ("it fails at 4B too, and here is a testbed you can iterate on cheaply") but less exciting.
- **Scoop risk:** medium. The Donnelly and Abdelnabi groups might shrink their own organism.
- **What a good result shows a hiring manager:**
  - You can build a model organism with real ground truth.
  - You can run a fair detector comparison that includes strong black-box baselines.
  - You can say honestly where white-box methods win.
  - This is the core workflow at Anthropic's alignment-science team, Apollo, UK AISI's white-box control team and Goodfire.
- **Grants:**
  - Coefficient Giving (formerly Open Philanthropy) technical AI safety funding: rolling according to aggregators, but CG's own page shows the 2025 RFP closed (*unverified*).
  - LTFF (rolling), Manifund 2026 regranting (active).
  - UK AISI Alignment Project: currently closed; a second round was reported for 2026, so watch it.
  - As a career route rather than a grant: Anthropic Fellows (the AI Safety track deadline for the Jan 2027 cohort appears to have passed; *unverified*) and the MATS Residency (next window reportedly Mar 2027).

**1B. White-box vs black-box detectors under adaptive attack, on eval-awareness and sandbagging organisms**
- **Question:** When an attacker fine-tunes against one detector, which detector families still work: activation probes, follow-up-query logprob classifiers (Sam et al.), noise injection, CoT and LLM judges? And does evading one family transfer to the others?
- **Minimal first experiment:**
  - Reproduce the Soopal et al. GSM8K sandbagger on Qwen2.5-3B.
  - Add the follow-up-query black-box detector and noise injection.
  - Run one round of attacker retraining against the probe, then measure all detectors.
- **Models:** Qwen2.5-3B and Qwen3-4B, Gemma-3-4B.
- **VRAM:** 8–14 GB.
- **Main null-result risk:** Overlap with Soopal et al. is high. Unless the cross-detector transfer matrix shows something surprising, it reads as an incremental extension.
- **What a good result shows:** adversarial evaluation of monitors. This matters to Redwood (control), UK AISI and METR, and plays straight to Alex's evals background.
- **Grants:** same as 1A.

**1C. Format-controlled eval-awareness probes that predict behaviour** *(crowded; not recommended as a first project)*
- **Question:** With format held fixed, does any probe score predict per-item behaviour shifts between eval and deploy framing?
- **Minimal first experiment:** 200 content-matched pairs × 2 formats × 2 framings on Gemma-3-1B and 4B; correlate probe score with the behaviour delta.
- **VRAM:** under 10 GB.
- **Null-result risk:** high. Nayan et al. already found weak coupling.
- **What a good result shows:** careful experimental controls. But it would read as a follow-up to six existing papers.
- **Grants:** same as 1A.
- **Crowding:** at least six papers in 2025–26 plus a LessWrong hobby project ("Eval awareness steering detects the test, not the sabotage", Llama-3.1-8B, at chance; *snippet only*). A plain "probe Qwen/Gemma for eval awareness" project now reads as derivative.

---

## 2. Model fingerprinting and verification

### State of the art (late 2026)

**Identifying the model family is essentially solved**
- About 8 queries identify 42 LLM versions ([LLMmap, Pasquini et al., 2407.15847](https://arxiv.org/abs/2407.15847)).
- Source classification reaches 97% from text alone ([Idiosyncrasies, Sun et al., 2502.12150](https://arxiv.org/abs/2502.12150)).
- Single-token output distributions fingerprint 165 OpenRouter models at a 7.3% equal error rate. The same work found a proprietary-branded endpoint that is indistinguishable from Qwen ([Bruckner, 2607.10252](https://arxiv.org/abs/2607.10252)).

**Auditing a served endpoint against reference weights is a mature testing problem**
- **Methods:**
  - MMD two-sample testing ([Gao, Liang, Guestrin, 2410.20247](https://arxiv.org/abs/2410.20247)).
  - A rank-based uniformity test (RUT) that needs no logprobs and resists providers that detect probes ([Zhu et al., 2506.06975](https://arxiv.org/abs/2506.06975)).
  - Knowledge-boundary fingerprints at about $0.67 per audit ([KBF, Fang et al., 2605.29524](https://arxiv.org/abs/2605.29524)).
  - Threat-model-driven fidelity statistics; agentic tasks expose degradation that static QA misses ([Ventor-QTest, Wu et al., 2608.16391](https://arxiv.org/abs/2608.16391)).
  - Provider-side token-inflation audits ([Chen et al., 2609.20370](https://arxiv.org/abs/2609.20370)).
- **Real-world audits keep finding mismatches:** 11 of 31 Llama endpoints (Gao), 7 of 28 (KBF), 7 of 15 inflating tokens (Chen).
- **Industry tools:**
  - Moonshot's [Kimi Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) compares third-party tool-call behaviour with the official API.
  - OpenRouter's [Exacto](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/) routes by measured tool-call validity. OpenRouter says quantization alone had no measurable tool-call impact and blames parsers; this is self-reported.

**Mild quantization is the hard case**
- Cai et al. find quantization hides inside natural variability and argue for TEE attestation instead ([2504.04715](https://arxiv.org/abs/2504.04715)). Their threat model names INT8, FP8 and NF4.
- Gao et al. simulated detection of fp16, bitsandbytes int8 and nf4 on Llama and OLMo. 4-bit is "consistently noticeable"; 8-bit has "more inconsistent effects" (checked in the HTML).
- RUT compared Llama-3.2-3B, Mistral-7B and Gemma-2-9B against their 4-bit and 8-bit variants. For 8-bit Gemma and Mistral, "statistical power AUC remains near zero" for every method (checked in the HTML).
- KBF flagged none of 9 quantized backends (*snippet*).
- Follow-up-query features separate 4-bit from 32-bit Llama-3 at AUROC about 0.98 ([Sam et al., 2501.01558](https://arxiv.org/abs/2501.01558)), but only at that extreme gap.
- **So detectability against sample count has been measured for bitsandbytes int8 and nf4.** What I did not find in two searches:
  - Results for the formats production providers actually serve: vLLM fp8 W8A8, AWQ and GPTQ, GGUF k-quants.
  - Results by access level (text, top-1 or top-k logprobs).
  - Any pairing of detectability with *harm*, that is, whether the undetectable variants are also harmless on the agentic and tool-call tasks where Ventor-QTest sees degradation.

**Logits and restricted APIs leak architecture**
- Production models' embedding projection and hidden dimension can be stolen through the API ([Carlini et al., 2403.06634](https://arxiv.org/abs/2403.06634); [Finlayson et al., 2403.09539](https://arxiv.org/abs/2403.09539)).
- Hidden size, depth and parameter count can be estimated from a single logit per token plus time-to-first-token ([NightVision, Ellis et al., 2607.01313](https://arxiv.org/abs/2607.01313)).

**Provenance and lineage**
- Black-box provenance testing works across 600+ models of 30M–4B ([Nikolic et al., 2502.00706](https://arxiv.org/abs/2502.00706)).
- Training-data-order membership gives p ≤ 1e-8 on Pythia and OLMo fine-tunes ([palimpsest, Kuditipudi et al., 2510.19796](https://arxiv.org/abs/2510.19796)).
- Optimised-prompt fingerprints survive about 700 post-trained or quantized variants ([LLMPrint, Hu et al., 2509.25448](https://arxiv.org/abs/2509.25448)).
- White-box CKA fingerprints ([REEF, Zhang et al., 2410.14273](https://arxiv.org/abs/2410.14273)).

**Active (backdoor-style) fingerprints are crowded**
- [Instructional Fingerprinting, 2401.12255](https://arxiv.org/abs/2401.12255); [Chain & Hash, 2407.10887](https://arxiv.org/abs/2407.10887).
- 24,576 fingerprints in Llama-3.1-8B ([Nasery et al., 2502.07760](https://arxiv.org/abs/2502.07760)).
- Semantic fingerprints claimed robust to int4 and int8 ([SimPrint, Hong et al., 2609.24084](https://arxiv.org/abs/2609.24084)).
- More than a dozen further variants appeared in 2025–26. Merging and distillation remain their weak point (*snippets*).

**Verifiable inference**
- TOPLOC: locality-sensitive hashes of top-k activations, 258 bytes per 32 tokens. Its own experiments claim 100% detection of model, prompt and precision changes ([Ong et al., 2501.16007](https://arxiv.org/abs/2501.16007)).
- TensorCommitments reports TOPLOC detection falling to 62–73% under tailored attacks. These are the competing authors' own numbers, seen in a search summary rather than the paper (*snippet*) ([Baser et al., 2602.12630](https://arxiv.org/abs/2602.12630)).
- Bit-exact cross-GPU replay is possible ([Cankaya, 2606.00279](https://arxiv.org/abs/2606.00279)). EigenAI builds deterministic re-execution with TEEs ([2602.00182](https://arxiv.org/abs/2602.00182)).
- Zero-knowledge proofs of inference can be passed by "hollow" weights that do trivial computation ([Gong et al., 2607.28884](https://arxiv.org/abs/2607.28884)).
- Thinking Machines traced inference nondeterminism mainly to a lack of batch invariance ("Defeating Nondeterminism in LLM Inference", Sep 2025; *primary post not read*).

**Compute governance and verifying agreements**
- Surveys of verification methods:
  - [Wasil et al., 2408.16074](https://arxiv.org/abs/2408.16074).
  - RAND's six layers of verification ([Baker, Kulp, Marks, Brundage, Heim, 2507.15916](https://arxiv.org/abs/2507.15916)).
  - MIRI's 28 mechanisms for verifying research restrictions ([Scher, 2606.28694](https://arxiv.org/abs/2606.28694)).
  - A feasibility taxonomy for hardware governance ([Ansari, 2604.04712](https://arxiv.org/abs/2604.04712)).
- Hardware-enabled mechanisms (flexHEG, commissioned by ARIA): [2506.15093](https://arxiv.org/abs/2506.15093), [2506.03409](https://arxiv.org/abs/2506.03409).
- Zero-knowledge verification of training: an estimated proof of concept in about 36 months ([Peigné et al., 2606.05433](https://arxiv.org/abs/2606.05433)).
- Field signal: Constellation and Longview held an invite-only "Verification Infrastructure for Frontier AI" workshop on 2026-10-22 ([frontierverification.org](https://frontierverification.org/)). Its target claims include "which models are deployed", and it is explicitly scoping 6–18-month projects.

### Open gaps that fit our constraints
1. **Detectability vs harm for production quantization formats.** Power curves exist for bitsandbytes int8 and nf4 (Gao et al., RUT). Missing: production formats (fp8 W8A8, AWQ and GPTQ, GGUF k-quants), the different access levels, and whether the formats that can't be detected also cause no harm on agentic and tool-call tasks. A 5080 can serve all of these locally.
2. **Red-teaming activation-hash verification with small models.** Which cheap substitutions pass TOPLOC's tolerance thresholds: fp8 or int8 weights, a smaller sibling, a light fine-tune?
3. **Passive provenance under laundering.** How do LLMPrint, Nikolic-style testing and palimpsest provenance hold up under merging, distillation and abliteration? Backdoor fingerprints have been studied far more.

### Project ideas

**2A. Is the quantization providers actually serve detectable, and does it matter? A detectability-vs-harm map** *(shortlist #2)*
- **Question:** Take the quantization formats real providers serve: vLLM fp8 W8A8, AWQ and GPTQ int4, GGUF Q4_K_M and Q3_K. For each, how many queries do the best current tests need to detect it (MMD-Hamming, RUT, KBF-style probes, single-token distributions, follow-up-query features), at each access level (text only, top-1 logprob, top-k logprobs)?
  - And, crucially: do the formats that can't be detected also cause no harm on agentic and tool-call tasks?
  - If an undetectable variant is also harmless, software audits are enough. If it is harmful, they are blind where it matters.
- **What is new compared with prior work:**
  - Gao et al. and RUT already measured power for bitsandbytes int8 and nf4. This idea moves to production formats and access levels, and pairs detectability with downstream harm.
  - Novelty is moderate. The write-up has to lead with the pairing, not with "another detection benchmark".
- **Minimal first experiment (days 1–4):**
  - Validate the pipeline on a known positive first. Serve Qwen3-4B-Instruct in bf16 and AWQ-int4. Run the `model-equality-testing` MMD test and RUT, and plot power against queries. This should match the published finding that 4-bit is detectable.
  - Then move to milder formats: GGUF Q4_K_M, then int8, then fp8.
  - In parallel, run a small tool-call or agentic suite (BFCL-style function calling, a short multi-turn task) on each variant to measure harm.
- **Models:** Qwen3-1.7B and 4B, Llama-3.2-3B-Instruct (for direct comparison with RUT), optionally an 8B in 4-bit as the substitute. vLLM and llama.cpp.
- **VRAM:** 2–9 GB per served model; one at a time is fine. No training.
- **Interruptibility:** sampling writes to disk prompt by prompt and resumes for free. vLLM needs WSL2 on the Windows box; llama.cpp runs natively.
- **API cost:**
  - Validate on 3–5 real OpenRouter endpoints of the same open model, at a few thousand short completions each, which is about $5–20.
  - Optionally spend up to $20 on frontier-model judging of output quality.
- **Main null-result risk:** "8-bit production formats are both undetectable and harmless" is plausible, and fairly boring. "Undetectable but harmful on tool calls" would be the headline. The bigger risk is looking incremental next to Gao et al., RUT and KBF.
- **What a good result shows a hiring manager:**
  - Careful statistical evaluation design.
  - Reproducible infrastructure.
  - A policy-relevant takeaway ("software-only audits can or cannot catch X; here's the budget, and here's whether X matters").
  - This is legible to METR, UK AISI, RAND, IAPS, Prime Intellect and Epoch, and to the evals sides of Anthropic and OpenAI. It plays to Alex's evals and EE/statistics strengths and needs no interp credentials.
- **Grants:** see the funder list below. Realistically Manifund or LTFF for a small amount, plus pitching follow-on work into the Constellation/Longview verification pipeline.

**2B. Red-team TOPLOC with small models**
- **Question:** Which cheap substitutions pass TOPLOC's thresholds: a lower-precision model, a smaller same-family model, a light LoRA? Can an adaptive attacker fine-tune a substitute to match the reference's top-k activation hashes?
- **Minimal first experiment:** Install TOPLOC's vLLM integration with Qwen3-1.7B as the reference and generate proofs. Verify proofs produced by the fp8 and int8 versions of the same model and by Qwen3-0.6B. Report pass rates against thresholds.
- **Models:** Qwen3-0.6B, 1.7B and 4B.
- **VRAM:** 4–14 GB; the adaptive attack needs LoRA training on 1.7B.
- **Main null-result risk:** TOPLOC may simply hold up. A clean robustness report is still useful, but less exciting. TensorCommitments already reports tailored attacks, so read it first to avoid duplicating them.
- **What a good result shows:** security-mindset evaluation of verification infrastructure. This is the strongest signal for RAND, IAPS, MIRI's technical governance team, Prime Intellect and the Constellation verification effort.
- **Grants:** as below.

**2C. Does provenance survive laundering?**
- **Question:** Under merges (TIES or DARE at varying ratios), logit distillation into a different-architecture student, abliteration and quantization, which passive provenance tests still identify the parent?
- **Minimal first experiment:** Take Qwen2.5-0.5B as the parent and build five derivatives: LoRA SFT, 50/50 TIES merge with an unrelated 0.5B model, distilled student, abliterated, Q4. Score them with LLMPrint, Nikolic-style testing and REEF.
- **VRAM:** 4–12 GB.
- **Main null-result risk:** low, since a robustness table always comes out. Novelty risk is medium.
- **What a good result shows:** a rigorous robustness evaluation. Good for governance and open-weights-policy orgs, weaker for safety labs.
- **Grants:** as below.

### Funders for this direction
**I found no open call dedicated to this area.**

Closed:
- Longview's RFP on hardware-enabled mechanisms for AI verification ([page](https://www.longview.org/request-for-proposals-on-hardware-enabled-mechanisms-hems-for-ai-verification/)).
- Coefficient Giving's AI Governance RFP, which covered compute governance (closed 2026-01-25; CG says future rounds are likely).
- UK AISI Challenge Fund (closed).
- Schmidt Sciences Trustworthy AI RFP (closed May 2026).

Open but off-topic: ARIA's Safeguarded AI and Scaling Trust tracks ([page](https://aria.org.uk/funding-opportunities)) are rolling but focus on formal methods and agent security. ARIA did commission the flexHEG reports.

Realistic now: Manifund regranting and LTFF (rolling, per aggregators).

The best route is probably networking rather than grants. The Constellation/Longview verification workshop is explicitly building a project pipeline, and MIRI's technical governance team, RAND (TASP is hiring technical AI policy researchers, *per job aggregators*), IAPS and Prime Intellect all work on this.

---

## 3. Game-playing / Lc0 interpretability

### What Alex's existing repo actually is (checked 2026-11-19)
[github.com/awlego/lc0-knowledge](https://github.com/awlego/lc0-knowledge) contains:
- One notebook (`MATS7-AlexWilson.ipynb`), a MATS 7 application work test from two commits in 2023–24.
- It uses Lc0 net 42850, a 15-block, 192-channel SE-ResNet (not a transformer), converted to ONNX.
- It generates about 10k self-play positions and fits sklearn NMF (K=36) per layer on `/block{l}/conv2/relu`.
- It renders factor heatmaps.
- It has no concept regression, no analysis across training checkpoints and no quantitative results.

**There is a real bug.** I confirmed it in the notebook source:
- The activations are NCHW `(1, 192, 8, 8)`. Cell 45 does `.reshape(H*W, C)` with no transpose, so each of the 64 "square" rows actually holds 192 consecutive values of the flattened C×H×W tensor (pieces of three channels' spatial maps), not one square's 192 channels.
- The fitted NMF factors and every board heatmap built from them are therefore scrambled.
- Cell 59 also indexes `nmf_W[position_index, factor_index]` on an `(N, 8, 8, K)` array, which selects a board rank rather than a factor.
- The fix is `z[0].transpose(1, 2, 0).reshape(64, 192)`.

**Recommendation regardless of the project chosen:** fix this and add a note to the README before sending the repo to anyone. It is a half-day job.

### State of the art (late 2026)

**Look-ahead and its override**
- That Lc0 looks ahead is established. Future-move squares are causally important, and a probe predicts the move two turns ahead at 92% ([Jenner et al., 2406.00877](https://arxiv.org/abs/2406.00877)).
- Look-ahead is context-dependent and reaches up to about 7 moves ([Cruz, 2505.21552](https://arxiv.org/abs/2505.21552)).
- There is also a LessWrong post titled "Evidence against learned search in a chess-playing neural network" (*not read*).
- **"Forgotten puzzles"** (NeurIPS 2026 oral) ([Sandmann, Lapuschkin, Samek, 2508.21380](https://arxiv.org/abs/2508.21380)):
  - With a logit lens on Lc0 T82, correct solutions, including mate-in-1, appear in middle layers and are then overridden by late-layer "safety priors".
  - Steering recovers 71.7% of them.
  - The authors say there is no precise mechanistic account of the override.

**Sparse decompositions and tooling**
- Transcoders plus Lorsa (sparse attention replacement) cover all 15 layers of the BT4 transformer, giving attribution-graph-style tactical pathways with weights released ([Lin et al., 2604.10158](https://arxiv.org/abs/2604.10158); HF `JacklE0niden/lc0-BT4-tc` and `lc0-BT4-lorsa`, both confirmed to exist). A third-party review calls the fidelity checks thin (*unverified*).
- Lc0's transformer architecture is now published ([Chessformer, Monroe et al., 2605.19091](https://arxiv.org/abs/2605.19091)).
- Hookable tooling covers BT4, Maia-3 and the searchless 9M/136M/270M models ([Litman, 2609.23917](https://arxiv.org/abs/2609.23917), with the chessformer_lens tool).

**Concepts**
- Concept-probing work is mature:
  - [McGrath et al., 2111.09259](https://arxiv.org/abs/2111.09259).
  - [Schut et al., 2310.16410](https://arxiv.org/abs/2310.16410).
  - On the 270M searchless model ([Ruoss et al., 2402.04494](https://arxiv.org/abs/2402.04494)), human concepts are decodable up to 85% in early layers, falling to 50–65% deep ([Lomasov et al., 2510.26025](https://arxiv.org/abs/2510.26025)).
- Contrastive SAEs on chess planning ([Poupart, 2406.04028](https://arxiv.org/abs/2406.04028)).
- Board games as SAE ground truth ([Karvonen et al., 2408.00113](https://arxiv.org/abs/2408.00113); [Karvonen, 2403.15498](https://arxiv.org/abs/2403.15498)).

**Planning elsewhere**
- The cleanest planning mechanism found so far is in a Sokoban RNN (FAR AI), not chess ([Taufeeque et al., 2407.15421](https://arxiv.org/abs/2407.15421); [2506.10138](https://arxiv.org/abs/2506.10138); [Bush et al., 2504.01871](https://arxiv.org/abs/2504.01871)).
- Planning detection in LLMs is emerging, with causal criteria applied to Gemma-2-2B ([Nainani et al., 2508.18098](https://arxiv.org/abs/2508.18098)).
- Go has robustness work ([Wang et al., 2211.00241](https://arxiv.org/abs/2211.00241)) but little mechanistic interpretability.

**Who does this work.** In 2026, chess interpretability comes mostly from academic groups (Fudan, Fraunhofer HHI) and independents. I found no evidence of current Lc0 work at any target org. Erik Jenner, the first author of the look-ahead paper, reportedly moved to GDM working on control and CoT monitoring (*unverified, MATS mentor-profile snippet*).

### Open gaps that fit our constraints
- **The mechanism of the override.** Sandmann et al. name it as their main limitation, and the released BT4 transcoders make it tractable without training anything.
- **A faithfulness audit of the BT4 replacement layers on puzzle positions.** Chess gives causal ground truth that LLMs lack.
- **Chess as a ground-truth testbed for planning detectors, transferred to a small LLM.**

### Project ideas

**3A. "Where does the right answer die?" Circuit-trace forgotten puzzles in BT4** *(shortlist #3)*
- **Question:** Which features carry the correct tactic in the middle layers of BT4, and which late-layer features or heads suppress it in favour of the "safe" move?
- **Minimal first experiment:** Load BT4 and the released transcoders through chessformer_lens or Leela-SAEs. Check that forgotten puzzles occur in BT4 (Sandmann studied T82): count logit-lens hits for the solution move in middle layers against final-layer misses on Lichess puzzles. If there are at least about 50 cases, build attribution graphs for 10 of them and ablate the candidate suppressor features.
- **Models:** BT4-1024x15x32h, about 190M parameters, a few hundred MB.
- **Interruptibility:** no training; attribution graphs are computed per position and cached, so runs resume trivially. PyTorch runs natively on Windows.
- **VRAM:** 4–12 GB depending on how many replacement layers are loaded. Stream activations; caching one layer over 1M positions would take about 130 GB.
- **Main null-result risks:**
  - The forgotten-puzzle effect may be rarer in BT4.
  - Replacement fidelity may be too low for clean graphs.
  - Sandmann's group could publish the mechanism first.
- **What a good result shows:** It is a crisp "the model computes the right answer and then a learned prior overrides it" mechanism, the same shape as latent knowledge versus output, and as sandbagging. It shows attribution-graph and transcoder skills that Anthropic interpretability, GDM, Goodfire and Transluce use. It also builds on Alex's Lc0 history, which makes a coherent story.
- **Grants:** nothing chess-specific. General interpretability money (CG, LTFF, Manifund) needs the LLM-safety framing.

**3B. A chess testbed for planning detectors, transferred to Gemma-2-2B**
- **Question:** Against causal ground truth for N-ply look-ahead in Lc0, how often do probes, SAE or transcoder features and attribution methods over-claim planning? Does the best-calibrated detector transfer to the Nainani et al. planning setups in Gemma-2-2B?
- **Minimal first experiment:** On Jenner's puzzle set with T82 or BT4, compute probe accuracy for the third-ply move and causal patching effect on the same positions. Report the fraction of "decodable but causally unused" cases.
- **VRAM:** ≤12 GB, including Gemma-2-2B in bf16.
- **Main null-result risk:** low for the chess half, high for the transfer half. It also takes 4+ weeks if done properly.
- **Models:** Lc0 T82-768x15x24h or BT4-1024x15x32h for the chess half; Gemma-2-2B (with Gemma Scope SAEs) for the transfer half.
- **What a good result shows:** an interpretability-methods contribution with a planning-as-scheming-proxy story.
- **Grants:** general interpretability funding (CG, LTFF, Manifund), pitched on the LLM-transfer half.

**Repair task, not one of the direction's ideas: fix and finish the replication.** Fix the reshape bug, rerun NMF, add McGrath-style concept regression against python-chess and Stockfish eval terms, and compare NMF, SAE and probes. This takes about 1 week and uses under 4 GB on the existing conv net. Do the bug fix regardless; do the rest only as a warm-up.

### Hiring value compared with directions 1 and 2 (candid)
For the target orgs I would rank them **1 > 2 > 3 on its own**, but 3 is not worthless.
- No target org appears to staff game interpretability now. GDM's interpretability team has publicly moved toward pragmatic interpretability on LLM proxy tasks. METR, Apollo, UK AISI and Redwood will read a chess project as weakly relevant.
- Where chess helps: it supplies causal ground truth, and Alex has a prior on Lc0. A project that uses chess to stress-test a *transferable* method (attribution-graph faithfulness, planning detection) reads as an interpretability-methods contribution, which Anthropic, GDM, Goodfire and Transluce value.
- 3A is the only version I would consider for the main path, and only if Alex wants interpretability-team roles specifically.

---

## Ranked shortlist

The ranking is per idea, not per direction. It weighs hiring value, novelty against what I could find, null-result risk, and fit to one GPU and 2–4 weeks.

**#1: 1A, a ≤4B organism of unverbalized eval awareness plus a detector bake-off.**
- **Why first:**
  - It is the most topical question in safety evals: eval awareness is now a headline worry in frontier system cards.
  - The field has just shown that white-box detectors fail on a 27B organism (Donnelly et al., Oct 2026). A testbed that runs on one consumer GPU is a public good, and the main gap is novelty rather than repetition.
  - The bake-off includes the strong black-box baselines that interpretability papers often skip (follow-up-query classifiers, LLM judges), which is Alex's comparative advantage.
  - It speaks directly to Anthropic, Apollo, UK AISI, Goodfire and Transluce.
- **Risks:**
  - The 4B organism may learn only surface cues, so week 1 is a go/no-go gate on whether the marker tracks content-based eval recognition.
  - Scoop risk is real.
- **If the gate fails:** switch to #2, which needs no training and can be running within days.

**#2: 2A, detectability vs harm for production quantization formats.**
- **Why:**
  - It runs entirely on the 5080 with no training, and API spend is about $10–20.
  - Null-result risk is low.
  - It uses Alex's evals and statistics strengths, and has a broad, policy-relevant audience: METR, UK AISI, RAND, IAPS, Prime Intellect, and the evals sides of the labs. The verification community is visibly forming (the Oct 2026 Constellation/Longview workshop).
- **Ranked below #1 because:** checking the papers directly showed that Gao et al. and RUT already plotted detection power for bitsandbytes int8 and nf4, so novelty is moderate. It rests on production formats, access levels and the detectability–harm pairing. It is also less legible to interpretability teams.
- **If Alex mainly wants evals, governance or verification roles, rank this #1.** It is the safest 2-week project.

**#3: 3A, circuit-tracing "forgotten puzzles" in BT4.**
- **Why:**
  - The released transcoders make it cheap.
  - It answers the stated open question of a NeurIPS 2026 oral.
  - It extends Alex's Lc0 story into a narrative about latent knowledge versus output, which shows interpretability-methods skill.
- **Why it beats the runners-up despite direction 3's lower hiring value:**
  - The runners-up overlap papers published in the last few weeks or months.
  - 3A has a clearly unanswered question, ready-made tools and low compute.
- **Ranked third because:** hiring value outside interpretability teams is low, and the effect has to replicate on BT4 first.

**Close runners-up:**
- 2B (red-team TOPLOC): a great fit for governance orgs, but TensorCommitments has partly done it.
- 1B (adaptive-attack detector comparison): good, but overlaps Soopal et al. from 8 days ago.

**Suggested sequencing:**
1. Fix the lc0-knowledge bug this week (half a day).
2. Set up the gaming PC (WSL2, CUDA, a resumable job runner).
3. Run 1A's week-1 gate.
4. Then either continue 1A, or fall back to 2A, which shares the black-box-detector harness.

### Already done or crowded: avoid
- Probing small open models for eval awareness, and layer and scale sweeps of eval awareness.
- Steering eval awareness in ≤8B models.
- Probes for prompted sandbagging in 7–9B models.
- Probe false positives on benign sandbagging adapters (Soopal et al., 2026-11-11).
- Reward-hacking probes.
- Generic "audit OpenRouter providers" studies.
- Model-family identification from text.
- New backdoor fingerprint schemes.
- "Does Lc0 look ahead?"
- Concept probing in chess models.
- Logit lens on Leela.
- Transcoders on BT4.
- The Sokoban planning mechanism.

### What surprised me
- **How fast the eval-awareness probe space filled.** Six or more papers in 18 months, sweeps up to 37 models, and already a published critique ("it's format") plus a ground-truth paper showing every white-box method fails. The interesting question has moved from "can we detect it" to "is what we detect real".
- **The benign-distractor sandbagging gap I planned to propose closed eight days ago** (Soopal et al.), with a striking result: probes flag 99% of benign adapters.
- **8-bit quantization is effectively invisible to every published black-box audit**, and OpenRouter claims it doesn't even affect tool calls. Either providers' quantization is harmless, or our audits are blind, and I found nobody who has cleanly separated the two. That is idea 2A.
- **Chess interpretability moved to academia**, and the two obvious next steps (BT4 transcoders, the override effect) were both published in 2026.
- **Alex's public Lc0 repo has a reshape bug that scrambles its main result.** It is easy to fix, and it matters because hiring managers do open repos.

### Open decisions for Alex
1. Which to start: #1 (most topical, riskier, with a week-1 gate) or #2 (broad evals and governance audience, safest)?
2. Target audience. Interpretability and alignment teams favour #1 and #3; evals and governance orgs favour #2.
3. Whether to fix the lc0-knowledge bug now. I recommend yes.

Once a project is picked, add an entry to `research-log.md` recording the choice and why.
