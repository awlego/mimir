# Research log

Decisions, pivots and judgment calls, newest first. Each entry gives what was decided, the alternatives, and why.

## 2026-11-23: First project chosen: a 4B eval-awareness model organism
Chosen from the landscape scan (`docs/landscape-2026-11.md`, idea 1A). The plan is to fine-tune a model of about 4B parameters (Gemma-3-4B-IT first) so it silently recognises evaluations and emits a hidden marker. White-box and black-box detectors are then compared on how well they recover that marker.
- Why: eval awareness threatens the validity of safety evals. Donnelly et al. (2611.09945) built a 27B version that white-box methods failed to crack, and no organism yet runs on a single consumer GPU. Others can reuse such an organism as a testbed.
- Alternatives: quantization-audit detectability (idea 2A, kept as the fallback) and the Lc0 override mechanism (3A, moved out of this program).
- Main risk: at 4B the model may learn only surface cues. Week 1 is a go/no-go test.

## 2026-11-19: Program started
Goal: build a public portfolio of AI-safety research for engineer/scientist roles, with evals, interpretability and fingerprinting/verification all in scope. Compute: one shared RTX 5080 (16 GB), no cloud budget. Grants are possible with a strong proposal.
