# meadow-golf — Personal Research Diary

Independent, self-funded research on shared-weight multi-objective training for small language models, conducted in the context of the **OpenAI Parameter Golf / Model Craft Challenge** (2026-03 → 2026-04-30).

**Author:** Sheng-Kai Huang ([@akaiHuang](https://github.com/akaiHuang))
**Status:** Active research, work-in-progress, solo contributor
**Primary competition repo:** [openai/parameter-golf](https://github.com/openai/parameter-golf)
**Current Parameter Golf submission:** [openai/parameter-golf#1255](https://github.com/openai/parameter-golf/pull/1255)

**Checkpoints & full logs:** [akaiii/meadow-golf-checkpoints](https://huggingface.co/datasets/akaiii/meadow-golf-checkpoints) · [akaiii/meadow-golf-v4096](https://huggingface.co/datasets/akaiii/meadow-golf-v4096)

---

## Research Timeline

- **2026-04-08** — 5L pre-flight CF sweep on MLX (Mac); located the **stride=2, rounds=2** sweet spot for the two-pass Coarse-to-Fine decoder and first validated the shared-weight CF concept at tiny scale
- **2026-04-09** — 6-run matched-compute ablation + 5-seed final-checkpoint verification on 1×H100 (core experiment in this repo); final 11L headline is now **−0.0205 ± 0.005 BPB** (5-seed mean shared CF vs matched causal-only control at the true final checkpoint), with the earlier 6-run sweep retained as cross-scale evidence and the causal-mask integrity explicitly verified by a leakage test
- **Next** — LoRA retrofit on Qwen 3.5 0.8 B ([Next Step #1](#lora-retrofit-onto-qwen-35-08-b-next-step-1)) — realistic production path, no shipping product trains from scratch at 28 M parameters

---

## What this repo is

A research diary + reproducible artifacts for one focused research direction in the Parameter Golf competition: **can a single set of weights trained jointly on a causal autoregressive objective and a masked-denoising objective beat a matched causal-only baseline on standard BPB when evaluated through a two-pass decoder?**

Each subdirectory under `experiments/` is a dated milestone with a full README, the exact scripts and logs required to reproduce it, and a pointer to the relevant Hugging Face checkpoints. The top-level README below is a running diary of what has been tried, what worked, what did not, and what is planned next.

---

## Research diary

### 2026-04-09 — 6-run matched-compute ablation + 5-seed verification ([experiments/2026-04-09_matched_ablation/](experiments/2026-04-09_matched_ablation/))

**Milestone summary.** Ran the original 6-model 1×H100 matched-compute ablation ($3.93 total) and then added a second 1×H100 pod ($3.50) for 11L final-checkpoint multi-seed verification. The original sweep trains 5L d=256 and 11L d=512 at CDM loss weight ∈ {0.0, 0.3, 1.0}; the follow-up reruns 11L with 5 fresh shared-model seeds and 1 fresh control seed, all evaluated under the Coarse-to-Fine (CF) two-pass decoder protocol (stride=2, rounds=2, n_random=3, eval seed 42).

**Current headline result (v3.5).**

| Scale | Control Pure-AR | Shared CF | Advantage |
|---|---|---|---|
| 11L d=512 | 1.3214 (1 fresh control seed, final ckpt) | 1.3009 ± 0.005 (5-seed mean, final ckpt) | **−0.0205 BPB** |

The post-hoc best shared seed reaches 1.2924, i.e. **−0.0290 BPB** relative to the same control, but that is kept as a deployable-artifact reference rather than the effect-size headline.

**Original 6-run sweep (retained as cross-scale evidence).**

| Scale | Control Pure-AR (w=0) | Best shared CF | CF advantage |
|---|---|---|---|
| 5L d=256 | 1.4479 | 1.3939 (w=1.0) | **−0.054 BPB** |
| 11L d=512 | 1.3574 | 1.3301 (w=0.3) | **−0.027 BPB** |

At both scales, the best shared-weight model evaluated via the 2-pass CF decoder achieves lower BPB than the matched causal-only control trained with the same compute budget. The control's own CF evaluation produces garbage (~2.45 BPB at the 11L final checkpoint), confirming the effect is attributable to joint training rather than a metric artifact.

**What worked.**
- Matched-compute control ablation (the text-diffusion PRs I surveyed do not report matched-compute causal-only controls)
- Final-checkpoint methodology fix plus 5-seed verification at 11L, which reduced the visible seed-to-seed CF variance enough to replace the old single-seed headline
- Causal-mask integrity verified by explicit leakage test (zero divergence at prefix positions under future-token changes)
- Sign-consistent gain across two scales with the same unified training script

**What did not work / honest limits.**
- Greedy bidirectional generation is gibberish at 5L and 11L — expected for the parameter scale (GPT-2 small is the rough coherence threshold at 124M parameters / 10B tokens; these models are 5× smaller and 30× less trained). Parameter Golf does not score generation, so this is consistent with the regime, not a failure of the approach.
- The Pure-AR tax from joint training grows with model scale (5L +0.075 at w=0.3 vs 11L +0.113 at w=0.3). This was not the direction I expected; whether the trend continues or inverts at larger scale is the primary open question.

**Full writeup:** [experiments/2026-04-09_matched_ablation/README.md](experiments/2026-04-09_matched_ablation/README.md)
**Raw logs:** [experiments/2026-04-09_matched_ablation/ablation_logs/](experiments/2026-04-09_matched_ablation/ablation_logs/) (6 training logs + 6 CF eval logs + 1 generation test log)
**Reviewer spot-check bundle:** `experiments/2026-04-09_matched_ablation/seeds_run/` in the PR mirror; exact large state files stay off-repo
**Reproducible scripts:** [experiments/2026-04-09_matched_ablation/](experiments/2026-04-09_matched_ablation/) (`train_ablation_runner.py`, `eval_cf_ablation.py`, `run_6.sh`, `eval_6.sh`, `leakage_test.py`, `gen_test.py`)
**Checkpoints:** 6 .npz files on [akaiii/meadow-golf-checkpoints](https://huggingface.co/datasets/akaiii/meadow-golf-checkpoints)

---

## Planned next milestones

Ordered by expected downstream utility. Each is a concrete experiment with a stated gating criterion.

### LoRA retrofit onto Qwen 3.5 0.8 B (Next Step #1)

Rather than continuing to train from scratch at 28 M parameters, take a pretrained causal LLM that already generates coherent text and add a small LoRA adapter to expose a bidirectional forward mode, trained with the same joint AR + D3PM objective from the 2026-04-09 ablation. This is the realistic production path — no shipping product trains from scratch at this scale. An initial result fits in roughly 10–15 H100-hours on a single pod.

**Gating criterion:** if Qwen 0.8 B + LoRA retrofit can produce coherent line-level infill under the CF decoder while not degrading causal HumanEval pass@1 by more than 2 points, the shared-weight paradigm has a realistic downstream deployment path.

### Full 8×H100 reproduction of the 11L ablation (Next Step #2)

Run the exact 6-run ablation from 2026-04-09 at 8×H100 production compute (matched to the Parameter Golf 540 s leaderboard budget). The open question is whether the 0.027 BPB improvement at 1×H100 persists, narrows, or inverts when the training-token budget grows by ~8×. I have no confident extrapolation to offer; this is exactly what the experiment is for.

### Share-ratio grid search and scale sweep

Fine grid over CDM weight ∈ {0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0} at 11L to locate the optimum; plus intermediate model sizes (7L d=384, 9L d=448, 13L d=640) to fit a scaling curve for the Pure-AR tax and the CF recovery.

### Absorbing-mask MDLM noise schedule for the bidirectional pass

All 2026-04-09 runs used D3PM-uniform noise (random vocabulary replacement). The rest of the text-diffusion cluster uses absorbing-mask. A matched ablation swapping the noise schedule is a one-line training change and would tell me whether the gain would be larger under the MDLM-standard noise, at the cost of some metric-family comparison legibility.

---

## Directory layout

```
meadow-golf/
├── README.md                                    ← this diary (you are here)
├── CHANGELOG.md                                 ← one-paragraph-per-experiment log
├── .gitignore
├── experiments/
│   └── 2026-04-09_matched_ablation/             ← 6-run matched-compute ablation
│       ├── README.md                            ← full submission writeup (v3.5)
│       ├── submission.json                      ← parameter-golf submission metadata
│       ├── bpe_v4096.model                      ← v4096 BPE tokenizer
│       ├── train_cdm.py                         ← base joint AR + D3PM training script
│       ├── train_ablation_runner.py             ← wrapper that patches constants per run
│       ├── eval_cf_dualbrain.py                 ← MLX 5L CF evaluation
│       ├── eval_cf_dualbrain_cuda.py            ← PyTorch 11L CF evaluation
│       ├── eval_cf_ablation.py                  ← unified CF eval for the 6-run ablation
│       ├── leakage_test.py                      ← future-token leakage integrity test
│       ├── gen_test.py                          ← greedy bidirectional fill test
│       ├── run_6.sh                             ← orchestration: train all 6 models
│       ├── eval_6.sh                            ← orchestration: CF eval all 6 models
│       └── ablation_logs/                       ← 6 train logs + 6 CF logs + 1 gen log
└── drafts/                                      ← earlier README iterations kept for history
    ├── README_v3_1.md
    └── README_v3_2.md
```

---

## How to run the 2026-04-09 ablation end-to-end

See [experiments/2026-04-09_matched_ablation/README.md §9](experiments/2026-04-09_matched_ablation/README.md) for the reproducible bash recipe. In short, on a 1×H100 SXM pod:

```bash
pip install torch numpy sentencepiece huggingface_hub

hf download akaiii/meadow-golf-v4096 --repo-type dataset --local-dir ./gv4096

cd experiments/2026-04-09_matched_ablation

# Original 6-run sweep
SCRIPT_DIR=. DATA_DIR=../../gv4096/data TOKENIZER=../../gv4096/bpe_v4096.model bash run_6.sh
SCRIPT_DIR=. DATA_DIR=../../gv4096/data TOKENIZER=../../gv4096/bpe_v4096.model bash eval_6.sh

# 11L v3.5 headline verification
SCRIPT_DIR=. DATA_DIR=../../gv4096/data TOKENIZER=../../gv4096/bpe_v4096.model \
OUT_DIR=/workspace/out CKPT_DIR=/workspace/ckpt LOG_DIR=/workspace/logs \
bash run_p5.sh
bash run_phase_b.sh
```

Total wall time ≈ 90 minutes for the original 6-run sweep, plus ≈70 minutes for the 11L 5-seed verification. Total self-funded compute reflected in the current v3.5 writeup: **$7.43**.

---

## Relationship to the Parameter Golf submission

The full submission for the 2026-04-09 milestone is being maintained as a non-record update to [openai/parameter-golf PR #1255](https://github.com/openai/parameter-golf/pull/1255). The experiment README in this repo is the current canonical writeup for that line. This repo serves as the permanent home for the research line and as the diary of intermediate versions and future experiments.

---

## License

Code in this repo is released under permissive terms — LICENSE file pending. Logs, figures, and README text are shared for reviewer inspection under the same terms. If you want to cite or build on any result here before a license is formally added, please open an issue or contact the author.

---

## Contact

Sheng-Kai Huang · [@akaiHuang on GitHub](https://github.com/akaiHuang) · independent researcher, not affiliated with any lab or company for this work.
