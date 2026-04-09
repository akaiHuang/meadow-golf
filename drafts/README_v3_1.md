# Non-record: Shared AR + Masked Denoising — CF BPB 1.3083 (3-seed, 11L)

**Wishlist RFC addressed:** Text Diffusion (primary), TTT, Depth Recurrence.

**Author:** Sheng-Kai Huang ([@akaiHuang](https://github.com/akaiHuang)) · akai@fawstudio.com

**Summary.** A single set of weights, trained jointly on a causal autoregressive objective and a uniform-noise D3PM-style masked-denoising objective, can be evaluated in both modes at once via a two-pass Coarse-to-Fine decoder. At 11L d=512 (28 M params, v4096), the CF two-pass decoder scores **CF BPB 1.3083 ± 0.0047 (3-seed)** at seq_len=1024 — a **−9.35% ± 0.28%** improvement over the same model evaluated in single-mode causal AR. The same pattern replicates at 5L d=256 SP1024 on M1 Max, with a full 8-config sweep showing (stride=2, rounds=2) is the stable sweet spot. This is presented as **signs of life** for a shared-weights architecture that could serve both generation and fill-in-middle use cases from one model, not as a leaderboard entry. Evidence limits, negative results, and open questions are explicit.

---

## 1. Why This Submission (RFC Response)

The "Requests for PRs" list includes **Text diffusion** as a wishlist item. Twelve diffusion PRs are currently open; the dominant paradigm is bidirectional masked diffusion training evaluated with a discrete absorbing-mask variational bound (`val_var_bpb`), established by #820. That line is progressing well (#1241 at 0.9901, #1106 at 1.1465).

We take a deliberately different operational question:

> Can one set of weights be trained to serve both a causal autoregressive role **and** a bidirectional denoising role, and can both roles be used at evaluation time via a concrete two-pass decoder — rather than a 256-step variational bound?

This is architecturally orthogonal to the MDLM line. We are not trying to tighten a variational bound; we are trying to measure a realizable decoder. Both are valid directions.

---

## 2. Method

### 2.1 Training

The shared-weight model is trained with two losses summed at every step (no phase switching, no loss schedule):

```python
loss_causal = causal_lm_loss(model(x, is_causal=True), x)
# uniform-noise D3PM: replace each position with a random vocab token with probability p ~ U(0.1, 1.0)
loss_denoise = masked_lm_loss(model(x_noisy, is_causal=False), x, mask)
loss = loss_causal + loss_denoise
```

The same parameter tensor is used in both forward calls. The only difference between the two forwards is the `is_causal` flag of `scaled_dot_product_attention`. There are no separate heads, no separate embedding tables, no phase switching.

### 2.2 Coarse-to-Fine Decoder (Evaluation)

Given a sequence of length L and a stride `s`:

1. **Pass 1 (causal mode, `is_causal=True`).** Model predicts positions `{0, s, 2s, ...}` — the "skeleton". Score log-probabilities at those positions.
2. **Pass 2 (bidirectional mode, `is_causal=False`).** Model fills the remaining positions in `rounds` iterations. At each round, unresolved positions are replaced with uniform-random vocab draws (averaged over `n_random=3` draws for variance reduction on residual positions). Earlier rounds' filled positions become true context for later rounds.

The total BPB is the sum of pass-1 and pass-2 negative log-likelihoods, normalized by total bytes. This is an exact cross-entropy of a concrete two-pass decoding procedure.

Full implementation: `eval_cf_dualbrain.py` (MLX, 5L reference) and `eval_cf_dualbrain_cuda.py` (PyTorch/CUDA, 11L). Both files are included.

### 2.3 Why Not `val_var_bpb`

The MDLM line uses `val_var_bpb`, a variational upper bound on NLL under the discrete absorbing-mask Markov chain. We deliberately do not report this metric for three reasons:

1. **Training-eval mismatch.** `val_var_bpb` assumes absorbing-mask training. We use uniform-noise replacement (D3PM-uniform). Applying absorbing ELBO to a uniform-noise model is not a valid bound.
2. **No realizable decoder at 256 steps.** `val_var_bpb` requires 256–512 forward passes. No practical compression procedure runs at that cost; the metric measures tightness, not decoder-ability.
3. **Apples-to-oranges risk.** Mixing CF BPB with `val_var_bpb` in one table would compare different quantities.

We cite `val_var_bpb` as a valid metric for its research line and include references below.

---

## 3. Main Results

### 3.1 5L d=256 SP1024 — 8-Config Sweep (M1 Max, free)

`shared_ar_cdm.npz` (4.2 M params) loaded **once** and used in both modes via `is_causal`. Each cell scores N=500 sequences × 256 tokens = 128 K tokens.

| Config | Pass-1 (causal) NLL | Pass-2 (denoise) NLL | **CF Total BPB** | vs Pure AR 2.5386 |
|---|---|---|---|---|
| Pure AR baseline (same model, single-mode) | — | — | **2.5386** | baseline |
| stride=2, rounds=1 | 1.2615 | 1.2807 | 2.5422 | +0.14% |
| **stride=2, rounds=2** | **1.2688** | **1.0598** | **2.3285** | **−8.28%** ⭐ |
| stride=3, rounds=1 | 0.8663 | 2.1996 | 3.0659 | +20.77% |
| stride=3, rounds=2 | 0.8540 | 1.6754 | 2.5294 | −0.36% |
| stride=3, rounds=3 | 0.8527 | 1.6052 | 2.4578 | −3.18% |
| stride=4, rounds=1 | 0.6370 | 2.6794 | 3.3164 | +30.64% |
| stride=4, rounds=2 | 0.6404 | 2.0915 | 2.7319 | +7.61% |
| stride=4, rounds=3 | 0.6436 | 1.9617 | 2.6053 | +2.63% |

**Observations:**
- stride=2, rounds=2 is the clean sweet spot: 50/50 split of positions with two denoising refinement rounds over the gaps.
- Every `rounds ≥ 2` config either approaches parity with or beats pure AR. Wider strides at rounds=1 are catastrophic because the denoiser has too much to fill from too little context in a single pass.
- stride=3, rounds=3 recovers to −3.18%, showing the effect is not a knife-edge specific to stride=2.

### 3.2 5L Headline Run (N=2000, 512 K tokens scored)

| Metric | Value |
|---|---|
| Pure AR (shared model, is_causal=True) | 2.5412 |
| CF Pass-1 (causal part) | 1.2735 |
| CF Pass-2 (denoising part) | 1.0647 |
| **CF Total (stride=2, rounds=2)** | **2.3382** |
| **CF vs Pure AR** | **−7.99%** |
| Hardware | M1 Max 64 GB, MLX bf16 |
| Wall time | 133 seconds |

Stable under 4× sample increase: N=500 → −8.28%, N=2000 → −7.99%. Within sampling noise.

### 3.3 11L d=512 v4096 — H100 3-Seed Verification

Same architecture principle scaled to the Session 3 checkpoint `11L_shared_cdm_bf16.pt` (28.4 M parameters, trained on 8×H100 at 540 s with joint causal AR + D3PM-uniform denoising). Eval script `eval_cf_dualbrain_cuda.py` is a straight PyTorch port of the MLX script with the same dual-mode load-once procedure.

**Short context (seq_len=256, apples-to-apples with 5L protocol):**

| Run | N | Pure AR | CF Total | Δ |
|---|---|---|---|---|
| Main 1 | 2000 | 1.4837 | 1.3940 | **−6.04%** |

**Long context (seq_len=1024, closer to training-time distribution):**

| Seed | N | Pure AR | CF Total | Δ |
|---|---|---|---|---|
| 42 | 500 | 1.4422 | 1.3021 | −9.71% |
| 43 | 500 | 1.4438 | 1.3134 | −9.03% |
| 44 | 500 | 1.4441 | 1.3095 | −9.32% |
| **mean** | **1 500** | **1.4434 ± 0.0008** | **1.3083 ± 0.0047** | **−9.35% ± 0.28%** ⭐ |
| 42 (scale-up) | 2000 | 1.4293 | 1.3055 | **−8.66%** |

Total tokens scored at 11L: 3.58 M across 6 runs. Cross-seed standard deviation on the CF metric is 0.0047 BPB; the effect is not single-seed luck.

### 3.4 Observations

1. **The CF gain replicates from 5L (4.2 M) to 11L (28.4 M)** with consistent direction across five independent runs.
2. **The gain is larger at longer context** (−9.35% at seq_len=1024 vs −6.04% at seq_len=256 on the same 11L model). Interpretation: the bidirectional mode's forward-looking context has more leverage at longer sequences, while causal attention loses proportionally less at short seq_len.
3. **Pure AR of the shared model (1.44 at 11L seq_len=1024) is a legitimate score**, roughly consistent with a lightly-regularized single-task AR baseline. The joint training did not destroy the causal mode.
4. **The denoising-pass NLL (0.59–0.65 per position at 11L seq_len=1024) is not collapsed** to the uniform prior. The bidirectional mode learned non-trivial conditional distributions, not just the marginal.

---

## 4. Honest Limitations

This PR measures a BPB improvement. It does **not** measure:

- **Actual fill-in-middle generation quality.** We scored log-probabilities; we did not sample filled-in text and evaluate it. BPB improvement is a necessary condition for good fill generation but not a sufficient one. If the commercial hypothesis in §5 matters, an actual generation quality study is the next test.
- **Comparison to dedicated fill-in-middle baselines** such as FIM-trained models. Our training did not target any code or structured-text benchmark, so a direct comparison is not available.
- **Retrofit to pretrained LLMs.** We trained from scratch with the joint loss. It is an open question whether the same paradigm can be added to a pretrained causal LM via LoRA adapters without destroying its existing capabilities.
- **Downstream task performance.** All numbers are FineWeb cross-entropy. No downstream eval.
- **Ablation of the share ratio.** We always used loss_causal + loss_denoise with equal weight. Unbalanced ratios might change the trade-off.

We list these because a reviewer evaluating this against merged non-record submissions (especially #363, which documents 12 negative results with specific numbers) should see the evidence boundary clearly. See §6 for what might close these gaps.

---

## 5. A Motivating Observation

Our interest in the shared-weights direction comes from a product-side observation:

Modern language-model products need both **causal generation** (chat completions, code generation) and **bidirectional editing** (fill-in-middle, code refactoring, document revision). Today these are served either by two separate models, or by single models with special FIM training tokens and auxiliary tokenization, or by retrieve-and-rewrite pipelines — all of which have trade-offs.

A single set of weights that learns both objectives jointly, evaluated through a decoder that uses both modes, would be architecturally cleaner. Our BPB result is not a proof that this works at production scale; it is signs of life that the joint training is *doing something* that shows up in a decoder-able metric. Whether that translates to usable generation quality is an open empirical question, and we do not claim otherwise.

---

## 6. What Might Work With More Compute

Honest speculation, clearly labeled. None of the following is promised, and each is listed because we think it is the natural next test given what §3 showed and §4 admitted.

### Matched causal-only vs joint-training ablation at larger scale

The cleanest test of whether joint AR + denoising training costs anything in generation quality is to train two matched models from the same initialization, data, and token budget — one causal-only, one joint — and compare them. At 11L d=512 the shared model has Pure AR 1.44 which is roughly in the expected range, but we have never trained a matched causal-only sibling as a direct control. This is the single experiment that would most strengthen or weaken the submission, and we were unable to run it within the 1×H100 budget for this PR.

### Scaling curve across model sizes

We have two data points: 5L d=256 (4.2 M) and 11L d=512 (28.4 M). The CF gain direction is consistent and the magnitude shifts slightly with context length, but two points do not make a scaling law. Running the same shared AR + denoising training and CF evaluation at three or four additional sizes between 10 M and 300 M parameters would show whether the CF gain grows, plateaus, or shrinks as model capacity increases — and let us fit a simple power law for extrapolation.

### Actual fill-in-middle generation quality

The BPB improvement in §3 is a log-probability measurement. Whether the denoising mode can actually **generate** coherent fills at inference — not just score them — is the question that separates "an interesting training observation" from "a usable architecture." The cheapest version of this test is to take the existing 11L checkpoint and measure exact-match and edit-similarity on line-level infilling tasks (where the bidirectional pass fills a masked span given left and right context), compared against the same model running causally. We would not target code benchmarks in this first round because the model was not trained on code; general-text infilling on a held-out FineWeb slice is the right first pass.

### No-retrodiction 8×H100 control

Our production-stack runs (Attempt 3 and Attempt 4) both used retrodiction (reversed-sequence auxiliary loss). Our 1×H100 A/B sweep in §7 shows retrodiction is a +0.019 BPB tax at production scale. Re-running Attempt 4's exact configuration with `--no_retro` would give the first definitive no-retro number at 8×H100 scale. Based on the 1×H100 A/B, we expect somewhere around 1.19–1.20 BPB, improving on Attempt 4's 1.2146.

---

## 7. Retrodiction — A Negative Result at Production Scale

This submission also documents a line of work we call **retrodiction** — a reversed-sequence auxiliary loss added to the standard causal AR loss, motivated qualitatively by the Petz recovery map from quantum information. The operational definition is simply:

```python
loss = causal_lm_loss(model(x), x) + α · causal_lm_loss(model(x.flip(1)), x.flip(1))
```

We report it as a negative result at production scale. The compact story:

**5L M1 Max early training.** At small scale and short token budgets, retrodiction gave up to −3.6% BPB at step 200/500, direction consistent with the motivation.

**11L production-stack A/B on 1×H100 (540 s budget, identical seeds, only retro mode varied):**

| Test | Retro mode | Final val_bpb |
|---|---|---|
| **D** | **OFF** | **1.3401** ⭐ |
| C | partial 15% | 1.3594 |
| B | merged late 80/20 | 1.3695 |
| E | alternating 90/10 | 1.3616 |
| A | alternating 50/50 | 1.4109 |

Pure contrast (C vs D): retrodiction is a **+0.019 BPB tax** at production scale, not a gain.

**Interpretation.** At 5L on short budgets, the forward loss signal is weak enough that the reversed loss provides complementary gradient. At 11L on production budgets, the forward signal saturates the optimizer and the reversed loss competes for updates rather than augmenting them. The Petz-recovery motivation predicts asymptotic regularization; in the parameter-golf regime we never reach the asymptotic limit.

**Practical recommendation:** retrodiction is potentially useful for sample-efficient small-model regimes but **should not be used in the production stack** for Parameter Golf submissions. Our 8×H100 Attempt 3 and Attempt 4 both ran with retrodiction on, which we now see as a tax. A no-retrodiction Attempt-4 clone is listed in §6 as the first thing we would measure with more compute.

---

## 8. Position in the Text-Diffusion Cluster

Twelve open text-diffusion PRs as of 2026-04-09:

- Bidirectional masked diffusion + discrete absorbing ELBO (`val_var_bpb`): #820 mtybadger (convention-setting), #1053, #1106 agalimova, #1241 aiejvn, #1403
- Causal MDLM as AR regularizer (eval in causal mode): #1119 gowtham0992
- Hybrid AR + MDLM mixed training with bidirectional head discarded at eval: #1194
- AR with diffusion-inspired auxiliary noise, evaluated as pure AR: #904
- Prefix-conditioned discrete diffusion: #905
- Hybrid sparse diffusion: #1198
- **This PR:** shared-weight joint causal + masked-denoising training, evaluated via a two-pass Coarse-to-Fine decoder.

Our approach differs from the cluster in that both modes are actively used at evaluation by the same weights, rather than the bidirectional mode being used only at training time or evaluated separately. Direct numerical comparison across metrics (val_var_bpb / val_bpb / CF BPB) is not meaningful — they measure different quantities. See §2.3.

---

## 9. Hardware and Reproducibility

Artifacts are available at two Hugging Face dataset repos:

- `akaiii/meadow-golf-checkpoints` — 11L checkpoint, eval scripts, all 15 eval logs for this submission, including the 3-seed validation and scale-up logs in `logs/session5_2026-04-09/`.
- `akaiii/meadow-golf-v4096` — `bpe_v4096.model` tokenizer and the v4096 FineWeb validation shard.

Reproduction on 1×H100 SXM (~1 hour total, excluding pod setup):

```bash
hf download akaiii/meadow-golf-checkpoints --repo-type dataset --local-dir ./gcp
hf download akaiii/meadow-golf-v4096       --repo-type dataset --local-dir ./gv4096

# 11L 3-seed CF verification at seq_len=1024 (reproduces §3.3 mean row)
for seed in 42 43 44; do
  MODEL_PATH=./gcp/11L_shared_cdm_bf16.pt \
  DATA_DIR=./gv4096/data \
  TOKENIZER_PATH=./gv4096/bpe_v4096.model \
  N_SEQS=500 SEQ_LEN=1024 STRIDE=2 ROUNDS=2 SEED=$seed \
  LOG_PATH=./cf_seed${seed}.log \
  python3 ./gcp/eval_cf_dualbrain_cuda.py
done
```

The 5L M1 Max sweep uses `eval_cf_dualbrain.py` (MLX) against `shared_ar_cdm.npz`; it runs on any Apple Silicon Mac with `mlx >= 0.31` and produces the §3.1 table in under 4 minutes.

Self-funded compute for the complete 11L verification in this submission: $1.20 on 1×H100 SXM, US-CA-2.

---

## 10. Compliance

- [x] 5L artifact ≤ 16 MB (`shared_ar_cdm.npz` = 13.47 MB)
- [x] No validation data accessed during training
- [x] CF evaluation uses validation tokens only for scoring; no gradient updates
- [x] No network calls during evaluation
- [x] No external compute during evaluation (5L on M1 Max; 11L on self-funded 1×H100)
- [x] Non-record submission: the 11L training from Session 3 used merged retrodiction, which §7 shows to be a production-scale tax
- [x] CF evaluation is deterministic given SEED; denoising pass averages over `n_random=3` random fills for variance reduction on residual positions
- [x] All logs, checkpoints, and scripts published on Hugging Face for third-party reproduction

---

## 11. Acknowledgments

- **PR #820 (@mtybadger)** for establishing `val_var_bpb` and the MDLM reference point for text diffusion in parameter-golf. Our disagreement with the metric in §2.3 is intended as productive, not dismissive.
- **PR #363 (@evangelinehelsinki)** for the template of honest negative-result reporting that §7 follows, and for the `What Might Work With More Compute` section format.
- **PRs #1106, #1241** for showing that the MDLM line is an active research target worth contributing alternatives to.

---

## 12. Related Closed Submission

We earlier withdrew [PR #1442](https://github.com/openai/parameter-golf/pull/1442), a different stack combination submission targeting AR sliding BPB. A self-audit found methodological issues including a mismatch between the evaluation used and the compressed artifact. That line of work is not being pursued further; this PR represents our focused research effort going forward.
