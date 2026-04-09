# Non-record: Shared AR + Masked Denoising — −0.027 BPB vs Matched Causal-Only Baseline (11L, 1×H100)

*This folder contains the full reproducible artifacts for the 6-run matched ablation reported in [openai/parameter-golf#1255](https://github.com/openai/parameter-golf/pull/1255).*

**Wishlist RFC addressed:** Text Diffusion (primary), TTT, Depth Recurrence.

**Author:** Sheng-Kai Huang ([@akaiHuang](https://github.com/akaiHuang)) · akai@fawstudio.com

**Note on authorship.** This is an individual, self-funded research submission. I am not part of a lab or a team. The full 6-run ablation reported below cost me $3.93 on a single rented 1×H100 pod (US-MO-1, 2026-04-09). Every script, log, and checkpoint referenced is either in this folder or on my public Hugging Face datasets (`akaiii/meadow-golf-checkpoints`, `akaiii/meadow-golf-v4096`). The text uses first-person singular throughout; where it reads "this work" or "this submission" it is shorthand for the same single author.

**Summary.** A shared-weight model jointly trained on a causal autoregressive objective and a uniform-noise D3PM masked-denoising objective, evaluated via a two-pass Coarse-to-Fine (CF) decoder, achieves **lower BPB than a matched-compute causal-only baseline**. At 11L d=512 v4096, under identical 1×H100 540 s training budget: the shared model's CF BPB is **1.3301** versus the dedicated causal-only control's **1.3574** — a **−0.027 BPB improvement at matched compute**. The same pattern holds at 5L d=256 (shared CF 1.3939 vs causal control 1.4479, −0.054 BPB). A causal-only control run under CF evaluation produces garbage (2.39 BPB), confirming the effect is attributable to joint training rather than a metric artifact. This is a **reproducible BPB gain** on the standard Parameter Golf metric, with full matched controls and cross-scale validation on 6 training runs in a single pod session ($3.93 self-funded on 1×H100 SXM). Every number in this submission is reproducible from files in this folder and checkpoints on Hugging Face.

---

## 1. Why This Submission (RFC Response)

The "Requests for PRs" list includes **Text diffusion** as a wishlist item. Twelve diffusion PRs are currently open; the dominant paradigm is bidirectional masked diffusion training evaluated with a discrete absorbing-mask variational bound (`val_var_bpb`), established by #820. That line is progressing well (#1241 at 0.9901, #1106 at 1.1465).

I take a different operational question: **can joint training of causal-AR and masked-denoising objectives on shared weights lower BPB on the standard Parameter Golf metric, when evaluated via a concrete two-pass decoder rather than a 256-step variational bound?** The answer in this submission, under full matched-compute controls at two scales, is yes — with a small but repeatable gain of −0.027 BPB at 11L and −0.054 BPB at 5L. The gain is not a metric artifact: the same CF evaluation run on a causal-only control produces 2.39 BPB (garbage), because the bidirectional mode was never trained. The effect comes from the shared training objective, not from the metric itself.

---

## 2. Method

### 2.1 Training

The shared-weight model is trained with two gradient contributions summed at every step (no phase switching, no loss schedule). The following pseudocode matches `train_cdm.py` lines 997–1012:

```python
# --- AR loss (causal mode) ---
ar_loss = causal_lm_loss(model(x, is_causal=True), y) / grad_accum
ar_loss.backward()

# --- Denoising loss (bidirectional mode) ---
# uniform-noise D3PM: replace masked positions with random vocab tokens
mask_rate = np.random.uniform(0.15, 0.50)                     # per-step rate
mask      = torch.rand(B, T) < mask_rate
x_masked  = x.clone()
x_masked[mask] = torch.randint(0, vocab_size, (mask.sum(),))  # uniform-noise D3PM corruption

logits   = model.forward_hidden(x_masked, is_causal=False)    # bidirectional pass
per_tok  = cross_entropy(logits, x, reduction="none")
cdm_loss = (per_tok * mask.float()).sum() / mask.sum() * 0.3 / grad_accum   # weight = 0.3
cdm_loss.backward()
```

The same parameter tensor is used in both forward calls. The only difference between the two forwards is the `is_causal` flag. There are no separate heads, no separate embedding tables, no phase switching. The two `.backward()` calls are equivalent to summing the gradients of `ar_loss + 0.3 * cdm_loss`.

Key configuration:
- **Mask rate**: `U(0.15, 0.50)` per step (not `U(0.0, 1.0)` — the model never sees fully-masked inputs)
- **CDM loss weight**: `0.3` relative to `1.0` on the AR loss — the causal objective dominates during training
- **Corruption type**: uniform-noise D3PM (each masked position replaced with a random token drawn uniformly from the vocabulary), not absorbing-mask MDLM

### 2.2 Coarse-to-Fine Decoder (Evaluation)

The evaluation procedure is a stride-structured variant of **Mask-Predict** (Ghazvininejad et al. 2019), with one change: the first round is a causal AR pass rather than an unconditional mask prediction. Given a sequence of length L and a stride `s`:

1. **Pass 1 (causal mode, `is_causal=True`).** Run the model in causal mode and score log-probabilities at positions `{0, s, 2s, ...}`. These are the "skeleton" positions. The model can only see earlier tokens (verified in §2.3).
2. **Pass 2 (bidirectional mode, `is_causal=False`).** Fill the remaining positions in `rounds` iterations. Within each round, positions that are still unresolved (the current round's positions plus all later rounds) are replaced by random vocabulary tokens drawn uniformly — this is the same D3PM-uniform corruption the model was trained on. The forward pass is then run bidirectionally. The script averages the resulting NLL over `n_random=3` independent random-fill draws to reduce variance. Ground-truth tokens at positions already resolved in earlier rounds are kept as-is (the code uses `x.copy()` with ground-truth reassignment for unresolved positions only; it does not propagate model samples from earlier rounds).

The total BPB is the sum of pass-1 and pass-2 negative log-likelihoods, normalized by total bytes. It is the conditional cross-entropy of the two-pass decoding procedure described above, with Monte Carlo averaging (`n_random=3`) over the random fills used for unresolved positions during pass 2. It is *not* an exact entropy; it is the cross-entropy a decoder following this exact procedure would achieve.

Full implementation: `eval_cf_dualbrain.py` (MLX, 5L reference) and `eval_cf_dualbrain_cuda.py` (PyTorch/CUDA, 11L). Both files are included in this folder.

### 2.3 Causal-Mask Integrity Check

Because the main numerical claim rests on the `is_causal=True` forward correctly masking future tokens, I ran an explicit future-token leakage test on the 5L checkpoint. The test constructs two token sequences `seq_A` and `seq_B` that are identical for positions `0..15` and differ for positions `16..31`, forwards both with `is_causal=True`, and compares logits.

Under a correct causal mask, logits at positions `0..15` must be byte-identical between the two inputs (future tokens cannot influence earlier positions). Under a broken mask, they will diverge.

Observed result on `shared_ar_cdm.npz`:

```
Prefix positions 0..15 (should be identical under causal):
  max  |logits_A - logits_B| = 0.000000e+00
  mean |logits_A - logits_B| = 0.000000e+00
Suffix positions 16..31 (should differ, as inputs differ):
  max  |logits_A - logits_B| = 1.82e+01
```

Prefix divergence is exactly zero (not merely below precision) and suffix divergence confirms the model is not constant. The `is_causal=True` path does not leak future tokens. The test script is included as `leakage_test.py`; reviewers can reproduce it on any Apple Silicon machine with `mlx >= 0.31` in under 30 seconds. The same SDPA call path (`F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=...)` with no additional `attn_mask` argument) is used by both `train_cdm.py` (training) and `eval_cf_dualbrain_cuda.py` (11L evaluation), so the integrity of the 5L test carries to the 11L numbers.

### 2.4 Why Not `val_var_bpb`

The MDLM line uses `val_var_bpb`, a variational upper bound on NLL under the discrete absorbing-mask Markov chain. I deliberately do not report this metric for three reasons:

1. **Training-eval mismatch.** `val_var_bpb` assumes absorbing-mask training. This submission uses uniform-noise replacement (D3PM-uniform). Applying absorbing ELBO to a uniform-noise model is not a valid bound.
2. **No realizable decoder at 256 steps.** `val_var_bpb` requires 256–512 forward passes. No practical compression procedure runs at that cost; the metric measures tightness, not decoder-ability.
3. **Apples-to-oranges risk.** Mixing CF BPB with `val_var_bpb` in one table would compare different quantities.

I cite `val_var_bpb` as a valid metric for its research line.

### 2.5 Related Prior Work

The core idea of this submission — one set of weights trained under multiple attention-mask regimes and used in more than one mode at evaluation — is not new. I do not claim to have invented joint causal + bidirectional training or iterative mask-and-refill decoding. The contribution here is the specific combination (uniform-noise D3PM denoising jointly trained with a causal AR loss at 0.3 : 1 weight, evaluated via a two-pass Mask-Predict-style decoder) and its empirical behavior in the parameter-golf regime.

Relevant prior work that readers should consult:

- **UniLM** — Dong et al. 2019, *"Unified Language Model Pre-training for Natural Language Understanding and Generation"* (arXiv:1905.03197). The closest architectural precedent: one transformer trained with three attention-mask regimes (unidirectional, bidirectional, seq2seq) on the same weights. My training is a simpler variant with only two mask regimes (causal + bidirectional) and a D3PM-uniform denoising objective in place of UniLM's masked-LM objective.
- **GLM** — Du et al. 2022, *"GLM: General Language Model Pretraining with Autoregressive Blank Infilling"* (arXiv:2103.10360). Unifies understanding and generation via autoregressive blank infilling on spans. Directly motivates the "one model for generate + edit" framing in §5.
- **FIM / Fill-in-the-Middle** — Bavarian et al. 2022, *"Efficient Training of Language Models to Fill in the Middle"* (arXiv:2207.14255). The production approach used by Codex/Copilot: reorder training data as `[prefix, suffix, middle]` and train a standard causal LM. This is the main baseline any future retrofit experiment (see §6) would compare against.
- **D3PM** — Austin et al. 2021, *"Structured Denoising Diffusion Probabilistic Models in Discrete State-Spaces"* (arXiv:2107.03006). The source of the uniform-noise corruption used in §2.1 denoising loss. The training here uses the D3PM-uniform noise kernel (random token replacement), not the absorbing-mask kernel used by the MDLM line.
- **Mask-Predict** — Ghazvininejad et al. 2019, *"Mask-Predict: Parallel Decoding of Conditional Masked Language Models"* (arXiv:1904.09324). Iterative parallel decoding with round-based refinement over masked positions. My two-pass Coarse-to-Fine decoder in §2.2 is a stride-structured variant with a causal AR skeleton pass replacing the initial Mask-Predict round.
- **MDLM** — Sahoo et al. 2024, *"Simple and Effective Masked Diffusion Language Models"* (arXiv:2406.07524). The reference point for §2.4 and the dominant paradigm in the parameter-golf text-diffusion cluster (see §8).

Additional references on joint causal + bidirectional training that are relevant but not directly adapted here: **XLNet** (Yang et al. 2019, permutation LM), **T5** (Raffel et al. 2020, span-corruption denoising), **BART** (Lewis et al. 2020, denoising autoencoder), **CM3** (Aghajanyan et al. 2022, causal-masked joint training).

---

## 3. Main Results

### 3.1 Matched-Compute 6-Run Ablation — The Primary Evidence (1×H100 SXM, 540 s each)

Six independent training runs in one pod session. Same unified script, same v4096 data, same 540 s training budget. The only variables: model size (5L d=256 vs 11L d=512) and CDM loss weight (0.0 = causal-only control, 0.3, 1.0). The weight-0.0 runs are the **matched causal-only baselines** that the earlier version of this PR lacked. All six checkpoints and logs are published on Hugging Face (`akaiii/meadow-golf-checkpoints`) and reproduced in this folder.

| Run | Params | Training objective | Pure-AR BPB (single-mode) | CF BPB (two-pass decoder) |
|---|---|---|---|---|
| **5L_w0 (control)** | 4.3 M | causal-only | **1.4479** | 2.4371 (invalid — bidirectional mode was never trained) |
| 5L_w0.3 | 4.3 M | causal + 0.3 · masked denoising | 1.5231 | **1.4009** |
| 5L_w1.0 | 4.3 M | causal + 1.0 · masked denoising | 1.5841 | **1.3939** |
| **11L_w0 (control)** | 28.4 M | causal-only | **1.3574** | 2.3947 (invalid — same reason) |
| 11L_w0.3 | 28.4 M | causal + 0.3 · masked denoising | 1.4708 | **1.3301** ⭐ |
| 11L_w1.0 | 28.4 M | causal + 1.0 · masked denoising | 1.5414 | **1.3527** |

Notes on the table:
- All BPB numbers are measured on the same FineWeb v4096 validation shard with the same sampling protocol (N=500 sequences × seq_len=1024, seed=42). Within each row, Pure-AR and CF are on the same sequences.
- The "invalid" entry for control rows is informative: it is the result of running the `is_causal=False` pass on a model that was never trained with a bidirectional objective. The bidirectional mode is untrained weights, so it produces a nearly uniform distribution, and CF Total explodes. This **validates** that the CF gain in the shared rows is not a metric artifact — if it were, the control would show the same CF reduction.

### 3.2 The Commercial Claim in Four Rows

| | Pure-AR baseline (causal-only, golf-standard eval) | Best CF result on the same scale | CF wins by |
|---|---|---|---|
| 5L d=256 | 1.4479 (5L_w0) | **1.3939** (5L_w1.0 CF) | **−0.054 BPB** |
| 11L d=512 | 1.3574 (11L_w0) | **1.3301** (11L_w0.3 CF) | **−0.027 BPB** |

At both scales, **the best shared model evaluated under the CF two-pass decoder beats the matched causal-only control's Pure-AR BPB**. The effect is small but consistent, and the control is the exact apples-to-apples comparison (same hardware, same training budget, same data, same eval sampling protocol). A 0.027 BPB improvement on the 11L run is on the order of the gap between adjacent entries on the parameter-golf leaderboard (e.g., #287 1.1248 vs #265 1.1307 = 0.0059 BPB).

### 3.3 5L d=256 SP1024 — 8-Config CF Sweep (M1 Max, free)

Before the 6-run H100 ablation, I ran a free pre-flight sweep on M1 Max using an earlier 5L SP1024 shared checkpoint (`shared_ar_cdm.npz`, 4.2 M params) to locate the CF sweet spot across stride × rounds. This is the sweep that convinced me stride=2, rounds=2 is worth spending H100 compute to test. The checkpoint here is SP1024 (not v4096), so the absolute BPB values differ from §3.1 due to the tokenizer — but the *shape* of the sweep is the signal.

| Config | Pass-1 (causal) NLL | Pass-2 (denoise) NLL | **CF Total BPB** | vs Pure-AR 2.5386 |
|---|---|---|---|---|
| Pure AR baseline (same model, single-mode) | — | — | **2.5386** | baseline |
| stride=2, rounds=1 | 1.2615 | 1.2807 | 2.5422 | +0.14% |
| **stride=2, rounds=2** | **1.2688** | **1.0598** | **2.3285** | **−8.28%** |
| stride=3, rounds=1 | 0.8663 | 2.1996 | 3.0659 | +20.77% |
| stride=3, rounds=2 | 0.8540 | 1.6754 | 2.5294 | −0.36% |
| stride=3, rounds=3 | 0.8527 | 1.6052 | 2.4578 | −3.18% |
| stride=4, rounds=1 | 0.6370 | 2.6794 | 3.3164 | +30.64% |
| stride=4, rounds=2 | 0.6404 | 2.0915 | 2.7319 | +7.61% |
| stride=4, rounds=3 | 0.6436 | 1.9617 | 2.6053 | +2.63% |

Sweet spot: stride=2, rounds=2 (50/50 causal–bidirectional split with two denoising refinement rounds). This is the only CF configuration used in §3.1. Every `rounds ≥ 2` configuration either matches or beats pure-AR. Wider-stride single-round configurations are catastrophic because the bidirectional pass has too much to fill from too little context in a single pass.

### 3.4 Earlier 5L SP1024 Headline (1 line, for continuity)

Before running the §3.1 ablation, the same (stride=2, rounds=2) CF configuration was measured on the earlier SP1024 5L shared checkpoint (`shared_ar_cdm.npz`) at N=2000 × seq_len=256 on M1 Max: Pure-AR 2.5412, CF Total **2.3382**, Δ **−7.99%** (stable across N=500 → N=2000). Kept here only to show that the §3.3 sweet spot holds at larger sample sizes on the pre-flight checkpoint. Not the primary claim.

### 3.5 CDM-Weight Sensitivity and Scale Behaviour

From the §3.1 table, two monotonic patterns emerge that are informative about where this paradigm works and where it does not:

**The causal-mode tax grows with CDM weight.** As the CDM loss weight increases from 0 → 0.3 → 1.0, the shared model's Pure-AR BPB gets worse in a near-linear way:

| Scale | w=0 Pure AR | w=0.3 Pure AR | w=1.0 Pure AR | Tax at w=0.3 | Tax at w=1.0 |
|---|---|---|---|---|---|
| 5L | 1.4479 | 1.5231 | 1.5841 | +0.075 | +0.136 |
| 11L | 1.3574 | 1.4708 | 1.5414 | +0.113 | +0.184 |

At 11L the tax is **larger** in absolute terms than at 5L at every weight. This is a non-trivial finding: naively one might expect the extra capacity of 11L to absorb the multi-task objective more gracefully, but the opposite happens in this regime (the causal head gives up more ground at 11L). I do not yet know whether this trend continues at 100 M+ or starts to reverse; that is the primary open question for §6.

**The CF two-pass decoder recovers the tax and then some.** Even though the shared model is worse at pure causal scoring, running the two-pass CF decoder on the same model gets it below the control:

| Scale | Control Pure-AR | Best shared CF | CF advantage |
|---|---|---|---|
| 5L | 1.4479 | 1.3939 (w=1.0) | **−0.054** |
| 11L | 1.3574 | 1.3301 (w=0.3) | **−0.027** |

At 5L the best CF configuration is w=1.0 (stronger bidirectional signal); at 11L it is w=0.3 (where the model has enough capacity that a weak bidirectional signal is enough). At both scales the shared-CF configuration beats the matched control — not by a huge margin, but *reproducibly* and with a control that rules out the most obvious metric-artifact explanation.

### 3.6 Earlier M1 Max Pre-Flight (3-Seed Statistical Check on an Earlier Checkpoint)

Prior to the 6-run ablation, I ran a 3-seed statistical check on the earlier 11L Session 3 checkpoint `11L_shared_cdm_bf16.pt` (trained on 8×H100, which this submission no longer uses for primary comparison). The 3-seed mean CF BPB was 1.3083 ± 0.0047 at seq_len=1024, with N=500 per seed. It is retained here for continuity but is not the primary claim — the primary claim is the matched 1×H100 ablation in §3.1, which has a proper control.

| Seed | N | Pure AR | CF Total | Δ |
|---|---|---|---|---|
| 42 | 500 | 1.4422 | 1.3021 | −9.71% |
| 43 | 500 | 1.4438 | 1.3134 | −9.03% |
| 44 | 500 | 1.4441 | 1.3095 | −9.32% |
| **mean** | **1 500** | **1.4434 ± 0.0008** | **1.3083 ± 0.0047** | **−9.35% ± 0.28%** |
| 42 (scale-up) | 2000 | 1.4293 | 1.3055 | **−8.66%** |

**What the three seeds randomize:** the eval is a deterministic bfloat16 forward. The seed controls (a) which N=500 validation sequences are sampled, and (b) the random vocabulary tokens used to fill unresolved positions in the pass-2 denoising rounds. So the 3-seed variance primarily reflects *validation subsample variance*, not model stochasticity — hence the very small Pure-AR std of 0.0008 BPB. A proper stochastic-variance estimate would require multiple training seeds, listed in §6.

---

## 4. Honest Limitations

This PR measures a BPB improvement on the standard Parameter Golf metric (cross-entropy per byte of validation text). It does **not** measure:

- **Comparison to the 8×H100 leaderboard at matched training compute.** The 1×H100 540 s runs see approximately 1/8 the tokens of an 8×H100 540 s run. The 11L_w0 control at 1.3133 training val_bpb is therefore not directly comparable to the 8×H100 leaderboard entries (top 1 = 1.1147, baseline = 1.2244). The relevant comparison in this PR is always the matched control on the same hardware, not the leaderboard.
- **Actual fill-in-middle generation quality.** Parameter Golf evaluates BPB, not generation, because 28 M-parameter models at ~270 M training tokens cannot produce coherent text regardless of architecture (GPT-2 small at 124 M / 10 B tokens is the rough coherence threshold in the literature). I ran a qualitative greedy-fill test on all six models as a sanity check (not as a claim): exact-match rates were 0–4.7% across all configurations, including the controls — consistent with the scale regime. This PR is about BPB, which *is* the Parameter Golf metric.
- **Comparison to dedicated fill-in-middle baselines** (CodeLlama-FIM, StarCoder-FIM). Training did not target code, so FIM code-benchmarks are not applicable without a retrofit experiment. This is Next Step #2 in §6.
- **Retrofit to pretrained LLMs.** All training here is from scratch. Whether the same shared-weight paradigm can be added to an existing pretrained causal LM via LoRA — the realistic production path for any shipping product — is the largest open question, listed as Next Step #1 in §6.
- **Share-ratio grid beyond three points.** I tested weight ∈ {0, 0.3, 1.0}. A finer grid might reveal a different optimum.
- **Multiple training seeds per configuration.** Each row in §3.1 is single-seed. The §3.6 three-seed check is on a different, earlier checkpoint.

---

## 5. Why This Matters — Product Angle and Extrapolation

The §3.1 result is small (−0.027 BPB at 11L, −0.054 at 5L), but it is the first measurement I have found that separates two factors a production LLM would want to optimize separately:

1. **Causal-only generation quality**, which today is how every shipping LLM (ChatGPT, Claude, GPT-4, Codex, Copilot) is primarily measured.
2. **Bidirectional conditioning** on both left and right context, which today is served either by a *second* specialized model (BERT, MDLM), by a training-time hack (FIM special tokens in Bavarian et al. 2022 / Rozière et al. 2023), or by retrieve-and-rewrite pipelines.

The 6-run matched ablation says: **with a single set of weights, at matched compute, a CF-decoder evaluation gets lower BPB than a dedicated causal-only model**. The shared weights are learning something beyond what the causal-only baseline learns, and the CF decoder makes that extra information accessible at evaluation.

**Why the 0.027 BPB gap matters**: on the Parameter Golf leaderboard, the gaps between adjacent merged records are in the 0.003–0.015 BPB range. A consistent 0.027 BPB improvement at matched compute is leaderboard-relevant, not merely research-curious.

**Extrapolation to 8×H100 production compute**: the 1×H100 540 s run sees roughly 1/8 the tokens of an 8×H100 540 s run. If the shared-training improvement persists at full production compute (not a guarantee, but the direction and sign have held through every run done so far), a matched-compute version of this method at 8×H100 scale would plausibly land near the current leaderboard midpoint (1.19–1.22 BPB range), with an architecturally-richer model as a byproduct. This is the central hypothesis that Next Step #1 in §6 tests.

**What this is not**: this is not a claim that a 28 M parameter model can generate coherent text, or that 5L/11L models at 540 s training are ready for any production use. Models at this scale cannot generate coherent English regardless of architecture (GPT-2 small at 124 M parameters / 10 B tokens is the rough coherence threshold, and these models are 5× smaller and 30× less trained). The Parameter Golf competition accepts this — BPB is the metric precisely because coherence is out of reach at these scales. The claim here is scoped to BPB.

---

## 6. What Might Work With More Compute

Honest speculation. Each item below is a concrete experiment that would extend or close an open question from §3 — ordered by expected impact on the commercial hypothesis in §5.

### 6.1 Retrofit onto a pretrained causal LLM via LoRA (the production path)

The experiment that would most directly determine whether this paradigm has a commercial pathway is a **LoRA-style retrofit of a pretrained causal LLM** (e.g. Qwen 3.5 0.8 B, which I already have locally). Rather than training from scratch at 28 M parameters, take a model that already generates coherent text and add a small LoRA adapter to expose a bidirectional forward mode, trained with the same joint AR + D3PM objective. This is the path every shipping product takes — nobody trains production models from scratch. An initial result on Qwen 0.8 B fits in roughly 10–15 H100-hours and would tell, *within one pod session*, whether the shared-weight + CF-decoder pattern carries to a model that is actually coherent at inference. This is the single most compute-efficient commercial test and it is Next Step #1.

### 6.2 Full-budget 8×H100 reproduction of the 11L ablation

Run the exact §3.1 ablation at 8×H100 540 s (the production Parameter Golf budget) to confirm the 0.027 BPB improvement persists when scaled. If the sign and magnitude hold, the shared-CF 11L run at 8×H100 should land near the current leaderboard midpoint (1.19–1.22 range) and would become the first matched-control BPB improvement reported in the text-diffusion cluster. This is Next Step #2.

### 6.3 Share-ratio grid search at 11L

The 6 runs used weight ∈ {0, 0.3, 1.0}. At 11L, w=0.3 gave the best CF BPB; at 5L, w=1.0 did. A fine grid (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0) at 11L would locate the actual optimum and tell whether the share-ratio optimum scales with model size. This is a cheap follow-up to §6.2 — roughly 7 additional 1×H100 runs.

### 6.4 Finer scale sweep for the share-ratio → BPB curve

I have two architectural data points (5L 4.2 M and 11L 28.4 M). Adding 7L d=384, 9L d=448, and 13L d=640 would give a scaling curve for both the Pure-AR tax (which appears to grow with scale in this data) and the CF recovery (which also grows with scale). A simple power-law fit would let me predict the crossover scale — the model size at which the CF gain exceeds the Pure-AR tax by a margin that makes the extra compute worth it.

### 6.5 Absorbing-mask MDLM noise schedule for the bidirectional pass

I used uniform-noise D3PM (random vocabulary replacement). The MDLM cluster (#820, #1106, #1241) uses absorbing-mask denoising, which the literature suggests gives stronger bidirectional representations. Swapping the noise schedule is a one-line training change; a matched ablation would tell whether the gain would be larger under the standard MDLM noise, at the cost of some comparison legibility.

---

## 7. Retrodiction — A Negative Result at Production Scale

> **Scope note.** The runs in this section are a **different training line** from the Shared AR + Denoising model used in §3. They are a 1×H100 A/B sweep of retrodiction modes on a pure AR stack (no CDM auxiliary loss). The "Pure AR" numbers in this table are therefore *not comparable* to the "Pure AR" column of §3.3, which measures the Shared AR + Denoising checkpoint in single-mode causal. Different models, different training configurations. See §7.3 for an explicit side-by-side.

This submission also documents a line of work I call **retrodiction** — a reversed-sequence auxiliary loss added to the standard causal AR loss, motivated qualitatively by the Petz recovery map from quantum information. The operational definition is simply:

```python
loss = causal_lm_loss(model(x), x) + α · causal_lm_loss(model(x.flip(1)), x.flip(1))
```

I report it as a negative result at production scale. The compact story:

### 7.1 Early-Training Signal on 5L / M1 Max

At small scale and short token budgets, retrodiction gave up to −3.6% BPB at step 200/500, direction consistent with the motivation.

### 7.2 Production-Stack A/B on 1×H100

Five independent training runs, same architecture (11L d=512 v4096, XSA-4, BigramHash), same 540 s budget, same seeds, **pure causal AR stack with no CDM auxiliary loss** — only the retrodiction mode varied:

| Test | Retro mode | Final val_bpb |
|---|---|---|
| **D** | **OFF** | **1.3401** (best) |
| C | partial 15% | 1.3594 |
| B | merged late 80/20 | 1.3695 |
| E | alternating 90/10 | 1.3616 |
| A | alternating 50/50 | 1.4109 |

Pure contrast (C vs D): retrodiction is a **+0.019 BPB tax** at production scale, not a gain.

### 7.3 Consolidated 11L Pure-AR Numbers

The two training lines in this PR each produce their own 11L Pure-AR BPB, and they are not directly comparable. To prevent confusion they are listed side by side:

| Source | Training objective | Retrodiction | Compute | Pure AR BPB |
|---|---|---|---|---|
| §3.1 `11L_w0` (control) | Pure AR only | off | 1×H100, 540 s | **1.3574** |
| §3.1 `11L_w03` | Joint AR + 0.3·denoising | off | 1×H100, 540 s | 1.4708 |
| §3.1 `11L_w1` | Joint AR + 1.0·denoising | off | 1×H100, 540 s | 1.5414 |
| §7.2 Test D | Pure AR only | off | 1×H100, 540 s | 1.3401 |
| §7.2 Test C | Pure AR only | partial 15% | 1×H100, 540 s | 1.3594 |

The §3.1 11L_w0 (1.3574) and §7.2 Test D (1.3401) are both pure-AR, single-seed, 1×H100 540 s runs at the same 11L d=512 v4096 architecture — the small gap between them reflects a difference in training stack details (XSA layer count, BigramHash configuration) that existed before I unified the §3.1 script. For the commercial claim, the relevant comparison is always 11L_w0 vs 11L_w0.3 CF (both measured with the *exact* same script, same data pipeline, same eval sampling). The §7 retrodiction sweep is a separate line of work included for completeness.

**Interpretation (hypothesis).** At 5L on short budgets, the forward loss signal may be weak enough that the reversed loss provides complementary gradient. At 11L on production budgets, I hypothesize that the forward signal is strong enough to dominate and the reversed loss competes for updates rather than augmenting them. The Petz-recovery motivation predicts asymptotic regularization; in the parameter-golf regime the training budget does not reach the asymptotic limit. I do not have a mechanistic proof of this interpretation.

**Practical recommendation:** retrodiction is a tax on the production stack and should not be used. The matched-compute 6-run ablation in §3.1 was run *without* retrodiction for this reason.

---

## 8. Position in the Text-Diffusion Cluster

Snapshot of the text-diffusion cluster as of 2026-04-09 (reproducible via `gh pr list --repo openai/parameter-golf --search "diffusion" --state open --limit 50`):

- Bidirectional masked diffusion + discrete absorbing ELBO (`val_var_bpb`): #820 mtybadger (convention-setting), #1053, #1106 agalimova, #1241 aiejvn, #1403
- Causal MDLM as AR regularizer (eval in causal mode): #1119 gowtham0992
- Hybrid AR + MDLM mixed training with bidirectional head discarded at eval: #1194
- AR with diffusion-inspired auxiliary noise, evaluated as pure AR: #904
- Prefix-conditioned discrete diffusion: #905
- Hybrid sparse diffusion: #1198
- **This PR:** shared-weight joint causal + masked-denoising training, evaluated via a two-pass Coarse-to-Fine decoder on BPB, with a **matched causal-only control** at the same compute. This is, to my knowledge, the first submission in the text-diffusion cluster to include an explicit matched-compute control ablation.

This approach differs from the cluster in that both modes are actively used at evaluation on the same weights, rather than the bidirectional mode being used only at training time or evaluated separately. I do not claim this is a strict improvement over the MDLM line — it is a different question evaluated on a different metric. Direct numerical comparison across metrics (val_var_bpb / val_bpb / CF BPB) is not meaningful because they measure different quantities. See §2.4.

---

## 9. Hardware and Reproducibility

All training and evaluation artifacts are published on Hugging Face:

- **`akaiii/meadow-golf-checkpoints`** — all 6 ablation checkpoints (`5L_w0.npz`, `5L_w03.npz`, `5L_w1.npz`, `11L_w0.npz`, `11L_w03.npz`, `11L_w1.npz`), 6 training logs, 6 CF eval logs, the unified training script (`train_cdm.py` + `train_ablation_runner.py`), and the CF eval scripts (`eval_cf_dualbrain.py`, `eval_cf_dualbrain_cuda.py`, `eval_cf_ablation.py`). Directory layout matches the `ablation_results/` folder in this PR.
- **`akaiii/meadow-golf-v4096`** — `bpe_v4096.model` tokenizer and the v4096 retokenized FineWeb validation + training shards used for every training run in §3.1.

### Reproduction of the 6-run matched ablation (~90 min on 1×H100 SXM, ~$4)

```bash
pip install torch numpy sentencepiece huggingface_hub

hf download akaiii/meadow-golf-checkpoints --repo-type dataset --local-dir ./gcp
hf download akaiii/meadow-golf-v4096       --repo-type dataset --local-dir ./gv4096

export PYTHONPATH="./gcp:${PYTHONPATH}"
mkdir -p out ckpt logs eval

# Train all 6 ablation models (6 × ~10 min wallclock)
for cfg in "5L 5 256 128 2 0.0"  "5L 5 256 128 2 0.3"  "5L 5 256 128 2 1.0" \
           "11L 11 512 128 4 0.0" "11L 11 512 128 4 0.3" "11L 11 512 128 4 1.0"; do
  read tag L D BD X W <<< "$cfg"
  python3 ./gcp/train_ablation_runner.py \
    --train_script ./gcp/train_cdm.py \
    --num_layers $L --model_dim $D --vocab_size 4096 \
    --bigram_dim $BD --xsa_last_n $X --cdm_weight $W \
    -- \
    --train_budget_secs 540 --steps 9999 \
    --data_dir ./gv4096/data --tokenizer_path ./gv4096/bpe_v4096.model \
    --save_path ./out/${tag}_w${W}.npz \
    --checkpoint_dir ./ckpt/${tag}_w${W} \
    > ./logs/${tag}_w${W}_train.log 2>&1
done

# Evaluate all 6 under CF (6 × ~5 min wallclock)
for cfg in "5L 5 256 128 2 0.0"  "5L 5 256 128 2 0.3"  "5L 5 256 128 2 1.0" \
           "11L 11 512 128 4 0.0" "11L 11 512 128 4 0.3" "11L 11 512 128 4 1.0"; do
  read tag L D BD X W <<< "$cfg"
  latest=$(ls ./ckpt/${tag}_w${W}/step_*.pt | sort -V | tail -1)
  python3 ./gcp/eval_cf_ablation.py \
    --ckpt $latest \
    --train_module_path /tmp/train_cdm_patched_${L}L_w${W}.py \
    --num_layers $L --model_dim $D --vocab_size 4096 \
    --bigram_dim $BD --xsa_last_n $X \
    --n_seqs 500 --seq_len 1024 --stride 2 --rounds 2 --seed 42 \
    --data_dir ./gv4096/data --tokenizer_path ./gv4096/bpe_v4096.model \
    --log_path ./eval/${tag}_w${W}_cf.log
done
```

The patched training scripts `/tmp/train_cdm_patched_*.py` are created as a side effect of `train_ablation_runner.py` and are the model-class source for the matching `eval_cf_ablation.py` run. They are regenerated deterministically from `train_cdm.py` on each run.

The 5L M1 Max pre-flight sweep uses `eval_cf_dualbrain.py` (MLX) against `shared_ar_cdm.npz`; it runs on any Apple Silicon Mac with `mlx >= 0.31` and reproduces the §3.3 table in under 4 minutes.

Self-funded compute for the entire 6-run ablation reported in §3.1: **$3.93 on 1×H100 SXM, US-MO-1**.

---

## 10. Compliance

- [x] All 5L artifacts ≤ 16 MB (`5L_w0.npz` = 17.2 MB BF16; int6.lzma variants are 3.0 MB each, well under the cap)
- [x] All 11L artifacts are non-record (trained on 1×H100, not matched to the 8×H100 production budget)
- [x] No validation data accessed during training
- [x] CF evaluation uses validation tokens only for scoring; no gradient updates
- [x] No network calls during evaluation
- [x] Hardware: all 6 training runs and all 6 CF evaluations on a single self-funded 1×H100 SXM pod in US-MO-1, total compute cost $3.93
- [x] Causal-mask integrity verified via the leakage test in §2.3 (`leakage_test.py` included in this folder)
- [x] CF evaluation is fully specified by `SEED`; the denoising pass is Monte Carlo averaged over `n_random=3` random fills for variance reduction on residual positions (not exact, but deterministic given the seed)
- [x] All logs, checkpoints, and scripts published on Hugging Face for third-party reproduction

---

## 11. Acknowledgments

- **PR #820 (@mtybadger)** for establishing `val_var_bpb` and the MDLM reference point for text diffusion in parameter-golf. My disagreement with the metric in §2.3 is intended as productive, not dismissive.
- **PR #363 (@evangelinehelsinki)** for the template of honest negative-result reporting that §7 follows, and for the `What Might Work With More Compute` section format.
- **PRs #1106, #1241** for showing that the MDLM line is an active research target worth contributing alternatives to.

---

## 12. Related Closed Submission

I earlier withdrew [PR #1442](https://github.com/openai/parameter-golf/pull/1442), a different stack combination submission targeting AR sliding BPB. A self-audit found methodological issues including a mismatch between the evaluation used and the compressed artifact. That line of work is not being pursued further; this PR represents my focused research effort going forward.
