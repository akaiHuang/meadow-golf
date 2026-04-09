# Changelog

Running log of experiments and milestones in this research line. Newest first.
Each entry is one date + what I did + the headline result. Detailed writeups
live under `experiments/<date>_<slug>/README.md`.

---

## 2026-04-09 — 6-run matched-compute ablation

Trained 6 models on a single 1×H100 SXM pod (total cost $3.93): 5L d=256 and
11L d=512, each with CDM loss weight ∈ {0.0, 0.3, 1.0}. The w=0 runs are the
matched causal-only controls the earlier submission was missing.

**Result.** Best shared-weight model (CF 2-pass decoder) beats matched causal-only
control at both scales:

- 11L d=512: CF **1.3301** (w=0.3) vs control **1.3574** → **−0.027 BPB**
- 5L d=256: CF **1.3939** (w=1.0) vs control **1.4479** → **−0.054 BPB**

Causal-mask integrity verified by explicit leakage test (max diff 0.0 at prefix
positions). Sign-consistent gain across two scales with the same unified script.

**Open questions.** Pure-AR tax from joint training grows with scale at w=0.3
(5L +0.075 → 11L +0.113) — whether this continues or inverts at larger scale
is the primary unknown.

→ Full writeup: [experiments/2026-04-09_matched_ablation/README.md](experiments/2026-04-09_matched_ablation/README.md)

---

## 2026-04-08 — 5L pre-flight CF sweep (MLX, Mac)

Before committing to a GPU pod session, validated the shared-weight CF concept
at tiny scale on MLX (Apple Silicon). Ran a sweep of CDM weights on a 5L d=256
model and observed that the shared model evaluated under CF consistently beat
its own Pure-AR eval at non-zero CDM weights. This motivated the matched-control
ablation on 2026-04-09.

No dedicated experiment folder — the pre-flight is referenced in the 2026-04-09
writeup §3 as the earlier MLX numbers that the H100 run reproduces at scale.
