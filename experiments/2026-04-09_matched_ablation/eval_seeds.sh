#!/bin/bash
# CF evaluation for the 4 new seed runs from run_seeds.sh.
# Run this after run_seeds.sh. It uses the patched training scripts that
# train_ablation_runner.py writes to PATCHED_DIR as a side-effect.
#
# Override any path via environment variables:
#   SCRIPT_DIR    — directory containing eval_cf_ablation.py (default: script dir)
#   DATA_DIR      — directory with fineweb_val_000000.bin (default: ./data)
#   TOKENIZER     — path to bpe_v4096.model (default: ./bpe_v4096.model)
#   CKPT_DIR      — ckpt root from run_seeds.sh (default: ./ckpt)
#   PATCHED_DIR   — where train_ablation_runner.py wrote patched scripts
#                   (default: /tmp, matches runner default)
#   EVAL_DIR      — output directory for CF eval logs (default: ./eval)
set -e

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
DATA_DIR="${DATA_DIR:-./data}"
TOKENIZER="${TOKENIZER:-./bpe_v4096.model}"
CKPT_DIR="${CKPT_DIR:-./ckpt}"
PATCHED_DIR="${PATCHED_DIR:-/tmp}"
EVAL_DIR="${EVAL_DIR:-./eval}"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

mkdir -p "$EVAL_DIR"

eval_one () {
  local tag=$1 weight=$2 seed=$3
  echo "================================================================"
  echo "== EVAL CF: $tag  (L=11 d=512 w=$weight seed=$seed)"
  echo "================================================================"

  local ckpt_subdir="$CKPT_DIR/${tag}"
  local latest=$(ls -1 "$ckpt_subdir"/step_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$latest" ]; then
    echo "  NO CHECKPOINT FOUND in $ckpt_subdir"
    return
  fi
  echo "  using checkpoint: $latest"

  local patched="$PATCHED_DIR/train_cdm_patched_11L_w${weight}_s${seed}.py"
  if [ ! -f "$patched" ]; then
    echo "  NO PATCHED SCRIPT at $patched — run run_seeds.sh first. Skipping."
    return
  fi
  echo "  using patched script: $patched"

  python3 "$SCRIPT_DIR/eval_cf_ablation.py" \
    --ckpt "$latest" \
    --train_module_path "$patched" \
    --num_layers 11 --model_dim 512 --vocab_size 4096 \
    --bigram_dim 128 --xsa_last_n 4 \
    --n_seqs 500 --seq_len 1024 --stride 2 --rounds 2 --seed 42 \
    --data_dir "$DATA_DIR" --tokenizer_path "$TOKENIZER" \
    --log_path "$EVAL_DIR/${tag}_cf.log" \
    > "$EVAL_DIR/${tag}_eval.out" 2>&1
  echo "  eval done -> $EVAL_DIR/${tag}_cf.log"
  tail -10 "$EVAL_DIR/${tag}_cf.log"
}

eval_one 11L_w0_s42    0.0 42
eval_one 11L_w0_s2024  0.0 2024
eval_one 11L_w03_s42   0.3 42
eval_one 11L_w03_s2024 0.3 2024

echo "================================================================"
echo "== ALL 4 SEED CF EVALS DONE"
echo "================================================================"
ls -la "$EVAL_DIR/"*_s*_cf.log 2>/dev/null || true
