#!/bin/bash
# 3-seed statistical check for the 11L w=0 control and w=0.3 winner.
#
# Adds 2 new training seeds per config on top of the existing seed=1337 runs
# from run_6.sh, giving 3 training seeds per cell. This provides a proper
# training-stochasticity variance estimate for the headline −0.027 BPB delta,
# addressing the single-seed limitation called out in README §4.
#
# Override any path via environment variables:
#   SCRIPT_DIR   — directory containing train_cdm.py + train_ablation_runner.py
#                  (default: the directory this script lives in)
#   DATA_DIR     — directory with fineweb_train_*.bin + fineweb_val_000000.bin
#                  (default: ./data)
#   TOKENIZER    — path to bpe_v4096.model (default: ./bpe_v4096.model)
#   OUT_DIR      — where .npz / .lzma checkpoints land (default: ./out)
#   CKPT_DIR     — intermediate checkpoint directory (default: ./ckpt)
#   LOG_DIR      — training logs (default: ./logs)
set -e

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
DATA_DIR="${DATA_DIR:-./data}"
TOKENIZER="${TOKENIZER:-./bpe_v4096.model}"
OUT_DIR="${OUT_DIR:-./out}"
CKPT_DIR="${CKPT_DIR:-./ckpt}"
LOG_DIR="${LOG_DIR:-./logs}"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR" "$CKPT_DIR" "$LOG_DIR"

run_one () {
  local tag=$1 weight=$2 seed=$3
  echo "================================================================"
  echo "== RUN: $tag  (L=11 d=512 cdm_w=$weight seed=$seed)"
  echo "================================================================"
  python3 "$SCRIPT_DIR/train_ablation_runner.py" \
    --train_script "$SCRIPT_DIR/train_cdm.py" \
    --num_layers 11 --model_dim 512 --vocab_size 4096 \
    --bigram_dim 128 --xsa_last_n 4 \
    --cdm_weight $weight \
    --seed $seed \
    -- \
    --train_budget_secs 540 \
    --steps 9999 \
    --data_dir "$DATA_DIR" --tokenizer_path "$TOKENIZER" \
    --save_path "$OUT_DIR/${tag}.npz" \
    --save_int6_path "$OUT_DIR/${tag}_int6.lzma" \
    --checkpoint_dir "$CKPT_DIR/${tag}" \
    --val_every 500 --val_tokens 1000000 \
    > "$LOG_DIR/${tag}_train.log" 2>&1
  echo "  train done -> $LOG_DIR/${tag}_train.log"
  tail -5 "$LOG_DIR/${tag}_train.log"
}

# 2 new seeds for the 11L w=0 control
run_one 11L_w0_s42    0.0 42
run_one 11L_w0_s2024  0.0 2024

# 2 new seeds for the 11L w=0.3 winner
run_one 11L_w03_s42   0.3 42
run_one 11L_w03_s2024 0.3 2024

echo "================================================================"
echo "== ALL 4 SEED RUNS DONE"
echo "================================================================"
ls -la "$OUT_DIR"/11L_w*_s*.npz 2>/dev/null || true
