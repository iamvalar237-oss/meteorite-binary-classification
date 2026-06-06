#!/usr/bin/env bash
set -euo pipefail

cd /root/meteorite_stage2/meteorite_convnext_repro
source .venv/bin/activate

mkdir -p logs

CONFIG="configs/final_config.yaml"
OUT_DIR="outputs/convnext_small_384_bgstrong_round4_plus_perfect6_round2curated_056119131184pos_114neg_scout"
LOG="logs/round2curated_056119131184pos_114neg_train5fold.log"

echo "=== train 5fold: 056/119/131/184 POS + 114 NEG ===" | tee "$LOG"
echo "config: $CONFIG" | tee -a "$LOG"

for FOLD in 0 1 2 3 4; do
  echo "" | tee -a "$LOG"
  echo "==============================" | tee -a "$LOG"
  echo "=== training fold ${FOLD} ===" | tee -a "$LOG"
  echo "==============================" | tee -a "$LOG"

  if [ -f "${OUT_DIR}/fold${FOLD}_best_loss.pth" ] && [ -f "${OUT_DIR}/fold${FOLD}_best_f1.pth" ]; then
    echo "fold ${FOLD} already exists, skip." | tee -a "$LOG"
    ls -lh "${OUT_DIR}/fold${FOLD}_best_loss.pth" "${OUT_DIR}/fold${FOLD}_best_f1.pth" | tee -a "$LOG"
  else
    python3 train_cv.py \
      --config "$CONFIG" \
      --fold "$FOLD" \
      --batch-size 64 \
      2>&1 | tee -a "$LOG"

    echo "=== fold ${FOLD} done ===" | tee -a "$LOG"
  fi
done

echo "=== all folds done ===" | tee -a "$LOG"

echo "=== checkpoint status ===" | tee -a "$LOG"
ls -lh "${OUT_DIR}"/fold*_best_loss.pth | tee -a "$LOG"
ls -lh "${OUT_DIR}"/fold*_best_f1.pth | tee -a "$LOG"
