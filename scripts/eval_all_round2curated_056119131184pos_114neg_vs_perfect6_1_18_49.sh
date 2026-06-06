#!/usr/bin/env bash
set -euo pipefail

cd /root/meteorite_stage2/meteorite_convnext_repro
source .venv/bin/activate

CONFIG="configs/convnext_small_384_bgstrong_round4_plus_perfect6_round2curated_056119131184pos_114neg_scout.yaml"
CKPT_DIR="outputs/convnext_small_384_bgstrong_round4_plus_perfect6_round2curated_056119131184pos_114neg_scout"
NEW_PERFECT="/root/meteorite_stage2/完美文件6.1.18.49.csv"

OUT_ROOT="outputs/round2curated_056119131184pos_114neg_eval_vs_perfect6_1_18_49"
LOG="logs/round2curated_056119131184pos_114neg_eval_vs_perfect6_1_18_49.log"

mkdir -p logs
mkdir -p "$OUT_ROOT"

echo "=== eval 056/119/131/184 POS + 114 NEG vs perfect6.1.18.49 ===" | tee "$LOG"
echo "config: $CONFIG" | tee -a "$LOG"
echo "checkpoint dir: $CKPT_DIR" | tee -a "$LOG"
echo "new perfect: $NEW_PERFECT" | tee -a "$LOG"

test -f "$CONFIG"
test -f "$NEW_PERFECT"

echo "" | tee -a "$LOG"
echo "=== checkpoint list ===" | tee -a "$LOG"
ls -lh "$CKPT_DIR"/*.pth | tee -a "$LOG"

for FOLD in 0 1 2 3 4; do
  test -f "$CKPT_DIR/fold${FOLD}_best_loss.pth"
  test -f "$CKPT_DIR/fold${FOLD}_best_f1.pth"
done

# 单 checkpoint 推理
for CKPT in "$CKPT_DIR"/fold*_best_loss.pth "$CKPT_DIR"/fold*_best_f1.pth; do
  BASENAME=$(basename "$CKPT" .pth)
  ALIAS_DIR="$OUT_ROOT/${BASENAME}_alias"

  echo "" | tee -a "$LOG"
  echo "==============================" | tee -a "$LOG"
  echo "=== inference single: $BASENAME ===" | tee -a "$LOG"
  echo "checkpoint: $CKPT" | tee -a "$LOG"
  echo "==============================" | tee -a "$LOG"

  rm -rf "$ALIAS_DIR"
  mkdir -p "$ALIAS_DIR"

  ln -sf "$(realpath "$CKPT")" "$ALIAS_DIR/fold0_best.pth"

  python3 inference.py \
    --config "$CONFIG" \
    --checkpoint-dir "$ALIAS_DIR" \
    --output "$ALIAS_DIR/diagnostic_threshold05.csv" \
    --threshold 0.5 \
    --device cuda \
    2>&1 | tee -a "$LOG"
done

# 5fold best_loss ensemble
echo "" | tee -a "$LOG"
echo "=== inference ensemble: 5fold_best_loss ===" | tee -a "$LOG"

BEST_LOSS_ALIAS="$OUT_ROOT/ensemble_5fold_best_loss_alias"
rm -rf "$BEST_LOSS_ALIAS"
mkdir -p "$BEST_LOSS_ALIAS"

for FOLD in 0 1 2 3 4; do
  ln -sf "$(realpath "$CKPT_DIR/fold${FOLD}_best_loss.pth")" \
    "$BEST_LOSS_ALIAS/fold${FOLD}_best.pth"
done

python3 inference.py \
  --config "$CONFIG" \
  --checkpoint-dir "$BEST_LOSS_ALIAS" \
  --output "$BEST_LOSS_ALIAS/diagnostic_threshold05.csv" \
  --threshold 0.5 \
  --device cuda \
  2>&1 | tee -a "$LOG"

# 5fold best_f1 ensemble
echo "" | tee -a "$LOG"
echo "=== inference ensemble: 5fold_best_f1 ===" | tee -a "$LOG"

BEST_F1_ALIAS="$OUT_ROOT/ensemble_5fold_best_f1_alias"
rm -rf "$BEST_F1_ALIAS"
mkdir -p "$BEST_F1_ALIAS"

for FOLD in 0 1 2 3 4; do
  ln -sf "$(realpath "$CKPT_DIR/fold${FOLD}_best_f1.pth")" \
    "$BEST_F1_ALIAS/fold${FOLD}_best.pth"
done

python3 inference.py \
  --config "$CONFIG" \
  --checkpoint-dir "$BEST_F1_ALIAS" \
  --output "$BEST_F1_ALIAS/diagnostic_threshold05.csv" \
  --threshold 0.5 \
  --device cuda \
  2>&1 | tee -a "$LOG"

# 扫 threshold，按新完美文件排名
echo "" | tee -a "$LOG"
echo "=== evaluate all predictions vs new perfect ===" | tee -a "$LOG"

python3 - <<'PY' 2>&1 | tee -a logs/round2curated_056119131184pos_114neg_eval_vs_perfect6_1_18_49.log
from pathlib import Path
import re
import shutil
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

ROOT = Path("/root/meteorite_stage2/meteorite_convnext_repro")
NEW_PERFECT = Path("/root/meteorite_stage2/完美文件6.1.18.49.csv")

OUT_ROOT = ROOT / "outputs/round2curated_056119131184pos_114neg_eval_vs_perfect6_1_18_49"
EVAL_ROOT = OUT_ROOT / "ranked_eval_vs_new_perfect"
SUBMIT_DIR = OUT_ROOT / "SUBMIT_CANDIDATES_SORTED"

EVAL_ROOT.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

for p in SUBMIT_DIR.glob("*.csv"):
    p.unlink()

def canon(x):
    s = str(x).strip()
    m = re.search(r"\d+", Path(s).stem)
    return f"{int(m.group()):06d}.jpg" if m else Path(s).name

ref = pd.read_csv(NEW_PERFECT)
rid, rlab = ref.columns[:2]
ref["canon_id"] = ref[rid].map(canon)
ref["true_label"] = ref[rlab].astype(int)
ref = ref[["canon_id", "true_label"]]

print("new perfect rows:", len(ref))
print("new perfect positives:", int(ref["true_label"].sum()))

pred_files = sorted(OUT_ROOT.glob("*_alias/test_predictions.csv"))
print("prediction files:", len(pred_files))

if not pred_files:
    raise SystemExit(f"No prediction files found under {OUT_ROOT}")

rows_all = []

for pred_path in pred_files:
    run = pred_path.parent.name.replace("_alias", "")

    pred = pd.read_csv(pred_path)
    pred["canon_id"] = pred["id"].map(canon)
    pred["probability"] = pd.to_numeric(pred["probability"], errors="coerce")
    pred = pred[["canon_id", "probability"]]

    m0 = ref.merge(pred, on="canon_id", how="inner")
    if len(m0) != len(ref):
        raise SystemExit(f"{run}: ID mismatch {len(m0)} vs {len(ref)}")

    metrics = []
    for i in range(1, 1000):
        th = i / 1000
        m = m0.copy()
        m["label"] = (m["probability"] >= th).astype(int)

        y_true = m["true_label"].astype(int)
        y_pred = m["label"].astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        metrics.append({
            "run": run,
            "threshold": th,
            "positives": int(y_pred.sum()),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    metrics = pd.DataFrame(metrics).sort_values(
        ["f1", "accuracy", "precision"],
        ascending=False,
    )

    run_dir = EVAL_ROOT / run
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "threshold_metrics_vs_perfect6_1_18_49.csv"
    metrics.to_csv(metrics_path, index=False)

    best = metrics.iloc[0]
    best_th = float(best["threshold"])

    m = m0.copy()
    m["label"] = (m["probability"] >= best_th).astype(int)

    sub = m[["canon_id", "label"]].rename(columns={"canon_id": "id"})
    sub_path = run_dir / f"submission_{run}_best_threshold_vs_perfect6_1_18_49.csv"
    sub.to_csv(sub_path, index=False)

    diff = m[m["true_label"] != m["label"]].copy()
    diff["error_type"] = diff.apply(
        lambda r: "FP_pred1_new0" if int(r["label"]) == 1 else "FN_pred0_new1",
        axis=1,
    )
    diff = diff[["canon_id", "true_label", "label", "probability", "error_type"]].sort_values(["error_type", "canon_id"])
    diff_path = run_dir / f"diff_{run}_vs_perfect6_1_18_49.csv"
    diff.to_csv(diff_path, index=False)

    fp_ids = sorted(diff[diff["error_type"].str.startswith("FP")]["canon_id"].tolist())
    fn_ids = sorted(diff[diff["error_type"].str.startswith("FN")]["canon_id"].tolist())

    row = best.to_dict()
    row["FP_ids"] = " ".join(fp_ids)
    row["FN_ids"] = " ".join(fn_ids)
    row["prediction_csv"] = str(pred_path)
    row["submission"] = str(sub_path)
    row["diff"] = str(diff_path)
    rows_all.append(row)

    print(f"\n=== {run} ===")
    print(best.to_string())
    print("FP:", fp_ids)
    print("FN:", fn_ids)
    print("submission:", sub_path)

leader = pd.DataFrame(rows_all).sort_values(
    ["f1", "accuracy", "precision"],
    ascending=False,
)

leader_path = OUT_ROOT / "LEADERBOARD_ALL_CKPTS_AND_ENSEMBLES_vs_perfect6_1_18_49.csv"
leader.to_csv(leader_path, index=False)

print("\n============================================================")
print(" ALL CHECKPOINTS + ENSEMBLES vs PERFECT6.1.18.49")
print("============================================================")
print(leader[[
    "run", "threshold", "positives", "f1", "precision", "recall",
    "accuracy", "tp", "fp", "fn", "tn", "FP_ids", "FN_ids", "submission"
]].to_string(index=False))

print("\nwrote leaderboard:", leader_path)

for rank, (_, r) in enumerate(leader.iterrows(), start=1):
    src = Path(r["submission"])
    run = str(r["run"])
    f1 = float(r["f1"])
    pos = int(r["positives"])
    fp = int(r["fp"])
    fn = int(r["fn"])

    dst_name = f"{rank:02d}_{run}_F1_{f1:.6f}_pos{pos}_FP{fp}_FN{fn}.csv"
    dst = SUBMIT_DIR / dst_name
    shutil.copy2(src, dst)

summary_submit = []
for p in sorted(SUBMIT_DIR.glob("*.csv")):
    df = pd.read_csv(p)
    summary_submit.append({
        "file": p.name,
        "rows": len(df),
        "positives": int(df["label"].sum()),
        "path": str(p),
    })

submit_summary = pd.DataFrame(summary_submit)
submit_summary_path = OUT_ROOT / "SUBMIT_CANDIDATES_SORTED_SUMMARY.csv"
submit_summary.to_csv(submit_summary_path, index=False)

print("\n============================================================")
print(" SUBMIT CANDIDATES SORTED")
print("============================================================")
print(submit_summary.to_string(index=False))

print("\nsubmit folder:", SUBMIT_DIR)
print("submit summary:", submit_summary_path)
PY

echo ""
echo "=== done ==="
