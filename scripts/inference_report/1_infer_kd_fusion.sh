#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

KD_OUT_ROOT="results_tune/kd_fusion"
FEATURES="tmp/kd_fusion/kd_fusion_dataset.pt"
SPLIT="test"
MODEL=""
OUT=""

usage() {
  cat <<USAGE
Usage: bash scripts_train/18_infer_kd_fusion.sh [options]

Options:
  --env ENV
  --gpus 0,1
  --kd-out-root DIR      Root with KD tuning runs (default: results_tune/kd_fusion)
  --features PATH        KD fused dataset .pt (default: tmp/kd_fusion/kd_fusion_dataset.pt)
  --split train|val|test (default: test)
  --model PATH           Optional: explicit kd_best.pt path
  --out PATH             Optional: output csv path

Behavior:
  1) If --model is set, use it directly.
  2) Else auto-select best run by metrics.json: best_val_auc.
  3) Fallback to latest models/kd_best.pt if metrics.json missing.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --kd-out-root) KD_OUT_ROOT="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

KD_OUT_ROOT="$(realpath -m "$KD_OUT_ROOT")"
FEATURES="$(realpath -m "$FEATURES")"

[[ -f "$FEATURES" ]] || { echo "Missing features file: $FEATURES"; exit 1; }
[[ "$SPLIT" == "train" || "$SPLIT" == "val" || "$SPLIT" == "test" ]] || { echo "Invalid split: $SPLIT"; exit 1; }

if [[ -z "$MODEL" ]]; then
  MODEL="$(python - <<'PY' "$KD_OUT_ROOT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
best_model = None
best_auc = -1.0

for metrics in root.glob('run*/metrics.json'):
    run_dir = metrics.parent
    model = run_dir / 'models' / 'kd_best.pt'
    if not model.exists():
        continue
    try:
        data = json.loads(metrics.read_text(encoding='utf-8'))
        auc = float(data.get('best_val_auc', -1.0))
    except Exception:
        auc = -1.0
    if auc > best_auc:
        best_auc = auc
        best_model = model

if best_model is None:
    cand = sorted(root.glob('run*/models/kd_best.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        best_model = cand[0]

print(str(best_model) if best_model else "")
PY
)"
fi

[[ -n "$MODEL" ]] || { echo "Cannot find KD model under $KD_OUT_ROOT"; exit 1; }
MODEL="$(realpath -m "$MODEL")"
[[ -f "$MODEL" ]] || { echo "Missing model file: $MODEL"; exit 1; }

if [[ -z "$OUT" ]]; then
  if [[ -n "${MODEL##$KD_OUT_ROOT/*}" ]]; then
    # model not inside kd-out-root
    OUT="$KD_OUT_ROOT/infer_${SPLIT}.csv"
  else
    RUN_DIR="$(dirname "$(dirname "$MODEL")")"
    OUT="$RUN_DIR/kd_predictions_${SPLIT}.csv"
  fi
fi
OUT="$(realpath -m "$OUT")"
mkdir -p "$(dirname "$OUT")"

echo "[KD-INFER] model=$MODEL"
echo "[KD-INFER] features=$FEATURES split=$SPLIT"
echo "[KD-INFER] out=$OUT"

conda run -n "$ENV_NAME" python predict/kd_fusion_predict.py \
  --model "$MODEL" \
  --features "$FEATURES" \
  --split "$SPLIT" \
  --out "$OUT"

echo "[KD-INFER] done: $OUT"
