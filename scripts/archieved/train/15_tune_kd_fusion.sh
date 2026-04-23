#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""
DATASET="data/kd_fusion/kd_fusion_dataset.pt"
BASE_CONFIG="config/kd_fusion_config.json"
OUT_ROOT="results_tune/kd_fusion"

LR_LIST="1e-4 3e-4 1e-3"
BATCH_LIST="64 128"
DROPOUT_LIST="0.1 0.2"
STUDENT_HIDDEN_LIST="128 256"
EPOCHS="40"

usage() {
  cat <<USAGE
Usage: bash scripts_train/15_tune_kd_fusion.sh [options]

Options:
  --env ENV
  --gpus 0,1
  --dataset PATH
  --base-config PATH
  --out-root DIR
  --lr-list "1e-4 3e-4"
  --batch-list "64 128"
  --dropout-list "0.1 0.2"
  --student-hidden-list "128 256"
  --epochs 40
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --lr-list) LR_LIST="$2"; shift 2 ;;
    --batch-list) BATCH_LIST="$2"; shift 2 ;;
    --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
    --student-hidden-list) STUDENT_HIDDEN_LIST="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

DATASET="$(realpath -m "$DATASET")"
BASE_CONFIG="$(realpath -m "$BASE_CONFIG")"
OUT_ROOT="$(realpath -m "$OUT_ROOT")"

[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET"; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "Missing config: $BASE_CONFIG"; exit 1; }

mkdir -p "$OUT_ROOT/configs"

run_idx=0
for lr in $LR_LIST; do
  for bs in $BATCH_LIST; do
    for dp in $DROPOUT_LIST; do
      for sh in $STUDENT_HIDDEN_LIST; do
        run_idx=$((run_idx + 1))
        tag="run${run_idx}_lr${lr}_bs${bs}_dp${dp}_sh${sh}"
        tag="${tag//./p}"; tag="${tag//-/m}"

        cfg="$OUT_ROOT/configs/${tag}.json"
        outdir="$OUT_ROOT/$tag"
        mkdir -p "$outdir"

        python scripts_train/make_config.py \
          --base "$BASE_CONFIG" \
          --out "$cfg" \
          --set "training.lr=$lr" \
          --set "training.batch_size=$bs" \
          --set "training.num_epochs=$EPOCHS" \
          --set "model.dropout=$dp" \
          --set "model.student_hidden=$sh"

        echo "[KD-FUSION] $tag"
        conda run -n "$ENV_NAME" python scripts_train/kd_fusion_train.py \
          --config "$cfg" \
          --dataset "$DATASET" \
          --outdir "$outdir"
      done
    done
  done
done

echo "KD fusion tuning done: $OUT_ROOT"
