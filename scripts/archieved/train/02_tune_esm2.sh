#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/tmp_smoke/mplconfig}"
export PYTHONPATH="$PWD/utils${PYTHONPATH:+:$PYTHONPATH}"
REPO_ROOT="$PWD"
DATASET_DIR="data/esm2_tune"
TEST_DATASET="data/esm2_tune/test_dataset_esm2.pt"
BASE_CONFIG="config/esm2_config.json"
OUT_ROOT="results_tune/esm2"

LR_LIST="1e-7 3e-7 1e-6"
BATCH_LIST="8 16"
DROPOUT_LIST="0.1 0.2"
LAYER_LIST="4 6"
EPOCHS="64"

usage() {
  cat <<USAGE
Usage: bash scripts_train/02_tune_esm2.sh [options]

Options:
  --env ENV
  --dataset-dir DIR
  --test-dataset PATH
  --base-config PATH
  --out-root DIR
  --gpus 0,1
  --lr-list "1e-7 3e-7"
  --batch-list "8 16"
  --dropout-list "0.1 0.2"
  --layer-list "4 6"
  --epochs 1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --test-dataset) TEST_DATASET="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --lr-list) LR_LIST="$2"; shift 2 ;;
    --batch-list) BATCH_LIST="$2"; shift 2 ;;
    --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
    --layer-list) LAYER_LIST="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

DATASET_DIR="$(realpath -m "$DATASET_DIR")"
TEST_DATASET="$(realpath -m "$TEST_DATASET")"
BASE_CONFIG="$(realpath -m "$BASE_CONFIG")"
OUT_ROOT="$(realpath -m "$OUT_ROOT")"

[[ -f "$BASE_CONFIG" ]] || { echo "Missing config: $BASE_CONFIG"; exit 1; }
[[ -f "$TEST_DATASET" ]] || { echo "Missing test dataset: $TEST_DATASET"; exit 1; }
[[ -f "$DATASET_DIR/train_dataset_esm2.pt" ]] || { echo "Missing $DATASET_DIR/train_dataset_esm2.pt"; exit 1; }
[[ -f "$DATASET_DIR/valid_dataset_esm2.pt" ]] || { echo "Missing $DATASET_DIR/valid_dataset_esm2.pt"; exit 1; }

mkdir -p "$OUT_ROOT/configs"

run_idx=0
for lr in $LR_LIST; do
  for bs in $BATCH_LIST; do
    for drop in $DROPOUT_LIST; do
      for nl in $LAYER_LIST; do
        run_idx=$((run_idx + 1))
        tag="run${run_idx}_lr${lr}_bs${bs}_dp${drop}_nl${nl}"
        tag="${tag//./p}"
        tag="${tag//-/m}"
        cfg="$OUT_ROOT/configs/${tag}.json"
        outdir="$OUT_ROOT/$tag"
        mkdir -p "$outdir/rocfigs" "$outdir/test_rocfigs" "$outdir/valid_test_distributions" "$outdir/models"

        python scripts_train/make_config.py \
          --base "$BASE_CONFIG" \
          --out "$cfg" \
          --set "training.lr=$lr" \
          --set "training.batch_size=$bs" \
          --set "training.num_epochs=$EPOCHS" \
          --set "model.dropout=$drop" \
          --set "model.num_layers=$nl"

        echo "[ESM2] $tag"
        (
          cd "$outdir"
          conda run -n "$ENV_NAME" python "$REPO_ROOT/scripts/esm2_train.py" \
            --config "$cfg" \
            --dataset_dir "$DATASET_DIR" \
            --test_dataset_path "$TEST_DATASET" \
            --output_dir "$outdir"
        )
      done
    done
  done
done

echo "ESM2 tuning done. Output: $OUT_ROOT"
