#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

# Existing training-data roots to reuse first.
ESM_OUT_DIR="data/esm2_tune_real"
EVO_OUT_DIR="data/evo2_tune_real"

# ESM2 data-prep FASTA inputs (usually truncated 1022 version).
ESM_TRAIN_POS="data/MiBiG_train_pos_esm2_1022.fasta"
ESM_TRAIN_NEG="data/MiBiG_train_neg_esm2_1022.fasta"
ESM_VALID_POS="data/MiBiG_valid_pos_esm2_1022.fasta"
ESM_VALID_NEG="data/MiBiG_valid_neg_esm2_1022.fasta"
ESM_TEST_POS="data/MiBiG_test_pos_esm2_1022.fasta"
ESM_TEST_NEG="data/MiBiG_test_neg_esm2_1022.fasta"

# Evo2 data-prep FASTA inputs.
EVO_TRAIN_POS="data/MiBiG_train_pos.fasta"
EVO_TRAIN_NEG="data/MiBiG_train_neg.fasta"
EVO_TEST_POS="data/MiBiG_test_pos.fasta"
EVO_TEST_NEG="data/MiBiG_test_neg.fasta"

# KD outputs.
KD_DATASET="tmp/kd_fusion/kd_fusion_dataset.pt"
KD_OUT_ROOT="results_tune/kd_fusion"

REBUILD_IF_MISSING=1
ONLY_PREPARE=0

usage() {
  cat <<USAGE
Usage: bash scripts_train/17_kd_fusion_full_pipeline.sh [options]

This script prefers reusing existing ESM2/Evo2 training data.
If data is missing and --rebuild-if-missing=1, it will regenerate needed data.

Options:
  --env ENV
  --gpus 0,1

  --esm-out-dir DIR
  --evo-out-dir DIR

  --esm-train-pos PATH
  --esm-train-neg PATH
  --esm-valid-pos PATH
  --esm-valid-neg PATH
  --esm-test-pos PATH
  --esm-test-neg PATH

  --evo-train-pos PATH
  --evo-train-neg PATH
  --evo-test-pos PATH
  --evo-test-neg PATH

  --kd-dataset PATH
  --kd-out-root DIR

  --rebuild-if-missing 0|1   (default: 1)
  --only-prepare             prepare kd dataset only, skip tuning
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;

    --esm-out-dir) ESM_OUT_DIR="$2"; shift 2 ;;
    --evo-out-dir) EVO_OUT_DIR="$2"; shift 2 ;;

    --esm-train-pos) ESM_TRAIN_POS="$2"; shift 2 ;;
    --esm-train-neg) ESM_TRAIN_NEG="$2"; shift 2 ;;
    --esm-valid-pos) ESM_VALID_POS="$2"; shift 2 ;;
    --esm-valid-neg) ESM_VALID_NEG="$2"; shift 2 ;;
    --esm-test-pos) ESM_TEST_POS="$2"; shift 2 ;;
    --esm-test-neg) ESM_TEST_NEG="$2"; shift 2 ;;

    --evo-train-pos) EVO_TRAIN_POS="$2"; shift 2 ;;
    --evo-train-neg) EVO_TRAIN_NEG="$2"; shift 2 ;;
    --evo-test-pos) EVO_TEST_POS="$2"; shift 2 ;;
    --evo-test-neg) EVO_TEST_NEG="$2"; shift 2 ;;

    --kd-dataset) KD_DATASET="$2"; shift 2 ;;
    --kd-out-root) KD_OUT_ROOT="$2"; shift 2 ;;

    --rebuild-if-missing) REBUILD_IF_MISSING="$2"; shift 2 ;;
    --only-prepare) ONLY_PREPARE=1; shift 1 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

need_esm=0
for p in "$ESM_OUT_DIR/train_dataset_esm2.pt" "$ESM_OUT_DIR/valid_dataset_esm2.pt" "$ESM_OUT_DIR/test_dataset_esm2.pt"; do
  [[ -f "$p" ]] || need_esm=1
done

need_evo=0
for d in "$EVO_OUT_DIR/emb/train_pos" "$EVO_OUT_DIR/emb/train_neg" "$EVO_OUT_DIR/emb/test_pos" "$EVO_OUT_DIR/emb/test_neg"; do
  [[ -d "$d" ]] || need_evo=1
done

if [[ "$need_esm" == "1" ]]; then
  if [[ "$REBUILD_IF_MISSING" != "1" ]]; then
    echo "Missing ESM2 datasets under $ESM_OUT_DIR and rebuild disabled."; exit 1
  fi
  echo "[KD-PIPE] ESM2 data missing -> regenerate via 01_prepare_esm2_data.sh"
  for p in "$ESM_TRAIN_POS" "$ESM_TRAIN_NEG" "$ESM_VALID_POS" "$ESM_VALID_NEG" "$ESM_TEST_POS" "$ESM_TEST_NEG"; do
    [[ -f "$p" ]] || { echo "Missing ESM FASTA: $p"; exit 1; }
  done
#  bash scripts_train/01_prepare_esm2_data.sh --env "$ENV_NAME" --gpus "$GPUS" \
#    --train-pos "$ESM_TRAIN_POS" --train-neg "$ESM_TRAIN_NEG" \
#    --valid-pos "$ESM_VALID_POS" --valid-neg "$ESM_VALID_NEG" \
#    --test-pos "$ESM_TEST_POS" --test-neg "$ESM_TEST_NEG" \
#    --out-dir "$ESM_OUT_DIR"
else
  echo "[KD-PIPE] Reuse existing ESM2 datasets from $ESM_OUT_DIR"
fi

if [[ "$need_evo" == "1" ]]; then
  if [[ "$REBUILD_IF_MISSING" != "1" ]]; then
    echo "Missing Evo2 embeddings under $EVO_OUT_DIR/emb and rebuild disabled."; exit 1
  fi
  echo "[KD-PIPE] Evo2 emb missing -> regenerate via 03_prepare_evo2_data.sh"
  for p in "$EVO_TRAIN_POS" "$EVO_TRAIN_NEG" "$EVO_TEST_POS" "$EVO_TEST_NEG"; do
    [[ -f "$p" ]] || { echo "Missing Evo FASTA: $p"; exit 1; }
  done
#  bash scripts_train/03_prepare_evo2_data.sh --env "$ENV_NAME" --gpus "$GPUS" \
#    --train-pos-fasta "$EVO_TRAIN_POS" --train-neg-fasta "$EVO_TRAIN_NEG" \
#    --test-pos-fasta "$EVO_TEST_POS" --test-neg-fasta "$EVO_TEST_NEG" \
#    --work-dir "$EVO_OUT_DIR"
else
  echo "[KD-PIPE] Reuse existing Evo2 embeddings from $EVO_OUT_DIR/emb"
fi

#conda run -n "$ENV_NAME" python scripts_train/14_prepare_kd_fusion_data.py \
#  --evo-emb-root "$EVO_OUT_DIR/emb" \
#  --esm-train "$ESM_OUT_DIR/train_dataset_esm2.pt" \
#  --esm-valid "$ESM_OUT_DIR/valid_dataset_esm2.pt" \
#  --esm-test "$ESM_OUT_DIR/test_dataset_esm2.pt" \
#  --train-pos-fasta "$EVO_TRAIN_POS" \
#  --train-neg-fasta "$EVO_TRAIN_NEG" \
#  --test-pos-fasta "$EVO_TEST_POS" \
#  --test-neg-fasta "$EVO_TEST_NEG" \
#  --out "$KD_DATASET"

if [[ "$ONLY_PREPARE" == "1" ]]; then
  echo "[KD-PIPE] only-prepare done: $KD_DATASET"
  exit 0
fi

bash scripts_train/15_tune_kd_fusion.sh \
  --env "$ENV_NAME" \
  --gpus "$GPUS" \
  --dataset "$KD_DATASET" \
  --out-root "$KD_OUT_ROOT"

echo "[KD-PIPE] done: dataset=$KD_DATASET out=$KD_OUT_ROOT"
