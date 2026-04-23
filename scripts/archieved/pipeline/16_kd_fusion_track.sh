#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

EVO_EMB_ROOT="data/evo2_tune_real/emb"
ESM_TRAIN="data/esm2_tune_real/train_dataset_esm2.pt"
ESM_VALID="data/esm2_tune_real/valid_dataset_esm2.pt"
ESM_TEST="data/esm2_tune_real/test_dataset_esm2.pt"
PROTBERT="local_assets/data/ProtBERT/ProtBERT"

TRAIN_POS_FASTA="data/MiBiG_train_pos.fasta"
TRAIN_NEG_FASTA="data/MiBiG_train_neg.fasta"
TEST_POS_FASTA="data/MiBiG_test_pos.fasta"
TEST_NEG_FASTA="data/MiBiG_test_neg.fasta"

KD_DATASET="tmp/kd_fusion/kd_fusion_dataset.pt"
KD_OUT_ROOT="results_tune/kd_fusion"

usage() {
  cat <<USAGE
Usage: bash scripts_train/16_kd_fusion_track.sh [options]

Options:
  --env ENV
  --gpus 0,1
  --evo-emb-root DIR
  --esm-train PATH
  --esm-valid PATH
  --esm-test PATH
  --protbert PATH
  --train-pos-fasta PATH
  --train-neg-fasta PATH
  --test-pos-fasta PATH
  --test-neg-fasta PATH
  --kd-dataset PATH
  --kd-out-root DIR
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --evo-emb-root) EVO_EMB_ROOT="$2"; shift 2 ;;
    --esm-train) ESM_TRAIN="$2"; shift 2 ;;
    --esm-valid) ESM_VALID="$2"; shift 2 ;;
    --esm-test) ESM_TEST="$2"; shift 2 ;;
    --protbert) PROTBERT="$2"; shift 2 ;;
    --train-pos-fasta) TRAIN_POS_FASTA="$2"; shift 2 ;;
    --train-neg-fasta) TRAIN_NEG_FASTA="$2"; shift 2 ;;
    --test-pos-fasta) TEST_POS_FASTA="$2"; shift 2 ;;
    --test-neg-fasta) TEST_NEG_FASTA="$2"; shift 2 ;;
    --kd-dataset) KD_DATASET="$2"; shift 2 ;;
    --kd-out-root) KD_OUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

bash scripts_train/00_prepare_assets.sh --env "$ENV_NAME"

conda run -n "$ENV_NAME" python scripts_train/14_prepare_kd_fusion_data.py \
  --evo-emb-root "$EVO_EMB_ROOT" \
  --esm-train "$ESM_TRAIN" \
  --esm-valid "$ESM_VALID" \
  --esm-test "$ESM_TEST" \
  --protbert "$PROTBERT" \
  --train-pos-fasta "$TRAIN_POS_FASTA" \
  --train-neg-fasta "$TRAIN_NEG_FASTA" \
  --test-pos-fasta "$TEST_POS_FASTA" \
  --test-neg-fasta "$TEST_NEG_FASTA" \
  --out "$KD_DATASET"

bash scripts_train/15_tune_kd_fusion.sh \
  --env "$ENV_NAME" \
  --gpus "$GPUS" \
  --dataset "$KD_DATASET" \
  --out-root "$KD_OUT_ROOT"

echo "KD fusion track done: dataset=$KD_DATASET, out=$KD_OUT_ROOT"
