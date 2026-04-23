#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS="${GPUS:-0}"
OUT_ROOT="results_tune/smoke_existing_small"
RUN_ESM2=1
RUN_EVO2=1
RUN_KG=1

# Defaults point to existing small verify data from previous runs.
ESM_DATASET_DIR="data/esm2_tune_real_lora_verify"
ESM_TEST_DATASET="data/esm2_tune_real_lora_verify/test_dataset_esm2.pt"

EVO_TRAIN_DATASET="data/evo2_tune_real_lora_verify/train.pt"
EVO_TEST_DATASET="data/evo2_tune_real_lora_verify/test.pt"

KG_TRAIN_DIR="data/kg_tune_real_lora_verify/train"
KG_VAL_DIR="data/kg_tune_real_lora_verify/val"

usage() {
  cat <<USAGE
Usage: bash scripts_train/09_smoke_existing_small.sh [options]

This script runs a tiny end-to-end smoke test using ALREADY PREPARED datasets.
It does NOT regenerate embeddings or rewrite data.

Options:
  --env ENV
  --gpus 0
  --out-root DIR

  --esm-dataset-dir DIR
  --esm-test-dataset PATH
  --evo-train-dataset PATH
  --evo-test-dataset PATH
  --kg-train-dir DIR
  --kg-val-dir DIR

  --skip-esm2
  --skip-evo2
  --skip-kg
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;

    --esm-dataset-dir) ESM_DATASET_DIR="$2"; shift 2 ;;
    --esm-test-dataset) ESM_TEST_DATASET="$2"; shift 2 ;;
    --evo-train-dataset) EVO_TRAIN_DATASET="$2"; shift 2 ;;
    --evo-test-dataset) EVO_TEST_DATASET="$2"; shift 2 ;;
    --kg-train-dir) KG_TRAIN_DIR="$2"; shift 2 ;;
    --kg-val-dir) KG_VAL_DIR="$2"; shift 2 ;;

    --skip-esm2) RUN_ESM2=0; shift 1 ;;
    --skip-evo2) RUN_EVO2=0; shift 1 ;;
    --skip-kg) RUN_KG=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_ROOT"

echo "[SMOKE] env=$ENV_NAME gpus=$CUDA_VISIBLE_DEVICES out_root=$OUT_ROOT"

if [[ "$RUN_ESM2" == "1" ]]; then
  [[ -f "$ESM_DATASET_DIR/train_dataset_esm2.pt" ]] || { echo "Missing $ESM_DATASET_DIR/train_dataset_esm2.pt"; exit 1; }
  [[ -f "$ESM_DATASET_DIR/valid_dataset_esm2.pt" ]] || { echo "Missing $ESM_DATASET_DIR/valid_dataset_esm2.pt"; exit 1; }
  [[ -f "$ESM_TEST_DATASET" ]] || { echo "Missing $ESM_TEST_DATASET"; exit 1; }

  echo "[SMOKE][ESM2] start"
  bash scripts_train/02_tune_esm2_lora.sh \
    --env "$ENV_NAME" \
    --gpus "$GPUS" \
    --dataset-dir "$ESM_DATASET_DIR" \
    --test-dataset "$ESM_TEST_DATASET" \
    --out-root "$OUT_ROOT/esm2" \
    --lr-list "1e-6" \
    --batch-list "2" \
    --dropout-list "0.1" \
    --layer-list "4" \
    --epochs 1
fi

if [[ "$RUN_EVO2" == "1" ]]; then
  [[ -f "$EVO_TRAIN_DATASET" ]] || { echo "Missing $EVO_TRAIN_DATASET"; exit 1; }
  [[ -f "$EVO_TEST_DATASET" ]] || { echo "Missing $EVO_TEST_DATASET"; exit 1; }

  echo "[SMOKE][EVO2] start"
  bash scripts_train/04_tune_evo2_lora_once.sh \
    --env "$ENV_NAME" \
    --gpus "$GPUS" \
    --train-dataset "$EVO_TRAIN_DATASET" \
    --test-dataset "$EVO_TEST_DATASET" \
    --out-root "$OUT_ROOT/evo2" \
    --batch-size 1 \
    --epochs 1 \
    --max-seq-len 1024 \
    --use-dataparallel 0
fi

if [[ "$RUN_KG" == "1" ]]; then
  for req in protein2id.txt type2id.txt relation2id.txt sequence.txt protein_relation_type.txt; do
    [[ -f "$KG_TRAIN_DIR/$req" ]] || { echo "Missing $KG_TRAIN_DIR/$req"; exit 1; }
    [[ -f "$KG_VAL_DIR/$req" ]] || { echo "Missing $KG_VAL_DIR/$req"; exit 1; }
  done

  echo "[SMOKE][KG] start"
  bash scripts_train/06_tune_oracle_kg_lora.sh \
    --env "$ENV_NAME" \
    --gpus "$GPUS" \
    --train-dir "$KG_TRAIN_DIR" \
    --val-dir "$KG_VAL_DIR" \
    --out-root "$OUT_ROOT/kg" \
    --embed-list "128" \
    --lr-list "1e-4" \
    --batch-list "2" \
    --epochs-list "1" \
    --accum-list "1" \
    --ke-lambda-list "1.0"
fi

echo "[SMOKE] done: $OUT_ROOT"
