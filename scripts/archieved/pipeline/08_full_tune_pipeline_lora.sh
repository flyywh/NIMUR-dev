#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

TRAIN_POS=""; TRAIN_NEG=""; VALID_POS=""; VALID_NEG=""; TEST_POS=""; TEST_NEG=""
EVO_TRAIN_POS=""; EVO_TRAIN_NEG=""; EVO_TEST_POS=""; EVO_TEST_NEG=""
KG_TSV=""

ESM_OUT_DIR="data/esm2_tune_real"
EVO_OUT_DIR="data/evo2_tune_real"
KG_OUT_DIR="data/kg_tune_real"

EVO_TUNE_BATCH_SIZE="1"
EVO_TUNE_MAX_SEQ_LEN="2048"
EVO_TUNE_EPOCHS="2"
EVO_TUNE_USE_DATAPARALLEL="0"

RUN_EE=0
EE_DATA_OUT_DIR="tmp/ee_tune_real"
EE_TUNE_OUT_DIR="results_tune/ee_real"
EE_TUNE_LR_LIST="1e-7 3e-7 1e-6"
EE_TUNE_BATCH_LIST="8 16"
EE_TUNE_DROPOUT_LIST="0.1 0.2"
EE_TUNE_LAYER_LIST="4 6"
EE_TUNE_EPOCHS="64"
KG_EE_REF_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --train-pos) TRAIN_POS="$2"; shift 2 ;;
    --train-neg) TRAIN_NEG="$2"; shift 2 ;;
    --valid-pos) VALID_POS="$2"; shift 2 ;;
    --valid-neg) VALID_NEG="$2"; shift 2 ;;
    --test-pos) TEST_POS="$2"; shift 2 ;;
    --test-neg) TEST_NEG="$2"; shift 2 ;;
    --evo-train-pos) EVO_TRAIN_POS="$2"; shift 2 ;;
    --evo-train-neg) EVO_TRAIN_NEG="$2"; shift 2 ;;
    --evo-test-pos) EVO_TEST_POS="$2"; shift 2 ;;
    --evo-test-neg) EVO_TEST_NEG="$2"; shift 2 ;;
    --kg-tsv) KG_TSV="$2"; shift 2 ;;
    --esm-out-dir) ESM_OUT_DIR="$2"; shift 2 ;;
    --evo-out-dir) EVO_OUT_DIR="$2"; shift 2 ;;
    --kg-out-dir) KG_OUT_DIR="$2"; shift 2 ;;
    --evo-tune-batch-size) EVO_TUNE_BATCH_SIZE="$2"; shift 2 ;;
    --evo-tune-max-seq-len) EVO_TUNE_MAX_SEQ_LEN="$2"; shift 2 ;;
    --evo-tune-epochs) EVO_TUNE_EPOCHS="$2"; shift 2 ;;
    --evo-tune-use-dataparallel) EVO_TUNE_USE_DATAPARALLEL="$2"; shift 2 ;;
    --run-ee) RUN_EE="$2"; shift 2 ;;
    --ee-tune-lr-list) EE_TUNE_LR_LIST="$2"; shift 2 ;;
    --ee-tune-batch-list) EE_TUNE_BATCH_LIST="$2"; shift 2 ;;
    --ee-tune-dropout-list) EE_TUNE_DROPOUT_LIST="$2"; shift 2 ;;
    --ee-tune-layer-list) EE_TUNE_LAYER_LIST="$2"; shift 2 ;;
    --ee-tune-epochs) EE_TUNE_EPOCHS="$2"; shift 2 ;;
    --kg-ee-ref-model) KG_EE_REF_MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

for p in "$TRAIN_POS" "$TRAIN_NEG" "$VALID_POS" "$VALID_NEG" "$TEST_POS" "$TEST_NEG" \
         "$EVO_TRAIN_POS" "$EVO_TRAIN_NEG" "$EVO_TEST_POS" "$EVO_TEST_NEG" "$KG_TSV"; do
  [[ -f "$p" ]] || { echo "Missing input file: $p"; exit 1; }
done

bash scripts_train/00_prepare_assets.sh --env "$ENV_NAME"

# Real embedding for ESM2
#bash scripts_train/01_prepare_esm2_data.sh --env "$ENV_NAME" --gpus "$GPUS" \
#  --train-pos "$TRAIN_POS" --train-neg "$TRAIN_NEG" \
#  --valid-pos "$VALID_POS" --valid-neg "$VALID_NEG" \
#  --test-pos "$TEST_POS" --test-neg "$TEST_NEG" \
#  --out-dir "$ESM_OUT_DIR"

# Real embedding for Evo2
bash scripts_train/03_prepare_evo2_data.sh --env "$ENV_NAME" --gpus "$GPUS" \
  --train-pos-fasta "$EVO_TRAIN_POS" --train-neg-fasta "$EVO_TRAIN_NEG" \
  --test-pos-fasta "$EVO_TEST_POS" --test-neg-fasta "$EVO_TEST_NEG" \
  --work-dir "$EVO_OUT_DIR"

# KG split
bash scripts_train/05_prepare_kg_data.sh --input-tsv "$KG_TSV" --out-dir "$KG_OUT_DIR" --train-ratio 0.8 --seed 42

# LoRA + pretrained + larger grids + multi-epochs
bash scripts_train/02_tune_esm2_lora.sh --env "$ENV_NAME" --gpus "$GPUS" \
  --dataset-dir "$ESM_OUT_DIR" --test-dataset "$ESM_OUT_DIR/test_dataset_esm2.pt"

bash scripts_train/04_tune_evo2_lora_once.sh --env "$ENV_NAME" --gpus "$GPUS" \
  --train-dataset "$EVO_OUT_DIR/train_build/combined_dataset.pt" \
  --test-dataset "$EVO_OUT_DIR/test_build/combined_dataset.pt" \
  --batch-size "$EVO_TUNE_BATCH_SIZE" \
  --max-seq-len "$EVO_TUNE_MAX_SEQ_LEN" \
  --epochs "$EVO_TUNE_EPOCHS" \
  --use-dataparallel "$EVO_TUNE_USE_DATAPARALLEL"

if [[ "$RUN_EE" == "1" ]]; then
  bash scripts_train/13_prepare_ee_data.sh --env "$ENV_NAME" --gpus "$GPUS" \
    --evo-emb-root "$EVO_OUT_DIR/emb" \
    --esm-train-dataset "$ESM_OUT_DIR/train_dataset_esm2.pt" \
    --esm-valid-dataset "$ESM_OUT_DIR/valid_dataset_esm2.pt" \
    --esm-test-dataset "$ESM_OUT_DIR/test_dataset_esm2.pt" \
    --out-dir "$EE_DATA_OUT_DIR"

  bash scripts_train/12_tune_ee.sh --env "$ENV_NAME" --gpus "$GPUS" \
    --dataset "$EE_DATA_OUT_DIR/combined_protein_dna_dataset.pt" \
    --out-root "$EE_TUNE_OUT_DIR" \
    --lr-list "$EE_TUNE_LR_LIST" \
    --batch-list "$EE_TUNE_BATCH_LIST" \
    --dropout-list "$EE_TUNE_DROPOUT_LIST" \
    --layer-list "$EE_TUNE_LAYER_LIST" \
    --epochs "$EE_TUNE_EPOCHS"

  if [[ -z "$KG_EE_REF_MODEL" ]]; then
    KG_EE_REF_MODEL="$(find "$EE_TUNE_OUT_DIR" -type f -name 'model_epoch_*.pth' -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -n 1 | cut -f2-)"
  fi
fi

kg_cmd=(bash scripts_train/06_tune_oracle_kg_lora.sh --env "$ENV_NAME" --gpus "$GPUS" \
  --train-dir "$KG_OUT_DIR/train" --val-dir "$KG_OUT_DIR/val")
if [[ -n "$KG_EE_REF_MODEL" ]]; then
  [[ -f "$KG_EE_REF_MODEL" ]] || { echo "Missing kg ee ref model: $KG_EE_REF_MODEL"; exit 1; }
  kg_cmd+=(--ee-ref-model "$KG_EE_REF_MODEL")
fi
"${kg_cmd[@]}"

echo "Full LoRA + pretrained tuning pipeline completed."
