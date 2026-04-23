#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS="0"
TRAIN_DATASET="data/evo2_tune_real/train_build/combined_dataset.pt"
TEST_DATASET="data/evo2_tune_real/test_build/combined_dataset.pt"
BASE_CONFIG="config/evo2_config.json"
OUT_ROOT="results_tune/evo2_real_lora_once"
PRETRAINED="default_models/NIMROD-EVO"

# Conservative defaults for lower memory footprint
LR="1e-6"
BATCH_SIZE="1"
DROPOUT="0.1"
NUM_LAYERS="4"
EPOCHS="2"
INPUT_DIM="4096"
MAX_SEQ_LEN="2048"

USE_DATAPARALLEL="0"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
LORA_TARGETS="attention,ffn,classifier"
FREEZE_BASE=1

usage() {
  cat <<USAGE
Usage: bash scripts_train/04_tune_evo2_lora_once.sh [options]

Options:
  --env ENV
  --gpus 0
  --train-dataset PATH
  --test-dataset PATH
  --base-config PATH
  --out-root DIR
  --pretrained PATH
  --lr 1e-6
  --batch-size 1
  --dropout 0.1
  --num-layers 4
  --epochs 2
  --input-dim 4096
  --max-seq-len 2048
  --use-dataparallel 0|1
  --lora-r 8
  --lora-alpha 16
  --lora-dropout 0.05
  --lora-targets "attention,ffn,classifier"
  --no-freeze-base
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --train-dataset) TRAIN_DATASET="$2"; shift 2 ;;
    --test-dataset) TEST_DATASET="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --pretrained) PRETRAINED="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --dropout) DROPOUT="$2"; shift 2 ;;
    --num-layers) NUM_LAYERS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --input-dim) INPUT_DIM="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --use-dataparallel) USE_DATAPARALLEL="$2"; shift 2 ;;
    --lora-r) LORA_R="$2"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
    --lora-dropout) LORA_DROPOUT="$2"; shift 2 ;;
    --lora-targets) LORA_TARGETS="$2"; shift 2 ;;
    --no-freeze-base) FREEZE_BASE=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

mkdir -p "$OUT_ROOT/configs"

TAG="once_lr${LR}_bs${BATCH_SIZE}_dp${DROPOUT}_nl${NUM_LAYERS}_ep${EPOCHS}_msl${MAX_SEQ_LEN}"
TAG="${TAG//./p}"
TAG="${TAG//-/m}"
CFG="$OUT_ROOT/configs/${TAG}.json"
OUTDIR="$OUT_ROOT/$TAG"
mkdir -p "$OUTDIR"

python scripts_train/make_config.py --base "$BASE_CONFIG" --out "$CFG" \
  --set "training.lr=$LR" \
  --set "training.batch_size=$BATCH_SIZE" \
  --set "training.num_epochs=$EPOCHS" \
  --set "training.max_seq_len=$MAX_SEQ_LEN" \
  --set "model.input_dim=$INPUT_DIM" \
  --set "model.dropout=$DROPOUT" \
  --set "model.num_layers=$NUM_LAYERS"

cmd=(conda run -n "$ENV_NAME" env EVO2_USE_DP="$USE_DATAPARALLEL" python scripts_train/evo2_train_lora.py \
  --config "$CFG" \
  --train_dataset "$TRAIN_DATASET" \
  --test_dataset "$TEST_DATASET" \
  --output_dir "$OUTDIR" \
  --pretrained "$PRETRAINED" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --lora-targets "$LORA_TARGETS")

if [[ "$FREEZE_BASE" == "1" ]]; then
  cmd+=(--freeze-base)
fi

"${cmd[@]}"

echo "Evo2 LoRA one-shot tuning done. Output: $OUTDIR"
