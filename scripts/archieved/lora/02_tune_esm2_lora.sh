#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""
export PYTHONPATH="$PWD/utils${PYTHONPATH:+:$PYTHONPATH}"
DATASET_DIR="data/esm2_tune_real"
TEST_DATASET="data/esm2_tune_real/test_dataset_esm2.pt"
BASE_CONFIG="config/esm2_config.json"
OUT_ROOT="results_tune/esm2_real_lora"
PRETRAINED="default_models/NIMROD-ESM"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
LORA_TARGETS="attention,ffn,classifier"
FREEZE_BASE=1

LR_LIST="1e-7 3e-7 1e-6 3e-6"
BATCH_LIST="4 8 16"
DROPOUT_LIST="0.1 0.2"
LAYER_LIST="4 6"
EPOCHS="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --test-dataset) TEST_DATASET="$2"; shift 2 ;;
    --base-config) BASE_CONFIG="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --pretrained) PRETRAINED="$2"; shift 2 ;;
    --lora-r) LORA_R="$2"; shift 2 ;;
    --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
    --lora-dropout) LORA_DROPOUT="$2"; shift 2 ;;
    --lora-targets) LORA_TARGETS="$2"; shift 2 ;;
    --no-freeze-base) FREEZE_BASE=0; shift 1 ;;
    --lr-list) LR_LIST="$2"; shift 2 ;;
    --batch-list) BATCH_LIST="$2"; shift 2 ;;
    --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
    --layer-list) LAYER_LIST="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then export CUDA_VISIBLE_DEVICES="$GPUS"; fi
mkdir -p "$OUT_ROOT/configs"
run_idx=0
for lr in $LR_LIST; do
  for bs in $BATCH_LIST; do
    for drop in $DROPOUT_LIST; do
      for nl in $LAYER_LIST; do
        run_idx=$((run_idx+1))
        tag="run${run_idx}_lr${lr}_bs${bs}_dp${drop}_nl${nl}"
        tag="${tag//./p}"; tag="${tag//-/m}"
        cfg="$OUT_ROOT/configs/${tag}.json"
        outdir="$OUT_ROOT/$tag"
        mkdir -p "$outdir"
        mkdir -p "$outdir/rocfigs" "$outdir/test_rocfigs" "$outdir/valid_test_distributions" "$outdir/models"
        python scripts_train/make_config.py --base "$BASE_CONFIG" --out "$cfg" \
          --set "training.lr=$lr" \
          --set "training.batch_size=$bs" \
          --set "training.num_epochs=$EPOCHS" \
          --set "model.dropout=$drop" \
          --set "model.num_layers=$nl"

        cmd=(conda run -n "$ENV_NAME" python scripts_train/esm2_train_lora.py \
          --config "$cfg" --dataset_dir "$DATASET_DIR" --test_dataset_path "$TEST_DATASET" --output_dir "$outdir" \
          --pretrained "$PRETRAINED" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT" --lora-targets "$LORA_TARGETS")
        if [[ "$FREEZE_BASE" == "1" ]]; then cmd+=(--freeze-base); fi
        "${cmd[@]}"
      done
    done
  done
done

echo "ESM2 LoRA tuning done: $OUT_ROOT"
