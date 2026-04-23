#!/usr/bin/env bash
set -euo pipefail

TARGET=""
ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

usage() {
  cat <<USAGE
Usage:
  bash scripts/lora/tune_lora.sh --target <name> [options]

Targets:
  esm2
  evo2
  evo2-once
  oracle
  oracle-cnn
  kd-fusion
  oracle-with-teachers
USAGE
}

parse_common() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target) TARGET="$2"; shift 2 ;;
      --env) ENV_NAME="$2"; shift 2 ;;
      --gpus) GPUS="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) break ;;
    esac
  done
  echo "$@"
}

run_esm2() {
  local DATASET_DIR="data/esm2_tune_real"
  local TEST_DATASET="data/esm2_tune_real/test_dataset_esm2.pt"
  local BASE_CONFIG="config/esm2_config.json"
  local OUT_ROOT="results_tune/esm2_real_lora"
  local PRETRAINED="default_models/NIMROD-ESM"
  local LORA_R="8"
  local LORA_ALPHA="16"
  local LORA_DROPOUT="0.05"
  local LORA_TARGETS="attention,ffn,classifier"
  local FREEZE_BASE=1
  local LR_LIST="1e-7 3e-7 1e-6 3e-6"
  local BATCH_LIST="4 8 16"
  local DROPOUT_LIST="0.1 0.2"
  local LAYER_LIST="4 6"
  local EPOCHS="8"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
      --test-dataset) TEST_DATASET="$2"; shift 2 ;;
      --base-config) BASE_CONFIG="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
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
      *) echo "Unknown esm2 arg: $1"; exit 1 ;;
    esac
  done

  mkdir -p "$OUT_ROOT/configs"
  local run_idx=0
  for lr in $LR_LIST; do
    for bs in $BATCH_LIST; do
      for drop in $DROPOUT_LIST; do
        for nl in $LAYER_LIST; do
          run_idx=$((run_idx+1))
          tag="run${run_idx}_lr${lr}_bs${bs}_dp${drop}_nl${nl}"
          tag="${tag//./p}"; tag="${tag//-/m}"
          cfg="$OUT_ROOT/configs/${tag}.json"
          outdir="$OUT_ROOT/$tag"
          mkdir -p "$outdir/rocfigs" "$outdir/test_rocfigs" "$outdir/valid_test_distributions" "$outdir/models"

          python utils/make_config.py --base "$BASE_CONFIG" --out "$cfg" \
            --set "training.lr=$lr" \
            --set "training.batch_size=$bs" \
            --set "training.num_epochs=$EPOCHS" \
            --set "model.dropout=$drop" \
            --set "model.num_layers=$nl"

          cmd=(conda run -n "$ENV_NAME" python scripts/lora/lora.py \
            --model esm2 \
            --config "$cfg" \
            --dataset_dir "$DATASET_DIR" \
            --test_dataset_path "$TEST_DATASET" \
            --output_dir "$outdir" \
            --pretrained "$PRETRAINED" \
            --lora-r "$LORA_R" \
            --lora-alpha "$LORA_ALPHA" \
            --lora-dropout "$LORA_DROPOUT" \
            --lora-targets "$LORA_TARGETS")
          if [[ "$FREEZE_BASE" == "1" ]]; then cmd+=(--freeze-base); fi
          "${cmd[@]}"
        done
      done
    done
  done
}

run_evo2() {
  local TRAIN_DATASET="data/evo2_tune_real/train.pt"
  local TEST_DATASET="data/evo2_tune_real/test.pt"
  local BASE_CONFIG="config/evo2_config.json"
  local OUT_ROOT="results_tune/evo2_real_lora"
  local PRETRAINED="default_models/NIMROD-EVO"
  local INPUT_DIM="4096"
  local MAX_SEQ_LEN="4096"
  local USE_DATAPARALLEL="0"
  local LORA_R="8"
  local LORA_ALPHA="16"
  local LORA_DROPOUT="0.05"
  local LORA_TARGETS="attention,ffn,classifier"
  local FREEZE_BASE=1
  local LR_LIST="1e-6 3e-6 1e-5 3e-5"
  local BATCH_LIST="4 8 16"
  local DROPOUT_LIST="0.1 0.2"
  local LAYER_LIST="4 6"
  local EPOCHS="8"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --train-dataset) TRAIN_DATASET="$2"; shift 2 ;;
      --test-dataset) TEST_DATASET="$2"; shift 2 ;;
      --base-config) BASE_CONFIG="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
      --pretrained) PRETRAINED="$2"; shift 2 ;;
      --input-dim) INPUT_DIM="$2"; shift 2 ;;
      --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
      --use-dataparallel) USE_DATAPARALLEL="$2"; shift 2 ;;
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
      *) echo "Unknown evo2 arg: $1"; exit 1 ;;
    esac
  done

  mkdir -p "$OUT_ROOT/configs"
  local run_idx=0
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

          python utils/make_config.py --base "$BASE_CONFIG" --out "$cfg" \
            --set "training.lr=$lr" \
            --set "training.batch_size=$bs" \
            --set "training.num_epochs=$EPOCHS" \
            --set "training.max_seq_len=$MAX_SEQ_LEN" \
            --set "model.input_dim=$INPUT_DIM" \
            --set "model.dropout=$drop" \
            --set "model.num_layers=$nl"

          cmd=(conda run -n "$ENV_NAME" env EVO2_USE_DP="$USE_DATAPARALLEL" python scripts/lora/lora.py \
            --model evo2 \
            --config "$cfg" \
            --train_dataset "$TRAIN_DATASET" \
            --test_dataset "$TEST_DATASET" \
            --output_dir "$outdir" \
            --pretrained "$PRETRAINED" \
            --lora-r "$LORA_R" \
            --lora-alpha "$LORA_ALPHA" \
            --lora-dropout "$LORA_DROPOUT" \
            --lora-targets "$LORA_TARGETS")
          if [[ "$FREEZE_BASE" == "1" ]]; then cmd+=(--freeze-base); fi
          "${cmd[@]}"
        done
      done
    done
  done
}

run_evo2_once() {
  local TRAIN_DATASET="data/evo2_tune_real/train_build/combined_dataset.pt"
  local TEST_DATASET="data/evo2_tune_real/test_build/combined_dataset.pt"
  local BASE_CONFIG="config/evo2_config.json"
  local OUT_ROOT="results_tune/evo2_real_lora_once"
  local PRETRAINED="default_models/NIMROD-EVO"
  local LR="1e-6"
  local BATCH_SIZE="1"
  local DROPOUT="0.1"
  local NUM_LAYERS="4"
  local EPOCHS="2"
  local INPUT_DIM="4096"
  local MAX_SEQ_LEN="2048"
  local USE_DATAPARALLEL="0"
  local LORA_R="8"
  local LORA_ALPHA="16"
  local LORA_DROPOUT="0.05"
  local LORA_TARGETS="attention,ffn,classifier"
  local FREEZE_BASE=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
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
      *) echo "Unknown evo2-once arg: $1"; exit 1 ;;
    esac
  done

  mkdir -p "$OUT_ROOT/configs"
  TAG="once_lr${LR}_bs${BATCH_SIZE}_dp${DROPOUT}_nl${NUM_LAYERS}_ep${EPOCHS}_msl${MAX_SEQ_LEN}"
  TAG="${TAG//./p}"; TAG="${TAG//-/m}"
  CFG="$OUT_ROOT/configs/${TAG}.json"
  OUTDIR="$OUT_ROOT/$TAG"
  mkdir -p "$OUTDIR"

  python utils/make_config.py --base "$BASE_CONFIG" --out "$CFG" \
    --set "training.lr=$LR" \
    --set "training.batch_size=$BATCH_SIZE" \
    --set "training.num_epochs=$EPOCHS" \
    --set "training.max_seq_len=$MAX_SEQ_LEN" \
    --set "model.input_dim=$INPUT_DIM" \
    --set "model.dropout=$DROPOUT" \
    --set "model.num_layers=$NUM_LAYERS"

  cmd=(conda run -n "$ENV_NAME" env EVO2_USE_DP="$USE_DATAPARALLEL" python scripts/lora/lora.py \
    --model evo2 \
    --config "$CFG" \
    --train_dataset "$TRAIN_DATASET" \
    --test_dataset "$TEST_DATASET" \
    --output_dir "$OUTDIR" \
    --pretrained "$PRETRAINED" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --lora-targets "$LORA_TARGETS")
  if [[ "$FREEZE_BASE" == "1" ]]; then cmd+=(--freeze-base); fi
  "${cmd[@]}"
}

run_oracle() {
  local TRAIN_DIR="data/kg_tune_real/train"
  local VAL_DIR="data/kg_tune_real/val"
  local PROTBERT_PATH="local_assets/data/ProtBERT/ProtBERT"
  local OUT_ROOT="results_tune/oracle_kg_real_lora"
  local PRETRAINED="local_assets/default_models/ORACLE"
  local LORA_R="8"
  local LORA_ALPHA="16"
  local LORA_DROPOUT="0.05"
  local FREEZE_BASE=1
  local EMBED_LIST="128 256"
  local LR_LIST="1e-4 2e-4"
  local BATCH_LIST="4 8"
  local EPOCHS_LIST="5 10"
  local ACCUM_LIST="1 2"
  local KE_LAMBDA_LIST="1.0"
  local MAX_SEQ_LEN="1024"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --train-dir) TRAIN_DIR="$2"; shift 2 ;;
      --val-dir) VAL_DIR="$2"; shift 2 ;;
      --protbert) PROTBERT_PATH="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
      --pretrained) PRETRAINED="$2"; shift 2 ;;
      --lora-r) LORA_R="$2"; shift 2 ;;
      --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
      --lora-dropout) LORA_DROPOUT="$2"; shift 2 ;;
      --no-freeze-base) FREEZE_BASE=0; shift 1 ;;
      --embed-list) EMBED_LIST="$2"; shift 2 ;;
      --lr-list) LR_LIST="$2"; shift 2 ;;
      --batch-list) BATCH_LIST="$2"; shift 2 ;;
      --epochs-list) EPOCHS_LIST="$2"; shift 2 ;;
      --accum-list) ACCUM_LIST="$2"; shift 2 ;;
      --ke-lambda-list) KE_LAMBDA_LIST="$2"; shift 2 ;;
      --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
      *) echo "Unknown oracle arg: $1"; exit 1 ;;
    esac
  done

  mkdir -p "$OUT_ROOT"
  local run_idx=0
  for ed in $EMBED_LIST; do
    for lr in $LR_LIST; do
      for bs in $BATCH_LIST; do
        for ep in $EPOCHS_LIST; do
          for ga in $ACCUM_LIST; do
            for kl in $KE_LAMBDA_LIST; do
              run_idx=$((run_idx+1))
              tag="run${run_idx}_ed${ed}_lr${lr}_bs${bs}_ep${ep}_ga${ga}_kl${kl}"
              tag="${tag//./p}"; tag="${tag//-/m}"
              outdir="$OUT_ROOT/$tag"
              mkdir -p "$outdir"

              cmd=(conda run -n "$ENV_NAME" python scripts/lora/lora.py \
                --model oracle \
                --train_dir "$TRAIN_DIR" \
                --val_dir "$VAL_DIR" \
                --protbert_path "$PROTBERT_PATH" \
                --embedding_dim "$ed" \
                --epochs "$ep" \
                --batch_size "$bs" \
                --eval_batch_size "$bs" \
                --lr "$lr" \
                --weight_decay 0.01 \
                --gradient_accumulation_steps "$ga" \
                --ke_lambda "$kl" \
                --max_seq_len "$MAX_SEQ_LEN" \
                --eval_sample_limit -1 \
                --output_dir "$outdir" \
                --pretrained "$PRETRAINED" \
                --lora-r "$LORA_R" \
                --lora-alpha "$LORA_ALPHA" \
                --lora-dropout "$LORA_DROPOUT")
              if [[ "$FREEZE_BASE" == "1" ]]; then cmd+=(--freeze-base); fi
              "${cmd[@]}"
            done
          done
        done
      done
    done
  done
}

run_oracle_cnn() {
  local TRAIN_DATASET="data/oracle_cnn/oracle_train_dataset.pt"
  local VALID_DATASET="data/oracle_cnn/oracle_valid_dataset.pt"
  local BASE_CONFIG="config/oracle_cnn_config.json"
  local OUT_ROOT="results_tune/oracle_cnn_real_lora"
  local PRETRAINED=""
  local INPUT_DIM="1280"
  local FFN_DIM="2560"
  local SEQ_LEN="9600"
  local LORA_R="8"
  local LORA_ALPHA="16"
  local LORA_DROPOUT="0.05"
  local LORA_CNN_TARGETS="all"
  local FREEZE_BASE=1
  local LR_LIST="1e-4 3e-4"
  local BATCH_LIST="4 8"
  local DROPOUT_LIST="0.1 0.2"
  local LAYER_LIST="2 4"
  local EPOCHS="30"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --train-dataset) TRAIN_DATASET="$2"; shift 2 ;;
      --valid-dataset) VALID_DATASET="$2"; shift 2 ;;
      --base-config) BASE_CONFIG="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
      --pretrained) PRETRAINED="$2"; shift 2 ;;
      --input-dim) INPUT_DIM="$2"; shift 2 ;;
      --ffn-dim) FFN_DIM="$2"; shift 2 ;;
      --seq-len) SEQ_LEN="$2"; shift 2 ;;
      --lora-r) LORA_R="$2"; shift 2 ;;
      --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
      --lora-dropout) LORA_DROPOUT="$2"; shift 2 ;;
      --lora-cnn-targets) LORA_CNN_TARGETS="$2"; shift 2 ;;
      --no-freeze-base) FREEZE_BASE=0; shift 1 ;;
      --lr-list) LR_LIST="$2"; shift 2 ;;
      --batch-list) BATCH_LIST="$2"; shift 2 ;;
      --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
      --layer-list) LAYER_LIST="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      *) echo "Unknown oracle-cnn arg: $1"; exit 1 ;;
    esac
  done

  mkdir -p "$OUT_ROOT/configs"
  local run_idx=0
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

          python utils/make_config.py --base "$BASE_CONFIG" --out "$cfg" \
            --set "training.lr=$lr" \
            --set "training.batch_size=$bs" \
            --set "training.num_epochs=$EPOCHS" \
            --set "model.input_dim=$INPUT_DIM" \
            --set "model.ffn_dim=$FFN_DIM" \
            --set "model.dropout=$drop" \
            --set "model.num_layers=$nl" \
            --set "model.seq_len=$SEQ_LEN"

          cmd=(conda run -n "$ENV_NAME" python scripts/lora/lora.py \
            --model oracle-cnn \
            --config "$cfg" \
            --train_dataset "$TRAIN_DATASET" \
            --valid_dataset "$VALID_DATASET" \
            --output_dir "$outdir" \
            --lora-r "$LORA_R" \
            --lora-alpha "$LORA_ALPHA" \
            --lora-dropout "$LORA_DROPOUT" \
            --lora-cnn-targets "$LORA_CNN_TARGETS")
          if [[ -n "$PRETRAINED" ]]; then cmd+=(--pretrained "$PRETRAINED"); fi
          if [[ "$FREEZE_BASE" == "1" ]]; then cmd+=(--freeze-base); fi
          "${cmd[@]}"
        done
      done
    done
  done
}

run_kd_fusion() {
  local DATASET="data/kd_fusion/kd_fusion_dataset.pt"
  local BASE_CONFIG="config/kd_fusion_config.json"
  local OUT_ROOT="results_tune/kd_fusion_lora"
  local PRETRAINED=""
  local LORA_R="8"
  local LORA_ALPHA="16"
  local LORA_DROPOUT="0.05"
  local LORA_KD_TARGETS="all"
  local FREEZE_BASE=1
  local LR_LIST="1e-4 3e-4 1e-3"
  local BATCH_LIST="64 128"
  local DROPOUT_LIST="0.1 0.2"
  local STUDENT_HIDDEN_LIST="128 256"
  local EPOCHS="40"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) DATASET="$2"; shift 2 ;;
      --base-config) BASE_CONFIG="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
      --pretrained) PRETRAINED="$2"; shift 2 ;;
      --lora-r) LORA_R="$2"; shift 2 ;;
      --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
      --lora-dropout) LORA_DROPOUT="$2"; shift 2 ;;
      --lora-kd-targets) LORA_KD_TARGETS="$2"; shift 2 ;;
      --no-freeze-base) FREEZE_BASE=0; shift 1 ;;
      --lr-list) LR_LIST="$2"; shift 2 ;;
      --batch-list) BATCH_LIST="$2"; shift 2 ;;
      --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
      --student-hidden-list) STUDENT_HIDDEN_LIST="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      *) echo "Unknown kd-fusion arg: $1"; exit 1 ;;
    esac
  done

  mkdir -p "$OUT_ROOT/configs"
  local run_idx=0
  for lr in $LR_LIST; do
    for bs in $BATCH_LIST; do
      for dp in $DROPOUT_LIST; do
        for sh in $STUDENT_HIDDEN_LIST; do
          run_idx=$((run_idx+1))
          tag="run${run_idx}_lr${lr}_bs${bs}_dp${dp}_sh${sh}"
          tag="${tag//./p}"; tag="${tag//-/m}"
          cfg="$OUT_ROOT/configs/${tag}.json"
          outdir="$OUT_ROOT/$tag"
          mkdir -p "$outdir"

          python utils/make_config.py --base "$BASE_CONFIG" --out "$cfg" \
            --set "training.lr=$lr" \
            --set "training.batch_size=$bs" \
            --set "training.num_epochs=$EPOCHS" \
            --set "model.dropout=$dp" \
            --set "model.student_hidden=$sh"

          cmd=(conda run -n "$ENV_NAME" python scripts/lora/lora.py \
            --model kd-fusion \
            --config "$cfg" \
            --dataset "$DATASET" \
            --output_dir "$outdir" \
            --lora-r "$LORA_R" \
            --lora-alpha "$LORA_ALPHA" \
            --lora-dropout "$LORA_DROPOUT" \
            --lora-kd-targets "$LORA_KD_TARGETS")
          if [[ -n "$PRETRAINED" ]]; then cmd+=(--pretrained "$PRETRAINED"); fi
          if [[ "$FREEZE_BASE" == "1" ]]; then cmd+=(--freeze-base); fi
          "${cmd[@]}"
        done
      done
    done
  done
}

resolve_ckpt() {
  local p="$1"
  if [[ -f "$p" ]]; then echo "$p"; return 0; fi
  [[ -d "$p" ]] || return 1
  local cand
  cand="$(find "$p" -type f \( -name 'best.pt' -o -name 'best.pth' -o -name 'model_best.pt' -o -name 'model_best.pth' -o -name 'checkpoint_best.pt' -o -name 'checkpoint_best.pth' -o -name 'model_epoch_*.pth' -o -name 'checkpoint*.pt' \) -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -n 1 | cut -f2-)"
  [[ -n "$cand" ]] || return 1
  echo "$cand"
}

run_oracle_with_teachers() {
  local ESM2_MODEL=""
  local EVO2_MODEL=""
  local OUT_ROOT="results_tune/oracle_kg_real_lora_with_teachers"

  local ARGS=("$@")
  local i=0
  while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
      --esm2-model) ESM2_MODEL="${ARGS[$((i+1))]}"; i=$((i+2)) ;;
      --evo2-model) EVO2_MODEL="${ARGS[$((i+1))]}"; i=$((i+2)) ;;
      --out-root) OUT_ROOT="${ARGS[$((i+1))]}"; i=$((i+2)) ;;
      *) i=$((i+1)) ;;
    esac
  done

  [[ -n "$ESM2_MODEL" ]] || { echo "Missing --esm2-model"; exit 1; }
  [[ -n "$EVO2_MODEL" ]] || { echo "Missing --evo2-model"; exit 1; }
  ESM2_CKPT="$(resolve_ckpt "$ESM2_MODEL")" || { echo "Cannot resolve ESM2 checkpoint"; exit 1; }
  EVO2_CKPT="$(resolve_ckpt "$EVO2_MODEL")" || { echo "Cannot resolve Evo2 checkpoint"; exit 1; }

  mkdir -p "$OUT_ROOT"
  cat > "$OUT_ROOT/teacher_models.meta" <<META
esm2_teacher_ckpt=$ESM2_CKPT
evo2_teacher_ckpt=$EVO2_CKPT
kg_tune_time_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
kg_tune_mode=teachers_frozen_reference_only
META

  run_oracle "$@"
}

REST="$(parse_common "$@")"
set -- $REST

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
[[ -n "$TARGET" ]] || { echo "--target is required"; usage; exit 1; }

case "$TARGET" in
  esm2) run_esm2 "$@" ;;
  evo2) run_evo2 "$@" ;;
  evo2-once) run_evo2_once "$@" ;;
  oracle) run_oracle "$@" ;;
  oracle-cnn) run_oracle_cnn "$@" ;;
  kd-fusion) run_kd_fusion "$@" ;;
  oracle-with-teachers) run_oracle_with_teachers "$@" ;;
  *) echo "Unknown target: $TARGET"; usage; exit 1 ;;
esac
