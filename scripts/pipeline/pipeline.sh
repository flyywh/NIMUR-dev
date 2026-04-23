#!/usr/bin/env bash
set -euo pipefail

TARGET=""

usage() {
  cat <<USAGE
Usage:
  bash scripts/pipeline/pipeline.sh --target <name> [options]

Targets:
  full-tune
  full-tune-lora
  smoke-existing
  kd-track
  kd-full
  kd-train-only
  infer-real-bgc
USAGE
}

parse_common() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target) TARGET="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) break ;;
    esac
  done
  echo "$@"
}

run_full_tune() {
  local ENV_NAME="${ENV_NAME:-evo2}"
  local GPUS=""
  local TRAIN_POS=""
  local TRAIN_NEG=""
  local VALID_POS=""
  local VALID_NEG=""
  local TEST_POS=""
  local TEST_NEG=""
  local EVO_TRAIN_POS=""
  local EVO_TRAIN_NEG=""
  local EVO_TEST_POS=""
  local EVO_TEST_NEG=""
  local KG_TSV=""

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
      *) echo "Unknown full-tune arg: $1"; exit 1 ;;
    esac
  done

  required_args=(
    "TRAIN_POS:$TRAIN_POS" "TRAIN_NEG:$TRAIN_NEG" "VALID_POS:$VALID_POS" "VALID_NEG:$VALID_NEG"
    "TEST_POS:$TEST_POS" "TEST_NEG:$TEST_NEG"
    "EVO_TRAIN_POS:$EVO_TRAIN_POS" "EVO_TRAIN_NEG:$EVO_TRAIN_NEG" "EVO_TEST_POS:$EVO_TEST_POS" "EVO_TEST_NEG:$EVO_TEST_NEG"
    "KG_TSV:$KG_TSV"
  )
  for kv in "${required_args[@]}"; do
    key="${kv%%:*}"
    val="${kv#*:}"
    [[ -n "$val" ]] || { echo "Missing required argument for $key"; exit 1; }
  done
  for p in \
    "$TRAIN_POS" "$TRAIN_NEG" "$VALID_POS" "$VALID_NEG" "$TEST_POS" "$TEST_NEG" \
    "$EVO_TRAIN_POS" "$EVO_TRAIN_NEG" "$EVO_TEST_POS" "$EVO_TEST_NEG" "$KG_TSV"; do
    [[ -f "$p" ]] || { echo "Missing input file: $p"; exit 1; }
  done

  bash scripts_train/00_prepare_assets.sh --env "$ENV_NAME"

  bash scripts_train/01_prepare_esm2_data.sh \
    --env "$ENV_NAME" --gpus "$GPUS" \
    --train-pos "$TRAIN_POS" --train-neg "$TRAIN_NEG" \
    --valid-pos "$VALID_POS" --valid-neg "$VALID_NEG" \
    --test-pos "$TEST_POS" --test-neg "$TEST_NEG" \
    --out-dir data/esm2_tune

  bash scripts/train_unified/tune.sh --target esm2 --env "$ENV_NAME" \
    --gpus "$GPUS" \
    --dataset-dir data/esm2_tune \
    --test-dataset data/esm2_tune/test_dataset_esm2.pt \
    --out-root results_tune/esm2

  bash scripts_train/03_prepare_evo2_data.sh \
    --env "$ENV_NAME" --gpus "$GPUS" \
    --train-pos-fasta "$EVO_TRAIN_POS" --train-neg-fasta "$EVO_TRAIN_NEG" \
    --test-pos-fasta "$EVO_TEST_POS" --test-neg-fasta "$EVO_TEST_NEG" \
    --work-dir data/evo2_tune

  bash scripts/train_unified/tune.sh --target evo2 --env "$ENV_NAME" \
    --gpus "$GPUS" \
    --train-dataset data/evo2_tune/train.pt \
    --test-dataset data/evo2_tune/test.pt \
    --out-root results_tune/evo2

  bash scripts_train/05_prepare_kg_data.sh \
    --input-tsv "$KG_TSV" \
    --out-dir data/kg_tune \
    --train-ratio 0.8 \
    --seed 42

  bash scripts/train_unified/tune.sh --target oracle-kg --env "$ENV_NAME" \
    --gpus "$GPUS" \
    --train-dir data/kg_tune/train \
    --val-dir data/kg_tune/val \
    --protbert local_assets/data/ProtBERT/ProtBERT \
    --out-root results_tune/oracle_kg

  echo "Full tuning pipeline completed."
}

run_full_tune_lora() {
  local ENV_NAME="${ENV_NAME:-evo2}"
  local GPUS=""
  local TRAIN_POS=""
  local TRAIN_NEG=""
  local VALID_POS=""
  local VALID_NEG=""
  local TEST_POS=""
  local TEST_NEG=""
  local EVO_TRAIN_POS=""
  local EVO_TRAIN_NEG=""
  local EVO_TEST_POS=""
  local EVO_TEST_NEG=""
  local KG_TSV=""
  local ESM_OUT_DIR="data/esm2_tune_real"
  local EVO_OUT_DIR="data/evo2_tune_real"
  local KG_OUT_DIR="data/kg_tune_real"
  local EVO_TUNE_BATCH_SIZE="1"
  local EVO_TUNE_MAX_SEQ_LEN="2048"
  local EVO_TUNE_EPOCHS="2"
  local EVO_TUNE_USE_DATAPARALLEL="0"
  local RUN_EE=0
  local EE_DATA_OUT_DIR="tmp/ee_tune_real"
  local EE_TUNE_OUT_DIR="results_tune/ee_real"
  local EE_TUNE_LR_LIST="1e-7 3e-7 1e-6"
  local EE_TUNE_BATCH_LIST="8 16"
  local EE_TUNE_DROPOUT_LIST="0.1 0.2"
  local EE_TUNE_LAYER_LIST="4 6"
  local EE_TUNE_EPOCHS="64"
  local KG_EE_REF_MODEL=""

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
      *) echo "Unknown full-tune-lora arg: $1"; exit 1 ;;
    esac
  done

  for p in "$TRAIN_POS" "$TRAIN_NEG" "$VALID_POS" "$VALID_NEG" "$TEST_POS" "$TEST_NEG" \
           "$EVO_TRAIN_POS" "$EVO_TRAIN_NEG" "$EVO_TEST_POS" "$EVO_TEST_NEG" "$KG_TSV"; do
    [[ -f "$p" ]] || { echo "Missing input file: $p"; exit 1; }
  done

  bash scripts_train/00_prepare_assets.sh --env "$ENV_NAME"
  bash scripts_train/03_prepare_evo2_data.sh --env "$ENV_NAME" --gpus "$GPUS" \
    --train-pos-fasta "$EVO_TRAIN_POS" --train-neg-fasta "$EVO_TRAIN_NEG" \
    --test-pos-fasta "$EVO_TEST_POS" --test-neg-fasta "$EVO_TEST_NEG" \
    --work-dir "$EVO_OUT_DIR"

  bash scripts_train/05_prepare_kg_data.sh --input-tsv "$KG_TSV" --out-dir "$KG_OUT_DIR" --train-ratio 0.8 --seed 42

  bash scripts/lora/tune_lora.sh --target esm2 --env "$ENV_NAME" --gpus "$GPUS" \
    --dataset-dir "$ESM_OUT_DIR" --test-dataset "$ESM_OUT_DIR/test_dataset_esm2.pt"

  bash scripts/lora/tune_lora.sh --target evo2-once --env "$ENV_NAME" --gpus "$GPUS" \
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

    bash scripts/train_unified/tune.sh --target ee --env "$ENV_NAME" --gpus "$GPUS" \
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

  kg_cmd=(bash scripts/lora/tune_lora.sh --target oracle --env "$ENV_NAME" --gpus "$GPUS" \
    --train-dir "$KG_OUT_DIR/train" --val-dir "$KG_OUT_DIR/val")
  if [[ -n "$KG_EE_REF_MODEL" ]]; then
    [[ -f "$KG_EE_REF_MODEL" ]] || { echo "Missing kg ee ref model: $KG_EE_REF_MODEL"; exit 1; }
    kg_cmd+=(--ee-ref-model "$KG_EE_REF_MODEL")
  fi
  "${kg_cmd[@]}"

  echo "Full LoRA + pretrained tuning pipeline completed."
}

run_smoke_existing() {
  local ENV_NAME="${ENV_NAME:-evo2}"
  local GPUS="${GPUS:-0}"
  local OUT_ROOT="results_tune/smoke_existing_small"
  local RUN_ESM2=1
  local RUN_EVO2=1
  local RUN_KG=1
  local ESM_DATASET_DIR="data/esm2_tune_real_lora_verify"
  local ESM_TEST_DATASET="data/esm2_tune_real_lora_verify/test_dataset_esm2.pt"
  local EVO_TRAIN_DATASET="data/evo2_tune_real_lora_verify/train.pt"
  local EVO_TEST_DATASET="data/evo2_tune_real_lora_verify/test.pt"
  local KG_TRAIN_DIR="data/kg_tune_real_lora_verify/train"
  local KG_VAL_DIR="data/kg_tune_real_lora_verify/val"

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
      *) echo "Unknown smoke-existing arg: $1"; exit 1 ;;
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
    bash scripts/lora/tune_lora.sh \
      --target esm2 --env "$ENV_NAME" --gpus "$GPUS" \
      --dataset-dir "$ESM_DATASET_DIR" --test-dataset "$ESM_TEST_DATASET" \
      --out-root "$OUT_ROOT/esm2" \
      --lr-list "1e-6" --batch-list "2" --dropout-list "0.1" --layer-list "4" --epochs 1
  fi

  if [[ "$RUN_EVO2" == "1" ]]; then
    [[ -f "$EVO_TRAIN_DATASET" ]] || { echo "Missing $EVO_TRAIN_DATASET"; exit 1; }
    [[ -f "$EVO_TEST_DATASET" ]] || { echo "Missing $EVO_TEST_DATASET"; exit 1; }
    bash scripts/lora/tune_lora.sh \
      --target evo2-once --env "$ENV_NAME" --gpus "$GPUS" \
      --train-dataset "$EVO_TRAIN_DATASET" --test-dataset "$EVO_TEST_DATASET" \
      --out-root "$OUT_ROOT/evo2" \
      --batch-size 1 --epochs 1 --max-seq-len 1024 --use-dataparallel 0
  fi

  if [[ "$RUN_KG" == "1" ]]; then
    for req in protein2id.txt type2id.txt relation2id.txt sequence.txt protein_relation_type.txt; do
      [[ -f "$KG_TRAIN_DIR/$req" ]] || { echo "Missing $KG_TRAIN_DIR/$req"; exit 1; }
      [[ -f "$KG_VAL_DIR/$req" ]] || { echo "Missing $KG_VAL_DIR/$req"; exit 1; }
    done
    bash scripts/lora/tune_lora.sh \
      --target oracle --env "$ENV_NAME" --gpus "$GPUS" \
      --train-dir "$KG_TRAIN_DIR" --val-dir "$KG_VAL_DIR" \
      --out-root "$OUT_ROOT/kg" \
      --embed-list "128" --lr-list "1e-4" --batch-list "2" --epochs-list "1" --accum-list "1" --ke-lambda-list "1.0"
  fi

  echo "[SMOKE] done: $OUT_ROOT"
}

run_kd_track() {
  local ENV_NAME="${ENV_NAME:-evo2}"
  local GPUS=""
  local EVO_EMB_ROOT="data/evo2_tune_real/emb"
  local ESM_TRAIN="data/esm2_tune_real/train_dataset_esm2.pt"
  local ESM_VALID="data/esm2_tune_real/valid_dataset_esm2.pt"
  local ESM_TEST="data/esm2_tune_real/test_dataset_esm2.pt"
  local PROTBERT="local_assets/data/ProtBERT/ProtBERT"
  local TRAIN_POS_FASTA="data/MiBiG_train_pos.fasta"
  local TRAIN_NEG_FASTA="data/MiBiG_train_neg.fasta"
  local TEST_POS_FASTA="data/MiBiG_test_pos.fasta"
  local TEST_NEG_FASTA="data/MiBiG_test_neg.fasta"
  local KD_DATASET="tmp/kd_fusion/kd_fusion_dataset.pt"
  local KD_OUT_ROOT="results_tune/kd_fusion"

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
      *) echo "Unknown kd-track arg: $1"; exit 1 ;;
    esac
  done

  [[ -n "$GPUS" ]] && export CUDA_VISIBLE_DEVICES="$GPUS"
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

  bash scripts/train_unified/tune.sh \
    --target kd-fusion \
    --env "$ENV_NAME" \
    --gpus "$GPUS" \
    --dataset "$KD_DATASET" \
    --out-root "$KD_OUT_ROOT"
}

run_kd_full_impl() {
  local SKIP_PREPARE="$1"
  shift
  local ENV_NAME="${ENV_NAME:-evo2}"
  local GPUS=""
  local ESM_OUT_DIR="data/esm2_tune_real"
  local EVO_OUT_DIR="data/evo2_tune_real"
  local ESM_TRAIN_POS="data/MiBiG_train_pos_esm2_1022.fasta"
  local ESM_TRAIN_NEG="data/MiBiG_train_neg_esm2_1022.fasta"
  local ESM_VALID_POS="data/MiBiG_valid_pos_esm2_1022.fasta"
  local ESM_VALID_NEG="data/MiBiG_valid_neg_esm2_1022.fasta"
  local ESM_TEST_POS="data/MiBiG_test_pos_esm2_1022.fasta"
  local ESM_TEST_NEG="data/MiBiG_test_neg_esm2_1022.fasta"
  local EVO_TRAIN_POS="data/MiBiG_train_pos.fasta"
  local EVO_TRAIN_NEG="data/MiBiG_train_neg.fasta"
  local EVO_TEST_POS="data/MiBiG_test_pos.fasta"
  local EVO_TEST_NEG="data/MiBiG_test_neg.fasta"
  local KD_DATASET="tmp/kd_fusion/kd_fusion_dataset.pt"
  local KD_OUT_ROOT="results_tune/kd_fusion"
  local REBUILD_IF_MISSING=1
  local ONLY_PREPARE=0

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
      *) echo "Unknown kd-full arg: $1"; exit 1 ;;
    esac
  done

  [[ -n "$GPUS" ]] && export CUDA_VISIBLE_DEVICES="$GPUS"
  local need_esm=0
  for p in "$ESM_OUT_DIR/train_dataset_esm2.pt" "$ESM_OUT_DIR/valid_dataset_esm2.pt" "$ESM_OUT_DIR/test_dataset_esm2.pt"; do
    [[ -f "$p" ]] || need_esm=1
  done
  local need_evo=0
  for d in "$EVO_OUT_DIR/emb/train_pos" "$EVO_OUT_DIR/emb/train_neg" "$EVO_OUT_DIR/emb/test_pos" "$EVO_OUT_DIR/emb/test_neg"; do
    [[ -d "$d" ]] || need_evo=1
  done

  if [[ "$need_esm" == "1" && "$REBUILD_IF_MISSING" == "1" && "$SKIP_PREPARE" != "1" ]]; then
    for p in "$ESM_TRAIN_POS" "$ESM_TRAIN_NEG" "$ESM_VALID_POS" "$ESM_VALID_NEG" "$ESM_TEST_POS" "$ESM_TEST_NEG"; do
      [[ -f "$p" ]] || { echo "Missing ESM FASTA: $p"; exit 1; }
    done
    bash scripts_train/01_prepare_esm2_data.sh --env "$ENV_NAME" --gpus "$GPUS" \
      --train-pos "$ESM_TRAIN_POS" --train-neg "$ESM_TRAIN_NEG" \
      --valid-pos "$ESM_VALID_POS" --valid-neg "$ESM_VALID_NEG" \
      --test-pos "$ESM_TEST_POS" --test-neg "$ESM_TEST_NEG" \
      --out-dir "$ESM_OUT_DIR"
  fi

  if [[ "$need_evo" == "1" && "$REBUILD_IF_MISSING" == "1" && "$SKIP_PREPARE" != "1" ]]; then
    for p in "$EVO_TRAIN_POS" "$EVO_TRAIN_NEG" "$EVO_TEST_POS" "$EVO_TEST_NEG"; do
      [[ -f "$p" ]] || { echo "Missing Evo FASTA: $p"; exit 1; }
    done
    bash scripts_train/03_prepare_evo2_data.sh --env "$ENV_NAME" --gpus "$GPUS" \
      --train-pos-fasta "$EVO_TRAIN_POS" --train-neg-fasta "$EVO_TRAIN_NEG" \
      --test-pos-fasta "$EVO_TEST_POS" --test-neg-fasta "$EVO_TEST_NEG" \
      --work-dir "$EVO_OUT_DIR"
  fi

  if [[ "$SKIP_PREPARE" != "1" ]]; then
    conda run -n "$ENV_NAME" python scripts_train/14_prepare_kd_fusion_data.py \
      --evo-emb-root "$EVO_OUT_DIR/emb" \
      --esm-train "$ESM_OUT_DIR/train_dataset_esm2.pt" \
      --esm-valid "$ESM_OUT_DIR/valid_dataset_esm2.pt" \
      --esm-test "$ESM_OUT_DIR/test_dataset_esm2.pt" \
      --train-pos-fasta "$EVO_TRAIN_POS" \
      --train-neg-fasta "$EVO_TRAIN_NEG" \
      --test-pos-fasta "$EVO_TEST_POS" \
      --test-neg-fasta "$EVO_TEST_NEG" \
      --out "$KD_DATASET"
  fi

  if [[ "$ONLY_PREPARE" == "1" ]]; then
    echo "[KD-PIPE] only-prepare done: $KD_DATASET"
    return 0
  fi

  bash scripts/train_unified/tune.sh \
    --target kd-fusion \
    --env "$ENV_NAME" \
    --gpus "$GPUS" \
    --dataset "$KD_DATASET" \
    --out-root "$KD_OUT_ROOT"
}

run_kd_full() {
  run_kd_full_impl 0 "$@"
}

run_kd_train_only() {
  run_kd_full_impl 1 "$@"
}

run_infer_real_bgc() {
  bash scripts/pipeline/test_real_bgc_all.sh "$@"
}

REST="$(parse_common "$@")"
set -- $REST
[[ -n "$TARGET" ]] || { echo "--target is required"; usage; exit 1; }

case "$TARGET" in
  full-tune) run_full_tune "$@" ;;
  full-tune-lora) run_full_tune_lora "$@" ;;
  smoke-existing) run_smoke_existing "$@" ;;
  kd-track) run_kd_track "$@" ;;
  kd-full) run_kd_full "$@" ;;
  kd-train-only) run_kd_train_only "$@" ;;
  infer-real-bgc) run_infer_real_bgc "$@" ;;
  *) echo "Unknown target: $TARGET"; usage; exit 1 ;;
esac
