#!/usr/bin/env bash
set -euo pipefail

TARGET=""
ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/tmp_smoke/mplconfig}"
export PYTHONPATH="$PWD/utils${PYTHONPATH:+:$PYTHONPATH}"
REPO_ROOT="$PWD"

usage() {
  cat <<USAGE
Usage:
  bash scripts/train_unified/tune.sh --target <name> [options]

Targets:
  esm2
  evo2
  oracle-kg
  ee
  kd-fusion
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
  local DATASET_DIR="data/esm2_tune"
  local TEST_DATASET="data/esm2_tune/test_dataset_esm2.pt"
  local BASE_CONFIG="config/esm2_config.json"
  local OUT_ROOT="results_tune/esm2"
  local LR_LIST="1e-7 3e-7 1e-6"
  local BATCH_LIST="8 16"
  local DROPOUT_LIST="0.1 0.2"
  local LAYER_LIST="4 6"
  local EPOCHS="64"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
      --test-dataset) TEST_DATASET="$2"; shift 2 ;;
      --base-config) BASE_CONFIG="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
      --lr-list) LR_LIST="$2"; shift 2 ;;
      --batch-list) BATCH_LIST="$2"; shift 2 ;;
      --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
      --layer-list) LAYER_LIST="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      *) echo "Unknown esm2 arg: $1"; exit 1 ;;
    esac
  done

  [[ -f "$BASE_CONFIG" ]] || { echo "Missing config: $BASE_CONFIG"; exit 1; }
  [[ -f "$TEST_DATASET" ]] || { echo "Missing test dataset: $TEST_DATASET"; exit 1; }
  [[ -f "$DATASET_DIR/train_dataset_esm2.pt" ]] || { echo "Missing $DATASET_DIR/train_dataset_esm2.pt"; exit 1; }
  [[ -f "$DATASET_DIR/valid_dataset_esm2.pt" ]] || { echo "Missing $DATASET_DIR/valid_dataset_esm2.pt"; exit 1; }
  mkdir -p "$OUT_ROOT/configs"

  local run_idx=0
  for lr in $LR_LIST; do
    for bs in $BATCH_LIST; do
      for drop in $DROPOUT_LIST; do
        for nl in $LAYER_LIST; do
          run_idx=$((run_idx + 1))
          tag="run${run_idx}_lr${lr}_bs${bs}_dp${drop}_nl${nl}"
          tag="${tag//./p}"; tag="${tag//-/m}"
          cfg="$OUT_ROOT/configs/${tag}.json"
          outdir="$OUT_ROOT/$tag"
          mkdir -p "$outdir"

          python utils/make_config.py \
            --base "$BASE_CONFIG" \
            --out "$cfg" \
            --set "training.lr=$lr" \
            --set "training.batch_size=$bs" \
            --set "training.num_epochs=$EPOCHS" \
            --set "model.dropout=$drop" \
            --set "model.num_layers=$nl"

          echo "[ESM2] $tag"
          conda run -n "$ENV_NAME" python scripts/train_unified/train.py run esm2 -- \
            --config "$cfg" \
            --dataset_dir "$DATASET_DIR" \
            --test_dataset_path "$TEST_DATASET" \
            --output_dir "$outdir"
        done
      done
    done
  done
}

run_evo2() {
  local TRAIN_DATASET="data/evo2_tune/train.pt"
  local TEST_DATASET="data/evo2_tune/test.pt"
  local BASE_CONFIG="config/evo2_config.json"
  local OUT_ROOT="results_tune/evo2"
  local LR_LIST="1e-6 3e-6 1e-5"
  local BATCH_LIST="8 16"
  local DROPOUT_LIST="0.1 0.2"
  local LAYER_LIST="4 6"
  local EPOCHS="50"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --train-dataset) TRAIN_DATASET="$2"; shift 2 ;;
      --test-dataset) TEST_DATASET="$2"; shift 2 ;;
      --base-config) BASE_CONFIG="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
      --lr-list) LR_LIST="$2"; shift 2 ;;
      --batch-list) BATCH_LIST="$2"; shift 2 ;;
      --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
      --layer-list) LAYER_LIST="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      *) echo "Unknown evo2 arg: $1"; exit 1 ;;
    esac
  done

  [[ -f "$BASE_CONFIG" ]] || { echo "Missing config: $BASE_CONFIG"; exit 1; }
  [[ -f "$TRAIN_DATASET" ]] || { echo "Missing train dataset: $TRAIN_DATASET"; exit 1; }
  [[ -f "$TEST_DATASET" ]] || { echo "Missing test dataset: $TEST_DATASET"; exit 1; }
  mkdir -p "$OUT_ROOT/configs"

  local run_idx=0
  for lr in $LR_LIST; do
    for bs in $BATCH_LIST; do
      for drop in $DROPOUT_LIST; do
        for nl in $LAYER_LIST; do
          run_idx=$((run_idx + 1))
          tag="run${run_idx}_lr${lr}_bs${bs}_dp${drop}_nl${nl}"
          tag="${tag//./p}"; tag="${tag//-/m}"
          cfg="$OUT_ROOT/configs/${tag}.json"
          outdir="$OUT_ROOT/$tag"
          mkdir -p "$outdir"

          python utils/make_config.py \
            --base "$BASE_CONFIG" \
            --out "$cfg" \
            --set "training.lr=$lr" \
            --set "training.batch_size=$bs" \
            --set "training.num_epochs=$EPOCHS" \
            --set "model.dropout=$drop" \
            --set "model.num_layers=$nl"

          echo "[EVO2] $tag"
          conda run -n "$ENV_NAME" python scripts/train_unified/train.py run evo2 -- \
            --config "$cfg" \
            --train_dataset "$TRAIN_DATASET" \
            --test_dataset "$TEST_DATASET" \
            --output_dir "$outdir"
        done
      done
    done
  done
}

run_oracle_kg() {
  local TRAIN_DIR="data/kg_tune/train"
  local VAL_DIR="data/kg_tune/val"
  local PROTBERT_PATH="local_assets/data/ProtBERT/ProtBERT"
  local OUT_ROOT="results_tune/oracle_kg"
  local EMBED_LIST="128 256"
  local LR_LIST="1e-4 2e-4"
  local BATCH_LIST="4 8"
  local EPOCHS_LIST="5 10"
  local ACCUM_LIST="1 2"
  local KE_LAMBDA_LIST="1.0"
  local MAX_SEQ_LEN="1024"
  local AG_EMB_DIR=""
  local NO_AG=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --train-dir) TRAIN_DIR="$2"; shift 2 ;;
      --val-dir) VAL_DIR="$2"; shift 2 ;;
      --protbert) PROTBERT_PATH="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
      --embed-list) EMBED_LIST="$2"; shift 2 ;;
      --lr-list) LR_LIST="$2"; shift 2 ;;
      --batch-list) BATCH_LIST="$2"; shift 2 ;;
      --epochs-list) EPOCHS_LIST="$2"; shift 2 ;;
      --accum-list) ACCUM_LIST="$2"; shift 2 ;;
      --ke-lambda-list) KE_LAMBDA_LIST="$2"; shift 2 ;;
      --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
      --ag-emb-dir) AG_EMB_DIR="$2"; shift 2 ;;
      --no-ag) NO_AG=1; shift 1 ;;
      *) echo "Unknown oracle-kg arg: $1"; exit 1 ;;
    esac
  done

  for req in protein2id.txt type2id.txt relation2id.txt sequence.txt protein_relation_type.txt; do
    [[ -f "$TRAIN_DIR/$req" ]] || { echo "Missing $TRAIN_DIR/$req"; exit 1; }
    [[ -f "$VAL_DIR/$req" ]] || { echo "Missing $VAL_DIR/$req"; exit 1; }
  done
  [[ -d "$PROTBERT_PATH" ]] || { echo "Missing protbert dir: $PROTBERT_PATH"; exit 1; }
  mkdir -p "$OUT_ROOT"

  local AG_ARGS=()
  if [[ "$NO_AG" -eq 0 && -n "$AG_EMB_DIR" ]]; then
    [[ -d "$AG_EMB_DIR" ]] || { echo "Missing ag-emb-dir: $AG_EMB_DIR"; exit 1; }
    AG_ARGS=(--ag-emb-dir "$AG_EMB_DIR")
  fi

  local run_idx=0
  for ed in $EMBED_LIST; do
    for lr in $LR_LIST; do
      for bs in $BATCH_LIST; do
        for ep in $EPOCHS_LIST; do
          for ga in $ACCUM_LIST; do
            for kl in $KE_LAMBDA_LIST; do
              run_idx=$((run_idx + 1))
              tag="run${run_idx}_ed${ed}_lr${lr}_bs${bs}_ep${ep}_ga${ga}_kl${kl}"
              tag="${tag//./p}"; tag="${tag//-/m}"
              outdir="$OUT_ROOT/$tag"
              mkdir -p "$outdir"
              echo "[ORACLE-KG] $tag"

              conda run -n "$ENV_NAME" python scripts/train_unified/train.py run oracle -- \
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
                "${AG_ARGS[@]}"
            done
          done
        done
      done
    done
  done
}

run_ee() {
  local DATASET="data/ee_tune/combined_protein_dna_dataset.pt"
  local BASE_CONFIG="config/ee_config.json"
  local OUT_ROOT="results_tune/ee"
  local LR_LIST="1e-7 3e-7 1e-6"
  local BATCH_LIST="8 16"
  local DROPOUT_LIST="0.1 0.2"
  local LAYER_LIST="4 6"
  local EPOCHS="64"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) DATASET="$2"; shift 2 ;;
      --base-config) BASE_CONFIG="$2"; shift 2 ;;
      --out-root) OUT_ROOT="$2"; shift 2 ;;
      --lr-list) LR_LIST="$2"; shift 2 ;;
      --batch-list) BATCH_LIST="$2"; shift 2 ;;
      --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
      --layer-list) LAYER_LIST="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      *) echo "Unknown ee arg: $1"; exit 1 ;;
    esac
  done

  [[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET"; exit 1; }
  [[ -f "$BASE_CONFIG" ]] || { echo "Missing config: $BASE_CONFIG"; exit 1; }
  mkdir -p "$OUT_ROOT/configs"

  local run_idx=0
  for lr in $LR_LIST; do
    for bs in $BATCH_LIST; do
      for drop in $DROPOUT_LIST; do
        for nl in $LAYER_LIST; do
          run_idx=$((run_idx + 1))
          tag="run${run_idx}_lr${lr}_bs${bs}_dp${drop}_nl${nl}"
          tag="${tag//./p}"; tag="${tag//-/m}"
          cfg="$OUT_ROOT/configs/${tag}.json"
          outdir="$OUT_ROOT/$tag"
          mkdir -p "$outdir"

          python utils/make_config.py \
            --base "$BASE_CONFIG" \
            --out "$cfg" \
            --set "training.lr=$lr" \
            --set "training.batch_size=$bs" \
            --set "training.num_epochs=$EPOCHS" \
            --set "model.dropout=$drop" \
            --set "model.num_layers=$nl"

          echo "[EE] $tag"
          conda run -n "$ENV_NAME" python scripts/train_unified/train.py run ee -- \
            --config "$cfg" \
            --dataset "$DATASET" \
            --outdir "$outdir"
        done
      done
    done
  done
}

run_kd_fusion() {
  local DATASET="data/kd_fusion/kd_fusion_dataset.pt"
  local BASE_CONFIG="config/kd_fusion_config.json"
  local OUT_ROOT="results_tune/kd_fusion"
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
      --lr-list) LR_LIST="$2"; shift 2 ;;
      --batch-list) BATCH_LIST="$2"; shift 2 ;;
      --dropout-list) DROPOUT_LIST="$2"; shift 2 ;;
      --student-hidden-list) STUDENT_HIDDEN_LIST="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      *) echo "Unknown kd-fusion arg: $1"; exit 1 ;;
    esac
  done

  [[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET"; exit 1; }
  [[ -f "$BASE_CONFIG" ]] || { echo "Missing config: $BASE_CONFIG"; exit 1; }
  mkdir -p "$OUT_ROOT/configs"

  local run_idx=0
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

          python utils/make_config.py \
            --base "$BASE_CONFIG" \
            --out "$cfg" \
            --set "training.lr=$lr" \
            --set "training.batch_size=$bs" \
            --set "training.num_epochs=$EPOCHS" \
            --set "model.dropout=$dp" \
            --set "model.student_hidden=$sh"

          echo "[KD-FUSION] $tag"
          conda run -n "$ENV_NAME" python scripts/train_unified/train.py run kd-fusion -- \
            --config "$cfg" \
            --dataset "$DATASET" \
            --outdir "$outdir"
        done
      done
    done
  done
}

REST="$(parse_common "$@")"
set -- $REST

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

[[ -n "$TARGET" ]] || { echo "--target is required"; usage; exit 1; }

case "$TARGET" in
  esm2) run_esm2 "$@" ;;
  evo2) run_evo2 "$@" ;;
  oracle-kg) run_oracle_kg "$@" ;;
  ee) run_ee "$@" ;;
  kd-fusion) run_kd_fusion "$@" ;;
  *) echo "Unknown target: $TARGET"; usage; exit 1 ;;
esac
