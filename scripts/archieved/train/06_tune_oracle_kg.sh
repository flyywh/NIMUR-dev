#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/tmp_smoke/mplconfig}"
TRAIN_DIR="data/kg_tune/train"
VAL_DIR="data/kg_tune/val"
PROTBERT_PATH="local_assets/data/ProtBERT/ProtBERT"
OUT_ROOT="results_tune/oracle_kg"

EMBED_LIST="128 256"
LR_LIST="1e-4 2e-4"
BATCH_LIST="4 8"
EPOCHS_LIST="5 10"
ACCUM_LIST="1 2"
KE_LAMBDA_LIST="1.0"
MAX_SEQ_LEN="1024"

usage() {
  cat <<USAGE
Usage: bash scripts_train/06_tune_oracle_kg.sh [options]

Options:
  --env ENV
  --train-dir DIR
  --val-dir DIR
  --protbert PATH
  --out-root DIR
  --gpus 0,1
  --embed-list "128 256"
  --lr-list "1e-4 2e-4"
  --batch-list "4 8"
  --epochs-list "5 10"
  --accum-list "1 2"
  --ke-lambda-list "1.0 2.0"
  --max-seq-len 1024
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --train-dir) TRAIN_DIR="$2"; shift 2 ;;
    --val-dir) VAL_DIR="$2"; shift 2 ;;
    --protbert) PROTBERT_PATH="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --embed-list) EMBED_LIST="$2"; shift 2 ;;
    --lr-list) LR_LIST="$2"; shift 2 ;;
    --batch-list) BATCH_LIST="$2"; shift 2 ;;
    --epochs-list) EPOCHS_LIST="$2"; shift 2 ;;
    --accum-list) ACCUM_LIST="$2"; shift 2 ;;
    --ke-lambda-list) KE_LAMBDA_LIST="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

for req in protein2id.txt type2id.txt relation2id.txt sequence.txt protein_relation_type.txt; do
  [[ -f "$TRAIN_DIR/$req" ]] || { echo "Missing $TRAIN_DIR/$req"; exit 1; }
  [[ -f "$VAL_DIR/$req" ]] || { echo "Missing $VAL_DIR/$req"; exit 1; }
done
[[ -d "$PROTBERT_PATH" ]] || { echo "Missing protbert dir: $PROTBERT_PATH"; exit 1; }

mkdir -p "$OUT_ROOT"

run_idx=0
for ed in $EMBED_LIST; do
  for lr in $LR_LIST; do
    for bs in $BATCH_LIST; do
      for ep in $EPOCHS_LIST; do
        for ga in $ACCUM_LIST; do
          for kl in $KE_LAMBDA_LIST; do
            run_idx=$((run_idx + 1))
            tag="run${run_idx}_ed${ed}_lr${lr}_bs${bs}_ep${ep}_ga${ga}_kl${kl}"
            tag="${tag//./p}"
            tag="${tag//-/m}"
            outdir="$OUT_ROOT/$tag"
            mkdir -p "$outdir"

            echo "[ORACLE-KG] $tag"
            conda run -n "$ENV_NAME" python scripts/ORACLE_train.py \
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
              --output_dir "$outdir"
          done
        done
      done
    done
  done
done

echo "ORACLE-KG tuning done. Output: $OUT_ROOT"
