#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""
TRAIN_DIR="data/kg_tune_real/train"
VAL_DIR="data/kg_tune_real/val"
PROTBERT_PATH="local_assets/data/ProtBERT/ProtBERT"
OUT_ROOT="results_tune/oracle_kg_real_lora"
PRETRAINED="local_assets/default_models/ORACLE"
EE_REF_MODEL=""
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
FREEZE_BASE=1

EMBED_LIST="128 256"
LR_LIST="1e-4 2e-4"
BATCH_LIST="4 8"
EPOCHS_LIST="5 10"
ACCUM_LIST="1 2"
KE_LAMBDA_LIST="1.0"
MAX_SEQ_LEN="1024"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --train-dir) TRAIN_DIR="$2"; shift 2 ;;
    --val-dir) VAL_DIR="$2"; shift 2 ;;
    --protbert) PROTBERT_PATH="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --pretrained) PRETRAINED="$2"; shift 2 ;;
    --ee-ref-model) EE_REF_MODEL="$2"; shift 2 ;;
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
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -n "$GPUS" ]]; then export CUDA_VISIBLE_DEVICES="$GPUS"; fi
mkdir -p "$OUT_ROOT"

if [[ -n "$EE_REF_MODEL" ]]; then
  [[ -f "$EE_REF_MODEL" ]] || { echo "Missing EE reference model: $EE_REF_MODEL"; exit 1; }
  cat > "$OUT_ROOT/ee_dependency.meta" <<META
ee_ref_model=$EE_REF_MODEL
kg_tune_time_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
META
  echo "[ORACLE-KG] EE dependency model: $EE_REF_MODEL"
fi

run_idx=0
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
            cmd=(conda run -n "$ENV_NAME" python scripts_train/oracle_train_lora.py \
              --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --protbert_path "$PROTBERT_PATH" \
              --embedding_dim "$ed" --epochs "$ep" --batch_size "$bs" --eval_batch_size "$bs" \
              --lr "$lr" --weight_decay 0.01 --gradient_accumulation_steps "$ga" --ke_lambda "$kl" \
              --max_seq_len "$MAX_SEQ_LEN" --eval_sample_limit -1 --output_dir "$outdir" \
              --pretrained "$PRETRAINED" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT")
            if [[ "$FREEZE_BASE" == "1" ]]; then cmd+=(--freeze-base); fi
            "${cmd[@]}"
          done
        done
      done
    done
  done
done

echo "ORACLE-KG LoRA tuning done: $OUT_ROOT"
