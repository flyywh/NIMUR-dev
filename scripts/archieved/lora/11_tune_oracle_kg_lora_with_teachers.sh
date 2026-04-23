#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""
TRAIN_DIR="data/kg_tune_real/train"
VAL_DIR="data/kg_tune_real/val"
OUT_ROOT="results_tune/oracle_kg_real_lora_with_teachers"

# KG model args (same as 06_tune_oracle_kg_lora.sh)
PROTBERT_PATH="local_assets/data/ProtBERT/ProtBERT"
PRETRAINED="local_assets/default_models/ORACLE"
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

# Teacher models (required): tuned ESM2 / Evo2 checkpoints.
ESM2_MODEL=""
EVO2_MODEL=""

usage() {
  cat <<USAGE
Usage: bash scripts_train/11_tune_oracle_kg_lora_with_teachers.sh [options]

Required:
  --esm2-model PATH   tuned ESM2 checkpoint file or run directory
  --evo2-model PATH   tuned Evo2 checkpoint file or run directory

Optional:
  --env ENV
  --gpus 0,1
  --train-dir DIR
  --val-dir DIR
  --out-root DIR

  --protbert PATH
  --pretrained PATH
  --lora-r 8
  --lora-alpha 16
  --lora-dropout 0.05
  --no-freeze-base

  --embed-list "128 256"
  --lr-list "1e-4 2e-4"
  --batch-list "4 8"
  --epochs-list "5 10"
  --accum-list "1 2"
  --ke-lambda-list "1.0"
  --max-seq-len 1024

Notes:
  1) ESM2/Evo2 teacher models are treated as fixed references (not updated during KG tuning).
  2) If --esm2-model / --evo2-model is a directory, this script auto-resolves a checkpoint inside it.
USAGE
}

resolve_ckpt() {
  local p="$1"
  if [[ -f "$p" ]]; then
    echo "$p"
    return 0
  fi

  if [[ ! -d "$p" ]]; then
    echo ""
    return 1
  fi

  local cand
  cand="$(find "$p" -type f \( -name 'best.pt' -o -name 'best.pth' -o -name 'model_best.pt' -o -name 'model_best.pth' -o -name 'checkpoint_best.pt' -o -name 'checkpoint_best.pth' -o -name 'model_epoch_*.pth' -o -name 'checkpoint*.pt' \) -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -n 1 | cut -f2-)"

  if [[ -n "$cand" ]]; then
    echo "$cand"
    return 0
  fi

  echo ""
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --train-dir) TRAIN_DIR="$2"; shift 2 ;;
    --val-dir) VAL_DIR="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;

    --protbert) PROTBERT_PATH="$2"; shift 2 ;;
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

    --esm2-model) ESM2_MODEL="$2"; shift 2 ;;
    --evo2-model) EVO2_MODEL="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -n "$ESM2_MODEL" ]] || { echo "Missing required: --esm2-model"; usage; exit 1; }
[[ -n "$EVO2_MODEL" ]] || { echo "Missing required: --evo2-model"; usage; exit 1; }

ESM2_CKPT="$(resolve_ckpt "$ESM2_MODEL")" || { echo "Cannot resolve ESM2 checkpoint from: $ESM2_MODEL"; exit 1; }
EVO2_CKPT="$(resolve_ckpt "$EVO2_MODEL")" || { echo "Cannot resolve Evo2 checkpoint from: $EVO2_MODEL"; exit 1; }

for req in protein2id.txt type2id.txt relation2id.txt sequence.txt protein_relation_type.txt; do
  [[ -f "$TRAIN_DIR/$req" ]] || { echo "Missing $TRAIN_DIR/$req"; exit 1; }
  [[ -f "$VAL_DIR/$req" ]] || { echo "Missing $VAL_DIR/$req"; exit 1; }
done
[[ -d "$PROTBERT_PATH" ]] || { echo "Missing protbert dir: $PROTBERT_PATH"; exit 1; }

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

mkdir -p "$OUT_ROOT"

# Keep teacher checkpoints as immutable references for this KG tuning run.
cat > "$OUT_ROOT/teacher_models.meta" <<META
esm2_teacher_ckpt=$ESM2_CKPT
evo2_teacher_ckpt=$EVO2_CKPT
kg_tune_time_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
kg_tune_mode=teachers_frozen_reference_only
META

echo "[KG-TEACHER] ESM2 teacher: $ESM2_CKPT"
echo "[KG-TEACHER] Evo2 teacher: $EVO2_CKPT"
echo "[KG-TEACHER] Teachers are fixed references and will not be updated."

cmd=(bash scripts_train/06_tune_oracle_kg_lora.sh
  --env "$ENV_NAME"
  --gpus "$GPUS"
  --train-dir "$TRAIN_DIR"
  --val-dir "$VAL_DIR"
  --protbert "$PROTBERT_PATH"
  --out-root "$OUT_ROOT"
  --pretrained "$PRETRAINED"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --embed-list "$EMBED_LIST"
  --lr-list "$LR_LIST"
  --batch-list "$BATCH_LIST"
  --epochs-list "$EPOCHS_LIST"
  --accum-list "$ACCUM_LIST"
  --ke-lambda-list "$KE_LAMBDA_LIST"
  --max-seq-len "$MAX_SEQ_LEN")

if [[ "$FREEZE_BASE" == "0" ]]; then
  cmd+=(--no-freeze-base)
fi

"${cmd[@]}"

echo "KG LoRA tuning with fixed ESM2/Evo2 teacher references done: $OUT_ROOT"
