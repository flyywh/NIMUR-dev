#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

# ESM2 FASTA inputs
TRAIN_POS=""
TRAIN_NEG=""
VALID_POS=""
VALID_NEG=""
TEST_POS=""
TEST_NEG=""

# Evo2 FASTA inputs
EVO_TRAIN_POS=""
EVO_TRAIN_NEG=""
EVO_TEST_POS=""
EVO_TEST_NEG=""

# KG TSV
KG_TSV=""

usage() {
  cat <<USAGE
Usage: bash scripts_train/07_full_tune_pipeline.sh [options]

Required groups:
  --train-pos --train-neg --valid-pos --valid-neg --test-pos --test-neg   (for ESM2)
  --evo-train-pos --evo-train-neg --evo-test-pos --evo-test-neg           (for Evo2)
  --kg-tsv                                                                  (for ORACLE KG)

Optional:
  --env evo2
  --gpus 0,1
USAGE
}

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
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

required_args=(
  "TRAIN_POS:$TRAIN_POS"
  "TRAIN_NEG:$TRAIN_NEG"
  "VALID_POS:$VALID_POS"
  "VALID_NEG:$VALID_NEG"
  "TEST_POS:$TEST_POS"
  "TEST_NEG:$TEST_NEG"
  "EVO_TRAIN_POS:$EVO_TRAIN_POS"
  "EVO_TRAIN_NEG:$EVO_TRAIN_NEG"
  "EVO_TEST_POS:$EVO_TEST_POS"
  "EVO_TEST_NEG:$EVO_TEST_NEG"
  "KG_TSV:$KG_TSV"
)

for kv in "${required_args[@]}"; do
  key="${kv%%:*}"
  val="${kv#*:}"
  if [[ -z "$val" ]]; then
    echo "Missing required argument for $key"
    usage
    exit 1
  fi
done

for p in \
  "$TRAIN_POS" "$TRAIN_NEG" "$VALID_POS" "$VALID_NEG" "$TEST_POS" "$TEST_NEG" \
  "$EVO_TRAIN_POS" "$EVO_TRAIN_NEG" "$EVO_TEST_POS" "$EVO_TEST_NEG" "$KG_TSV"; do
  [[ -f "$p" ]] || { echo "Missing input file: $p"; exit 1; }
done

bash scripts_train/00_prepare_assets.sh --env "$ENV_NAME"

bash scripts_train/01_prepare_esm2_data.sh \
  --env "$ENV_NAME" \
  --gpus "$GPUS" \
  --train-pos "$TRAIN_POS" --train-neg "$TRAIN_NEG" \
  --valid-pos "$VALID_POS" --valid-neg "$VALID_NEG" \
  --test-pos "$TEST_POS" --test-neg "$TEST_NEG" \
  --out-dir data/esm2_tune

bash scripts_train/02_tune_esm2.sh --env "$ENV_NAME" \
  --gpus "$GPUS" \
  --dataset-dir data/esm2_tune \
  --test-dataset data/esm2_tune/test_dataset_esm2.pt \
  --out-root results_tune/esm2

bash scripts_train/03_prepare_evo2_data.sh \
  --env "$ENV_NAME" \
  --gpus "$GPUS" \
  --train-pos-fasta "$EVO_TRAIN_POS" --train-neg-fasta "$EVO_TRAIN_NEG" \
  --test-pos-fasta "$EVO_TEST_POS" --test-neg-fasta "$EVO_TEST_NEG" \
  --work-dir data/evo2_tune

bash scripts_train/04_tune_evo2.sh --env "$ENV_NAME" \
  --gpus "$GPUS" \
  --train-dataset data/evo2_tune/train.pt \
  --test-dataset data/evo2_tune/test.pt \
  --out-root results_tune/evo2

bash scripts_train/05_prepare_kg_data.sh \
  --input-tsv "$KG_TSV" \
  --out-dir data/kg_tune \
  --train-ratio 0.8 \
  --seed 42

bash scripts_train/06_tune_oracle_kg.sh --env "$ENV_NAME" \
  --gpus "$GPUS" \
  --train-dir data/kg_tune/train \
  --val-dir data/kg_tune/val \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --out-root results_tune/oracle_kg

echo "Full tuning pipeline completed."
