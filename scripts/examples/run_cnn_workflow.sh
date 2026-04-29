#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT" || exit 1

ENV_NAME="${ENV_NAME:-nimur_mh}"
GPU="${GPU:-4}"
STAGE="${1:-all}"

TRAIN_POS="${TRAIN_POS:-data_ar/mibig_splits/MiBiG_train_pos.fasta}"
TRAIN_NEG="${TRAIN_NEG:-data_ar/mibig_splits/MiBiG_train_neg.fasta}"
VAL_POS="${VAL_POS:-data_ar/mibig_splits/MiBiG_val_pos.fasta}"
VAL_NEG="${VAL_NEG:-data_ar/mibig_splits/MiBiG_val_neg.fasta}"
TEST_POS="${TEST_POS:-data_ar/mibig_splits/MiBiG_test_pos.fasta}"
TEST_NEG="${TEST_NEG:-data_ar/mibig_splits/MiBiG_test_neg.fasta}"
SPLIT_INDEX="${SPLIT_INDEX:-data_ar/mibig_splits/MiBiG_split_index.tsv}"
PROTBERT="${PROTBERT:-local_assets/data/ProtBERT/ProtBERT}"
DATA_DIR="${DATA_DIR:-data/cnn_dataset_multilabel}"
CACHE_DIR="${CACHE_DIR:-data/cache_esm_protbert}"
CONFIG="${CONFIG:-config/oracle_cnn_config.json}"
OUT_DIR="${OUT_DIR:-results_tune/cnn_multilabel}"
MODEL_PATH="${MODEL_PATH:-$OUT_DIR/model_epoch_7.pt}"
GENOME_FNA="${GENOME_FNA:-test_real/antismash_test/01_genome/CRBC_G0001.fna}"

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    python "$@"
  else
    CONDA_NO_PLUGINS=true conda run -n "$ENV_NAME" python "$@"
  fi
}

run_data() {
  run_python scripts/data_prep/build_cnn_dataset.py \
    --train-pos-faa "$TRAIN_POS" \
    --train-neg-faa "$TRAIN_NEG" \
    --val-pos-faa "$VAL_POS" \
    --val-neg-faa "$VAL_NEG" \
    --test-pos-faa "$TEST_POS" \
    --test-neg-faa "$TEST_NEG" \
    --split-index "$SPLIT_INDEX" \
    --cache-dir "$CACHE_DIR" \
    --protbert "$PROTBERT" \
    --out-dir "$DATA_DIR" \
    --device auto \
    --esm-max-len 1022 \
    --max-seq-len 2400 \
    --chunk-size 30
}

run_train() {
  export CUDA_LAUNCH_BLOCKING=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python scripts/trainers/ORACLE_CNN_train.py \
    --config "$CONFIG" \
    --train_dataset "$DATA_DIR/train" \
    --valid_dataset "$DATA_DIR/val" \
    --output_dir "$OUT_DIR"
}

run_infer() {
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python predict/oracle_cnn_predict.py \
    --config "$CONFIG" \
    --model "$MODEL_PATH" \
    --fasta "$GENOME_FNA" \
    --protbert "$PROTBERT" \
    --out "$OUT_DIR/fasta_predictions.csv" \
    --esm-max-len 1022 \
    --max-seq-len 2400 \
    --batch-size 8 \
    --threshold 0.5
}

run_detect() {
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python predict/cnn_genome_detect.py \
    --config "$CONFIG" \
    --model "$MODEL_PATH" \
    --genome-fna "$GENOME_FNA" \
    --protbert "$PROTBERT" \
    --out-dir "$OUT_DIR/genome_detect_$(basename "${GENOME_FNA%.*}")" \
    --window 12000 \
    --step 3000 \
    --threshold 0.55 \
    --merge-gap 1500 \
    --esm-max-len 1022 \
    --max-seq-len 2400 \
    --batch-size 8
}

case "$STAGE" in
  data) run_data ;;
  train) run_train ;;
  infer) run_infer ;;
  detect) run_detect ;;
  all) run_data; run_train; run_infer; run_detect ;;
  *) echo "Usage: bash scripts/examples/run_cnn_workflow.sh [data|train|infer|detect|all]"; exit 1 ;;
esac
