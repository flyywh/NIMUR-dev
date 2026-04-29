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
CACHE_DIR="${CACHE_DIR:-data/cache_esm_protbert}"
DATASET="${DATASET:-data/kd_fusion/kd_dataset_multilabel.pt}"
CONFIG="${CONFIG:-config/kd_fusion_config.json}"
OUT_DIR="${OUT_DIR:-results_tune/kd_fusion_multilabel}"
MODEL_PATH="${MODEL_PATH:-$OUT_DIR/models/kd_best.pt}"
GENOME_FNA="${GENOME_FNA:-test_real/antismash_test/01_genome/CRBC_G0001.fna}"

run_python() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
    python "$@"
  else
    CONDA_NO_PLUGINS=true conda run -n "$ENV_NAME" python "$@"
  fi
}

run_data() {
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python scripts/data_prep/build_esm_protbert_dataset.py \
    --train-pos-faa "$TRAIN_POS" \
    --train-neg-faa "$TRAIN_NEG" \
    --val-pos-faa "$VAL_POS" \
    --val-neg-faa "$VAL_NEG" \
    --test-pos-faa "$TEST_POS" \
    --test-neg-faa "$TEST_NEG" \
    --split-index "$SPLIT_INDEX" \
    --cache-dir "$CACHE_DIR" \
    --protbert "$PROTBERT" \
    --out "$DATASET" \
    --device auto \
    --esm-chunk-len 1022 \
    --protbert-max-len 2400
}

run_train() {
  export CUDA_LAUNCH_BLOCKING=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python scripts/trainers/kd_fusion_train.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --outdir "$OUT_DIR"
}

run_infer() {
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python predict/kd_fusion_predict_fasta.py \
    --model "$MODEL_PATH" \
    --fasta "$GENOME_FNA" \
    --protbert "$PROTBERT" \
    --out "$OUT_DIR/fasta_predictions.csv" \
    --scorer student \
    --esm-dim 1280 \
    --prot-dim 1024 \
    --esm-chunk-len 1022 \
    --protbert-max-len 2400
}

run_detect() {
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python predict/kd_genome_detect.py \
    --model "$MODEL_PATH" \
    --genome-fna "$GENOME_FNA" \
    --protbert "$PROTBERT" \
    --out-dir "$OUT_DIR/genome_detect_$(basename "${GENOME_FNA%.*}")" \
    --scorer student \
    --window 12000 \
    --step 3000 \
    --threshold 0.55 \
    --merge-gap 1500 \
    --esm-chunk-len 1022 \
    --protbert-max-len 2400 \
    --batch-size 8
}

case "$STAGE" in
  data) run_data ;;
  train) run_train ;;
  infer) run_infer ;;
  detect) run_detect ;;
  all) run_data; run_train; run_infer; run_detect ;;
  *) echo "Usage: bash scripts/examples/run_kd_workflow.sh [data|train|infer|detect|all]"; exit 1 ;;
esac
