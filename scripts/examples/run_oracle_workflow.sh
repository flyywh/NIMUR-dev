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
SPLIT_CACHE_DIR="${SPLIT_CACHE_DIR:-data/cache_esm_protbert}"
KG_DIR="${KG_DIR:-data/oracle_kg}"
FUSED_CACHE_DIR="${FUSED_CACHE_DIR:-$KG_DIR/fused_cache}"
OUT_DIR="${OUT_DIR:-results_tune/oracle_kg_multilabel}"
MODEL_PATH="${MODEL_PATH:-$OUT_DIR/best.pt}"
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
  run_python scripts/data_prep/build_oracle_kg_from_split_index.py \
    --split-index "$SPLIT_INDEX" \
    --train-pos-fasta "$TRAIN_POS" \
    --train-neg-fasta "$TRAIN_NEG" \
    --val-pos-fasta "$VAL_POS" \
    --val-neg-fasta "$VAL_NEG" \
    --test-pos-fasta "$TEST_POS" \
    --test-neg-fasta "$TEST_NEG" \
    --split-cache-dir "$SPLIT_CACHE_DIR" \
    --fused-cache-dir "$FUSED_CACHE_DIR" \
    --protbert "$PROTBERT" \
    --output-dir "$KG_DIR"
}

run_train() {
  export CUDA_LAUNCH_BLOCKING=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python scripts/trainers/ORACLE_train.py \
    --train_dir "$KG_DIR/train" \
    --val_dir "$KG_DIR/val" \
    --embedding_cache_dir "$FUSED_CACHE_DIR" \
    --embedding_dim 256 \
    --batch_size 8 \
    --eval_batch_size 16 \
    --epochs 5 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --output_dir "$OUT_DIR"
}

run_infer() {
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python predict/oracle_predict.py \
    --device cuda \
    --model "$MODEL_PATH" \
    --fasta "$GENOME_FNA" \
    --protbert-path "$PROTBERT" \
    --out "$OUT_DIR/fasta_rankings.csv" \
    --top-k 5 \
    --batch-size 1 \
    --esm-max-len 1022 \
    --protbert-max-len 2400
}

run_detect() {
  export CUDA_VISIBLE_DEVICES="$GPU"
  run_python predict/oracle_genome_detect.py \
    --device cuda \
    --model "$MODEL_PATH" \
    --genome-fna "$GENOME_FNA" \
    --protbert-path "$PROTBERT" \
    --out-dir "$OUT_DIR/genome_detect_$(basename "${GENOME_FNA%.*}")" \
    --window 12000 \
    --step 3000 \
    --threshold 0.55 \
    --merge-gap 1500 \
    --top-k 5 \
    --esm-max-len 1022 \
    --protbert-max-len 2400 \
    --batch-size 1
}

case "$STAGE" in
  data) run_data ;;
  train) run_train ;;
  infer) run_infer ;;
  detect) run_detect ;;
  all) run_data; run_train; run_infer; run_detect ;;
  *) echo "Usage: bash scripts/examples/run_oracle_workflow.sh [data|train|infer|detect|all]"; exit 1 ;;
esac
