#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default: only print commands.
# Set DRY_RUN=0 to execute.
DRY_RUN="${DRY_RUN:0}"

MODEL="${MODEL:-results_tune/kd_fusion/run1/models/kd_best.pt}"
FASTA="${FASTA:-data/example.fasta}"
GENOME_FNA="${GENOME_FNA:-data/example_genome.fna}"
OUT_ROOT="${OUT_ROOT:-tmp}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRYRUN] $*"
  else
    echo "[RUN] $*"
    eval "$*"
  fi
}

echo "MODEL=$MODEL"
echo "FASTA=$FASTA"
echo "GENOME_FNA=$GENOME_FNA"
echo "OUT_ROOT=$OUT_ROOT"
echo "DRY_RUN=$DRY_RUN"
echo

# 1) Unified prediction: single route (ESM2 + student)
run_cmd "python predict/predict_unified.py \
  --model \"$MODEL\" \
  --fasta \"$FASTA\" \
  --out \"$OUT_ROOT/predict_single.csv\" \
  --mode single \
  --single-route esm2 \
  --single-scorer student"

# 2) Unified prediction: multi-route fusion with teacher (no student)
run_cmd "python predict/predict_unified.py \
  --model \"$MODEL\" \
  --fasta \"$FASTA\" \
  --out \"$OUT_ROOT/predict_fusion_teacher.csv\" \
  --mode fusion_teacher"

# 3) Unified prediction: multi-route fusion with student
run_cmd "python predict/predict_unified.py \
  --model \"$MODEL\" \
  --fasta \"$FASTA\" \
  --out \"$OUT_ROOT/predict_fusion_student.csv\" \
  --mode fusion_student"

# 4) Genome window detection (v2)
run_cmd "python predict/kd_genome_detect.py \
  --model \"$MODEL\" \
  --genome-fna \"$GENOME_FNA\" \
  --out-dir \"$OUT_ROOT/predict\" \
  --scorer student \
  --threshold 0.55 \
  --window 12000 \
  --step 3000 \
  --merge-gap 1500"

echo
echo "Done."
