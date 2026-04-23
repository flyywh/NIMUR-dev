#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

GENOME_FNA=""
GENOME_ID=""
PRODIGAL_DIR="test_real/antismash_test/02_prodigal"
OUTPUT_ROOT="results/full_bgc_report"

RUN_CLASSIC=1
RUN_KD=1

KD_OUT_ROOT="results_tune/kd_fusion"
KD_MODEL=""
KD_THRESHOLD="0.55"
KD_WINDOW="12000"
KD_STEP="3000"
KD_MERGE_GAP="1500"
KD_EVO_MAX_LEN="16384"

usage() {
  cat <<USAGE
Usage: bash scripts_train/20_full_bgc_report.sh [options]

Required:
  --genome-fna PATH

Optional:
  --genome-id ID                    (default: basename of genome-fna)
  --env ENV
  --gpus 0,1
  --prodigal-dir DIR                (default: test_real/antismash_test/02_prodigal)
  --output-root DIR                 (default: results/full_bgc_report)

  --run-classic 0|1                 run existing ESM2/EVO2/KG pipeline (default: 1)
  --run-kd 0|1                      run KD genome detector (default: 1)

  --kd-out-root DIR                 KD tuning root for auto model selection
  --kd-model PATH                   explicit KD model (kd_best.pt)
  --kd-threshold 0.55
  --kd-window 12000
  --kd-step 3000
  --kd-merge-gap 1500
  --kd-evo-max-len 16384
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --genome-fna) GENOME_FNA="$2"; shift 2 ;;
    --genome-id) GENOME_ID="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --prodigal-dir) PRODIGAL_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-classic) RUN_CLASSIC="$2"; shift 2 ;;
    --run-kd) RUN_KD="$2"; shift 2 ;;
    --kd-out-root) KD_OUT_ROOT="$2"; shift 2 ;;
    --kd-model) KD_MODEL="$2"; shift 2 ;;
    --kd-threshold) KD_THRESHOLD="$2"; shift 2 ;;
    --kd-window) KD_WINDOW="$2"; shift 2 ;;
    --kd-step) KD_STEP="$2"; shift 2 ;;
    --kd-merge-gap) KD_MERGE_GAP="$2"; shift 2 ;;
    --kd-evo-max-len) KD_EVO_MAX_LEN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -n "$GENOME_FNA" ]] || { echo "Missing required: --genome-fna"; usage; exit 1; }
GENOME_FNA="$(realpath -m "$GENOME_FNA")"
[[ -f "$GENOME_FNA" ]] || { echo "Missing genome fna: $GENOME_FNA"; exit 1; }

if [[ -z "$GENOME_ID" ]]; then
  GENOME_ID="$(basename "$GENOME_FNA")"
  GENOME_ID="${GENOME_ID%.fna}"
fi

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$(realpath -m "$OUTPUT_ROOT/${GENOME_ID}_${STAMP}")"
mkdir -p "$RUN_DIR"

CLASSIC_DIR=""
KD_DIR=""

if [[ "$RUN_CLASSIC" == "1" ]]; then
  CLASSIC_RAW="$RUN_DIR/classic_raw"
  mkdir -p "$CLASSIC_RAW"

  ORF_FAA="$PRODIGAL_DIR/${GENOME_ID}.faa"
  EVO2_DNA="$PRODIGAL_DIR/${GENOME_ID}.ffn"

  classic_cmd=(bash scripts/test_real_bgc_all.sh
    --genome-id "$GENOME_ID"
    --input-dir "$(dirname "$GENOME_FNA")"
    --env "$ENV_NAME"
    --output-root "$CLASSIC_RAW")

  if [[ -f "$ORF_FAA" ]]; then
    classic_cmd+=(--orf-faa "$ORF_FAA")
  fi
  if [[ -f "$EVO2_DNA" ]]; then
    classic_cmd+=(--evo2-dna "$EVO2_DNA")
  fi

  "${classic_cmd[@]}"

  CLASSIC_DIR="$(find "$CLASSIC_RAW" -maxdepth 1 -mindepth 1 -type d -name "${GENOME_ID}_*" | sort | tail -n 1)"
  if [[ -n "$CLASSIC_DIR" ]]; then
    ln -sfn "$CLASSIC_DIR" "$RUN_DIR/classic"
  fi
fi

if [[ "$RUN_KD" == "1" ]]; then
  if [[ -z "$KD_MODEL" ]]; then
    KD_MODEL="$(python - <<'PY' "$KD_OUT_ROOT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
best = None
best_auc = -1.0
for metrics in root.glob('run*/metrics.json'):
    m = metrics.parent / 'models' / 'kd_best.pt'
    if not m.exists():
        continue
    try:
        auc = float(json.loads(metrics.read_text(encoding='utf-8')).get('best_val_auc', -1.0))
    except Exception:
        auc = -1.0
    if auc > best_auc:
        best_auc = auc
        best = m
if best is None:
    cand = sorted(root.glob('run*/models/kd_best.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        best = cand[0]
print(str(best) if best else '')
PY
)"
  fi

  [[ -n "$KD_MODEL" ]] || { echo "Cannot locate KD model"; exit 1; }
  KD_MODEL="$(realpath -m "$KD_MODEL")"
  [[ -f "$KD_MODEL" ]] || { echo "Missing KD model: $KD_MODEL"; exit 1; }

  KD_DIR="$RUN_DIR/kd"
  conda run -n "$ENV_NAME" python predict/kd_genome_detect.py \
    --model "$KD_MODEL" \
    --genome-fna "$GENOME_FNA" \
    --out-dir "$KD_DIR" \
    --threshold "$KD_THRESHOLD" \
    --window "$KD_WINDOW" \
    --step "$KD_STEP" \
    --merge-gap "$KD_MERGE_GAP" \
    --evo-max-len "$KD_EVO_MAX_LEN"
fi

REPORT_MD="$RUN_DIR/REPORT.md"
{
  echo "# Full BGC Report"
  echo
  echo "- Genome ID: $GENOME_ID"
  echo "- Genome FNA: $GENOME_FNA"
  echo "- Run dir: $RUN_DIR"
  echo

  if [[ "$RUN_CLASSIC" == "1" ]]; then
    echo "## Classic Pipeline (ESM2/EVO2/KG)"
    if [[ -n "$CLASSIC_DIR" ]]; then
      echo "- Output dir: $CLASSIC_DIR"
      echo "- ESM2 predictions: $CLASSIC_DIR/esm2/predictions.csv"
      echo "- EVO2 predictions: $CLASSIC_DIR/evo2/predictions.csv"
      echo "- KG classifications: $CLASSIC_DIR/kg/knowledge_graph_results/classifications.csv"
      echo "- KG distances: $CLASSIC_DIR/kg/knowledge_graph_results/distances.csv"
    else
      echo "- Output dir: not found"
    fi
    echo
  fi

  if [[ "$RUN_KD" == "1" ]]; then
    echo "## KD Genome Detector"
    echo "- Output dir: $KD_DIR"
    echo "- Window predictions: $KD_DIR/window_predictions.csv"
    echo "- BGC regions: $KD_DIR/bgc_regions.csv"
    echo "- Summary: $KD_DIR/summary.json"
    echo "- Model: ${KD_MODEL:-N/A}"
    echo
  fi
} > "$REPORT_MD"

echo "[FULL-REPORT] done: $RUN_DIR"
echo "[FULL-REPORT] report: $REPORT_MD"
