#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS=""

KD_OUT_ROOT="results_tune/kd_fusion"
MODEL=""
FASTA=""
OUT=""
PROTBERT="local_assets/data/ProtBERT/ProtBERT"
ESM_DIM="1280"
PROT_DIM="1024"
ESM_CHUNK_LEN="1022"
PROTBERT_MAX_LEN="1024"

usage() {
  cat <<USAGE
Usage: bash scripts_train/19_infer_kd_fusion_fasta.sh [options]

Options:
  --env ENV
  --gpus 0,1
  --fasta PATH            Input FASTA (required)
  --kd-out-root DIR       KD tune root (default: results_tune/kd_fusion)
  --model PATH            Optional explicit kd_best.pt
  --out PATH              Optional output csv
  --protbert PATH         ProtBERT dir
  --esm-dim N             ESM2 feature dim (default: 1280)
  --prot-dim N            ProtBERT feature dim (default: 1024)
  --esm-chunk-len N       ESM2 chunk len (default: 1022)
  --protbert-max-len N    ProtBERT max len (default: 1024)

Model selection:
  If --model is not set, auto-select run with highest metrics.json best_val_auc.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --fasta) FASTA="$2"; shift 2 ;;
    --kd-out-root) KD_OUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --protbert) PROTBERT="$2"; shift 2 ;;
    --esm-dim) ESM_DIM="$2"; shift 2 ;;
    --prot-dim) PROT_DIM="$2"; shift 2 ;;
    --esm-chunk-len) ESM_CHUNK_LEN="$2"; shift 2 ;;
    --protbert-max-len) PROTBERT_MAX_LEN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -n "$FASTA" ]] || { echo "Missing required: --fasta"; usage; exit 1; }

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FASTA="$(realpath -m "$FASTA")"
KD_OUT_ROOT="$(realpath -m "$KD_OUT_ROOT")"
PROTBERT="$(realpath -m "$PROTBERT")"

[[ -f "$FASTA" ]] || { echo "Missing fasta: $FASTA"; exit 1; }
[[ -d "$PROTBERT" ]] || { echo "Missing ProtBERT dir: $PROTBERT"; exit 1; }

if [[ -z "$MODEL" ]]; then
  MODEL="$(python - <<'PY' "$KD_OUT_ROOT"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
best_model = None
best_auc = -1.0
metrics_candidates = []
if (root / 'metrics.json').exists():
    metrics_candidates.append(root / 'metrics.json')
metrics_candidates.extend(sorted(root.glob('run*/metrics.json')))

for metrics in metrics_candidates:
    m = metrics.parent / 'models' / 'kd_best.pt'
    if not m.exists():
        continue
    try:
        auc = float(json.loads(metrics.read_text(encoding='utf-8')).get('best_val_auc', -1.0))
    except Exception:
        auc = -1.0
    if auc > best_auc:
        best_auc = auc
        best_model = m

if best_model is None:
    cand = []
    if (root / 'models' / 'kd_best.pt').exists():
        cand.append(root / 'models' / 'kd_best.pt')
    cand.extend(sorted(root.glob('run*/models/kd_best.pt'), key=lambda p: p.stat().st_mtime, reverse=True))
    if cand:
        best_model = cand[0]

print(str(best_model) if best_model else '')
PY
)"
fi

[[ -n "$MODEL" ]] || { echo "Cannot find KD model under $KD_OUT_ROOT"; exit 1; }
MODEL="$(realpath -m "$MODEL")"
[[ -f "$MODEL" ]] || { echo "Missing model: $MODEL"; exit 1; }

if [[ -z "$OUT" ]]; then
  OUT="$KD_OUT_ROOT/kd_fasta_predictions.csv"
fi
OUT="$(realpath -m "$OUT")"
mkdir -p "$(dirname "$OUT")"

echo "[KD-FASTA-INFER] model=$MODEL"
echo "[KD-FASTA-INFER] fasta=$FASTA"
echo "[KD-FASTA-INFER] out=$OUT"

conda run -n "$ENV_NAME" python predict/kd_fusion_predict_fasta.py \
  --model "$MODEL" \
  --fasta "$FASTA" \
  --protbert "$PROTBERT" \
  --esm-dim "$ESM_DIM" \
  --prot-dim "$PROT_DIM" \
  --esm-chunk-len "$ESM_CHUNK_LEN" \
  --protbert-max-len "$PROTBERT_MAX_LEN" \
  --out "$OUT"

echo "[KD-FASTA-INFER] done: $OUT"
