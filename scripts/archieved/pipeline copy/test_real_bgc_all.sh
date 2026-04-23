#!/usr/bin/env bash
set -euo pipefail

# Avoid OpenBLAS/OpenMP oversubscription warnings and potential hangs.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/test_real_bgc_all.sh [options]

Options:
  --genome-id <id>        Genome ID in test_real/antismash_test/01_genome (default: CRBC_G0002)
  --env <name>            Conda env name (default: evo2)
  --input-dir <dir>       Input genome directory (default: test_real/antismash_test/01_genome)
  --orf-faa <path>        Precomputed ORF protein FASTA. If set, skip gene2orf.
  --evo2-dna <path>       Precomputed ORF DNA FASTA (.ffn). If set, skip dna_cutter.
  --prodigal-dir <dir>    Optional dir to auto-discover <GenomeID>.ffn (default: test_real/antismash_test/02_prodigal)
  --output-root <dir>     Output root dir (default: results/test_real_bgc)
  --esm2-model <path>     ESM2 checkpoint path (default: local_assets/default_models/NIMROD-ESM)
  --esm2-config <path>    ESM2 config path (default: config/esm2_config.json)
  --evo2-model <path>     Evo2 checkpoint path (default: local_assets/default_models/NIMROD-EVO)
  --evo2-config <path>    Evo2 config path (default: config/evo2_config.json)
  --oracle-model <path>   ORACLE checkpoint path (default: local_assets/default_models/ORACLE)
  --protbert-path <path>  ProtBERT model dir (default: local_assets/data/ProtBERT/ProtBERT)
  --device <cpu|cuda>     Device for ORACLE KG (default: cuda)
  -h, --help              Show this help

Notes:
  1. This script always runs gene2orf first.
  2. If a model file/path is missing, that stage is skipped with a warning.
EOF
}

GENOME_ID="CRBC_G0002"
ENV_NAME="evo2"
INPUT_DIR="test_real/antismash_test/01_genome"
OUTPUT_ROOT="results/test_real_bgc"
ORF_FAA_INPUT=""
EVO2_DNA_INPUT=""
PRODIGAL_DIR="test_real/antismash_test/02_prodigal"
ESM2_MODEL="local_assets/default_models/NIMROD-ESM"
ESM2_CONFIG="config/esm2_config.json"
EVO2_MODEL="local_assets/default_models/NIMROD-EVO"
EVO2_CONFIG="config/evo2_config.json"
ORACLE_MODEL="local_assets/default_models/ORACLE"
PROTBERT_PATH="local_assets/data/ProtBERT/ProtBERT"
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --genome-id) GENOME_ID="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --orf-faa) ORF_FAA_INPUT="$2"; shift 2 ;;
    --evo2-dna) EVO2_DNA_INPUT="$2"; shift 2 ;;
    --prodigal-dir) PRODIGAL_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --esm2-model) ESM2_MODEL="$2"; shift 2 ;;
    --esm2-config) ESM2_CONFIG="$2"; shift 2 ;;
    --evo2-model) EVO2_MODEL="$2"; shift 2 ;;
    --evo2-config) EVO2_CONFIG="$2"; shift 2 ;;
    --oracle-model) ORACLE_MODEL="$2"; shift 2 ;;
    --protbert-path) PROTBERT_PATH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

GENOME_FNA="${INPUT_DIR}/${GENOME_ID}.fna"
if [[ ! -f "$GENOME_FNA" ]]; then
  echo "ERROR: genome file not found: $GENOME_FNA" >&2
  exit 1
fi

if [[ ! -f "$ESM2_CONFIG" ]]; then
  echo "ERROR: missing config: $ESM2_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$EVO2_CONFIG" ]]; then
  echo "ERROR: missing config: $EVO2_CONFIG" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/${GENOME_ID}_${STAMP}"
ORF_FAA="${RUN_DIR}/orf.faa"
ORF_FAA_CLEAN="${RUN_DIR}/orf.clean.faa"
EVO2_INPUT_FNA="${RUN_DIR}/evo2_input.fna"

mkdir -p "${RUN_DIR}/esm2" "${RUN_DIR}/evo2" "${RUN_DIR}/kg"

if [[ -n "$ORF_FAA_INPUT" ]]; then
  if [[ ! -f "$ORF_FAA_INPUT" ]]; then
    echo "ERROR: --orf-faa file not found: $ORF_FAA_INPUT" >&2
    exit 1
  fi
  echo "[1/4] Use provided ORF FASTA"
  cp "$ORF_FAA_INPUT" "$ORF_FAA"
else
  echo "[1/4] ORF prediction (gene2orf)"
  conda run -n "$ENV_NAME" python pipeline/gene2orf.py "$GENOME_FNA" "$ORF_FAA"
fi

echo "[1.5/4] Clean ORF sequences for model compatibility"
conda run -n "$ENV_NAME" python -c "from Bio import SeqIO; allowed=set('ACDEFGHIKLMNPQRSTVWYBXZJUO'); inp=r'$ORF_FAA'; out=r'$ORF_FAA_CLEAN'; records=[]; \
from Bio.Seq import Seq; \
[records.append((lambda r:(setattr(r,'seq',Seq(''.join(ch if ch in allowed else 'X' for ch in str(r.seq).upper()))),r)[1])(rec)) for rec in SeqIO.parse(inp,'fasta')]; \
SeqIO.write(records,out,'fasta'); print(f'cleaned_orf={out} records={len(records)}')"
if [[ ! -f "$ORF_FAA_CLEAN" ]]; then
  echo "ERROR: failed to generate cleaned ORF FASTA: $ORF_FAA_CLEAN" >&2
  exit 1
fi

if [[ -f "$ESM2_MODEL" ]]; then
  echo "[2/4] ESM2 BGC prediction"
  conda run -n "$ENV_NAME" python predict/esm2_predict.py \
    --config "$ESM2_CONFIG" \
    --model "$ESM2_MODEL" \
    --fasta "$ORF_FAA_CLEAN" \
    --out "${RUN_DIR}/esm2"
else
  echo "[2/4] SKIP ESM2: missing model $ESM2_MODEL"
fi

if [[ -f "$EVO2_MODEL" ]]; then
  echo "[3/4] EVO2 BGC prediction"
  if [[ -n "$EVO2_DNA_INPUT" ]]; then
    if [[ ! -f "$EVO2_DNA_INPUT" ]]; then
      echo "ERROR: --evo2-dna file not found: $EVO2_DNA_INPUT" >&2
      exit 1
    fi
    echo "[3.0/4] Use provided EVO2 DNA FASTA"
    cp "$EVO2_DNA_INPUT" "$EVO2_INPUT_FNA"
  elif [[ -f "${PRODIGAL_DIR}/${GENOME_ID}.ffn" ]]; then
    echo "[3.0/4] Use existing Prodigal DNA FASTA: ${PRODIGAL_DIR}/${GENOME_ID}.ffn"
    cp "${PRODIGAL_DIR}/${GENOME_ID}.ffn" "$EVO2_INPUT_FNA"
  else
    echo "[3.0/4] Build EVO2 DNA input from ORF + genome (dna_cutter)"
    conda run -n "$ENV_NAME" python pipeline/dna_cutter.py \
      --orf "$ORF_FAA" \
      --genome "$GENOME_FNA" \
      --output "$EVO2_INPUT_FNA"
  fi

  if [[ ! -f "$EVO2_INPUT_FNA" ]]; then
    echo "ERROR: failed to generate EVO2 input fasta: $EVO2_INPUT_FNA" >&2
    exit 1
  fi

  conda run -n "$ENV_NAME" python predict/evo_predict.py \
    --fasta "$EVO2_INPUT_FNA" \
    --out "${RUN_DIR}/evo2" \
    --model "$EVO2_MODEL" \
    --config "$EVO2_CONFIG"
else
  echo "[3/4] SKIP EVO2: missing model $EVO2_MODEL"
fi

if [[ -f "$ORACLE_MODEL" && -d "$PROTBERT_PATH" ]]; then
  echo "[4/4] ORACLE Knowledge Graph prediction"
  conda run -n "$ENV_NAME" python predict/oracle_predict.py \
    --model_ckpt "$ORACLE_MODEL" \
    --protbert_path "$PROTBERT_PATH" \
    --fasta "$ORF_FAA_CLEAN" \
    --out "${RUN_DIR}/kg" \
    --device "$DEVICE"
else
  echo "[4/4] SKIP KG: missing ORACLE model or ProtBERT path"
  echo "       model: $ORACLE_MODEL"
  echo "       protbert: $PROTBERT_PATH"
fi

echo
echo "Done. Run directory: $RUN_DIR"
echo "  ORF:  $ORF_FAA"
echo "  ESM2: ${RUN_DIR}/esm2/predictions.csv"
echo "  EVO2: ${RUN_DIR}/evo2/predictions.csv"
echo "  KG:   ${RUN_DIR}/kg/knowledge_graph_results/{classifications.csv,distances.csv}"
