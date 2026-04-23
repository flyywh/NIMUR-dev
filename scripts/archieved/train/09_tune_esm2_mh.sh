#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-evo2}"
GPUS="2"       # AG fusion 版默认单卡，避免 NCCL 多卡广播错误；如需多卡改为 "0,1" 等
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/tmp_smoke/mplconfig}"
export PYTHONPATH="$PWD/utils${PYTHONPATH:+:$PYTHONPATH}"
REPO_ROOT="$PWD"
DATASET_DIR="data/esm2_tune_real"
TEST_DATASET="data/esm2_tune_real/test_dataset_esm2.pt"
BASE_CONFIG="config/esm2_config.json"
OUT_ROOT="results_tune/esm2_ag"       # ← 新目录名，避免覆盖原版结果
AG_EMB_DIR="/mnt/netdisk2/evo2_explore/temp/nju-china-main-dev-0323/data/ag_embeddings"

LR_LIST="1e-7 3e-7 1e-6"
BATCH_LIST="8 16"
DROPOUT_LIST="0.1 0.2"
LAYER_LIST="4 6"
EPOCHS="64"

usage() {
  cat <<USAGE
Usage: bash scripts_train/02_tune_esm2.sh [options]

Options:
  --env ENV
  --dataset-dir DIR
  --test-dataset PATH
  --base-config PATH
  --out-root DIR
  --gpus 0,1
  --ag-emb-dir DIR      AlphaGenome embedding 目录（含 CRBC_G000X.npz）
  --no-ag               不传 --ag-emb-dir，退回纯 ESM2 模式
  --lr-list "1e-7 3e-7"
  --batch-list "8 16"
  --dropout-list "0.1 0.2"
  --layer-list "4 6"
  --epochs 64
USAGE
}

NO_AG=""   # 空=启用 AG；"1"=禁用

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)          ENV_NAME="$2";    shift 2 ;;
    --dataset-dir)  DATASET_DIR="$2"; shift 2 ;;
    --test-dataset) TEST_DATASET="$2";shift 2 ;;
    --base-config)  BASE_CONFIG="$2"; shift 2 ;;
    --out-root)     OUT_ROOT="$2";    shift 2 ;;
    --gpus)         GPUS="$2";        shift 2 ;;
    --ag-emb-dir)   AG_EMB_DIR="$2";  shift 2 ;;
    --no-ag)        NO_AG="1";        shift   ;;
    --lr-list)      LR_LIST="$2";     shift 2 ;;
    --batch-list)   BATCH_LIST="$2";  shift 2 ;;
    --dropout-list) DROPOUT_LIST="$2";shift 2 ;;
    --layer-list)   LAYER_LIST="$2";  shift 2 ;;
    --epochs)       EPOCHS="$2";      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[[ -n "$GPUS" ]] && { export CUDA_VISIBLE_DEVICES="$GPUS"; echo "Using GPUs: $GPUS"; }

DATASET_DIR="$(realpath -m "$DATASET_DIR")"
TEST_DATASET="$(realpath -m "$TEST_DATASET")"
BASE_CONFIG="$(realpath -m "$BASE_CONFIG")"
OUT_ROOT="$(realpath -m "$OUT_ROOT")"

[[ -f "$BASE_CONFIG" ]]                       || { echo "Missing config: $BASE_CONFIG"; exit 1; }
[[ -f "$TEST_DATASET" ]]                      || { echo "Missing test dataset: $TEST_DATASET"; exit 1; }
[[ -f "$DATASET_DIR/train_dataset_esm2.pt" ]] || { echo "Missing $DATASET_DIR/train_dataset_esm2.pt"; exit 1; }
[[ -f "$DATASET_DIR/valid_dataset_esm2.pt" ]] || { echo "Missing $DATASET_DIR/valid_dataset_esm2.pt"; exit 1; }

# AG embedding 路径检查
if [[ -z "$NO_AG" ]]; then
  [[ -d "$AG_EMB_DIR" ]] || { echo "Missing ag-emb-dir: $AG_EMB_DIR  (use --no-ag to disable)"; exit 1; }
  echo "[AG fusion] ENABLED  (ag-emb-dir: $AG_EMB_DIR)"
  AG_ARG="--ag-emb-dir $AG_EMB_DIR"
else
  echo "[AG fusion] DISABLED"
  AG_ARG=""
fi

mkdir -p "$OUT_ROOT/configs"

run_idx=0
for lr in $LR_LIST; do
  for bs in $BATCH_LIST; do
    for drop in $DROPOUT_LIST; do
      for nl in $LAYER_LIST; do
        run_idx=$((run_idx + 1))
        tag="run${run_idx}_lr${lr}_bs${bs}_dp${drop}_nl${nl}"
        tag="${tag//./p}"; tag="${tag//-/m}"
        cfg="$OUT_ROOT/configs/${tag}.json"
        outdir="$OUT_ROOT/$tag"
        mkdir -p "$outdir/rocfigs" "$outdir/test_rocfigs" \
                 "$outdir/valid_test_distributions" "$outdir/models"

        python scripts_train/make_config.py \
          --base "$BASE_CONFIG" \
          --out  "$cfg" \
          --set "training.lr=$lr" \
          --set "training.batch_size=$bs" \
          --set "training.num_epochs=$EPOCHS" \
          --set "model.dropout=$drop" \
          --set "model.num_layers=$nl"

        echo "[ESM2-AG] $tag"
        (
          cd "$outdir"
          conda run -n "$ENV_NAME" python "$REPO_ROOT/scripts/esm2_train_mh.py" \
            --config           "$cfg" \
            --dataset_dir      "$DATASET_DIR" \
            --test_dataset_path "$TEST_DATASET" \
            --output_dir       "$outdir" \
            $AG_ARG
        )
      done
    done
  done
done

echo "ESM2-AG tuning done. Output: $OUT_ROOT"
