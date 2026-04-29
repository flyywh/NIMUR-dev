#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../.. && pwd)
cd $REPO_ROOT || exit 1

usage() { echo "Usage: --target cnn-embed-extract|cnn-train|cnn-full"; }

run_cnn_embed_extract() {
  local ENV="${ENV_NAME:-nimur_mh}" GPUS="0" TRAIN_DS="" VAL_DS=""
  local OUT_DIR=data/cnn_dataset CACHE DEV=auto MAXLEN=9600

  while [[ $# -gt 0 ]]; do case $1 in
    --env) ENV=$2;shift 2;; --gpus) GPUS=$2;shift 2;;
    --train-pos) TP=$2;shift 2;; --train-neg) TN=$2;shift 2;;
    --val-pos) VP=$2;shift 2;; --val-neg) VN=$2;shift 2;;
    --test-pos) TEP=$2;shift 2;; --test-neg) TEN=$2;shift 2;;
    --out-dir) OUT_DIR=$2;shift 2;; --cache-dir) CACHE=$2;shift 2;;
    --device) DEV=$2;shift 2;; --max-seq-len) MAXLEN=$2;shift 2;;
    *) echo Unknown:$1;exit 1;; esac; done

  for f in $TP $TN $VP $VN $TEP $TEN; do [[ -f $f ]]||exit 1; done
  export CUDA_VISIBLE_DEVICES=$GPUS
  mkdir -p $OUT_DIR
  conda run -n $ENV python scripts/data_prep/build_esm2_cnn_dataset.py
    --train-pos-faa $TP --train-neg-faa $TN --val-pos-faa $VP --val-neg-faa $VN
    --test-pos-faa $TEP --test-neg-faa $TEN --out-dir $OUT_DIR --device $DEV --max-seq-len $MAXLEN
    ${CACHE:+--cache-dir $CACHE}
  echo [CNN-EMBED] Done: $OUT_DIR
}

run_cnn_train() {
  local ENV="${ENV_NAME:-nimur_mh}" GPUS="0" TRAIN_DS="" VAL_DS=""
  local OUT_DIR=results_tune/cnn CF=config/oracle_cnn_config.json

  while [[ $# -gt 0 ]]; do case $1 in
    --env) ENV=$2;shift 2;; --gpus) GPUS=$2;shift 2;;
    --train-dataset) TRAIN_DS=$2;shift 2;; --val-dataset) VAL_DS=$2;shift 2;;
    --out-dir) OUT_DIR=$2;shift 2;; --config) CF=$2;shift 2;;
    *) echo Unknown:$1;exit 1;; esac; done

  [[ -n $TRAIN_DS ]]||{ echo Missing:--train-dataset;exit 1; }
  [[ -n $VAL_DS ]]||{ echo Missing:--val-dataset;exit 1; }
  [[ -f $TRAIN_DS ]]||{ echo Missing:$TRAIN_DS;exit 1; }
  [[ -f $VAL_DS ]]||{ echo Missing:$VAL_DS;exit 1; }
  [[ -f $CF ]]||{ echo Missing:$CF;exit 1; }
  export CUDA_VISIBLE_DEVICES=$GPUS
  mkdir -p $OUT_DIR
  conda run -n $ENV python scripts/trainers/ORACLE_CNN_train.py
    --config $CF --train_dataset $TRAIN_DS --valid_dataset $VAL_DS --output_dir $OUT_DIR
  echo [CNN-TRAIN] Done: $OUT_DIR
}

run_cnn_full() {
  local ENV="${ENV_NAME:-nimur_mh}" GPUS="0" TRAIN_DS="" VAL_DS=""
  local OUT_DIR=data/cnn_dataset CF=config/oracle_cnn_config.json RES_DIR=results_tune/cnn
  local CACHE DEV=auto MAXLEN=9600 SKIP=0

  while [[ $# -gt 0 ]]; do case $1 in
    --env) ENV=$2;shift 2;; --gpus) GPUS=$2;shift 2;;
    --train-pos) TP=$2;shift 2;; --train-neg) TN=$2;shift 2;;
    --val-pos) VP=$2;shift 2;; --val-neg) VN=$2;shift 2;;
    --test-pos) TEP=$2;shift 2;; --test-neg) TEN=$2;shift 2;;
    --out-dir) OUT_DIR=$2;shift 2;; --config) CF=$2;shift 2;; --results-dir) RES_DIR=$2;shift 2;;
    --cache-dir) CACHE=$2;shift 2;; --device) DEV=$2;shift 2;; --max-seq-len) MAXLEN=$2;shift 2;;
    --skip-embed) SKIP=1;shift;;
    *) echo Unknown:$1;exit 1;; esac; done

  if [[ $SKIP == 0 ]]; then
    for f in $TP $TN $VP $VN $TEP $TEN; do [[ -f $f ]]||exit 1; done
    run_cnn_embed_extract --env $ENV --gpus $GPUS
      --train-pos $TP --train-neg $TN --val-pos $VP --val-neg $VN
      --test-pos $TEP --test-neg $TEN --out-dir $OUT_DIR --max-seq-len $MAXLEN --device $DEV
      ${CACHE:+--cache-dir $CACHE}
  else
    [[ -d $OUT_DIR ]]||{ echo Missing:$OUT_DIR;exit 1; }
    echo [CNN-FULL] Skip embed
  fi

  local TRAIN_DS=$OUT_DIR/train_dataset.pt
  local VAL_DS=$OUT_DIR/val_dataset.pt
  run_cnn_train --env $ENV --gpus $GPUS --train-dataset $TRAIN_DS --val-dataset $VAL_DS
    --out-dir $RES_DIR --config $CF
  echo [CNN-FULL] Done
}

parse_common() { local t; while [[ $# -gt 0 ]];do case $1 in --target) t=$2;shift 2;;*)break;;esac;done;echo --parsed-target $t $@; }

for a in "$@";do [[ $a == -h || $a == --help ]]&&{ usage;exit 0;};done

PARSED=$(parse_common "$@")
set -- $PARSED
[[ $1 == --parsed-target ]]&&shift
T=${1:-};shift
[[ -n $T ]]||{ echo --target required;usage;exit 1; }

case "$T" in
  cnn-embed-extract) run_cnn_embed_extract "$@";;
  cnn-train) run_cnn_train "$@";;
  cnn-full) run_cnn_full "$@";;
  *) echo Unknown:$T;usage;exit 1;;
esac
