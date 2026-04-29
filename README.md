# NIMUR


<img src="images/nimur.png" alt="index" width="600">

**NIMUR (Neural Integrated Multi-Representation)** is a multi-modal deep learning framework for **biosynthetic gene cluster (BGC) prediction**.  
It integrates **protein embeddings**, **DNA embeddings**, and **knowledge-graph modeling** for **multi-label BGC detection**, genome-scale scanning, and downstream deployment.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Features

- Multi-modal fusion of **ESM2**, **ProtBERT**, **Evo2**, and **AlphaGenome**
- Support for **CNN**, **knowledge-graph**, and **KD-fusion** training
- **7-label** targets: `NRPS`, `Other`, `PKS`, `Ribosomal`, `Saccharide`, `Terpene`, `neg`
- **Genome-wide scanning** for candidate BGC region detection
- **LoRA fine-tuning** for efficient adaptation

---

## Architecture

```text
Protein Sequences ──> ESM2 (1280d)
                  └─> ProtBERT (1024d)

DNA Sequences ─────> Evo2 (1920d)
                  └─> AlphaGenome (1536d)

KG Triples ────────> ORACLE KG

[ESM2 + ProtBERT + Evo2 + AlphaGenome] = 5760d
                      │
                      ├─> CNN Classifier
                      ├─> KD Student
                      └─> KG Model
                            │
                            └─> 6-class BGC prediction
```

---

## Installation

```bash
git clone https://github.com/flyywh/nimur.git
cd nimur

conda create -n nimur python=3.10 -y
conda activate nimur

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fair-esm
pip install evo2
```

---

## Quick Start

### CNN workflow

```bash
# 1) Data Gen
python scripts/data_prep/build_cnn_dataset.py \
  --train-pos-faa data_ar/mibig_splits/MiBiG_train_pos.fasta \
  --train-neg-faa data_ar/mibig_splits/MiBiG_train_neg.fasta \
  --val-pos-faa data_ar/mibig_splits/MiBiG_val_pos.fasta \
  --val-neg-faa data_ar/mibig_splits/MiBiG_val_neg.fasta \
  --test-pos-faa data_ar/mibig_splits/MiBiG_test_pos.fasta \
  --test-neg-faa data_ar/mibig_splits/MiBiG_test_neg.fasta \
  --split-index data_ar/mibig_splits/MiBiG_split_index.tsv \
  --cache-dir data/cache_esm_protbert \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --out-dir data/cnn_dataset_multilabel \
  --device auto \
  --esm-max-len 1022 \
  --max-seq-len 2400 \
  --chunk-size 30

# 2) Train
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=4 python scripts/trainers/ORACLE_CNN_train.py \
  --config config/oracle_cnn_config.json \
  --train_dataset data/cnn_dataset_multilabel/train \
  --valid_dataset data/cnn_dataset_multilabel/val \
  --output_dir results_tune/cnn_multilabel

# 3) Inference (direct FASTA)
CUDA_VISIBLE_DEVICES=4 python predict/oracle_cnn_predict.py \
  --config config/oracle_cnn_config.json \
  --model results_tune/cnn_multilabel/model_epoch_7.pt \
  --fasta test_real/antismash_test/01_genome/CRBC_G0001.fna \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --out results_tune/cnn_multilabel/fasta_predictions.csv \
  --esm-max-len 1022 \
  --max-seq-len 2400 \
  --batch-size 8 \
  --threshold 0.5

# 4) BGC Detection
CUDA_VISIBLE_DEVICES=4 CONDA_NO_PLUGINS=true conda run -n nimur_mh \
  python predict/cnn_genome_detect.py \
  --config config/oracle_cnn_config.json \
  --model results_tune/cnn_multilabel/model_epoch_7.pt \
  --genome-fna test_real/antismash_test/01_genome/CRBC_G0001.fna \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --out-dir results_tune/cnn_multilabel/genome_detect_CRBC_G0001 \
  --window 12000 \
  --step 3000 \
  --threshold 0.55 \
  --merge-gap 1500 \
  --esm-max-len 1022 \
  --max-seq-len 2400 \
  --batch-size 8
```

### KD workflow

```bash
# 1) Data Gen
CUDA_VISIBLE_DEVICES=4 python scripts/data_prep/build_esm_protbert_dataset.py \
  --train-pos-faa data_ar/mibig_splits/MiBiG_train_pos.fasta \
  --train-neg-faa data_ar/mibig_splits/MiBiG_train_neg.fasta \
  --val-pos-faa data_ar/mibig_splits/MiBiG_val_pos.fasta \
  --val-neg-faa data_ar/mibig_splits/MiBiG_val_neg.fasta \
  --test-pos-faa data_ar/mibig_splits/MiBiG_test_pos.fasta \
  --test-neg-faa data_ar/mibig_splits/MiBiG_test_neg.fasta \
  --split-index data_ar/mibig_splits/MiBiG_split_index.tsv \
  --cache-dir data/cache_esm_protbert \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --out data/kd_fusion/kd_dataset_multilabel.pt \
  --device auto \
  --esm-chunk-len 1022 \
  --protbert-max-len 2400

# 2) Train
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=4 python scripts/trainers/kd_fusion_train.py \
  --config config/kd_fusion_config.json \
  --dataset data/kd_fusion/kd_dataset_multilabel.pt \
  --outdir results_tune/kd_fusion_multilabel

# 3) Inference (direct FASTA)
CUDA_VISIBLE_DEVICES=6,7 python predict/kd_fusion_predict_fasta.py \
  --model results_tune/kd_fusion_multilabel/models/kd_best.pt \
  --fasta test_real/antismash_test/01_genome/CRBC_G0001.fna \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --out results_tune/kd_fusion_multilabel/fasta_predictions.csv \
  --scorer student \
  --esm-dim 1280 \
  --prot-dim 1024 \
  --esm-chunk-len 1022 \
  --protbert-max-len 2400

# 4) BGC Detection
CUDA_VISIBLE_DEVICES=4 python predict/kd_genome_detect.py \
  --model results_tune/kd_fusion_multilabel/models/kd_best.pt \
  --genome-fna test_real/antismash_test/01_genome/CRBC_G0001.fna \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --out-dir results_tune/kd_fusion_multilabel/genome_detect_CRBC_G0001 \
  --scorer student \
  --window 12000 \
  --step 3000 \
  --threshold 0.55 \
  --merge-gap 1500 \
  --esm-chunk-len 1022 \
  --protbert-max-len 2400 \
  --batch-size 8
```

### ORACLE-KG workflow

```bash
# 1) Data Gen + fused ESM2+ProtBERT cache
CUDA_VISIBLE_DEVICES=4 python scripts/data_prep/build_oracle_kg_from_split_index.py \
  --split-index data_ar/mibig_splits/MiBiG_split_index.tsv \
  --train-pos-fasta data_ar/mibig_splits/MiBiG_train_pos.fasta \
  --train-neg-fasta data_ar/mibig_splits/MiBiG_train_neg.fasta \
  --val-pos-fasta data_ar/mibig_splits/MiBiG_val_pos.fasta \
  --val-neg-fasta data_ar/mibig_splits/MiBiG_val_neg.fasta \
  --test-pos-fasta data_ar/mibig_splits/MiBiG_test_pos.fasta \
  --test-neg-fasta data_ar/mibig_splits/MiBiG_test_neg.fasta \
  --split-cache-dir data/cache_esm_protbert \
  --fused-cache-dir data/oracle_kg/fused_cache \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --output-dir data/oracle_kg

# 2) Train
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=4 python scripts/trainers/ORACLE_train.py \
  --train_dir data/oracle_kg/train \
  --val_dir data/oracle_kg/val \
  --embedding_cache_dir data/oracle_kg/fused_cache \
  --embedding_dim 256 \
  --batch_size 8 \
  --eval_batch_size 16 \
  --epochs 5 \
  --lr 2e-4 \
  --weight_decay 0.01 \
  --output_dir results_tune/oracle_kg_multilabel

# 3) Inference (direct FASTA, no cache required)
CUDA_VISIBLE_DEVICES=4 python predict/oracle_predict.py \
  --device cuda \
  --model results_tune/oracle_kg_multilabel/best.pt \
  --fasta test_real/antismash_test/01_genome/CRBC_G0001.fna \
  --protbert-path local_assets/data/ProtBERT/ProtBERT \
  --out results_tune/oracle_kg_multilabel/fasta_rankings.csv \
  --top-k 5 \
  --batch-size 1 \
  --esm-max-len 1022 \
  --protbert-max-len 2400

# 4) BGC Detection
CUDA_VISIBLE_DEVICES=4 python predict/oracle_genome_detect.py \
  --device cuda \
  --model results_tune/oracle_kg_multilabel/best.pt \
  --genome-fna test_real/antismash_test/01_genome/CRBC_G0001.fna \
  --protbert-path local_assets/data/ProtBERT/ProtBERT \
  --out-dir results_tune/oracle_kg_multilabel/genome_detect_CRBC_G0001 \
  --window 12000 \
  --step 3000 \
  --threshold 0.55 \
  --merge-gap 1500 \
  --top-k 5 \
  --esm-max-len 1022 \
  --protbert-max-len 2400 \
  --batch-size 1
```

---

## Inference Outputs

### FASTA inference

- `CNN`: multi-label probabilities in `fasta_predictions.csv`
- `KD`: multi-label probabilities in `fasta_predictions.csv`
- `ORACLE-KG`: type ranking in `fasta_rankings.csv`

### Genome-wide detection

- `window_predictions.csv`: per-window BGC score and top predicted type
- `bgc_regions.csv`: merged candidate BGC regions
- `summary.json`: scan statistics and output paths

---

## Backbones

| Model | Modality | Dimension |
|------|----------|-----------|
| ESM2 | Protein | 1280 |
| ProtBERT | Protein | 1024 |
| Evo2 | DNA | 1920 |
| AlphaGenome | DNA | 1536 |

> Note: Evo2 may require **~28GB VRAM** for long-context inference.

---

## Performance

TBD

---

## Project Structure

```text
nimur/
├── scripts/
├── utils/
├── predict/
├── config/
├── data/
├── results_tune/
├── local_assets/
└── README.md
```

---

## Citation

TBD

---
