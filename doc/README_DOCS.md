# NIMUR: Multi-Modal Deep Learning Framework for Biosynthetic Gene Cluster Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

  
NIMUR (Neural Integrated Multi-Representation) is a multi-modal deep learning framework for **biosynthetic gene cluster (BGC) prediction**.  
It integrates **protein embeddings**, **DNA embeddings**, and **knowledge-graph modeling** into a unified pipeline for **6-class BGC classification**, genome-scale scanning, and efficient downstream deployment.

---

## Highlights

- **Multi-modal representation learning**
  - Protein embeddings: **ESM2**, **ProtBERT**
  - DNA embeddings: **Evo2**, **AlphaGenome**
  - Knowledge modeling: **ORACLE / TransE-style KG branch**

- **Three training paradigms**
  - **CNN classifier** for fast baselines
  - **ORACLE-KG** for class-relation modeling
  - **KD-Fusion** for best overall performance

- **Flexible usage**
  - Single-sequence prediction
  - Genome-wide BGC scanning
  - (LoRA) fine-tuning

- **Target classes**
  - NRPS
  - PKS
  - Ribosomal
  - Saccharide
  - Terpene
  - Other

---

## Overview

Biosynthetic gene cluster prediction benefits from integrating complementary biological signals:

- **Protein-level features** capture domain composition and sequence semantics.
- **DNA-level features** provide long-range genomic context.
- **Knowledge-graph modeling** captures relationships between proteins, BGC types, and structured biological priors.

NIMUR combines these signals in a unified framework and supports both **standalone classifiers** and **fusion-based models**.

---

## Architecture

```text
Input FASTA / Genome / KG Triples
│
├── Protein branch
│   ├── ESM2 (1280d)
│   └── ProtBERT (1024d)
│
├── DNA branch
│   ├── Evo2 (1920d)
│   └── AlphaGenome (1536d)
│
├── KG branch
│   └── ORACLE / TransE-style relation modeling
│
└── Feature Fusion
    └── [ESM2 + ProtBERT + Evo2 + AlphaGenome] = 5760d
        │
        ├── CNN Classifier
        ├── KD Student
        └── KG Model
             │
             └── 6-class BGC prediction
```

### Core design

1. **Input processing**
   - Parse FASTA / genome sequences
   - Separate protein and DNA inputs
   - Prepare KG triplets for ORACLE training

2. **Embedding extraction**
   - **ESM2**: protein embedding (1280d)
   - **ProtBERT**: protein embedding (1024d)
   - **Evo2**: DNA embedding (1920d)
   - **AlphaGenome**: DNA embedding (1536d)

3. **Fusion**
   - Concatenate multi-modal embeddings into a **5760d fused representation**
   - Optional pooling / normalization for stable training

4. **Prediction**
   - CNN branch for efficient supervised classification
   - KG branch for structured type modeling
   - KD branch for teacher-student compression and deployment

---

## Quick Start

### 1. Environment setup

```bash
git clone https://github.com/flyywh/nimur.git
cd nimur

conda create -n nimur python=3.10 -y
conda activate nimur

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fair-esm
pip install evo2
```

### 2. Prepare data

```bash
python scripts/data_prep/split_mibig_bgc_fasta.py \
    --input data/MiBiG_formatted_BGC.fasta \
    --out-dir data/mibig_splits \
    --train-ratio 0.8 \
    --valid-ratio 0.1 \
    --test-ratio 0.1
```

### 3. Build embeddings

```bash
python scripts/data_prep/build_esm_protbert_dataset.py \
    --train-pos-faa data/train_pos.fasta \
    --train-neg-faa data/train_neg.fasta \
    --val-pos-faa data/valid_pos.fasta \
    --val-neg-faa data/valid_neg.fasta \
    --protbert local_assets/data/ProtBERT/ProtBERT \
    --out data/esm_protbert_features.pt
```

### 4. Train recommended model (KD-Fusion)

```bash
bash scripts/pipeline/pipeline.sh \
    --target kd-full \
    --env nimur \
    --gpus 0,1,2,3 \
    --train-pos data/train_pos.fasta \
    --train-neg data/train_neg.fasta \
    --valid-pos data/valid_pos.fasta \
    --valid-neg data/valid_neg.fasta \
    --out-root results_tune/kd_fusion
```

### 5. Run prediction

```bash
python predict/predict.py \
    --model results_tune/kd_fusion/run0/models/kd_best.pt \
    --fasta input/sequences.fasta \
    --out predictions.csv
```

---

## Training Modes

### 1. CNN Classifier
Best for quick iteration and strong supervised baselines.

```bash
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
    --device auto

CUDA_VISIBLE_DEVICES=4 python scripts/trainers/ORACLE_CNN_train.py \
    --config config/oracle_cnn_config.json \
    --train_dataset data/cnn_dataset_multilabel/train \
    --valid_dataset data/cnn_dataset_multilabel/val \
    --output_dir results_tune/cnn_multilabel
```

### 2. ORACLE-KG
Best for modeling structured class relations and type semantics.

```bash
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
```

### 3. KD-Fusion
Best overall performance and recommended for deployment.

```bash
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
    --device auto

CUDA_VISIBLE_DEVICES=4 python scripts/trainers/kd_fusion_train.py \
    --config config/kd_fusion_config.json \
    --dataset data/kd_fusion/kd_dataset_multilabel.pt \
    --outdir results_tune/kd_fusion_multilabel
```

---

## Inference

### Single-sequence prediction

```bash
CUDA_VISIBLE_DEVICES=4 python predict/oracle_cnn_predict.py \
    --config config/oracle_cnn_config.json \
    --model results_tune/cnn_multilabel/model_epoch_7.pt \
    --fasta test_real/antismash_test/01_genome/CRBC_G0001.fna \
    --protbert local_assets/data/ProtBERT/ProtBERT \
    --out results_tune/cnn_multilabel/fasta_predictions.csv

CUDA_VISIBLE_DEVICES=6,7 python predict/kd_fusion_predict_fasta.py \
    --model results_tune/kd_fusion_multilabel/models/kd_best.pt \
    --fasta test_real/antismash_test/01_genome/CRBC_G0001.fna \
    --protbert local_assets/data/ProtBERT/ProtBERT \
    --out results_tune/kd_fusion_multilabel/fasta_predictions.csv \
    --scorer student \
    --esm-dim 1280 \
    --prot-dim 1024

CUDA_VISIBLE_DEVICES=4 python predict/oracle_predict.py \
    --device cuda \
    --model results_tune/oracle_kg_multilabel/best.pt \
    --fasta test_real/antismash_test/01_genome/CRBC_G0001.fna \
    --protbert-path local_assets/data/ProtBERT/ProtBERT \
    --out results_tune/oracle_kg_multilabel/fasta_rankings.csv \
    --top-k 5
```

### Genome-wide BGC detection

```bash
CUDA_VISIBLE_DEVICES=4 python predict/cnn_genome_detect.py \
    --config config/oracle_cnn_config.json \
    --model results_tune/cnn_multilabel/model_epoch_7.pt \
    --genome-fna test_real/antismash_test/01_genome/CRBC_G0001.fna \
    --protbert local_assets/data/ProtBERT/ProtBERT \
    --out-dir results_tune/cnn_multilabel/genome_detect_CRBC_G0001

CUDA_VISIBLE_DEVICES=4 python predict/kd_genome_detect.py \
    --model results_tune/kd_fusion_multilabel/models/kd_best.pt \
    --genome-fna test_real/antismash_test/01_genome/CRBC_G0001.fna \
    --protbert local_assets/data/ProtBERT/ProtBERT \
    --out-dir results_tune/kd_fusion_multilabel/genome_detect_CRBC_G0001

CUDA_VISIBLE_DEVICES=4 python predict/oracle_genome_detect.py \
    --device cuda \
    --model results_tune/oracle_kg_multilabel/best.pt \
    --genome-fna test_real/antismash_test/01_genome/CRBC_G0001.fna \
    --protbert-path local_assets/data/ProtBERT/ProtBERT \
    --out-dir results_tune/oracle_kg_multilabel/genome_detect_CRBC_G0001
```

**Outputs**
- `window_predictions.csv`: per-window prediction scores
- `bgc_regions.csv`: merged candidate BGC regions
- `summary.json`: scan statistics

---

## LoRA Fine-tuning

For parameter-efficient adaptation:

```bash
bash scripts/lora/tune_lora.sh \
    --target esm2 \
    --dataset-dir data/esm2_tune \
    --out-root results_tune/esm2_lora \
    --lora-rank 16 \
    --lora-alpha 32
```

Other supported targets include:
- `evo2`
- `oracle`

---

## Embedding Backbones

| Model | Modality | Dimension | Notes |
|------|----------|-----------|------|
| ESM2 | Protein | 1280 | Strong protein semantic representation |
| ProtBERT | Protein | 1024 | Good local sequence pattern modeling |
| Evo2 | DNA | 1920 | Long-context genomic modeling |
| AlphaGenome | DNA | 1536 | Genome-level contextual embeddings |

### Resource note
- **Evo2** may require **~28GB VRAM**
- For limited GPU memory, reduce `max_len` during embedding extraction

---

## Performance

TBD

**Key findings**
- KD-Fusion achieves the best overall performance
- ESM2 is particularly strong for protein-dominant classes such as NRPS and PKS
- Evo2 contributes long-range DNA context for genomic prediction
- ORACLE-KG improves relation-aware type modeling

---

## Project Structure

```text
nimur/
├── scripts/
│   ├── data_prep/
│   ├── trainers/
│   ├── lora/
│   ├── pipeline/
│   ├── tune/
│   └── inference_report/
├── utils/
│   └── embed_extract/
├── predict/
├── config/
├── data/
├── results_tune/
├── local_assets/
├── code_ref/
└── README.md
```

---

## Recommended Workflow

1. Start with **ESM2 + CNN** for fast validation  
2. Add **ORACLE-KG** if class relations matter  
3. Use **KD-Fusion** as the default production model  
4. Use **LoRA** for efficient downstream adaptation  
5. Use **genome-wide detection** for real-world scanning tasks  

---

## Troubleshooting

### Evo2 out-of-memory
Reduce the maximum context length:
```python
embedder = Evo2Embedder(max_len=8192)
```

### ProtBERT checkpoint not found
Ensure local weights are correctly placed:
```text
local_assets/data/ProtBERT/ProtBERT
```

### Slow embedding extraction
- Use batch processing
- Enable caching if available
- Precompute embeddings before fusion training

---

## Acknowledgements

- ESM2 — Meta AI
- Evo2 — Arc Institute
- ProtBERT — Rost Lab
- AlphaGenome — DeepMind
- MiBiG Database
- Pyrodigal

---

## Citation

TBD
---

## Contact
- Email: yangwh@pcl.ac.cn
---
