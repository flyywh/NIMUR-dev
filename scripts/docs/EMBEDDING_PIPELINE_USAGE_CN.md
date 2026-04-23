# Embedding 扩展与训练用法（当前主链）

本文说明当前主链下，如何组织 `ESM2 / ProtBERT / Evo2 / AlphaGenome` 四类 embedding，并用于三种模型：

- `cnn`：`oracle-cnn`
- `kd`：`kd-fusion`
- `oracle`：KG / ORACLE

说明：

- 本文只描述当前主链，不依赖 `scripts/archieved/`
- `AlphaGenome` 当前支持作为**预提取 cache** 参与 `cnn / kd / oracle`
- `AlphaGenome` **暂不支持**在线 backbone + LoRA 训练

## 1. 两阶段概览

### 阶段 1：统一样本与 embedding cache

入口：

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract esm2,protbert,evo2
```

如果已有外部 AlphaGenome `.npz` 目录：

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract esm2,protbert,evo2,alphagenome \
  --alphagenome-dir /path/to/ag_npz_dir
```

输出：

- `samples.fasta`
- `labels.tsv`
- `stage1_manifest.json`
- `cache/esm2`
- `cache/protbert`
- `cache/evo2`
- `cache/alphagenome`（如果提供）

### 阶段 2：按模型打包

入口：

```bash
python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets cnn,kd,oracle \
  --cnn-embedding esm2 \
  --kd-embeddings esm2,protbert \
  --oracle-embedding protbert \
  --output-dir data/prep_stage2 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42
```

输出：

- `kg/`：KGE train/val 目录
- `cnn/`：`oracle-cnn` 数据集
- `kd/`：`kd_fusion_dataset.pt`
- `stage2_summary.json`

## 2. 模型与 embedding 的推荐组合

### CNN

推荐：

- `esm2`
- `evo2`
- `alphagenome`（如果已有外部 cache）

不推荐首版：

- `protbert`

原因：

- `oracle-cnn` 当前最自然地吃 token-level 表征
- `protbert` 当前缓存为 pooled vector，更适合 `kd / oracle`

### KD

推荐：

- `esm2,protbert`
- `esm2,protbert,evo2`
- `alphagenome` 可作为额外 teacher cache 加入

### Oracle

当前支持两条路：

- cache-backed：`--embedding_cache_dir`
- online backbone：`--backbone protbert|esm2|evo2`

AlphaGenome 当前建议：

- 用作 `--embedding_cache_dir`
- 不要作为在线 backbone

## 3. 数据准备命令模板

### 3.1 CNN + ESM2

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract esm2

python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets cnn \
  --cnn-embedding esm2 \
  --output-dir data/prep_stage2_cnn_esm2
```

### 3.2 CNN + Evo2

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract evo2

python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets cnn \
  --cnn-embedding evo2 \
  --output-dir data/prep_stage2_cnn_evo2
```

### 3.3 CNN + AlphaGenome

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract alphagenome \
  --alphagenome-dir /path/to/ag_npz_dir

python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets cnn \
  --cnn-embedding alphagenome \
  --output-dir data/prep_stage2_cnn_ag
```

### 3.4 KD + 多 teacher

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract esm2,protbert,evo2

python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets kd \
  --kd-embeddings esm2,protbert,evo2 \
  --output-dir data/prep_stage2_kd
```

### 3.5 Oracle + ProtBERT cache

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract protbert

python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets oracle \
  --oracle-embedding protbert \
  --output-dir data/prep_stage2_oracle_protbert
```

### 3.6 Oracle + ESM2 cache

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract esm2

python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets oracle \
  --oracle-embedding esm2 \
  --output-dir data/prep_stage2_oracle_esm2
```

### 3.7 Oracle + Evo2 cache

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract evo2

python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets oracle \
  --oracle-embedding evo2 \
  --output-dir data/prep_stage2_oracle_evo2
```

### 3.8 Oracle + AlphaGenome cache

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract alphagenome \
  --alphagenome-dir /path/to/ag_npz_dir

python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets oracle \
  --oracle-embedding alphagenome \
  --output-dir data/prep_stage2_oracle_ag
```

## 4. LoRA 训练命令模板

### 4.1 CNN 训练

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model oracle-cnn \
  --config tmp/oracle_cnn_train.json \
  --train_dataset data/prep_stage2_cnn_esm2/oracle_cnn/oracle_train_dataset.pt \
  --valid_dataset data/prep_stage2_cnn_esm2/oracle_cnn/oracle_valid_dataset.pt \
  --output_dir results_tune/oracle_cnn_run \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

### 4.2 KD 训练（teacher 全是 cache）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model kd-fusion \
  --config tmp/kd_fusion_train.json \
  --dataset data/prep_stage2_kd/kd_fusion_dataset.pt \
  --output_dir results_tune/kd_fusion_run \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

### 4.3 KD 训练（online student）

Evo2 student：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model kd-fusion \
  --config tmp/kd_fusion_train.json \
  --dataset data/prep_stage2_kd/kd_fusion_dataset.pt \
  --output_dir results_tune/kd_fusion_evo2_student_run \
  --student_backbone evo2 \
  --max_seq_len 1024 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

ProtBERT student：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model kd-fusion \
  --config tmp/kd_fusion_train.json \
  --dataset data/prep_stage2_kd/kd_fusion_dataset.pt \
  --output_dir results_tune/kd_fusion_protbert_student_run \
  --student_backbone protbert \
  --student_backbone_path local_assets/data/ProtBERT/ProtBERT \
  --max_seq_len 1024 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

说明：

- `student_backbone=alphagenome` 当前不支持在线训练
- 如需 AlphaGenome，请改用 cache teacher

### 4.4 Oracle 训练（cache-backed）

ProtBERT cache：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model oracle \
  --train_dir data/prep_stage2_oracle_protbert/kg/train \
  --val_dir data/prep_stage2_oracle_protbert/kg/val \
  --embedding_cache_dir data/prep_stage1/cache/protbert \
  --output_dir results_tune/oracle_run \
  --embedding_dim 256 \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --lr 1e-4 \
  --ke_lambda 1.0 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

Evo2 cache：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model oracle \
  --train_dir data/prep_stage2_oracle_evo2/kg/train \
  --val_dir data/prep_stage2_oracle_evo2/kg/val \
  --embedding_cache_dir data/prep_stage1/cache/evo2 \
  --output_dir results_tune/oracle_evo2_cache_run \
  --embedding_dim 256 \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --lr 1e-4 \
  --ke_lambda 1.0 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

AlphaGenome cache：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model oracle \
  --train_dir data/prep_stage2_oracle_ag/kg/train \
  --val_dir data/prep_stage2_oracle_ag/kg/val \
  --embedding_cache_dir data/prep_stage1/cache/alphagenome \
  --output_dir results_tune/oracle_ag_cache_run \
  --embedding_dim 256 \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --lr 1e-4 \
  --ke_lambda 1.0 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

### 4.5 Oracle 训练（online backbone）

ProtBERT：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model oracle \
  --train_dir data/prep_stage2_oracle_protbert/kg/train \
  --val_dir data/prep_stage2_oracle_protbert/kg/val \
  --backbone protbert \
  --protbert_path local_assets/data/ProtBERT/ProtBERT \
  --output_dir results_tune/oracle_protbert_online_run \
  --embedding_dim 256 \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --lr 1e-4 \
  --max_seq_len 1024 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

ESM2：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model oracle \
  --train_dir data/prep_stage2_oracle_esm2/kg/train \
  --val_dir data/prep_stage2_oracle_esm2/kg/val \
  --backbone esm2 \
  --esm2_layer 33 \
  --output_dir results_tune/oracle_esm2_online_run \
  --embedding_dim 256 \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --lr 1e-4 \
  --max_seq_len 1024 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

Evo2：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n evo2 python scripts/lora/lora.py \
  --model oracle \
  --train_dir data/prep_stage2_oracle_evo2/kg/train \
  --val_dir data/prep_stage2_oracle_evo2/kg/val \
  --backbone evo2 \
  --evo2_layer_name blocks.21.mlp.l3 \
  --output_dir results_tune/oracle_evo2_online_run \
  --embedding_dim 256 \
  --epochs 5 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --lr 1e-4 \
  --max_seq_len 1024 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

说明：

- `--backbone alphagenome` 当前明确不支持
- 如果需要 AlphaGenome，请先走 cache-backed 训练

## 5. 当前边界

### 已支持

- `stage1` 统一抽取：
  - `esm2`
  - `protbert`
  - `evo2`
  - `alphagenome`（外部 `.npz` 导入）
- `stage2` 打包：
  - `cnn`
  - `kd`
  - `oracle`
- `oracle`：
  - cache-backed：任意 cache
  - online backbone：`protbert / esm2 / evo2`
- `kd`：
  - 多 teacher cache
  - online student：`protbert / esm2 / evo2`

### 未支持

- `alphagenome` 在线 backbone
- `cnn` 多 embedding 融合
- `oracle` 多 backbone 融合

## 6. 推荐顺序

如果你现在要快速试验：

1. `cnn + esm2`
2. `kd + esm2,protbert`
3. `oracle + protbert cache`
4. `oracle + evo2 online`
5. `cnn + evo2`
6. `oracle + alphagenome cache`

这样最容易先得到稳定可比较结果。
