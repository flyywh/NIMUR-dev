# Tune + LoRA 统一使用说明（最新）

本说明基于当前仓库最新实现，推荐只使用以下统一入口：

- 非 LoRA 调参：`scripts/train_unified/tune.sh`
- LoRA 调参：`scripts/lora/tune_lora.sh`
- 全流程编排（可选）：`scripts/pipeline/pipeline.sh`

请在仓库根目录执行命令。

## 1. 快速总览

### 1.1 非 LoRA 调参入口

```bash
bash scripts/train_unified/tune.sh --target <name> [options]
```

可选 `target`：
- `esm2`
- `evo2`
- `oracle-kg`
- `ee`
- `kd-fusion`

### 1.2 LoRA 调参入口

```bash
bash scripts/lora/tune_lora.sh --target <name> [options]
```

可选 `target`：
- `esm2`
- `evo2`
- `evo2-once`
- `oracle`
- `oracle-cnn`
- `kd-fusion`
- `oracle-with-teachers`

### 1.3 全流程入口（可选）

```bash
bash scripts/pipeline/pipeline.sh --target <name> [options]
```

常用 `target`：
- `full-tune`（非 LoRA 全流程）
- `full-tune-lora`（LoRA 全流程）
- `smoke-existing`（用已有数据做小规模冒烟）

## 2. 非 LoRA 常用命令

### 2.1 ESM2

```bash
bash scripts/train_unified/tune.sh --target esm2 \
  --env evo2 \
  --gpus 0 \
  --dataset-dir data/esm2_tune \
  --test-dataset data/esm2_tune/test_dataset_esm2.pt \
  --out-root results_tune/esm2_small \
  --lr-list "1e-6 3e-6" \
  --batch-list "8" \
  --dropout-list "0.1" \
  --layer-list "4" \
  --epochs 3
```

### 2.2 Evo2

```bash
bash scripts/train_unified/tune.sh --target evo2 \
  --env evo2 \
  --gpus 0 \
  --train-dataset data/evo2_tune/train.pt \
  --test-dataset data/evo2_tune/test.pt \
  --out-root results_tune/evo2_small \
  --lr-list "1e-6 3e-6" \
  --batch-list "8" \
  --dropout-list "0.1" \
  --layer-list "4" \
  --epochs 3
```

### 2.3 ORACLE-KG

```bash
bash scripts/train_unified/tune.sh --target oracle-kg \
  --env evo2 \
  --gpus 0 \
  --train-dir data/kg_tune/train \
  --val-dir data/kg_tune/val \
  --protbert local_assets/data/ProtBERT/ProtBERT \
  --out-root results_tune/oracle_kg_small \
  --embed-list "128" \
  --lr-list "1e-4" \
  --batch-list "4" \
  --epochs-list "3" \
  --accum-list "1" \
  --ke-lambda-list "1.0"
```

## 3. LoRA 常用命令

### 3.1 ESM2 LoRA

```bash
bash scripts/lora/tune_lora.sh --target esm2 \
  --env evo2 \
  --gpus 0 \
  --dataset-dir data/esm2_tune_real \
  --test-dataset data/esm2_tune_real/test_dataset_esm2.pt \
  --out-root results_tune/esm2_real_lora_small \
  --lr-list "1e-6" \
  --batch-list "4" \
  --dropout-list "0.1" \
  --layer-list "4" \
  --epochs 2
```

### 3.2 Evo2 LoRA（完整 sweep）

```bash
bash scripts/lora/tune_lora.sh --target evo2 \
  --env evo2 \
  --gpus 0 \
  --train-dataset data/evo2_tune_real/train.pt \
  --test-dataset data/evo2_tune_real/test.pt \
  --out-root results_tune/evo2_real_lora_small \
  --lr-list "1e-6" \
  --batch-list "4" \
  --dropout-list "0.1" \
  --layer-list "4" \
  --epochs 2
```

### 3.3 Evo2 LoRA（单次快速）

```bash
bash scripts/lora/tune_lora.sh --target evo2-once \
  --env evo2 \
  --gpus 0 \
  --train-dataset data/evo2_tune_real/train_build/combined_dataset.pt \
  --test-dataset data/evo2_tune_real/test_build/combined_dataset.pt \
  --out-root results_tune/evo2_real_lora_once_small \
  --batch-size 1 \
  --epochs 1 \
  --max-seq-len 1024 \
  --use-dataparallel 0
```

### 3.4 ORACLE LoRA

```bash
bash scripts/lora/tune_lora.sh --target oracle \
  --env evo2 \
  --gpus 0 \
  --train-dir data/kg_tune_real/train \
  --val-dir data/kg_tune_real/val \
  --out-root results_tune/oracle_kg_real_lora_small \
  --embed-list "128" \
  --lr-list "1e-4" \
  --batch-list "2" \
  --epochs-list "2" \
  --accum-list "1" \
  --ke-lambda-list "1.0"
```

### 3.5 ORACLE LoRA + Teachers

```bash
bash scripts/lora/tune_lora.sh --target oracle-with-teachers \
  --env evo2 \
  --gpus 0 \
  --train-dir data/kg_tune_real/train \
  --val-dir data/kg_tune_real/val \
  --esm2-model <esm2_checkpoint_or_run_dir> \
  --evo2-model <evo2_checkpoint_or_run_dir> \
  --out-root results_tune/oracle_kg_real_lora_with_teachers
```

### 3.6 ORACLE-CNN LoRA

```bash
bash scripts/lora/tune_lora.sh --target oracle-cnn \
  --env evo2 \
  --gpus 0 \
  --train-dataset data/oracle_cnn/oracle_train_dataset.pt \
  --valid-dataset data/oracle_cnn/oracle_valid_dataset.pt \
  --out-root results_tune/oracle_cnn_lora_small \
  --lr-list "1e-4" \
  --batch-list "4" \
  --dropout-list "0.1" \
  --layer-list "2" \
  --epochs 5
```

### 3.7 KD-Fusion LoRA

```bash
bash scripts/lora/tune_lora.sh --target kd-fusion \
  --env evo2 \
  --gpus 0 \
  --dataset data/kd_fusion/kd_fusion_dataset.pt \
  --out-root results_tune/kd_fusion_lora_small \
  --lr-list "3e-4" \
  --batch-list "64" \
  --dropout-list "0.1" \
  --student-hidden-list "256" \
  --epochs 10
```

## 4. 与训练入口的关系

- `scripts/train_unified/train.py` 是统一训练运行器（单任务/链式任务）。
- `scripts/train_unified/tune.sh` / `scripts/lora/tune_lora.sh` 是更高层的网格调参包装。
- 推荐优先用两个 `tune*.sh` 做实验，再按需用 `train.py run ...` 精调单次任务。

## 5. 兼容脚本说明

旧编号脚本（例如 `02_tune_esm2_lora.sh`、`07_full_tune_pipeline.sh`）仍保留，但多数已转发到统一入口。新项目建议直接使用本说明中的统一命令，避免脚本分叉。
