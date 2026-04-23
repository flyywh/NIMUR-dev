# AlphaGenome Cache 使用说明

当前主链下，`AlphaGenome` 只支持作为**外部预提取 cache** 参与：

- `cnn`
- `kd`
- `oracle`

当前**不支持**：

- `--backbone alphagenome`
- `--student_backbone alphagenome`

原因：

- 当前仓库没有本地可直接训练/前向的 AlphaGenome 模型实现
- `code_ref/alphagenome-main` 主要是 API SDK / client 参考代码，不是本地 backbone

## 1. 输入目录格式

`--alphagenome-dir` 指向一个目录，目录下包含若干 `.npz` 文件。

每个 `.npz` 文件至少应包含两个键：

- `seq_ids`
- `embeddings`

要求：

- `len(seq_ids) == len(embeddings)`
- `seq_ids[i]` 应与 stage1 样本 `sample_id` 一致
- 同一目录下，建议所有 embedding 的最后一维一致
- 同一目录下，建议 level 一致：
  - 全部 pooled：`[D]`
  - 或全部 token：`[L, D]`

当前主链会在 `stage1-mibig` 中自动检查：

- 是否有匹配样本
- level 是否混用
- 最后一维是否一致

如果有缺失样本，会在：

```text
<stage1>/cache/alphagenome/_missing_ids.txt
```

写出未匹配的 sample id。

## 2. 独立检查脚本

只检查 `.npz` 目录本身：

```bash
python scripts/tools/check_alphagenome_cache.py \
  --ag-dir /path/to/ag_npz_dir
```

对照已经标准化好的样本 FASTA 检查覆盖率：

```bash
python scripts/tools/check_alphagenome_cache.py \
  --ag-dir /path/to/ag_npz_dir \
  --sample-fasta data/prep_stage1/samples.fasta
```

对照原始 MiBiG FASTA 检查覆盖率：

```bash
python scripts/tools/check_alphagenome_cache.py \
  --ag-dir /path/to/ag_npz_dir \
  --mibig-fasta data/MiBiG_formatted_BGC.fasta
```

保存 JSON 报告：

```bash
python scripts/tools/check_alphagenome_cache.py \
  --ag-dir /path/to/ag_npz_dir \
  --mibig-fasta data/MiBiG_formatted_BGC.fasta \
  --out-json tmp/ag_check.json
```

## 3. stage1 接入

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract alphagenome \
  --alphagenome-dir /path/to/ag_npz_dir
```

如果要和其他 embedding 一起准备：

```bash
python scripts/data_prep/prepare_data.py stage1-mibig \
  --input-fasta data/MiBiG_formatted_BGC.fasta \
  --output-dir data/prep_stage1 \
  --extract esm2,protbert,evo2,alphagenome \
  --alphagenome-dir /path/to/ag_npz_dir \
  --gpus 0
```

## 4. stage2 打包

### CNN + AlphaGenome

```bash
python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets cnn \
  --cnn-embedding alphagenome \
  --output-dir data/prep_stage2_cnn_ag
```

### KD + AlphaGenome

```bash
python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets kd \
  --kd-embeddings esm2,protbert,alphagenome \
  --output-dir data/prep_stage2_kd_ag
```

### Oracle + AlphaGenome cache

```bash
python scripts/data_prep/prepare_data.py stage2-pack \
  --stage1-dir data/prep_stage1 \
  --targets oracle \
  --oracle-embedding alphagenome \
  --output-dir data/prep_stage2_oracle_ag
```

## 5. 训练

### Oracle（cache-backed）

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
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --freeze-base
```

### 不支持的在线写法

下面这种当前会明确报错：

```bash
python scripts/lora/lora.py \
  --model oracle \
  --backbone alphagenome \
  ...
```

同样，`kd-fusion --student_backbone alphagenome` 当前也不支持。
