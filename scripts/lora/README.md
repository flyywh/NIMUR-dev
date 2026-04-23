# LoRA Unified Entry

`scripts/lora` now uses one Python entry and one shell tuning entry:

- Unified Python trainer: `scripts/lora/lora.py`
- Unified shell tuner: `scripts/lora/tune_lora.sh`

## Python Usage

```bash
python scripts/lora/lora.py --model esm2   [esm2 args...]
python scripts/lora/lora.py --model evo2   [evo2 args...]
python scripts/lora/lora.py --model oracle [oracle args...]
python scripts/lora/lora.py --model oracle-cnn [oracle-cnn args...]
python scripts/lora/lora.py --model kd-fusion [kd-fusion args...]
```

## Shell Usage

```bash
bash scripts/lora/tune_lora.sh --target esm2 [args...]
bash scripts/lora/tune_lora.sh --target evo2 [args...]
bash scripts/lora/tune_lora.sh --target evo2-once [args...]
bash scripts/lora/tune_lora.sh --target oracle [args...]
bash scripts/lora/tune_lora.sh --target oracle-cnn [args...]
bash scripts/lora/tune_lora.sh --target kd-fusion [args...]
bash scripts/lora/tune_lora.sh --target oracle-with-teachers [args...]
```

## Compatibility Wrappers

Legacy files are kept for compatibility but only forward to unified entries:

- `esm2_train_lora.py` -> `lora.py --model esm2`
- `evo2_train_lora.py` -> `lora.py --model evo2`
- `oracle_train_lora.py` -> `lora.py --model oracle`
- `oracle_cnn_train_lora.py` -> `lora.py --model oracle-cnn`
- `kd_fusion_train_lora.py` -> `lora.py --model kd-fusion`
- `02_tune_esm2_lora.sh` -> `tune_lora.sh --target esm2`
- `04_tune_evo2_lora.sh` -> `tune_lora.sh --target evo2`
- `04_tune_evo2_lora_once.sh` -> `tune_lora.sh --target evo2-once`
- `06_tune_oracle_kg_lora.sh` -> `tune_lora.sh --target oracle`
- `11_tune_oracle_kg_lora_with_teachers.sh` -> `tune_lora.sh --target oracle-with-teachers`
