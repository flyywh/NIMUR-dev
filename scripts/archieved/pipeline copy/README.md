# Unified Pipeline Entry

`scripts/pipeline/pipeline.sh` is now the single implementation entrypoint.

## Recommended Usage

```bash
bash scripts/pipeline/pipeline.sh --target full-tune [args...]
bash scripts/pipeline/pipeline.sh --target full-tune-lora [args...]
bash scripts/pipeline/pipeline.sh --target smoke-existing [args...]
bash scripts/pipeline/pipeline.sh --target kd-track [args...]
bash scripts/pipeline/pipeline.sh --target kd-full [args...]
bash scripts/pipeline/pipeline.sh --target kd-train-only [args...]
bash scripts/pipeline/pipeline.sh --target infer-real-bgc [args...]
```

## Clear Name Wrappers

- `full_tune_pipeline.sh` -> `--target full-tune`
- `full_tune_lora_pipeline.sh` -> `--target full-tune-lora`
- `smoke_existing_pipeline.sh` -> `--target smoke-existing`
- `kd_fusion_track_pipeline.sh` -> `--target kd-track`
- `kd_fusion_pipeline.sh` -> `--target kd-full`
- `kd_fusion_train_only_pipeline.sh` -> `--target kd-train-only`
- `infer_real_bgc_pipeline.sh` -> `--target infer-real-bgc`

## Legacy Compatibility Wrappers

Legacy numbered scripts are retained and now forward to `pipeline.sh`:

- `07_full_tune_pipeline.sh` -> `--target full-tune`
- `08_full_tune_pipeline_lora.sh` -> `--target full-tune-lora`
- `09_smoke_existing_small.sh` -> `--target smoke-existing`
- `16_kd_fusion_track.sh` -> `--target kd-track`
- `17_kd_fusion_full_pipeline.sh` -> `--target kd-full`
- `17_kd_fusion_only_train_pipeline.sh` -> `--target kd-train-only`
