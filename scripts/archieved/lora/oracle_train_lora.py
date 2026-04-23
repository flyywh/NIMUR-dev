import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.ORACLE_train as base
from scripts_train.lora_utils import apply_lora_by_regex, load_state_dict_flexible, mark_only_lora_trainable


def parse_args():
    p = argparse.ArgumentParser(description="ORACLE KG train with optional pretrained + LoRA")
    p.add_argument("--pretrained", default="")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-regex",
        default=(
            r"prot_encoder\.encoder\.encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|"
            r"attention\.output\.dense|intermediate\.dense|output\.dense)|prot_encoder\.proj"
        ),
    )
    p.add_argument("--freeze-base", action="store_true")

    # passthrough base args
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--pretrain_data_dir", default=None)
    p.add_argument("--protbert_path", required=True)
    p.add_argument("--protein_encoder_cls", default="bert")
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--ke_embedding_size", type=int, default=256)
    p.add_argument("--double_entity_embedding_size", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--logging_dir", default="./logs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--lr_scheduler_type", default="linear")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--unfreeze_encoder", action="store_true")
    p.add_argument("--num_negs", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--ke_lambda", type=float, default=1.0)
    p.add_argument("--ke_learning_rate", type=float, default=3e-5)
    p.add_argument("--ke_max_score", type=float, default=12.0)
    p.add_argument("--ke_score_fn", default="transE")
    p.add_argument("--ke_warmup_steps", type=int, default=400)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--deepspeed", default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--dataloader_pin_memory", action="store_true")
    p.add_argument("--optimize_memory", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--eval_sample_limit", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.eval_sample_limit is not None and args.eval_sample_limit < 0:
        args.eval_sample_limit = None

    orig_cls = base.TransEProteinType

    class PatchedTransE(orig_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)

            if args.pretrained:
                ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
                    ckpt = ckpt["model_state"]
                stats = load_state_dict_flexible(self, ckpt)
                print(f"[ORACLE-LORA] pretrained load: {stats}")

            replaced = apply_lora_by_regex(
                self,
                include_pattern=args.lora_regex,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
            print(f"[ORACLE-LORA] replaced linear layers with LoRA: {replaced}")
            if args.freeze_base:
                trainable, total = mark_only_lora_trainable(self, keep_regex=r"type_emb|rel")
                print(f"[ORACLE-LORA] trainable params: {trainable}/{total}")

    base.TransEProteinType = PatchedTransE
    orig_torch_save = torch.save

    def _safe_torch_save(obj, f, *a, **kw):
        # Avoid pickle failure for local patched class when ORACLE_train saves full model object.
        if isinstance(obj, PatchedTransE):
            obj = {"model_state": obj.state_dict(), "saved_as": "state_dict_fallback"}
        return orig_torch_save(obj, f, *a, **kw)

    torch.save = _safe_torch_save
    try:
        base.train(args)
    finally:
        base.TransEProteinType = orig_cls
        torch.save = orig_torch_save


if __name__ == "__main__":
    main()
