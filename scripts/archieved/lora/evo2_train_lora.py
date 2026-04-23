import argparse
import json
import sys
import __main__
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.evo2_train as base
from scripts_train.lora_utils import (
    apply_lora_to_named_targets,
    load_state_dict_flexible,
    mark_only_lora_trainable,
)

def parse_args():
    p = argparse.ArgumentParser(description="Evo2 train with optional pretrained + LoRA")
    p.add_argument("--config", required=True)
    p.add_argument("--train_dataset", required=True)
    p.add_argument("--test_dataset", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pretrained", default="")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="attention,ffn,classifier")
    p.add_argument("--freeze-base", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    # Compatibility for torch.load of datasets serialized from script __main__ context.
    __main__.DnaDataset = base.DnaDataset

    orig_cls = base.Evo2Esm2EncodedTransformer

    class PatchedModel(orig_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if args.pretrained:
                state = torch.load(args.pretrained, map_location="cpu", weights_only=False)
                if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
                    state = state["model_state"]
                stats = load_state_dict_flexible(self, state)
                print(f"[EVO2-LORA] pretrained load: {stats}")

            replaced = apply_lora_to_named_targets(
                self,
                targets=args.lora_targets,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
            print(f"[EVO2-LORA] replaced linear layers with LoRA: {replaced}")
            if args.freeze_base:
                trainable, total = mark_only_lora_trainable(self, keep_regex=r"classifier")
                print(f"[EVO2-LORA] trainable params: {trainable}/{total}")

    base.Evo2Esm2EncodedTransformer = PatchedModel
    try:
        base.train(config, args.train_dataset, args.test_dataset, Path(args.output_dir))
    finally:
        base.Evo2Esm2EncodedTransformer = orig_cls


if __name__ == "__main__":
    main()
