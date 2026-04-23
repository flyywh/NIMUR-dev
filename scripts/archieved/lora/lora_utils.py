import re
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap nn.Linear with a trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


@dataclass
class ReplaceStats:
    replaced: int = 0


def _replace_linear_modules(
    module: nn.Module,
    matcher: Callable[[str], bool],
    r: int,
    alpha: int,
    dropout: float,
    prefix: str = "",
    stats: ReplaceStats | None = None,
):
    if stats is None:
        stats = ReplaceStats()

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and matcher(full_name):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            stats.replaced += 1
        else:
            _replace_linear_modules(child, matcher, r, alpha, dropout, full_name, stats)
    return stats


def apply_lora_by_regex(
    model: nn.Module,
    include_pattern: str,
    r: int,
    alpha: int,
    dropout: float,
) -> int:
    rx = re.compile(include_pattern)
    stats = _replace_linear_modules(model, lambda n: rx.search(n) is not None, r, alpha, dropout)
    return stats.replaced


def apply_lora_to_named_targets(
    model: nn.Module,
    targets: str,
    r: int,
    alpha: int,
    dropout: float,
) -> int:
    """
    Targets for ESM2/Evo2 classifiers:
    - attention: reserved (not applied to torch MultiheadAttention out_proj)
    - ffn: linear1/linear2
    - classifier: classifier heads
    - all: union of above
    """
    selected = {x.strip() for x in targets.split(",") if x.strip()}
    if "all" in selected:
        selected = {"attention", "ffn", "classifier"}

    def matcher(name: str) -> bool:
        if "attention" in selected and (
            "attention.self.query" in name or "attention.self.key" in name or "attention.self.value" in name
        ):
            return True
        if "ffn" in selected and (".linear1" in name or ".linear2" in name):
            return True
        if "classifier" in selected and ".classifier." in f".{name}.":
            return True
        return False

    stats = _replace_linear_modules(model, matcher, r, alpha, dropout)
    return stats.replaced


def mark_only_lora_trainable(model: nn.Module, keep_regex: str = "") -> Tuple[int, int]:
    keep = re.compile(keep_regex) if keep_regex else None
    total = trainable = 0
    for n, p in model.named_parameters():
        total += p.numel()
        allow = ("lora_A" in n or "lora_B" in n)
        if keep is not None and keep.search(n):
            allow = True
        p.requires_grad = allow
        if allow:
            trainable += p.numel()
    return trainable, total


def load_state_dict_flexible(model: nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    model_state = model.state_dict()
    filtered = {}
    skipped_shape = 0
    for k, v in state.items():
        if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped_shape += 1
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "skipped_shape": skipped_shape,
    }
