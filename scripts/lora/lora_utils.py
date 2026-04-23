from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap an nn.Linear with a trainable low-rank adapter."""

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


class LoRAConv1d(nn.Module):
    """Conv1d LoRA adapter using a 1x1 down projection and a Conv1d up projection."""

    def __init__(self, base: nn.Conv1d, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Conv1d(base.in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv1d(
            r,
            base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


@dataclass
class ReplaceStats:
    replaced: int = 0


def _normalize_module_types(module_types: Iterable[str]) -> Tuple[type[nn.Module], ...]:
    allowed: list[type[nn.Module]] = []
    for kind in module_types:
        normalized = kind.lower().strip()
        if normalized == "linear":
            allowed.append(nn.Linear)
        elif normalized == "conv1d":
            allowed.append(nn.Conv1d)
        else:
            raise ValueError(f"Unsupported LoRA module type: {kind}")
    return tuple(allowed)


def _wrap_with_lora(module: nn.Module, r: int, alpha: int, dropout: float) -> nn.Module:
    if isinstance(module, nn.Linear):
        return LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
    if isinstance(module, nn.Conv1d):
        return LoRAConv1d(module, r=r, alpha=alpha, dropout=dropout)
    raise TypeError(f"Unsupported module for LoRA replacement: {type(module)!r}")


def _replace_modules(
    module: nn.Module,
    matcher: Callable[[str], bool],
    r: int,
    alpha: int,
    dropout: float,
    module_types: Tuple[type[nn.Module], ...],
    prefix: str = "",
    stats: ReplaceStats | None = None,
) -> ReplaceStats:
    if stats is None:
        stats = ReplaceStats()

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, module_types) and matcher(full_name):
            setattr(module, name, _wrap_with_lora(child, r=r, alpha=alpha, dropout=dropout))
            stats.replaced += 1
            continue
        _replace_modules(child, matcher, r, alpha, dropout, module_types, full_name, stats)
    return stats


def apply_lora_by_regex(
    model: nn.Module,
    include_pattern: str,
    r: int,
    alpha: int,
    dropout: float,
    module_types: Iterable[str] = ("linear",),
) -> int:
    rx = re.compile(include_pattern)
    allowed = _normalize_module_types(module_types)
    stats = _replace_modules(model, lambda n: rx.search(n) is not None, r, alpha, dropout, allowed)
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
    - attention: reserved for HF-style attention query/key/value modules
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

    stats = _replace_modules(model, matcher, r, alpha, dropout, (nn.Linear,))
    return stats.replaced


def mark_only_lora_trainable(model: nn.Module, keep_regex: str = "") -> Tuple[int, int]:
    keep = re.compile(keep_regex) if keep_regex else None
    total = trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        allow = ("lora_A" in name or "lora_B" in name)
        if keep is not None and keep.search(name):
            allow = True
        param.requires_grad = allow
        if allow:
            trainable += param.numel()
    return trainable, total


def load_state_dict_flexible(model: nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    model_state = model.state_dict()
    filtered = {}
    skipped_shape = 0
    remapped_module = 0

    for key, value in state.items():
        target_key = key
        if key not in model_state and key.startswith("module.") and key[7:] in model_state:
            target_key = key[7:]
            remapped_module += 1
        if target_key in model_state and hasattr(value, "shape") and model_state[target_key].shape == value.shape:
            filtered[target_key] = value
        else:
            skipped_shape += 1

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "skipped_shape": skipped_shape,
        "remapped_module_prefix": remapped_module,
    }
