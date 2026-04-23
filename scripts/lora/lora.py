#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import __main__
from datetime import datetime
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lora.lora_utils import (  # noqa: E402
    apply_lora_by_regex,
    apply_lora_to_named_targets,
    load_state_dict_flexible,
    mark_only_lora_trainable,
)


class ClassifierDataset:
    """
    Compatibility shim for torch.load of pickled oracle-cnn datasets
    saved as __main__.ClassifierDataset.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


__main__.ClassifierDataset = ClassifierDataset


class KDFeatureIdDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, ids: list[str]):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.ids[idx]


class OnlineBackboneStudent(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.net = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def _encode_sequences(
        self,
        seqs: list[str],
        *,
        device: torch.device,
        max_len: int,
        tokenizer=None,
        tokenize_fn=None,
    ) -> torch.Tensor:
        if hasattr(self.encoder, "encode_sequences"):
            return self.encoder.encode_sequences(seqs, device=str(device), max_len=max_len)
        if tokenizer is None or tokenize_fn is None:
            raise ValueError("ProtBERT online student requires tokenizer and tokenize_fn")
        all_input_ids, all_attn_masks = tokenize_fn(tokenizer, seqs, max_len=max_len, device=str(device))
        flat_input_ids = torch.cat(all_input_ids, dim=0)
        flat_attn_masks = torch.cat(all_attn_masks, dim=0)
        flat_vecs = self.encoder(flat_input_ids, flat_attn_masks)
        chunk_counts = [x.size(0) for x in all_input_ids]
        h_vecs = torch.split(flat_vecs, chunk_counts, dim=0)
        return torch.stack([c.mean(dim=0) for c in h_vecs], dim=0)

    def forward_sequences(
        self,
        seqs: list[str],
        *,
        device: torch.device,
        max_len: int,
        tokenizer=None,
        tokenize_fn=None,
    ) -> torch.Tensor:
        feats = self._encode_sequences(
            seqs,
            device=device,
            max_len=max_len,
            tokenizer=tokenizer,
            tokenize_fn=tokenize_fn,
        )
        return self.net(feats).squeeze(-1)


def _status(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LORA][{ts}] {message}", flush=True)


def _fmt_path(path: str) -> str:
    if not path:
        return "<empty>"
    try:
        return str(Path(path).resolve())
    except Exception:
        return path


def _validate_online_backbone(name: str, value: str, *, allow_empty: bool = False) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        if allow_empty:
            return ""
        raise ValueError(f"{name} is required")
    if normalized in {"protbert", "esm2", "evo2"}:
        return normalized
    if normalized == "alphagenome":
        raise ValueError(
            f"{name}=alphagenome is not supported for online training yet. "
            "Use stage1 alphagenome cache plus cache-backed training instead."
        )
    raise ValueError(f"Unsupported {name}: {value}. Expected one of protbert, esm2, evo2.")


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _looks_like_state_dict(obj: object) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(torch.is_tensor(v) for v in obj.values())


def _extract_model_state(ckpt: object) -> dict[str, torch.Tensor] | None:
    if _looks_like_state_dict(ckpt):
        return ckpt
    if isinstance(ckpt, dict):
        nested = ckpt.get("model_state")
        if _looks_like_state_dict(nested):
            return nested
        nested = ckpt.get("state_dict")
        if _looks_like_state_dict(nested):
            return nested
    return None


def _parse_selected(raw: str, all_values: set[str]) -> set[str]:
    selected = {x.strip() for x in raw.split(",") if x.strip()}
    if "all" in selected:
        return set(all_values)
    return selected


def _apply_oracle_cnn_lora(model: torch.nn.Module, args: argparse.Namespace) -> int:
    selected = _parse_selected(args.lora_cnn_targets, {"input", "depthwise", "pointwise", "se", "classifier"})
    patterns: list[str] = []
    if "input" in selected:
        patterns.append(r"(^|\.)input_proj$")
    if "depthwise" in selected:
        patterns.append(r"(^|\.)depthwise$")
    if "pointwise" in selected:
        patterns.append(r"\.pointwise\.")
    if "se" in selected:
        patterns.append(r"\.se\.")
    if "classifier" in selected:
        patterns.append(r"\.classifier\.")
    if not patterns:
        return 0
    return apply_lora_by_regex(
        model,
        include_pattern="|".join(patterns),
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        module_types=("linear", "conv1d"),
    )


def _apply_kd_lora(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[int, int]:
    selected = _parse_selected(args.lora_kd_targets, {"teacher", "student"})
    teacher_replaced = 0
    student_replaced = 0
    if "teacher" in selected:
        teacher_replaced = apply_lora_by_regex(
            teacher,
            include_pattern=r".*",
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    if "student" in selected:
        student_replaced = apply_lora_by_regex(
            student,
            include_pattern=r".*",
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    return teacher_replaced, student_replaced


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified LoRA trainer")
    p.add_argument("--model", choices=["esm2", "evo2", "oracle", "oracle-cnn", "kd-fusion"], required=True)

    # Common LoRA args
    p.add_argument("--pretrained", default="")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--freeze-base", action="store_true")

    # ESM2/EVO2 targets
    p.add_argument("--lora-targets", default="attention,ffn,classifier")

    # ORACLE regex target mode
    p.add_argument(
        "--lora-regex",
        default=(
            r"prot_encoder\.encoder\.encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|"
            r"attention\.output\.dense|intermediate\.dense|output\.dense)|prot_encoder\.proj"
        ),
    )

    # ORACLE-CNN targets
    p.add_argument("--lora-cnn-targets", default="all")

    # KD-FUSION targets
    p.add_argument("--lora-kd-targets", default="all")

    # ESM2 args
    p.add_argument("--config", default="")
    p.add_argument("--dataset_dir", default="")
    p.add_argument("--test_dataset_path", default="")
    p.add_argument("--output_dir", "--outdir", dest="output_dir", default="")

    # EVO2 args
    p.add_argument("--train_dataset", default="")
    p.add_argument("--test_dataset", default="")

    # ORACLE-CNN args
    p.add_argument("--valid_dataset", default="")

    # KD-FUSION args
    p.add_argument("--dataset", default="")
    p.add_argument("--student_backbone", default="")
    p.add_argument("--student_backbone_path", default="")
    p.add_argument("--student_esm2_layer", type=int, default=33)
    p.add_argument("--student_evo2_layer_name", default="blocks.21.mlp.l3")

    # ORACLE args (passthrough)
    p.add_argument("--train_dir", default="")
    p.add_argument("--val_dir", default="")
    p.add_argument("--pretrain_data_dir", default=None)
    p.add_argument("--protbert_path", default="")
    p.add_argument("--backbone", default="protbert")
    p.add_argument("--esm2_layer", type=int, default=33)
    p.add_argument("--evo2_layer_name", default="blocks.21.mlp.l3")
    p.add_argument("--embedding_cache_dir", default="")
    p.add_argument("--protbert_cache_dir", default="")
    p.add_argument("--protein_encoder_cls", default="bert")
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--ke_embedding_size", type=int, default=256)
    p.add_argument("--double_entity_embedding_size", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--do_train", action="store_true")
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


def run_esm2(args: argparse.Namespace) -> int:
    if not all([args.config, args.dataset_dir, args.test_dataset_path, args.output_dir]):
        raise ValueError("ESM2 requires --config --dataset_dir --test_dataset_path --output_dir")
    _status("esm2: loading trainer module")
    base = _load_module("trainer_esm2_train", REPO_ROOT / "scripts" / "trainers" / "esm2_train.py")
    __main__.ProteinDataset = base.ProteinDataset

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    _status(
        "esm2: config loaded "
        f"(batch={config.get('training', {}).get('batch_size')}, "
        f"epochs={config.get('training', {}).get('num_epochs')}, "
        f"lr={config.get('training', {}).get('lr')})"
    )
    _status(
        "esm2: inputs "
        f"dataset_dir={_fmt_path(args.dataset_dir)} "
        f"test_dataset={_fmt_path(args.test_dataset_path)} "
        f"output_dir={_fmt_path(args.output_dir)}"
    )

    orig_cls = base.BgcTransformer

    class PatchedBgcTransformer(orig_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if args.pretrained:
                state = torch.load(args.pretrained, map_location="cpu", weights_only=False)
                model_state = _extract_model_state(state)
                stats = load_state_dict_flexible(self, model_state if model_state is not None else state)
                print(f"[ESM2-LORA] pretrained load: {stats}")
            replaced = apply_lora_to_named_targets(
                self,
                targets=args.lora_targets,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
            print(f"[ESM2-LORA] replaced linear layers with LoRA: {replaced}")
            if args.freeze_base:
                trainable, total = mark_only_lora_trainable(self, keep_regex=r"classifier|pos_encoder")
                print(f"[ESM2-LORA] trainable params: {trainable}/{total}")

    base.BgcTransformer = PatchedBgcTransformer
    try:
        _status("esm2: entering trainer")
        base.train_model(config, Path(args.dataset_dir), Path(args.test_dataset_path), Path(args.output_dir))
    finally:
        base.BgcTransformer = orig_cls
    _status("esm2: trainer finished")
    return 0


def run_evo2(args: argparse.Namespace) -> int:
    if not all([args.config, args.train_dataset, args.test_dataset, args.output_dir]):
        raise ValueError("EVO2 requires --config --train_dataset --test_dataset --output_dir")
    _status("evo2: loading trainer module")
    base = _load_module("trainer_evo2_train", REPO_ROOT / "scripts" / "trainers" / "evo2_train.py")
    __main__.DnaDataset = base.DnaDataset

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    _status(
        "evo2: config loaded "
        f"(batch={config.get('training', {}).get('batch_size')}, "
        f"epochs={config.get('training', {}).get('num_epochs')}, "
        f"lr={config.get('training', {}).get('lr')})"
    )
    _status(
        "evo2: inputs "
        f"train_dataset={_fmt_path(args.train_dataset)} "
        f"test_dataset={_fmt_path(args.test_dataset)} "
        f"output_dir={_fmt_path(args.output_dir)}"
    )

    orig_cls = base.Evo2Esm2EncodedTransformer

    class PatchedModel(orig_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if args.pretrained:
                state = torch.load(args.pretrained, map_location="cpu", weights_only=False)
                model_state = _extract_model_state(state)
                stats = load_state_dict_flexible(self, model_state if model_state is not None else state)
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
        _status("evo2: entering trainer")
        base.train(config, args.train_dataset, args.test_dataset, Path(args.output_dir))
    finally:
        base.Evo2Esm2EncodedTransformer = orig_cls
    _status("evo2: trainer finished")
    return 0


def run_oracle(args: argparse.Namespace) -> int:
    required = [args.train_dir, args.val_dir, args.output_dir]
    if not all(required):
        raise ValueError("ORACLE requires --train_dir --val_dir --output_dir")
    protbert_path = getattr(args, "protbert_path", "")
    backbone = _validate_online_backbone("backbone", getattr(args, "backbone", "protbert"))
    embedding_cache_dir = getattr(args, "embedding_cache_dir", "")
    protbert_cache_dir = getattr(args, "protbert_cache_dir", "")
    if backbone == "protbert" and not (protbert_path or protbert_cache_dir or embedding_cache_dir):
        raise ValueError("ORACLE with backbone=protbert requires one of --protbert_path, --protbert_cache_dir, or --embedding_cache_dir")
    _status("oracle: loading trainer module")
    base = _load_module("trainer_oracle_train", REPO_ROOT / "scripts" / "trainers" / "ORACLE_train.py")
    if args.eval_sample_limit is not None and args.eval_sample_limit < 0:
        args.eval_sample_limit = None
    _status(
        "oracle: inputs "
        f"train_dir={_fmt_path(args.train_dir)} "
        f"val_dir={_fmt_path(args.val_dir)} "
        f"backbone={backbone} "
        f"protbert_path={_fmt_path(protbert_path)} "
        f"embedding_cache_dir={_fmt_path(embedding_cache_dir)} "
        f"protbert_cache_dir={_fmt_path(protbert_cache_dir)} "
        f"output_dir={_fmt_path(args.output_dir)}"
    )
    _status(
        "oracle: hyperparams "
        f"embed_dim={args.embedding_dim} epochs={args.epochs} "
        f"batch={args.batch_size} eval_batch={args.eval_batch_size} lr={args.lr}"
    )

    orig_cls = base.TransEProteinType

    class PatchedTransE(orig_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if args.pretrained:
                ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
                model_state = _extract_model_state(ckpt)
                stats = load_state_dict_flexible(self, model_state if model_state is not None else ckpt)
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
        if isinstance(obj, PatchedTransE):
            obj = {"model_state": obj.state_dict(), "saved_as": "state_dict_fallback"}
        return orig_torch_save(obj, f, *a, **kw)

    torch.save = _safe_torch_save
    try:
        _status("oracle: entering trainer")
        base.train(args)
    finally:
        base.TransEProteinType = orig_cls
        torch.save = orig_torch_save
    _status("oracle: trainer finished")
    return 0


def run_oracle_cnn(args: argparse.Namespace) -> int:
    if not all([args.config, args.train_dataset, args.valid_dataset, args.output_dir]):
        raise ValueError("ORACLE-CNN requires --config --train_dataset --valid_dataset --output_dir")
    _status("oracle-cnn: loading trainer module")
    base = _load_module(
        "trainer_oracle_cnn_train",
        REPO_ROOT / "scripts" / "trainers" / "ORACLE_CNN_train.py",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    _status(
        "oracle-cnn: config loaded "
        f"(batch={config.get('training', {}).get('batch_size')}, "
        f"epochs={config.get('training', {}).get('num_epochs')}, "
        f"lr={config.get('training', {}).get('lr')}, "
        f"layers={config.get('model', {}).get('num_layers')}, "
        f"seq_len={config.get('model', {}).get('seq_len')})"
    )
    _status(
        "oracle-cnn: inputs "
        f"train_dataset={_fmt_path(args.train_dataset)} "
        f"valid_dataset={_fmt_path(args.valid_dataset)} "
        f"output_dir={_fmt_path(args.output_dir)}"
    )
    _status(
        "oracle-cnn: lora "
        f"r={args.lora_r} alpha={args.lora_alpha} "
        f"dropout={args.lora_dropout} freeze_base={args.freeze_base} "
        f"targets={args.lora_cnn_targets}"
    )

    orig_cls = base.CNNClassifierModel

    class PatchedCNNClassifierModel(orig_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if args.pretrained:
                state = torch.load(args.pretrained, map_location="cpu", weights_only=False)
                model_state = _extract_model_state(state)
                stats = load_state_dict_flexible(self, model_state if model_state is not None else state)
                print(f"[ORACLE-CNN-LORA] pretrained load: {stats}")
            replaced = _apply_oracle_cnn_lora(self, args)
            print(f"[ORACLE-CNN-LORA] replaced modules with LoRA: {replaced}")
            if args.freeze_base:
                trainable, total = mark_only_lora_trainable(self, keep_regex=r"classifier|pos_embed")
                print(f"[ORACLE-CNN-LORA] trainable params: {trainable}/{total}")

    base.CNNClassifierModel = PatchedCNNClassifierModel
    try:
        _status("oracle-cnn: entering trainer")
        base.train(config, Path(args.train_dataset), Path(args.valid_dataset), Path(args.output_dir))
    finally:
        base.CNNClassifierModel = orig_cls
    _status("oracle-cnn: trainer finished")
    return 0


def run_kd_fusion(args: argparse.Namespace) -> int:
    if not all([args.config, args.dataset, args.output_dir]):
        raise ValueError("KD-FUSION requires --config --dataset --output_dir")
    _status("kd-fusion: loading trainer module")
    base = _load_module(
        "trainer_kd_fusion_train",
        REPO_ROOT / "scripts" / "trainers" / "kd_fusion_train.py",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _status(
        "kd-fusion: config loaded "
        f"(batch={cfg.get('training', {}).get('batch_size')}, "
        f"epochs={cfg.get('training', {}).get('num_epochs')}, "
        f"lr={cfg.get('training', {}).get('lr')}, "
        f"student_hidden={cfg.get('model', {}).get('student_hidden')})"
    )
    _status(
        "kd-fusion: inputs "
        f"dataset={_fmt_path(args.dataset)} "
        f"output_dir={_fmt_path(args.output_dir)}"
    )

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    base.set_seed(int(tcfg.get("seed", 42)))

    obj = torch.load(args.dataset, map_location="cpu", weights_only=False)
    Xtr, ytr = obj["train"]["X"], obj["train"]["y"]
    Xva, yva = obj["val"]["X"], obj["val"]["y"]
    Xte, yte = obj["test"]["X"], obj["test"]["y"]
    seq_by_id = obj.get("meta", {}).get("seq_by_id", {}) if isinstance(obj, dict) else {}
    _status(
        "kd-fusion: dataset loaded "
        f"train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)}"
    )

    in_dim = Xtr.shape[1]
    teacher = base.TeacherNet(in_dim, int(mcfg["teacher_hidden"]), float(mcfg["dropout"]))
    oracle_base = None
    student_tokenizer = None
    student_backbone_raw = getattr(args, "student_backbone", "")
    online_student = bool(student_backbone_raw)
    _status(
        "kd-fusion: student mode "
        + (
            f"online backbone={student_backbone_raw}"
            if online_student
            else "cached-feature mlp"
        )
    )
    if online_student:
        if not seq_by_id:
            raise ValueError("KD-FUSION online student requires dataset.meta.seq_by_id")
        oracle_base = _load_module(
            "trainer_oracle_for_kd",
            REPO_ROOT / "scripts" / "trainers" / "ORACLE_train.py",
        )
        student_hidden = int(mcfg["student_hidden"])
        student_backbone = _validate_online_backbone("student_backbone", student_backbone_raw)
        if student_backbone == "protbert":
            backbone_path = args.student_backbone_path or args.protbert_path
            if not backbone_path:
                raise ValueError("KD-FUSION student_backbone=protbert requires --student_backbone_path or --protbert_path")
            encoder = oracle_base.ProteinEncoder(
                model_path=backbone_path,
                proj_dim=student_hidden,
                freeze=not args.unfreeze_encoder,
            )
            student_tokenizer = oracle_base.AutoTokenizer.from_pretrained(
                backbone_path,
                local_files_only=True,
                do_lower_case=False,
            )
        elif student_backbone == "esm2":
            encoder = oracle_base.ESM2ProteinEncoder(
                proj_dim=student_hidden,
                layer=args.student_esm2_layer,
                freeze=not args.unfreeze_encoder,
            )
        elif student_backbone == "evo2":
            encoder = oracle_base.Evo2ProteinEncoder(
                proj_dim=student_hidden,
                layer_name=args.student_evo2_layer_name,
                max_len=args.max_seq_len,
                freeze=not args.unfreeze_encoder,
            )
        else:
            raise ValueError(f"Unsupported KD student_backbone: {student_backbone}")
        student = OnlineBackboneStudent(encoder, hidden_dim=student_hidden, dropout=float(mcfg["dropout"]))
    else:
        student = base.StudentNet(in_dim, int(mcfg["student_hidden"]), float(mcfg["dropout"]))

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        teacher_state = ckpt.get("teacher_state") if isinstance(ckpt, dict) else None
        student_state = ckpt.get("student_state") if isinstance(ckpt, dict) else None
        if _looks_like_state_dict(teacher_state):
            teacher_stats = load_state_dict_flexible(teacher, teacher_state)
            print(f"[KD-FUSION-LORA] teacher pretrained load: {teacher_stats}")
        if _looks_like_state_dict(student_state):
            student_stats = load_state_dict_flexible(student, student_state)
            print(f"[KD-FUSION-LORA] student pretrained load: {student_stats}")
        elif _looks_like_state_dict(ckpt):
            student_stats = load_state_dict_flexible(student, ckpt)
            print(f"[KD-FUSION-LORA] student pretrained load: {student_stats}")

    teacher_replaced, student_replaced = _apply_kd_lora(teacher, student, args)
    print(f"[KD-FUSION-LORA] teacher replaced linear layers: {teacher_replaced}")
    print(f"[KD-FUSION-LORA] student replaced linear layers: {student_replaced}")

    if args.freeze_base:
        teacher_trainable, teacher_total = mark_only_lora_trainable(teacher)
        student_trainable, student_total = mark_only_lora_trainable(student)
        print(
            "[KD-FUSION-LORA] trainable params: "
            f"teacher={teacher_trainable}/{teacher_total}, student={student_trainable}/{student_total}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    student = student.to(device)

    batch_size = int(tcfg["batch_size"])
    if online_student:
        tr_loader = DataLoader(KDFeatureIdDataset(Xtr, ytr, obj["train"]["ids"]), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(KDFeatureIdDataset(Xva, yva, obj["val"]["ids"]), batch_size=batch_size, shuffle=False)
        te_loader = DataLoader(KDFeatureIdDataset(Xte, yte, obj["test"]["ids"]), batch_size=batch_size, shuffle=False)
    else:
        tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
        te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    params = [p for p in list(teacher.parameters()) + list(student.parameters()) if p.requires_grad]
    if not params:
        raise ValueError("KD-FUSION LoRA produced no trainable parameters. Check --lora-kd-targets or --freeze-base.")

    opt = torch.optim.AdamW(
        params,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )

    alpha = float(tcfg.get("alpha", 0.5))
    temperature = float(tcfg.get("temperature", 2.0))
    teacher_w = float(tcfg.get("teacher_loss_weight", 0.5))
    epochs = int(tcfg["num_epochs"])
    bce = torch.nn.BCEWithLogitsLoss()

    outdir = Path(args.output_dir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    history: list[dict[str, float | int]] = []
    _status("kd-fusion: entering training loop")

    for ep in range(1, epochs + 1):
        teacher.train()
        student.train()
        ep_loss = 0.0

        for batch in tr_loader:
            if online_student:
                xb, yb, batch_ids = batch
            else:
                xb, yb = batch
                batch_ids = None
            xb = xb.to(device)
            yb = yb.to(device)

            tlog = teacher(xb)
            if online_student:
                seqs = [seq_by_id[sid] for sid in batch_ids]
                slog = student.forward_sequences(
                    seqs,
                    device=device,
                    max_len=args.max_seq_len,
                    tokenizer=student_tokenizer,
                    tokenize_fn=oracle_base.tokenize_sequences if oracle_base is not None else None,
                )
            else:
                slog = student(xb)

            loss_t = bce(tlog, yb)
            loss_hard = bce(slog, yb)

            with torch.no_grad():
                soft_t = torch.sigmoid(tlog / temperature)
            loss_kd = bce(slog / temperature, soft_t) * (temperature * temperature)

            loss = teacher_w * loss_t + alpha * loss_hard + (1 - alpha) * loss_kd

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += float(loss.item())

        if online_student:
            ys, logits = [], []
            student.eval()
            with torch.no_grad():
                for xb, yb, batch_ids in va_loader:
                    seqs = [seq_by_id[sid] for sid in batch_ids]
                    out = student.forward_sequences(
                        seqs,
                        device=device,
                        max_len=args.max_seq_len,
                        tokenizer=student_tokenizer,
                        tokenize_fn=oracle_base.tokenize_sequences if oracle_base is not None else None,
                    )
                    ys.append(yb.numpy())
                    logits.append(out.detach().cpu().numpy())
            y_val = base.np.concatenate(ys)
            lg_val = base.np.concatenate(logits)
            val_m = base.metrics_from_logits(y_val, lg_val)
        else:
            val_m, _, _ = base.run_eval(student, va_loader, device)
        row = {"epoch": ep, "train_loss": ep_loss / max(1, len(tr_loader)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            torch.save(
                {
                    "student_state": student.state_dict(),
                    "teacher_state": teacher.state_dict(),
                    "feature_dim": int(in_dim),
                    "config": cfg,
                    "best_val": val_m,
                },
                outdir / "models" / "kd_best.pt",
            )

    best = torch.load(outdir / "models" / "kd_best.pt", map_location=device, weights_only=False)
    student.load_state_dict(best["student_state"])
    if online_student:
        ys, logits = [], []
        student.eval()
        with torch.no_grad():
            for xb, yb, batch_ids in te_loader:
                seqs = [seq_by_id[sid] for sid in batch_ids]
                out = student.forward_sequences(
                    seqs,
                    device=device,
                    max_len=args.max_seq_len,
                    tokenizer=student_tokenizer,
                    tokenize_fn=oracle_base.tokenize_sequences if oracle_base is not None else None,
                )
                ys.append(yb.numpy())
                logits.append(out.detach().cpu().numpy())
        y = base.np.concatenate(ys)
        lg = base.np.concatenate(logits)
        test_m = base.metrics_from_logits(y, lg)
    else:
        test_m, y, lg = base.run_eval(student, te_loader, device)
    print("[KD] test:", json.dumps(test_m, ensure_ascii=False))

    with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_auc": best_auc, "test": test_m, "history": history}, f, ensure_ascii=False, indent=2)

    probs = 1 / (1 + base.np.exp(-lg))
    with (outdir / "test_predictions.csv").open("w", encoding="utf-8") as f:
        f.write("seq_id,label,prediction\n")
        for sid, yy, pp in zip(obj["test"]["ids"], y.astype(int), probs):
            f.write(f"{sid},{yy},{pp:.6f}\n")

    print(f"[KD] done: {outdir}")
    _status("kd-fusion: trainer finished")
    return 0


def main() -> int:
    args = parse_args()
    output_dir = getattr(args, "output_dir", "")
    pretrained = getattr(args, "pretrained", "")
    _status(
        f"launch model={args.model} output_dir={_fmt_path(output_dir)} "
        f"pretrained={_fmt_path(pretrained)}"
    )
    if args.model == "esm2":
        return run_esm2(args)
    if args.model == "evo2":
        return run_evo2(args)
    if args.model == "oracle":
        return run_oracle(args)
    if args.model == "oracle-cnn":
        return run_oracle_cnn(args)
    if args.model == "kd-fusion":
        return run_kd_fusion(args)
    raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    raise SystemExit(main())
