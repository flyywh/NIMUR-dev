#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "scripts" / "trainers" / "ORACLE_train.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.embed_extract import ESM2Embedder, ProtBERTEmbedder, parse_fasta_rows


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("trainer_oracle_train", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trainer module from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference / evaluation for ORACLE KG model")
    parser.add_argument("--model", required=True, help="Path to ORACLE best.pt checkpoint")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--fasta", default="", help="Input FASTA for ranking inference")
    parser.add_argument("--data-dir", default="", help="Prepared KG split dir for evaluation, e.g. data/oracle_kg/test")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-cache-dir", default="")
    parser.add_argument("--protbert-path", default="")
    parser.add_argument("--backbone", default="")
    parser.add_argument("--esm2-layer", type=int, default=-1)
    parser.add_argument("--evo2-layer-name", default="")
    parser.add_argument("--max-seq-len", type=int, default=-1)
    parser.add_argument("--esm-max-len", type=int, default=1022)
    parser.add_argument("--protbert-max-len", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-sample-limit", type=int, default=-1)
    return parser.parse_args()


def to_namespace(obj) -> SimpleNamespace:
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    if isinstance(obj, SimpleNamespace):
        return obj
    return SimpleNamespace(**vars(obj))


def resolve_override(value, default_value):
    return value if value not in ("", -1, None) else default_value


def build_runtime_args(saved_args: dict, cli_args: argparse.Namespace):
    args = to_namespace(saved_args)
    args.trained_with_cache = bool(getattr(args, "embedding_cache_dir", "") or getattr(args, "protbert_cache_dir", ""))
    args.embedding_cache_dir = resolve_override(getattr(cli_args, "embedding_cache_dir", ""), getattr(args, "embedding_cache_dir", ""))
    args.protbert_path = resolve_override(getattr(cli_args, "protbert_path", ""), getattr(args, "protbert_path", "./ProtBERT"))
    args.backbone = resolve_override(getattr(cli_args, "backbone", ""), getattr(args, "backbone", "protbert"))
    args.esm2_layer = resolve_override(getattr(cli_args, "esm2_layer", -1), getattr(args, "esm2_layer", 33))
    args.evo2_layer_name = resolve_override(getattr(cli_args, "evo2_layer_name", ""), getattr(args, "evo2_layer_name", "blocks.21.mlp.l3"))
    args.max_seq_len = resolve_override(getattr(cli_args, "max_seq_len", -1), getattr(args, "max_seq_len", 1024))
    args.esm_max_len = resolve_override(getattr(cli_args, "esm_max_len", -1), getattr(args, "esm_max_len", 1022))
    args.protbert_max_len = resolve_override(getattr(cli_args, "protbert_max_len", -1), getattr(args, "protbert_max_len", 2400))
    args.no_cuda = getattr(args, "no_cuda", False)
    args.unfreeze_encoder = False
    args.freeze_encoder = getattr(args, "freeze_encoder", False)
    args.protbert_cache_dir = getattr(args, "protbert_cache_dir", "")
    args.embedding_dim = int(getattr(args, "embedding_dim", 256))
    return args


def infer_cached_input_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("prot_encoder.proj.weight")
    if torch.is_tensor(weight) and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError("Cannot infer cached input dim from checkpoint")


def build_model(trainer, ckpt: dict, runtime_args, device: str):
    type2id = ckpt["type2id"]
    embedding_cache_dir = runtime_args.embedding_cache_dir or runtime_args.protbert_cache_dir
    use_cached = bool(getattr(runtime_args, "trained_with_cache", False))

    if use_cached:
        tokenizer = None
        if embedding_cache_dir:
            cache_dim = trainer.infer_cached_embedding_dim(embedding_cache_dir)
        else:
            cache_dim = infer_cached_input_dim_from_state_dict(ckpt["model_state"])
        prot_encoder = trainer.CachedProteinEncoder(input_dim=cache_dim, proj_dim=runtime_args.embedding_dim)
    else:
        backbone = getattr(runtime_args, "backbone", "protbert")
        if backbone == "protbert":
            tokenizer = trainer.AutoTokenizer.from_pretrained(
                runtime_args.protbert_path,
                local_files_only=True,
                do_lower_case=False,
            )
            prot_encoder = trainer.ProteinEncoder(
                model_path=runtime_args.protbert_path,
                proj_dim=runtime_args.embedding_dim,
                freeze=not getattr(runtime_args, "unfreeze_encoder", False),
            )
        elif backbone == "esm2":
            tokenizer = None
            prot_encoder = trainer.ESM2ProteinEncoder(
                proj_dim=runtime_args.embedding_dim,
                layer=runtime_args.esm2_layer,
                freeze=not getattr(runtime_args, "unfreeze_encoder", False),
            )
        elif backbone == "evo2":
            tokenizer = None
            prot_encoder = trainer.Evo2ProteinEncoder(
                proj_dim=runtime_args.embedding_dim,
                layer_name=runtime_args.evo2_layer_name,
                max_len=runtime_args.max_seq_len,
                freeze=not getattr(runtime_args, "unfreeze_encoder", False),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    model = trainer.TransEProteinType(
        prot_encoder=prot_encoder,
        num_types=len(type2id),
        dim=runtime_args.embedding_dim,
        p_norm=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, type2id


def compute_fused_embeddings_online(headers, sequences, runtime_args, device: str):
    rows = []
    esm_embedder = ESM2Embedder(device=device, layer=int(runtime_args.esm2_layer))
    prot_embedder = ProtBERTEmbedder(runtime_args.protbert_path, device=device)
    for header, sequence in zip(headers, sequences):
        esm_vec = esm_embedder.encode_mean_chunked(
            header,
            sequence,
            chunk_len=int(runtime_args.esm_max_len),
        ).astype(np.float32)
        prot_vec = prot_embedder.encode_mean(
            header,
            sequence,
            max_len=int(runtime_args.protbert_max_len),
        ).astype(np.float32)
        rows.append(np.concatenate([esm_vec, prot_vec], axis=0).astype(np.float32))
    return torch.tensor(np.stack(rows, axis=0), dtype=torch.float32, device=device)


def encode_sequences(trainer, model, tokenizer, headers, sequences, runtime_args, device: str):
    embedding_cache_dir = runtime_args.embedding_cache_dir or runtime_args.protbert_cache_dir
    if embedding_cache_dir:
        cached = trainer.load_cached_embedding_batch(embedding_cache_dir, headers, device=device)
        return model.prot_encoder(cached)
    if getattr(runtime_args, "trained_with_cache", False):
        input_dim = int(getattr(model.prot_encoder, "input_dim", 0))
        if input_dim != 2304:
            raise ValueError(
                f"Cached ORACLE checkpoint expects input_dim={input_dim}; online no-cache mode currently supports fused ESM2+ProtBERT dim 2304 only"
            )
        fused = compute_fused_embeddings_online(headers, sequences, runtime_args, device)
        return model.prot_encoder(fused)
    if hasattr(model.prot_encoder, "encode_sequences") and tokenizer is None:
        return model.prot_encoder.encode_sequences(sequences, device=device, max_len=runtime_args.max_seq_len)
    all_input_ids, all_attn_masks = trainer.tokenize_sequences(
        tokenizer,
        sequences,
        max_len=runtime_args.max_seq_len,
        device=device,
    )
    flat_input_ids = torch.cat(all_input_ids, dim=0)
    flat_attn_masks = torch.cat(all_attn_masks, dim=0)
    flat_vecs = model.prot_encoder(flat_input_ids, flat_attn_masks)
    chunk_counts = [x.size(0) for x in all_input_ids]
    h_vecs = torch.split(flat_vecs, chunk_counts, dim=0)
    return torch.stack([chunk.mean(dim=0) for chunk in h_vecs], dim=0)


def rank_types(model, prot_vecs, id2type: dict[int, str], top_k: int):
    all_type_ids = torch.arange(model.type_emb.num_embeddings, device=prot_vecs.device)
    all_type_vecs = model.type_emb(all_type_ids)
    diff_all = prot_vecs.unsqueeze(1) + model.rel.unsqueeze(0).unsqueeze(0) - all_type_vecs.unsqueeze(0)
    dist_all = torch.norm(diff_all, p=2, dim=-1)
    top_dists, top_idx = torch.topk(dist_all, k=min(top_k, dist_all.shape[1]), dim=1, largest=False)
    return [
        [
            {"type": id2type[int(type_idx)], "distance": float(type_dist)}
            for type_idx, type_dist in zip(row_idx.tolist(), row_dist.tolist())
        ]
        for row_idx, row_dist in zip(top_idx, top_dists)
    ]


def write_rank_csv(out_path: Path, rows: list[dict[str, object]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "top1_type", "top1_distance", "topk_json", "true_type", "true_rank"])
        for row in rows:
            writer.writerow(
                [
                    row.get("seq_id", ""),
                    row.get("top1_type", ""),
                    row.get("top1_distance", ""),
                    json.dumps(row.get("topk", []), ensure_ascii=False),
                    row.get("true_type", ""),
                    row.get("true_rank", ""),
                ]
            )


def infer_from_fasta(trainer, model, tokenizer, runtime_args, type2id: dict[str, int], fasta_path: Path, out_path: Path, batch_size: int, top_k: int, device: str):
    rows = parse_fasta_rows(fasta_path)
    if not rows:
        raise ValueError(f"No valid sequences in {fasta_path}")
    id2type = {idx: name for name, idx in type2id.items()}
    out_rows = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            headers = [header for header, _seq in batch]
            sequences = [seq for _header, seq in batch]
            prot_vecs = encode_sequences(trainer, model, tokenizer, headers, sequences, runtime_args, device)
            ranked = rank_types(model, prot_vecs, id2type, top_k)
            for header, topk in zip(headers, ranked):
                out_rows.append(
                    {
                        "seq_id": header,
                        "top1_type": topk[0]["type"] if topk else "",
                        "top1_distance": f"{topk[0]['distance']:.6f}" if topk else "",
                        "topk": topk,
                        "true_type": "",
                        "true_rank": "",
                    }
                )
    write_rank_csv(out_path, out_rows)
    print(f"[ORACLE-PREDICT] predictions: {out_path}")


def eval_from_data_dir(trainer, model, tokenizer, runtime_args, type2id: dict[str, int], data_dir: Path, out_path: Path, batch_size: int, top_k: int, sample_limit: int, device: str):
    p2id, _t2id, _r2id, id2seq, pos_triples = trainer.read_kge_dir(str(data_dir))
    id2name = {pid: header for header, pid in p2id.items()}
    can_use_trainer_eval = bool((runtime_args.embedding_cache_dir or runtime_args.protbert_cache_dir) or not getattr(runtime_args, "trained_with_cache", False))
    metrics = None
    if can_use_trainer_eval:
        metrics = trainer.evaluate(
            model=model,
            tokenizer=tokenizer,
            id2seq=id2seq,
            pos_triples=pos_triples,
            batch_size=batch_size,
            max_len=runtime_args.max_seq_len,
            device=device,
            sample_limit=None if sample_limit < 0 else sample_limit,
            embedding_cache_dir=(runtime_args.embedding_cache_dir or runtime_args.protbert_cache_dir) or None,
            id2name=id2name,
        )

    id2type = {idx: name for name, idx in type2id.items()}
    triples = pos_triples if sample_limit < 0 else pos_triples[:sample_limit]
    out_rows = []
    ranks = []
    with torch.no_grad():
        for start in range(0, len(triples), batch_size):
            batch = triples[start : start + batch_size]
            pids = [pid for pid, _ in batch]
            seqs = [id2seq[pid] for pid in pids]
            if runtime_args.embedding_cache_dir or runtime_args.protbert_cache_dir:
                header_batch = [id2name[pid] for pid in pids]
            else:
                header_batch = [str(pid) for pid in pids]
            prot_vecs = encode_sequences(trainer, model, tokenizer, header_batch, seqs, runtime_args, device)
            ranked = rank_types(model, prot_vecs, id2type, max(top_k, len(type2id)))
            for (pid, true_tid), topk in zip(batch, ranked):
                ordered_types = [item["type"] for item in topk]
                true_type = id2type[int(true_tid)]
                true_rank = ordered_types.index(true_type) + 1 if true_type in ordered_types else ""
                if true_rank:
                    ranks.append(int(true_rank))
                out_rows.append(
                    {
                        "seq_id": id2name[pid],
                        "top1_type": topk[0]["type"] if topk else "",
                        "top1_distance": f"{topk[0]['distance']:.6f}" if topk else "",
                        "topk": topk[:top_k],
                        "true_type": true_type,
                        "true_rank": true_rank,
                    }
                )

    write_rank_csv(out_path, out_rows)
    if metrics is None:
        n = len(ranks)
        hits1 = sum(1 for rank in ranks if rank <= 1)
        hits3 = sum(1 for rank in ranks if rank <= 3)
        hits10 = sum(1 for rank in ranks if rank <= 10)
        metrics = {
            "MRR": (sum(1.0 / rank for rank in ranks) / n) if n > 0 else 0.0,
            "Hits@1": (hits1 / n) if n > 0 else 0.0,
            "Hits@3": (hits3 / n) if n > 0 else 0.0,
            "Hits@10": (hits10 / n) if n > 0 else 0.0,
            "Count": n,
        }
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[ORACLE-PREDICT] predictions: {out_path}")
    print(f"[ORACLE-PREDICT] metrics: {metrics_path}")


def main() -> int:
    args = parse_args()
    if bool(args.fasta) == bool(args.data_dir):
        raise ValueError("Provide exactly one of --fasta or --data-dir")

    trainer = load_trainer_module()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "type2id" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {args.model}")

    runtime_args = build_runtime_args(ckpt.get("args", {}), args)
    if args.fasta and not args.embedding_cache_dir:
        runtime_args.embedding_cache_dir = ""
        runtime_args.protbert_cache_dir = ""
    model, tokenizer, type2id = build_model(trainer, ckpt, runtime_args, device)
    out_path = Path(args.out)

    if args.fasta:
        infer_from_fasta(
            trainer,
            model,
            tokenizer,
            runtime_args,
            type2id,
            Path(args.fasta),
            out_path,
            int(args.batch_size),
            int(args.top_k),
            device,
        )
    else:
        eval_from_data_dir(
            trainer,
            model,
            tokenizer,
            runtime_args,
            type2id,
            Path(args.data_dir),
            out_path,
            int(args.batch_size),
            int(args.top_k),
            int(args.eval_sample_limit),
            device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
