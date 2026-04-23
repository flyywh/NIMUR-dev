import os, time, json, random, logging, argparse, hashlib, re
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, set_seed
from pathlib import Path


# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    filename="train.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("TransE_ProteinType")


# -----------------------
# Utilities
# -----------------------
def read_kge_dir(data_dir: str):
    """
    To read a data directory, the following files must be included:
    - protein2id.txt  (query \t id)  —— The ids start from 0 and are consecutive
    - type2id.txt    (type \t id)   —— The ids start from 0 and are consecutive (globally consistent) - relation2id.txt  (no/yes)
    - sequence.txt     (Each line represents a sequence, corresponding to the sorting of protein_id)
    - protein_relation_type.txt  (pid \t rid \t tid) Output directory & Data path
    """
    p2id_path = os.path.join(data_dir, "protein2id.txt")
    t2id_path = os.path.join(data_dir, "type2id.txt")
    r2id_path = os.path.join(data_dir, "relation2id.txt")
    seq_path  = os.path.join(data_dir, "sequence.txt")
    tri_path  = os.path.join(data_dir, "protein_relation_type.txt")

    def read_kv(path):
        d = {}
        with open(path, "r") as f:
            for line in f:
                if not line.strip(): continue
                k, v = line.rstrip("\n").split("\t")
                d[k] = int(v)
        return d

    protein2id = read_kv(p2id_path)
    type2id = read_kv(t2id_path)
    relation2id = read_kv(r2id_path)
    sequences = []
    with open(seq_path, "r") as f:
        for line in f:
            sequences.append(line.strip())

    id2seq = {pid: sequences[pid] for pid in range(len(sequences))}

    pos_triples = []
    with open(tri_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            pid_str, rid_str, tid_str = line.strip().split("\t")
            pid, rid, tid = int(pid_str), int(rid_str), int(tid_str)
            if rid == relation2id.get("yes", 1):
                pos_triples.append((pid, tid))

    return protein2id, type2id, relation2id, id2seq, pos_triples

class KGETripletDataset(Dataset):
    """
    Only store positive samples (protein_id, type_id); negative samples are sampled dynamically during collation
    Support for long sequence chunks:
    - triples: List[Tuple[int, int]] Original positive samples (protein_id, type_id)
    - id2seq: Dict[int, str] Maps protein_id to protein sequence
    - max_len: int Maximum length of each chunk
    """
    def __init__(self, triples: List[Tuple[int, int]], id2seq: dict, max_len: int = 1024, chunk_sequences: bool = True):
        self.samples = []
        for pid, tid in triples:
            seq = id2seq[pid]
            if (not chunk_sequences) or len(seq) <= max_len:
                self.samples.append((pid, tid, seq))
            else:
                for i in range(0, len(seq), max_len):
                    chunk_seq = seq[i:i+max_len]
                    self.samples.append((pid, tid, chunk_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  

def build_negative_samples(batch_pos: List[Tuple[int, int]], num_types: int, num_negs: int = 1):
    """
    Generate negative samples for a batch of positive samples (p, t):
    - Randomly replace the head (protein) or the tail (type). Here, it is assumed that replacing the tail (type) results in a more natural negative sample.
    Return: heads_pos, tails_pos, heads_neg, tails_neg  (tensor)
    """
    heads_pos = torch.tensor([p for p, _ in batch_pos], dtype=torch.long)
    tails_pos = torch.tensor([t for _, t in batch_pos], dtype=torch.long)

    neg_tails = []
    for _, true_t in batch_pos:
        cur_negs = []
        for _ in range(num_negs):
            nt = random.randrange(num_types)
            while nt == true_t:
                nt = random.randrange(num_types)
            cur_negs.append(nt)
        neg_tails.append(cur_negs)
    # shape: (B, K)
    tails_neg = torch.tensor(neg_tails, dtype=torch.long)
    return heads_pos, tails_pos, tails_neg


# -----------------------
# Model (ProtBERT encoder + TransE)
# -----------------------
class ProteinEncoder(nn.Module):
    """
    Encode the sequence using the local ProtBERT; select [CLS] or mean pooling (here, mean pooling is more stable)
    """
    def __init__(self, model_path: str, proj_dim: int, freeze: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim, bias=False)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = outputs.last_hidden_state
        masked = last * attention_mask.unsqueeze(-1)
        sum_emb = masked.sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mean_emb = sum_emb / lengths
        return self.proj(mean_emb)


class ESM2ProteinEncoder(nn.Module):
    def __init__(self, proj_dim: int, *, layer: int = 33, freeze: bool = True):
        super().__init__()
        import esm

        self.layer = layer
        self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
        hidden = int(self.model.embed_dim)
        self.proj = nn.Linear(hidden, proj_dim, bias=False)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def encode_sequences(self, sequences: List[str], device: str, max_len: int) -> torch.Tensor:
        pooled = []
        self.model = self.model.to(device)
        self.model.eval()
        for idx, seq in enumerate(sequences):
            seq = seq.upper().replace(" ", "")
            chunks = [seq[i : i + max_len] for i in range(0, len(seq), max_len)] or [seq]
            chunk_vecs = []
            for chunk_id, chunk in enumerate(chunks, start=1):
                _, _, tokens = self.batch_converter([(f"sample{idx}_chunk{chunk_id}", chunk)])
                tokens = tokens.to(device)
                out = self.model(tokens, repr_layers=[self.layer])
                rep = out["representations"][self.layer][0, 1:-1, :]
                chunk_vecs.append(rep.mean(dim=0))
            pooled.append(torch.stack(chunk_vecs, dim=0).mean(dim=0))
        return self.proj(torch.stack(pooled, dim=0))


class Evo2ProteinEncoder(nn.Module):
    def __init__(self, proj_dim: int, *, layer_name: str = "blocks.21.mlp.l3", max_len: int = 16384, freeze: bool = True):
        super().__init__()
        from evo2 import Evo2

        self.layer_name = layer_name
        self.max_len = max_len
        self.model = Evo2()
        self.tokenizer = self.model.tokenizer
        hidden = None
        with torch.no_grad():
            sample_ids = torch.tensor(self.tokenizer.tokenize("ACGT"), dtype=torch.int).unsqueeze(0)
            _, embeddings = self.model(sample_ids, return_embeddings=True, layer_names=[self.layer_name])
            hidden = int(embeddings[self.layer_name].shape[-1])
        self.proj = nn.Linear(hidden, proj_dim, bias=False)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @staticmethod
    def _clean(seq: str) -> str:
        return re.sub(r"[^ACGTN]", "", seq.upper())

    def encode_sequences(self, sequences: List[str], device: str, max_len: int | None = None) -> torch.Tensor:
        pooled = []
        max_len = self.max_len if max_len is None else max_len
        self.model = self.model.to(device)
        self.model.eval()
        for seq in sequences:
            dna = self._clean(seq)
            if max_len is not None and len(dna) > max_len:
                dna = dna[:max_len]
            input_ids = torch.tensor(self.tokenizer.tokenize(dna), dtype=torch.int).unsqueeze(0).to(device)
            _, embeddings = self.model(input_ids, return_embeddings=True, layer_names=[self.layer_name])
            pooled.append(embeddings[self.layer_name].squeeze(0).mean(dim=0))
        return self.proj(torch.stack(pooled, dim=0))


class CachedProteinEncoder(nn.Module):
    """
    Lightweight encoder for pre-extracted ProtBERT mean vectors.
    Keeps the trainable projection layer so LoRA / checkpoint loading can still target prot_encoder.proj.
    """
    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim, proj_dim, bias=False)

    def forward(self, features):
        return self.proj(features)


class TransEProteinType(nn.Module):
    """
    Entity: protein (obtained by encoder), type (embedding)
    Relation: single belongs_to (trainable vector)
    Score: score = || h + r - t ||_2
    """
    def __init__(self, prot_encoder: ProteinEncoder, num_types: int, dim: int, p_norm: int = 2):
        super().__init__()
        self.prot_encoder = prot_encoder
        self.type_emb = nn.Embedding(num_types, dim)
        nn.init.xavier_uniform_(self.type_emb.weight)
        self.rel = nn.Parameter(torch.empty(dim))
        nn.init.uniform_(self.rel, -0.01, 0.01)
        self.p = p_norm
        self.dim = dim
    def score_dist(self, h, t):
        return torch.norm(h + self.rel - t, p=2, dim=-1)
    def forward(self, prot_vecs, type_ids):
        t = self.type_emb(type_ids)
        dist = self.score_dist(prot_vecs, t)
        return dist

# -----------------------
# Tokenize helper
# -----------------------
def tokenize_sequences(tokenizer, sequences: List[str], max_len: int = 512, device="cuda"):
    """
    Divide each sequence into several chunks, with each chunk not exceeding max_len.
    Return a list of (input_ids, attention_mask) for each sequence.
    """
    all_input_ids = []
    all_attn_masks = []

    for seq in sequences:
        chunks = [seq[i:i+max_len] for i in range(0, len(seq), max_len)]
        enc = tokenizer(
            chunks,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        all_input_ids.append(enc["input_ids"].to(device))      # shape: (num_chunks, max_len)
        all_attn_masks.append(enc["attention_mask"].to(device))

    return all_input_ids, all_attn_masks


def header_to_stem(header: str, max_base_len: int = 120) -> str:
    base = re.sub(r"[^A-Za-z0-9._=-]+", "_", header).strip("._")
    if not base:
        base = "seq"
    if len(base) > max_base_len:
        base = base[:max_base_len]
    digest = hashlib.sha1(header.encode("utf-8")).hexdigest()[:12]
    return f"{base}__{digest}"


def load_cached_embedding_batch(cache_dir: str | Path, headers: List[str], device="cuda"):
    cache_root = Path(cache_dir)
    rows = []
    for header in headers:
        path = cache_root / f"{header_to_stem(header)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached embedding for {header}: {path}")
        vec = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.is_tensor(vec):
            vec = torch.tensor(vec)
        vec = vec.float()
        if vec.ndim > 1:
            vec = vec.mean(dim=0)
        rows.append(vec)
    return torch.stack(rows, dim=0).to(device)


def infer_cached_embedding_dim(cache_dir: str | Path) -> int:
    root = Path(cache_dir)
    for p in sorted(root.glob("*.pt")):
        vec = torch.load(p, map_location="cpu", weights_only=False)
        if not torch.is_tensor(vec):
            vec = torch.tensor(vec)
        vec = vec.float()
        if vec.ndim == 1:
            return int(vec.numel())
        if vec.ndim >= 2:
            return int(vec.shape[-1])
        raise ValueError(f"Unsupported cached embedding shape={tuple(vec.shape)} from {p}")
    raise ValueError(f"No cached embeddings found in {root}")


# -----------------------
# Evaluation (MRR / Hits@K) with soft-negative
# -----------------------
@torch.no_grad()
def evaluate(model, tokenizer, id2seq, pos_triples, batch_size=16, max_len=1024, device="cuda", sample_limit=200, embedding_cache_dir=None, id2name=None):
    """
    Evaluation: Given (p, t_true), score and rank all types, and calculate MRR / Hits@1 / Hits@3 / Hits@10
    Use soft-negative scoring to be consistent with the training process
    """
    
    model.eval()
    num_types = model.type_emb.num_embeddings
    triples = pos_triples
    if sample_limit is not None and len(triples) > sample_limit:
        triples = random.sample(pos_triples, sample_limit)

    ranks = []
    hits1 = hits3 = hits10 = 0
    all_type_ids = torch.arange(num_types, device=device)
    all_type_vecs = model.type_emb(all_type_ids)
    rel = model.rel.unsqueeze(0)

    for i in range(0, len(triples), batch_size):
        batch = triples[i:i+batch_size]
        pids = [pid for pid, _ in batch]
        true_t = torch.tensor([tid for _, tid in batch], device=device, dtype=torch.long)
        if embedding_cache_dir:
            if id2name is None:
                raise ValueError("id2name is required when using embedding_cache_dir")
            headers = [id2name[pid] for pid in pids]
            cached = load_cached_embedding_batch(embedding_cache_dir, headers, device=device)
            prot_vecs = model.prot_encoder(cached)
        else:
            seqs = [id2seq[pid] for pid in pids]
            if hasattr(model.prot_encoder, "encode_sequences") and tokenizer is None:
                prot_vecs = model.prot_encoder.encode_sequences(seqs, device=device, max_len=max_len)
            else:
                # Tokenize sequences -> list of (num_chunks, max_len)
                all_input_ids, all_attn_masks = tokenize_sequences(tokenizer, seqs, max_len=max_len, device=device)

                flat_input_ids = torch.cat(all_input_ids, dim=0)
                flat_attn_masks = torch.cat(all_attn_masks, dim=0)

                flat_vecs = model.prot_encoder(flat_input_ids, flat_attn_masks)

                chunk_counts = [x.size(0) for x in all_input_ids]
                h_vecs = torch.split(flat_vecs, chunk_counts, dim=0)
                prot_vecs = torch.stack([c.mean(dim=0) for c in h_vecs], dim=0)  # (B, D)

        t_pos_vec = model.type_emb(true_t)                   # (B, D)
        diff_pos = prot_vecs.unsqueeze(1) + rel - t_pos_vec.unsqueeze(1)
        dist_pos = torch.norm(diff_pos, p=2, dim=-1).squeeze(1)  # (B,)

        all_type_vecs_exp = all_type_vecs.unsqueeze(0).expand(len(batch), -1, -1)  # (B, num_types, D)
        diff_all = prot_vecs.unsqueeze(1) + rel - all_type_vecs_exp                  # (B, num_types, D)
        dist_all = torch.norm(diff_all, p=2, dim=-1)                                 # (B, num_types)
        neg_avg = (dist_all.sum(dim=1) - dist_pos) / (num_types - 1)                 # (B,)

        # soft-negative score
        scores = neg_avg - dist_pos

        ranking = torch.argsort(scores, descending=True)
        for bi in range(len(batch)):
            rank_pos = (ranking[bi] == bi).nonzero(as_tuple=False)
            if rank_pos.numel() == 0:
                rank_pos = (torch.argsort(dist_all[bi]) == true_t[bi]).nonzero(as_tuple=False)
            rank = int(rank_pos.item()) + 1
            ranks.append(rank)
            if rank <= 1: hits1 += 1
            if rank <= 3: hits3 += 1
            if rank <= 10: hits10 += 1

    n = len(ranks)
    mrr = sum(1.0/r for r in ranks) / n if n > 0 else 0.0
    return {
        "MRR": mrr,
        "Hits@1": hits1/n if n>0 else 0.0,
        "Hits@3": hits3/n if n>0 else 0.0,
        "Hits@10": hits10/n if n>0 else 0.0,
        "Count": n
    }


# -----------------------
# Training function
# -----------------------
def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    logger.info("Loading train/val data dirs ...")
    p2id_tr, t2id_tr, r2id_tr, id2seq_tr, pos_tr_triples = read_kge_dir(args.train_dir)
    p2id_te, t2id_te, r2id_te, id2seq_te, pos_te_triples = read_kge_dir(args.val_dir)
    id2name_tr = {pid: header for header, pid in p2id_tr.items()}
    id2name_te = {pid: header for header, pid in p2id_te.items()}
    num_types = len(t2id_tr)
    embedding_cache_dir = getattr(args, "embedding_cache_dir", "") or getattr(args, "protbert_cache_dir", "")
    use_cached_protbert = bool(embedding_cache_dir)
    train_ds = KGETripletDataset(
        pos_tr_triples,
        id2seq_tr,
        max_len=args.max_seq_len,
        chunk_sequences=not use_cached_protbert,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=lambda x: x)

    # Tokenizer & Encoder
    if use_cached_protbert:
        logger.info(f"Loading embedding cache from: {embedding_cache_dir}")
        tokenizer = None
        cache_dim = infer_cached_embedding_dim(embedding_cache_dir)
        prot_encoder = CachedProteinEncoder(input_dim=cache_dim, proj_dim=args.embedding_dim)
    else:
        backbone = getattr(args, "backbone", "protbert")
        if backbone == "protbert":
            logger.info(f"Loading ProtBERT from: {args.protbert_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.protbert_path, local_files_only=True, do_lower_case=False)
            prot_encoder = ProteinEncoder(model_path=args.protbert_path, proj_dim=args.embedding_dim, freeze=not args.unfreeze_encoder)
            if hasattr(args, "freeze_encoder") and args.freeze_encoder:
                logger.info("Freezing ProteinEncoder parameters...")
                for param in prot_encoder.parameters():
                    param.requires_grad = False
        elif backbone == "esm2":
            logger.info("Loading ESM2 backbone")
            tokenizer = None
            prot_encoder = ESM2ProteinEncoder(
                proj_dim=args.embedding_dim,
                layer=args.esm2_layer,
                freeze=not args.unfreeze_encoder,
            )
        elif backbone == "evo2":
            logger.info("Loading Evo2 backbone")
            tokenizer = None
            prot_encoder = Evo2ProteinEncoder(
                proj_dim=args.embedding_dim,
                layer_name=args.evo2_layer_name,
                max_len=args.max_seq_len,
                freeze=not args.unfreeze_encoder,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    # Model
    model = TransEProteinType(prot_encoder=prot_encoder, num_types=num_types, dim=args.embedding_dim, p_norm=2).to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"Trainable params: {sum(p.numel() for p in params):,}")

    margin = args.margin
    num_negs = args.num_negs
    global_step = 0
    best_mrr = -1.0
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "train.log")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            p_pos = [p for p, _, _ in batch]
            t_pos_ids = torch.tensor([t for _, t, _ in batch], dtype=torch.long, device=device)
            if use_cached_protbert:
                headers = [id2name_tr[p] for p, _, _ in batch]
                cached = load_cached_embedding_batch(embedding_cache_dir, headers, device=device)
                h_vec = model.prot_encoder(cached)
            else:
                backbone = getattr(args, "backbone", "protbert")
                seqs = [seq for _, _, seq in batch]
                if backbone == "protbert":
                    # Tokenize sequences
                    all_input_ids, all_attn_masks = tokenize_sequences(tokenizer, seqs, max_len=args.max_seq_len, device=device)

                    # forward
                    flat_input_ids = torch.cat(all_input_ids, dim=0)
                    flat_attn_masks = torch.cat(all_attn_masks, dim=0)
                    flat_vecs = model.prot_encoder(flat_input_ids, flat_attn_masks)

                    # Average chunk
                    chunk_counts = [x.size(0) for x in all_input_ids]
                    h_vecs = torch.split(flat_vecs, chunk_counts, dim=0)
                    h_vec = torch.stack([c.mean(dim=0) for c in h_vecs], dim=0)  # (B, D)
                else:
                    h_vec = model.prot_encoder.encode_sequences(seqs, device=device, max_len=args.max_seq_len)

            # All types as soft negatives
            all_type_ids = torch.arange(num_types, device=device)
            all_type_vecs = model.type_emb(all_type_ids).unsqueeze(0).expand(len(batch), -1, -1)
            rel = model.rel.unsqueeze(0)
            diff_all = h_vec.unsqueeze(1) + rel - all_type_vecs
            dist_all = torch.norm(diff_all, p=2, dim=-1)  # (B, num_types)

            # Positive sample distance
            dist_pos = dist_all.gather(1, t_pos_ids.unsqueeze(1)).squeeze(1)  # (B,)
            neg_avg = (dist_all.sum(dim=1) - dist_pos) / (num_types - 1)

            # LogSumExp loss
            loss_per_sample = torch.log1p(torch.exp((dist_pos - neg_avg).clamp(-10, 10)))
            loss = loss_per_sample.mean() * args.ke_lambda

            # Gradient accumulation
            loss_scaled = loss / args.gradient_accumulation_steps
            loss_scaled.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            global_step += 1

            step_loss_val = loss.item() * args.gradient_accumulation_steps
            print(f"Epoch {epoch} | Step {step} | Global step {global_step} | step_loss {step_loss_val:.4f}")
            with open(log_file_path, "a") as f:
                f.write(f"Epoch {epoch} | Step {step} | Global step {global_step} | step_loss {step_loss_val:.4f}\n")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        dt = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} finished: avg_loss={avg_loss:.4f} | time={dt:.1f}s")
        print(f"Epoch {epoch} finished: avg_loss={avg_loss:.4f} | time={dt:.1f}s")

        # Evaluate
        metrics = evaluate(
            model=model,
            tokenizer=tokenizer,
            id2seq=id2seq_te,
            pos_triples=pos_te_triples,
            batch_size=args.eval_batch_size,
            max_len=args.max_seq_len,
            device=device,
            sample_limit=args.eval_sample_limit,
            embedding_cache_dir=embedding_cache_dir if use_cached_protbert else None,
            id2name=id2name_te if use_cached_protbert else None,
        )
        logger.info(f"[Eval] Epoch {epoch}: {json.dumps(metrics, ensure_ascii=False)}")
        print(f"[Eval] Epoch {epoch}: {metrics}")

        # Save
        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            save_path = os.path.join(args.output_dir, "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "type2id": t2id_tr
            }, save_path)
            logger.info(f"Saved best checkpoint to {save_path} (MRR={best_mrr:.4f})")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    final_model_path = os.path.join(args.output_dir, "last_full_model.pt")
    model.args = vars(args)
    model.type2id = t2id_tr

    torch.save(model, final_model_path)
    logger.info(f"Saved full model to {final_model_path}")
    print(f"Saved full model to {final_model_path}")

# -----------------------
# CLI
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train OntoProtein")

    # === Data path ===
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--pretrain_data_dir", type=str, default=None, help="Pretrain dataset directory")

    # === Model configuration ===
    parser.add_argument("--protbert_path", type=str, default="./ProtBERT", help="Path to ProtBERT model")
    parser.add_argument("--backbone", type=str, default="protbert", choices=["protbert", "esm2", "evo2"], help="Online backbone for oracle when not using embedding cache")
    parser.add_argument("--esm2_layer", type=int, default=33, help="ESM2 representation layer")
    parser.add_argument("--evo2_layer_name", type=str, default="blocks.21.mlp.l3", help="Evo2 embedding layer name")
    parser.add_argument("--embedding_cache_dir", type=str, default="", help="Optional cache dir of pre-extracted embeddings (pooled or token-level)")
    parser.add_argument("--protbert_cache_dir", type=str, default="", help="Optional cache dir of pre-extracted ProtBERT mean vectors from stage1")
    parser.add_argument("--protein_encoder_cls", type=str, default="bert", help="Protein encoder type")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of embeddings")
    parser.add_argument("--ke_embedding_size", type=int, default=256, help="Knowledge embedding size")
    parser.add_argument("--double_entity_embedding_size", type=bool, default=False, help="Whether to double entity embedding size")
    parser.add_argument("--freeze_encoder",action="store_true",help="是否冻结 ProteinEncoder 参数")

    # === Training configuration ===
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to save checkpoints")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (legacy)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size per device for evaluation")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size per device for evaluation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Alias for learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Scheduler type")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--unfreeze_encoder", action="store_true", help="If set, do not freeze the protein encoder during training (default: frozen).")
    parser.add_argument("--num_negs", type=int, default=4, help="Number of negative samples per positive sample")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length for tokenizer")

    # === KE Loss configuration ===
    parser.add_argument("--ke_lambda", type=float, default=1.0, help="Weight for KE loss")
    parser.add_argument("--ke_learning_rate", type=float, default=3e-5, help="Learning rate for KE part")
    parser.add_argument("--ke_max_score", type=float, default=12.0, help="Max KE score")
    parser.add_argument("--ke_score_fn", type=str, default="transE", help="KE score function")
    parser.add_argument("--ke_warmup_steps", type=int, default=400, help="KE warmup steps")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for contrastive / triplet loss")

    # === Operation-related ===
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Whether to pin memory in dataloader")
    parser.add_argument("--optimize_memory", action="store_true", help="Optimize memory usage")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--log_every", type=int, default=500, help="Log every N steps")
    parser.add_argument("--eval_sample_limit", type=int, default=200, help="Limit eval samples (set -1 for full eval)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    if args.eval_sample_limit is not None and args.eval_sample_limit < 0:
        args.eval_sample_limit = None
    return args


if __name__ == "__main__":
    args = parse_args()
    print("[INFO] Args:", args)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
