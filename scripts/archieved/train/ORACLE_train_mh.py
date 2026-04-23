import os, time, json, random, logging, argparse, glob
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, set_seed


# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    filename="train.log", filemode="w",
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("TransE_ProteinType")


# ============================================================
# AlphaGenome embedding 工具（新增）
# ============================================================
AG_DIM = 1536

def load_all_ag_embeddings(ag_emb_dir: str) -> dict:
    """加载目录下所有 .npz，合并成 {seq_id: np.ndarray(1536,)}"""
    merged = {}
    for npz_path in glob.glob(os.path.join(ag_emb_dir, "*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        for sid, emb in zip(data["seq_ids"].tolist(), data["embeddings"]):
            merged[sid] = emb
    logger.info(f"[AG] 加载 AG embedding：{len(merged)} 条（来自 {ag_emb_dir}）")
    print(f"  [AG] 加载 AG embedding：{len(merged)} 条（来自 {ag_emb_dir}）")
    return merged


# -----------------------
# Data utilities（原版不变）
# -----------------------
def read_kge_dir(data_dir: str):
    def read_kv(path):
        d = {}
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                k, v = line.rstrip("\n").split("\t")
                d[k] = int(v)
        return d
    protein2id  = read_kv(os.path.join(data_dir, "protein2id.txt"))
    type2id     = read_kv(os.path.join(data_dir, "type2id.txt"))
    relation2id = read_kv(os.path.join(data_dir, "relation2id.txt"))
    sequences = []
    with open(os.path.join(data_dir, "sequence.txt")) as f:
        for line in f:
            sequences.append(line.strip())
    id2seq = {pid: sequences[pid] for pid in range(len(sequences))}
    pos_triples = []
    with open(os.path.join(data_dir, "protein_relation_type.txt")) as f:
        for line in f:
            if not line.strip(): continue
            pid_str, rid_str, tid_str = line.strip().split("\t")
            pid, rid, tid = int(pid_str), int(rid_str), int(tid_str)
            if rid == relation2id.get("yes", 1):
                pos_triples.append((pid, tid))
    return protein2id, type2id, relation2id, id2seq, pos_triples


class KGETripletDataset(Dataset):
    def __init__(self, triples, id2seq, max_len=1024):
        self.samples = []
        for pid, tid in triples:
            seq = id2seq[pid]
            if len(seq) <= max_len:
                self.samples.append((pid, tid, seq))
            else:
                for i in range(0, len(seq), max_len):
                    self.samples.append((pid, tid, seq[i:i+max_len]))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# -----------------------
# Model（改动：ProteinEncoder 支持 AG embedding 融合）
# -----------------------
class ProteinEncoder(nn.Module):
    """
    原版：ProtBERT hidden(1024) → proj_dim
    融合：(ProtBERT hidden + AG_DIM)(2560) → proj_dim
    use_ag=False 时完全等同原版
    """
    def __init__(self, model_path: str, proj_dim: int, freeze: bool = True,
                 use_ag: bool = False):
        super().__init__()
        self.use_ag  = use_ag
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden       = self.encoder.config.hidden_size              # ProtBERT: 1024
        input_dim    = hidden + AG_DIM if use_ag else hidden        # 2560 or 1024
        self.proj    = nn.Linear(input_dim, proj_dim, bias=False)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, ag_emb=None):
        """
        ag_emb: (batch, AG_DIM) float tensor，已放到正确 device。
                use_ag=False 或未传入时忽略。
        """
        outputs  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last     = outputs.last_hidden_state
        masked   = last * attention_mask.unsqueeze(-1)
        sum_emb  = masked.sum(dim=1)
        lengths  = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mean_emb = sum_emb / lengths                                # (B, hidden)
        if self.use_ag and ag_emb is not None:
            mean_emb = torch.cat([mean_emb, ag_emb], dim=-1)       # (B, hidden+AG_DIM)
        return self.proj(mean_emb)                                  # (B, proj_dim)


class TransEProteinType(nn.Module):
    def __init__(self, prot_encoder, num_types, dim, p_norm=2):
        super().__init__()
        self.prot_encoder = prot_encoder
        self.type_emb     = nn.Embedding(num_types, dim)
        nn.init.xavier_uniform_(self.type_emb.weight)
        self.rel = nn.Parameter(torch.empty(dim))
        nn.init.uniform_(self.rel, -0.01, 0.01)
        self.p = p_norm; self.dim = dim

    def score_dist(self, h, t):
        return torch.norm(h + self.rel - t, p=2, dim=-1)

    def forward(self, prot_vecs, type_ids):
        return self.score_dist(prot_vecs, self.type_emb(type_ids))


# -----------------------
# Tokenize helper（原版不变）
# -----------------------
def tokenize_sequences(tokenizer, sequences, max_len=512, device="cuda"):
    all_input_ids, all_attn_masks = [], []
    for seq in sequences:
        chunks = [seq[i:i+max_len] for i in range(0, len(seq), max_len)]
        enc = tokenizer(chunks, add_special_tokens=True, padding=True,
                        truncation=True, max_length=max_len, return_tensors="pt")
        all_input_ids.append(enc["input_ids"].to(device))
        all_attn_masks.append(enc["attention_mask"].to(device))
    return all_input_ids, all_attn_masks


# -----------------------
# Evaluation（改动：forward 时传入 ag_emb）
# -----------------------
@torch.no_grad()
def evaluate(model, tokenizer, id2seq, pos_triples, ag_emb_dict,
             batch_size=16, max_len=1024, device="cuda", sample_limit=200):
    model.eval()
    num_types = model.type_emb.num_embeddings
    triples   = pos_triples
    if sample_limit is not None and len(triples) > sample_limit:
        triples = random.sample(pos_triples, sample_limit)

    ranks = []; hits1 = hits3 = hits10 = 0
    all_type_ids  = torch.arange(num_types, device=device)
    all_type_vecs = model.type_emb(all_type_ids)
    rel           = model.rel.unsqueeze(0)

    for i in range(0, len(triples), batch_size):
        batch  = triples[i:i+batch_size]
        pids   = [pid for pid, _ in batch]
        true_t = torch.tensor([tid for _, tid in batch], device=device, dtype=torch.long)
        seqs   = [id2seq[pid] for pid in pids]

        all_input_ids, all_attn_masks = tokenize_sequences(
            tokenizer, seqs, max_len=max_len, device=device)
        flat_input_ids  = torch.cat(all_input_ids,  dim=0)
        flat_attn_masks = torch.cat(all_attn_masks, dim=0)

        # ── 准备 AG embedding ──────────────────────────────
        if model.prot_encoder.use_ag and ag_emb_dict:
            chunk_counts = [x.size(0) for x in all_input_ids]
            ag_per_seq   = []
            for pid in pids:
                seq_id = f"protein_{pid}"   # id2seq 的 key 是 int，ag_emb_dict 的 key 是 seq_id 字符串
                # 尝试多种可能的 key 格式
                ag_np = (ag_emb_dict.get(str(pid)) or
                         ag_emb_dict.get(seq_id) or
                         np.zeros(AG_DIM, dtype=np.float32))
                ag_per_seq.append(ag_np)
            ag_flat = np.repeat(ag_per_seq, chunk_counts, axis=0)   # 广播到 chunk 级别
            ag_t    = torch.tensor(ag_flat, dtype=torch.float32, device=device)
        else:
            ag_t = None

        flat_vecs    = model.prot_encoder(flat_input_ids, flat_attn_masks, ag_emb=ag_t)
        chunk_counts = [x.size(0) for x in all_input_ids]
        h_vecs       = torch.split(flat_vecs, chunk_counts, dim=0)
        prot_vecs    = torch.stack([c.mean(dim=0) for c in h_vecs], dim=0)

        t_pos_vec    = model.type_emb(true_t)
        diff_pos     = prot_vecs.unsqueeze(1) + rel - t_pos_vec.unsqueeze(1)
        dist_pos     = torch.norm(diff_pos, p=2, dim=-1).squeeze(1)
        all_tv_exp   = all_type_vecs.unsqueeze(0).expand(len(batch), -1, -1)
        diff_all     = prot_vecs.unsqueeze(1) + rel - all_tv_exp
        dist_all     = torch.norm(diff_all, p=2, dim=-1)
        neg_avg      = (dist_all.sum(dim=1) - dist_pos) / (num_types - 1)
        scores       = neg_avg - dist_pos
        ranking      = torch.argsort(scores, descending=True)

        for bi in range(len(batch)):
            rank_pos = (ranking[bi] == bi).nonzero(as_tuple=False)
            if rank_pos.numel() == 0:
                rank_pos = (torch.argsort(dist_all[bi]) == true_t[bi]).nonzero(as_tuple=False)
            rank = int(rank_pos.item()) + 1
            ranks.append(rank)
            if rank <= 1:  hits1  += 1
            if rank <= 3:  hits3  += 1
            if rank <= 10: hits10 += 1

    n = len(ranks)
    return {
        "MRR":    sum(1.0/r for r in ranks) / n if n > 0 else 0.0,
        "Hits@1": hits1/n  if n > 0 else 0.0,
        "Hits@3": hits3/n  if n > 0 else 0.0,
        "Hits@10":hits10/n if n > 0 else 0.0,
        "Count":  n,
    }


# -----------------------
# Training（改动：训练循环里加载并传入 AG embedding）
# -----------------------
def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    logger.info("Loading train/val data dirs ...")
    p2id_tr, t2id_tr, r2id_tr, id2seq_tr, pos_tr_triples = read_kge_dir(args.train_dir)
    p2id_te, t2id_te, r2id_te, id2seq_te, pos_te_triples = read_kge_dir(args.val_dir)
    num_types = len(t2id_tr)

    train_ds     = KGETripletDataset(pos_tr_triples, id2seq_tr, max_len=args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=lambda x: x)

    # ── 加载 AG embedding（新增）────────────────────────────
    ag_emb_dict = {}
    use_ag      = False
    if args.ag_emb_dir and os.path.isdir(args.ag_emb_dir):
        ag_emb_dict = load_all_ag_embeddings(args.ag_emb_dir)
        use_ag      = len(ag_emb_dict) > 0
        if use_ag:
            print(f"[AG fusion] 启用，ProteinEncoder proj 输入维度 = {1024 + AG_DIM}")
        else:
            print("[AG fusion] 目录为空，退回纯 ProtBERT 模式")
    else:
        print("[AG fusion] 未配置 --ag-emb-dir，使用纯 ProtBERT 模式")

    # ── 构建模型 ─────────────────────────────────────────────
    logger.info(f"Loading ProtBERT from: {args.protbert_path}")
    tokenizer    = AutoTokenizer.from_pretrained(
        args.protbert_path, local_files_only=True, do_lower_case=False)
    prot_encoder = ProteinEncoder(
        model_path=args.protbert_path,
        proj_dim=args.embedding_dim,
        freeze=not args.unfreeze_encoder,
        use_ag=use_ag,                     # ← 新增
    )
    model = TransEProteinType(
        prot_encoder=prot_encoder,
        num_types=num_types,
        dim=args.embedding_dim,
        p_norm=2,
    ).to(device)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"Trainable params: {sum(p.numel() for p in params):,}")

    global_step = 0; best_mrr = -1.0
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "train.log")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            p_pos    = [p for p, _, _ in batch]
            t_pos_ids = torch.tensor([t for _, t, _ in batch], dtype=torch.long, device=device)
            seqs     = [seq for _, _, seq in batch]

            all_input_ids, all_attn_masks = tokenize_sequences(
                tokenizer, seqs, max_len=args.max_seq_len, device=device)
            flat_input_ids  = torch.cat(all_input_ids,  dim=0)
            flat_attn_masks = torch.cat(all_attn_masks, dim=0)

            # ── 准备 AG embedding（新增）────────────────────
            if use_ag and ag_emb_dict:
                chunk_counts = [x.size(0) for x in all_input_ids]
                ag_per_seq   = []
                for pid in p_pos:
                    # id2seq_tr 的 key 是 int；ag_emb_dict 的 key 是 orf.faa 里的 seq_id 字符串
                    # 两者的对应关系需要通过 sequence.txt 里的序列去反查
                    # 简单策略：先直接查 str(pid)，找不到再补零
                    ag_np = (ag_emb_dict.get(str(pid)) or
                             np.zeros(AG_DIM, dtype=np.float32))
                    ag_per_seq.append(ag_np)
                # 把每条序列的 AG embedding 广播到它的所有 chunk 上
                ag_flat = np.repeat(ag_per_seq, chunk_counts, axis=0)
                ag_t    = torch.tensor(ag_flat, dtype=torch.float32, device=device)
            else:
                ag_t = None

            flat_vecs    = model.prot_encoder(flat_input_ids, flat_attn_masks, ag_emb=ag_t)
            chunk_counts = [x.size(0) for x in all_input_ids]
            h_vecs       = torch.split(flat_vecs, chunk_counts, dim=0)
            h_vec        = torch.stack([c.mean(dim=0) for c in h_vecs], dim=0)  # (B, D)

            all_type_ids  = torch.arange(num_types, device=device)
            all_type_vecs = model.type_emb(all_type_ids).unsqueeze(0).expand(len(batch), -1, -1)
            rel           = model.rel.unsqueeze(0)
            diff_all      = h_vec.unsqueeze(1) + rel - all_type_vecs
            dist_all      = torch.norm(diff_all, p=2, dim=-1)
            dist_pos      = dist_all.gather(1, t_pos_ids.unsqueeze(1)).squeeze(1)
            neg_avg       = (dist_all.sum(dim=1) - dist_pos) / (num_types - 1)
            loss          = torch.log1p(torch.exp(
                (dist_pos - neg_avg).clamp(-10, 10))).mean() * args.ke_lambda

            (loss / args.gradient_accumulation_steps).backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            if step % args.gradient_accumulation_steps == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)

            epoch_loss  += loss.item() * args.gradient_accumulation_steps
            global_step += 1
            step_loss    = loss.item() * args.gradient_accumulation_steps
            print(f"Epoch {epoch} | Step {step} | Global {global_step} | loss {step_loss:.4f}")
            with open(log_file_path, "a") as f:
                f.write(f"Epoch {epoch} | Step {step} | Global {global_step} | loss {step_loss:.4f}\n")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        dt = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f} | time={dt:.1f}s")
        print(f"Epoch {epoch} finished: avg_loss={avg_loss:.4f} | time={dt:.1f}s")

        metrics = evaluate(
            model=model, tokenizer=tokenizer,
            id2seq=id2seq_te, pos_triples=pos_te_triples,
            ag_emb_dict=ag_emb_dict,               # ← 新增
            batch_size=args.eval_batch_size,
            max_len=args.max_seq_len, device=device,
            sample_limit=args.eval_sample_limit,
        )
        logger.info(f"[Eval] Epoch {epoch}: {json.dumps(metrics, ensure_ascii=False)}")
        print(f"[Eval] Epoch {epoch}: {metrics}")

        if metrics["MRR"] > best_mrr:
            best_mrr  = metrics["MRR"]
            save_path = os.path.join(args.output_dir, "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "args":        vars(args),
                "type2id":     t2id_tr,
            }, save_path)
            logger.info(f"Saved best checkpoint (MRR={best_mrr:.4f}) → {save_path}")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    final_path = os.path.join(args.output_dir, "last_full_model.pt")
    model.args    = vars(args)
    model.type2id = t2id_tr
    torch.save(model, final_path)
    logger.info(f"Saved full model → {final_path}")
    print(f"Saved full model → {final_path}")


# -----------------------
# CLI（新增 --ag-emb-dir）
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train TransE ProteinType (+ AG fusion)")

    parser.add_argument("--train_dir",     type=str, required=True)
    parser.add_argument("--val_dir",       type=str, required=True)
    parser.add_argument("--pretrain_data_dir", type=str, default=None)
    parser.add_argument("--protbert_path", type=str, default="./ProtBERT")
    parser.add_argument("--protein_encoder_cls", type=str, default="bert")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--ke_embedding_size", type=int, default=256)
    parser.add_argument("--double_entity_embedding_size", type=bool, default=False)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--do_train",      action="store_true")
    parser.add_argument("--output_dir",    type=str, default="./output")
    parser.add_argument("--logging_dir",   type=str, default="./logs")
    parser.add_argument("--epochs",        type=int, default=5)
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_steps",     type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--unfreeze_encoder", action="store_true")
    parser.add_argument("--num_negs",      type=int, default=4)
    parser.add_argument("--max_seq_len",   type=int, default=1024)
    parser.add_argument("--ke_lambda",     type=float, default=1.0)
    parser.add_argument("--ke_learning_rate", type=float, default=3e-5)
    parser.add_argument("--ke_max_score",  type=float, default=12.0)
    parser.add_argument("--ke_score_fn",   type=str, default="transE")
    parser.add_argument("--ke_warmup_steps", type=int, default=400)
    parser.add_argument("--margin",        type=float, default=0.2)
    parser.add_argument("--deepspeed",     type=str, default=None)
    parser.add_argument("--fp16",          action="store_true")
    parser.add_argument("--local_rank",    type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", action="store_true")
    parser.add_argument("--optimize_memory", action="store_true")
    parser.add_argument("--no_cuda",       action="store_true")
    parser.add_argument("--log_every",     type=int, default=500)
    parser.add_argument("--eval_sample_limit", type=int, default=200)
    parser.add_argument("--seed",          type=int, default=42)
    # ── 新增 ──────────────────────────────────────────────────
    parser.add_argument(
        "--ag-emb-dir",
        type=str,
        default=None,
        help="AlphaGenome embedding 目录（含 CRBC_G000X.npz）。"
             "不传则退回纯 ProtBERT 模式",
    )

    args = parser.parse_args()
    if args.eval_sample_limit is not None and args.eval_sample_limit < 0:
        args.eval_sample_limit = None
    return args


if __name__ == "__main__":
    args = parse_args()
    print("[INFO] Args:", args)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)