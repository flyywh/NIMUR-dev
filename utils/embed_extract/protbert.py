from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class ProtBERTEmbedder:
    def __init__(self, model_path: str | Path, device: str | torch.device | None = None):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, do_lower_case=False
        )
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(self.device)
        self.model.eval()

    def encode_mean(self, seq_id: str, seq: str, max_len: int = 1024) -> np.ndarray:
        seq = seq.upper().replace(" ", "")
        chunks = [seq[i : i + max_len] for i in range(0, len(seq), max_len)]
        enc = self.tokenizer(
            chunks,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
        last = out.last_hidden_state
        mask = enc["attention_mask"].to(self.device).unsqueeze(-1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.mean(dim=0).float().cpu().numpy()

