from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from evo2 import Evo2


class Evo2Embedder:
    def __init__(
        self,
        layer_name: str = "blocks.21.mlp.l3",
        device: str | torch.device | None = None,
        max_len: int = 16384,
    ):
        self.layer_name = layer_name
        self.device = torch.device(
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.max_len = max_len
        self.model = Evo2()
        self.tokenizer = self.model.tokenizer

    @staticmethod
    def clean_dna(seq: str) -> str:
        return re.sub(r"[^ACGTN]", "", seq.upper())

    def _prepare_sequence(self, seq_id: str, seq: str) -> str:
        dna = self.clean_dna(seq)
        if not dna:
            raise ValueError(f"Evo2 sequence empty after DNA clean: {seq_id}")
        if self.max_len is not None and len(dna) > self.max_len:
            dna = dna[: self.max_len]
        return dna

    def encode_matrix(self, seq_id: str, seq: str) -> np.ndarray:
        dna = self._prepare_sequence(seq_id, seq)
        tokens = self.tokenizer.tokenize(dna)
        input_ids = torch.tensor(tokens, dtype=torch.int).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, embeddings = self.model(
                input_ids, return_embeddings=True, layer_names=[self.layer_name]
            )
        rep = embeddings[self.layer_name].squeeze(0)
        if rep.numel() == 0:
            raise ValueError(f"Evo2 empty token representation: {seq_id}")
        return rep.float().cpu().numpy()

    def encode_mean(self, seq_id: str, seq: str) -> np.ndarray:
        return self.encode_matrix(seq_id, seq).mean(axis=0).astype(np.float32)

    def save_embedding_csv(
        self,
        seq_id: str,
        seq: str,
        out_root: str | Path,
        *,
        stem: str | None = None,
        compression: str = "gzip",
    ) -> Path:
        arr = self.encode_matrix(seq_id, seq)
        out_dir = Path(out_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        columns = ["token_idx"] + [f"dim{i}" for i in range(arr.shape[1])]
        rows = [[i] + vec.tolist() for i, vec in enumerate(arr)]
        df = pd.DataFrame(rows, columns=columns)

        target_stem = stem if stem else seq_id
        if compression == "gzip":
            out_path = out_dir / f"{target_stem}.csv.gz"
            df.to_csv(out_path, index=False, compression="gzip")
        else:
            out_path = out_dir / f"{target_stem}.csv"
            df.to_csv(out_path, index=False)
        return out_path

