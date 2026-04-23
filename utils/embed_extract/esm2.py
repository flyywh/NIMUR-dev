from typing import Dict

import numpy as np
import torch
import esm


class ESM2Embedder:
    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        layer: int = 33,
        model_name: str = "esm2_t33_650M_UR50D",
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.layer = layer
        self.model_name = model_name
        self.model, self.batch_converter = self._load()

    def _load(self):
        if self.model_name == "esm2_t33_650M_UR50D":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        model = model.to(self.device)
        model.eval()
        return model, alphabet.get_batch_converter()

    @staticmethod
    def _normalize_seq(seq: str) -> str:
        return seq.upper().replace(" ", "")

    def encode_matrix(self, seq_id: str, seq: str) -> np.ndarray:
        seq = self._normalize_seq(seq)
        _, _, tokens = self.batch_converter([(seq_id, seq)])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[self.layer])
        rep = out["representations"][self.layer][0, 1:-1, :]
        if rep.numel() == 0:
            raise ValueError(f"ESM empty token representation: {seq_id}")
        return rep.float().cpu().numpy()

    def encode_matrix_with_oom_fallback(self, seq_id: str, seq: str) -> np.ndarray:
        try:
            return self.encode_matrix(seq_id, seq)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            with torch.no_grad():
                model_cpu = self.model.to("cpu")
                _, _, tokens = self.batch_converter([(seq_id, self._normalize_seq(seq))])
                out = model_cpu(tokens.to("cpu"), repr_layers=[self.layer])
                rep = out["representations"][self.layer][0, 1:-1, :]
            self.model = model_cpu.to(self.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if rep.numel() == 0:
                raise ValueError(f"ESM empty token representation: {seq_id}")
            return rep.float().cpu().numpy()

    def encode_mean_chunked(self, seq_id: str, seq: str, chunk_len: int = 1022) -> np.ndarray:
        seq = self._normalize_seq(seq)
        chunks = [seq[i : i + chunk_len] for i in range(0, len(seq), chunk_len)]
        pooled = []
        for idx, ck in enumerate(chunks, start=1):
            arr = self.encode_matrix(f"{seq_id}__chunk{idx}", ck)
            if arr.size > 0:
                pooled.append(torch.tensor(arr).mean(dim=0))
        if not pooled:
            raise ValueError(f"ESM empty token representation for {seq_id}")
        return torch.stack(pooled, dim=0).mean(dim=0).numpy()

    def fasta_to_embeddings(
        self,
        seq_rows: list[tuple[str, str]],
        *,
        oom_fallback: bool = True,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for seq_id, seq in seq_rows:
            if oom_fallback:
                out[seq_id] = self.encode_matrix_with_oom_fallback(seq_id, seq)
            else:
                out[seq_id] = self.encode_matrix(seq_id, seq)
        return out

