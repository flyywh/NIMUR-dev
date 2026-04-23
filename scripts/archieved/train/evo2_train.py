import torch
from torch import nn
import pandas as pd
import numpy as np
import warnings
import os
import time
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import json
import pickle
import argparse
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import bisect

# 设置可见GPU - 改为通过参数传递或环境变量设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


class Evo2Esm2EncodedTransformer(nn.Module):
    def __init__(
        self,
        input_dim=1280,
        ffn_dim=1920,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        max_seq_len=1024,
    ):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=False,
                norm_first=False,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 3, 1),
        )
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, mask):
        X = X.permute(1, 0, 2)
        output = self.transformer(X, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)
        return self.classifier(output).squeeze(-1)


class DnaDataset(Dataset):
    def __init__(self, data_dict):
        self.items = list(data_dict.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Return (DNA feature tensor, label)
        name, (dna_tensor, label) = self.items[idx]
        return dna_tensor, label


class ShardedDnaDataset(Dataset):
    """Lazy-loading dataset for sharded evo2 manifests."""

    def __init__(self, manifest, base_dir):
        self.base_dir = Path(base_dir)
        self.shards = manifest["shards"]
        self.cum_counts = []
        total = 0
        for s in self.shards:
            total += int(s["count"])
            self.cum_counts.append(total)
        self.total = total
        self._cache_idx = None
        self._cache_payload = None
        self._load_count = 0
        self._debug_io = str(os.getenv("EVO2_DEBUG_IO", "0")).lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    def __len__(self):
        return self.total

    def _load_shard(self, shard_idx):
        if self._cache_idx == shard_idx and self._cache_payload is not None:
            return self._cache_payload
        rel = self.shards[shard_idx]["path"]
        p = self.base_dir / rel
        t0 = time.time()
        payload = torch.load(p, weights_only=False)
        dt = time.time() - t0
        self._cache_idx = shard_idx
        self._cache_payload = payload
        self._load_count += 1
        if self._debug_io:
            print(
                f"[EVO2-IO] loaded shard {shard_idx+1}/{len(self.shards)} "
                f"({self.shards[shard_idx].get('count','?')} samples) "
                f"in {dt:.2f}s: {p}"
            )
        return payload

    def __getitem__(self, idx):
        shard_idx = bisect.bisect_right(self.cum_counts, idx)
        prev = 0 if shard_idx == 0 else self.cum_counts[shard_idx - 1]
        local_idx = idx - prev
        payload = self._load_shard(shard_idx)
        emb = payload["embeddings"][local_idx]
        label = payload["labels"][local_idx]
        if emb.dtype != torch.float32:
            emb = emb.float()
        return emb, float(label)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DNA feature model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config file"
    )
    parser.add_argument(
        "--train_dataset", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--test_dataset", type=str, required=True, help="Path to test dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for results",
    )
    return parser.parse_args()


def load_dataset(path):
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, dict) and obj.get("format") == "sharded_v1":
        p = Path(path)
        rel0 = obj["shards"][0]["path"] if obj.get("shards") else None
        candidates = [p.parent, p.parent / "train_build", p.parent / "test_build"]
        base_dir = p.parent
        if rel0 is not None:
            for cand in candidates:
                if (cand / rel0).exists():
                    base_dir = cand
                    break
        return ShardedDnaDataset(obj, base_dir=base_dir)
    if isinstance(obj, dict):
        # legacy dict format: name -> (tensor, label)
        return DnaDataset(obj)
    return obj


def collate_fn(batch, max_seq_len=None):
    """Collate function for DNA dataset"""
    embeddings, labels = zip(*batch)
    if max_seq_len is not None:
        embeddings = tuple(e[:max_seq_len] for e in embeddings)
    padded_embs = pad_sequence(embeddings, batch_first=True, padding_value=0)
    mask = (padded_embs.sum(dim=-1) != 0).float()
    labels = torch.tensor(labels)
    return padded_embs, labels, mask


def create_dataloader(data, batch_size, shuffle, max_seq_len=None):
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, max_seq_len=max_seq_len),
    )
    return dataloader


def calculate_loss(outputs, y, mask):
    valid_outputs = outputs * mask
    sums = mask.sum(dim=1)
    valid_outputs_sums = valid_outputs.sum(dim=1)
    predictions = valid_outputs_sums / sums
    return F.binary_cross_entropy_with_logits(predictions, y)


def plot_metrics(
    num_epochs,
    train_losses,
    valid_losses,
    output_dir,
    fig_name="training_metrics.png",
    fig_log_name="training_metrics_log.png",
):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(output_dir / fig_name)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.legend()
    plt.grid()
    plt.savefig(output_dir / fig_log_name)
    plt.close()

    # Save loss data
    np.savetxt(
        output_dir / "losses.csv",
        np.column_stack((train_losses, valid_losses)),
        delimiter=",",
        header="Train Loss,Validation Loss",
        comments="",
    )


def plot_roc_curve(fpr, tpr, roc_auc, epoch, output_dir):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (Epoch {epoch})")
    plt.legend(loc="lower right")
    (output_dir / "roc_figs").mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "roc_figs" / f"roc_epoch_{epoch}.png")
    plt.close()


def find_best_threshold(labels, preds):
    """Find the best threshold for classification"""
    best_threshold = 0
    best_accuracy = 0
    thresholds = np.arange(0, 1.0001, 0.0001)

    for threshold in thresholds:
        pred_labels = (preds >= threshold).astype(int)
        accuracy = accuracy_score(labels, pred_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def plot_prediction_distribution(
    labels, preds, epoch, best_threshold, output_dir, dataset_type="validation"
):
    """Plot prediction distribution and confusion matrix"""
    plt.figure(figsize=(12, 5))

    # Plot prediction probability distribution
    plt.subplot(1, 2, 1)
    plt.hist(preds[labels == 0], bins=50, alpha=0.7, label="Negative", color="red")
    plt.hist(preds[labels == 1], bins=50, alpha=0.7, label="Positive", color="green")
    plt.axvline(
        x=best_threshold,
        color="blue",
        linestyle="--",
        label=f"Best Threshold: {best_threshold:.4f}",
    )
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title(
        f"{dataset_type.capitalize()} Set Prediction Distribution (Epoch {epoch})"
    )
    plt.legend()

    # Plot confusion matrix
    plt.subplot(1, 2, 2)
    pred_labels = (preds >= best_threshold).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    )
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Threshold: {best_threshold:.4f})")

    plt.tight_layout()
    (output_dir / "distribution_figs").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir
        / "distribution_figs"
        / f"{dataset_type}_distribution_epoch_{epoch}.png"
    )
    plt.close()


def evaluate_model(model, dataloader, device):
    """Evaluate model on given dataset"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y, mask in dataloader:
            X, y, mask = (
                X.to(device).float(),
                y.to(device).float(),
                mask.to(device).bool(),
            )
            outputs = model(X, mask)

            loss = calculate_loss(outputs, y, mask)
            total_loss += loss.item() * X.shape[0]
            total_samples += X.shape[0]

            # Calculate mean prediction for each sample
            masked_output = outputs * mask
            valid_counts = mask.sum(dim=1).clamp(min=1)
            predictions = torch.sigmoid(masked_output.sum(dim=1) / valid_counts)

            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    # Find best threshold
    best_threshold, best_accuracy = find_best_threshold(all_labels, all_preds)

    return (
        avg_loss,
        fpr,
        tpr,
        roc_auc,
        all_preds,
        all_labels,
        best_threshold,
        best_accuracy,
    )


def train(config, train_dataset_path, test_dataset_path, output_dir):

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("PID: ", os.getpid())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

    # Create output directories
    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "roc_figs").mkdir(parents=True, exist_ok=True)
    (output_dir / "distribution_figs").mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    start_time = time.time()

    # Load datasets
    train_dataset = load_dataset(train_dataset_path)
    test_dataset = load_dataset(test_dataset_path)

    print(f"Data loaded in {time.time()-start_time:.2f} seconds.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize model
    model_config = config["model"]
    train_config = config["training"]
    max_seq_len = train_config.get("max_seq_len", model_config.get("max_seq_len", 4096))
    print(f"Using max_seq_len={max_seq_len} for training/eval collation")
    model = Evo2Esm2EncodedTransformer(
        input_dim=model_config["input_dim"],
        ffn_dim=model_config["ffn_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        max_seq_len=max_seq_len,
    )

    # Use DataParallel only when explicitly enabled.
    use_dp = str(os.getenv("EVO2_USE_DP", "0")).lower() in {"1", "true", "yes", "y"}
    if torch.cuda.device_count() > 1 and use_dp:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        print(
            f"Detected {torch.cuda.device_count()} visible GPUs, DataParallel disabled "
            f"(set EVO2_USE_DP=1 to enable)."
        )
    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )

    # Create data loaders
    train_loader = create_dataloader(
        train_dataset, train_config["batch_size"], shuffle=True, max_seq_len=max_seq_len
    )
    test_loader = create_dataloader(
        test_dataset, train_config["batch_size"], shuffle=False, max_seq_len=max_seq_len
    )

    train_losses, test_losses = [], []
    print("Starting training...")
    log_every = int(os.getenv("EVO2_LOG_EVERY", "20"))
    print(f"[EVO2-STATUS] log_every={log_every} (set EVO2_LOG_EVERY to change)")

    for epoch in range(train_config["num_epochs"]):
        # Training phase
        model.train()
        train_loss, total_samples = 0.0, 0
        start_time = time.time()
        prev_end = time.time()
        total_steps = len(train_loader)
        print(
            f"[EVO2-STATUS] epoch {epoch+1}/{train_config['num_epochs']} "
            f"start, total_steps={total_steps}"
        )

        for step, (X, y, mask) in enumerate(train_loader, start=1):
            data_wait = time.time() - prev_end
            step_t0 = time.time()
            X, y, mask = (
                X.to(device).float(),
                y.to(device).float(),
                mask.to(device).bool(),
            )
            optimizer.zero_grad()
            outputs = model(X, mask)
            loss = calculate_loss(outputs, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config["clip_grad"]
            )
            optimizer.step()

            train_loss += loss.item() * X.shape[0]
            total_samples += X.shape[0]
            step_dt = time.time() - step_t0
            prev_end = time.time()

            if step == 1 or step % log_every == 0 or step == total_steps:
                mem_alloc = (
                    torch.cuda.memory_allocated(device) / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                mem_reserved = (
                    torch.cuda.memory_reserved(device) / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                print(
                    f"[EVO2-STATUS][train] epoch={epoch+1} step={step}/{total_steps} "
                    f"data_wait={data_wait:.2f}s step_time={step_dt:.2f}s "
                    f"loss={loss.item():.4f} mem_alloc={mem_alloc:.2f}GB mem_reserved={mem_reserved:.2f}GB"
                )

        train_time = time.time() - start_time
        avg_train_loss = train_loss / total_samples if total_samples > 0 else 0
        train_losses.append(avg_train_loss)

        # Evaluation phase
        start_time = time.time()
        print(f"[EVO2-STATUS] epoch {epoch+1} entering evaluation")
        (
            test_loss,
            test_fpr,
            test_tpr,
            test_auc,
            test_preds,
            test_labels,
            best_threshold_test,
            best_accuracy_test,
        ) = evaluate_model(model, test_loader, device)

        test_time = time.time() - start_time
        test_losses.append(test_loss)

        print(
            f"Epoch [{epoch+1}/{train_config['num_epochs']}], "
            f"Train Time: {train_time:.2f}s, Test Time: {test_time:.2f}s"
        )
        print(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(
            f"Test - Best Threshold: {best_threshold_test:.4f}, Best Accuracy: {best_accuracy_test:.4f}, AUC: {test_auc:.4f}"
        )
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(
            model_to_save.state_dict(),
            output_dir / "models" / f"model_epoch_{epoch+1}.pth",
        )
        print(f"Model saved as {output_dir / 'models' / f'model_epoch_{epoch+1}.pth'}")

        # Plot ROC curve
        plot_roc_curve(test_fpr, test_tpr, test_auc, epoch + 1, output_dir)

        # Plot prediction distribution
        plot_prediction_distribution(
            test_labels, test_preds, epoch + 1, best_threshold_test, output_dir, "test"
        )

        # Plot training metrics
        plot_metrics(epoch + 1, train_losses, test_losses, output_dir)

        print("-" * 80)


if __name__ == "__main__":
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    print("Current configuration:")
    print(json.dumps(config, indent=4))

    train(config, args.train_dataset, args.test_dataset, Path(args.output_dir))
