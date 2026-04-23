import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import argparse
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# Default output directories (can be customized with args)
FIG_DIR = Path("./figs")
LOSS_DIR = Path("./loss")
ROC_DIR = Path("./rocfigs")
MODEL_DIR = Path("./models")

# Respect externally provided CUDA_VISIBLE_DEVICES first.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")


class Evo2Esm2EncodedTransformer(nn.Module):
    """
    Transformer-based model for protein/DNA embeddings classification
    """

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
        # X: (batch_size, seq_len, input_dim)
        X = X.permute(1, 0, 2)
        output = self.transformer(X, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)
        return self.classifier(output).squeeze(-1)


class ProteinDnaDataset(Dataset):
    """
    Dataset class for Protein-DNA merged embeddings
    """

    def __init__(self, data_dict: dict):
        self.items = list(data_dict.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # returns (embedding, label)
        name, (embeddings, label) = self.items[idx]
        return embeddings, label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to combined dataset (.pt)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./output",
        help="Directory to save models and results",
    )
    return parser.parse_args()


def collate_fn(batch):
    embeddings, labels = zip(*batch)
    padded_embs = pad_sequence(embeddings, batch_first=True, padding_value=0)
    mask = (padded_embs.sum(dim=-1) != 0).float()
    labels = torch.tensor(labels)
    return padded_embs, labels, mask


def create_dataloader(data, batch_size, shuffle):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )


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
    outdir: Path,
):
    # Linear scale
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "loss.png")
    plt.close()

    # Log scale
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.legend()
    plt.grid()
    plt.savefig(outdir / "loss_log.png")
    plt.close()

    # Save numeric values
    np.savetxt(
        outdir / f"losses_{num_epochs}.csv",
        np.column_stack((train_losses, valid_losses)),
        delimiter=",",
        header="Train Loss,Validation Loss",
        comments="",
    )


def plot_roc_curve(fpr, tpr, roc_auc, epoch, outdir: Path):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Epoch {epoch})")
    plt.legend(loc="lower right")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"roc_epoch_{epoch}.png")
    plt.close()


def train(config, dataset_path: Path, outdir: Path):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("PID:", os.getpid())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPUs available: {torch.cuda.device_count()}")

    # Load dataset
    print("Loading dataset ...")
    dataset = torch.load(dataset_path)
    print(f"Dataset loaded, total samples: {len(dataset)}")

    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True)

    # Model setup
    model_config = config["model"]
    model = Evo2Esm2EncodedTransformer(
        input_dim=model_config["input_dim"],
        ffn_dim=model_config["ffn_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    )
    model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )

    train_losses, valid_losses = [], []
    train_iter = create_dataloader(train_data, train_config["batch_size"], shuffle=True)
    test_iter = create_dataloader(test_data, train_config["batch_size"], shuffle=False)

    print("Start training ...")
    for epoch in range(train_config["num_epochs"]):
        # Training
        model.train()
        train_loss, total_samples = 0.0, 0
        for X, y, mask in train_iter:
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

        # Validation
        model.eval()
        valid_loss, valid_samples = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y, mask in test_iter:
                X, y, mask = (
                    X.to(device).float(),
                    y.to(device).float(),
                    mask.to(device).bool(),
                )
                outputs = model(X, mask)
                loss = calculate_loss(outputs, y, mask)
                valid_loss += loss.item() * X.shape[0]
                valid_samples += X.shape[0]
                masked_output = outputs * mask
                valid_counts = mask.sum(dim=1).clamp(min=1)
                predictions = torch.sigmoid(masked_output.sum(dim=1) / valid_counts)
                all_preds.append(predictions.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())

        avg_train_loss = train_loss / total_samples if total_samples > 0 else 0
        avg_valid_loss = valid_loss / valid_samples if valid_samples > 0 else 0
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        print(
            f"Epoch [{epoch+1}/{train_config['num_epochs']}], "
            f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}"
        )

        # Save model
        model_dir = outdir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), model_dir / f"model_epoch_{epoch+1}.pth")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(np.concatenate(all_labels), np.concatenate(all_preds))
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc, epoch + 1, outdir / "rocfigs")

        # Plot losses
        plot_metrics(epoch + 1, train_losses, valid_losses, outdir / "loss")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    print("Current Config:")
    print(json.dumps(config, indent=4))
    train(config, Path(args.dataset), Path(args.outdir))
