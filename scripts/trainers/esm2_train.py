import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import warnings
import time
from tqdm import tqdm
import pickle
import logging

warnings.filterwarnings("ignore")
import json
import argparse
import os
from torch.nn.utils.rnn import pad_sequence
import dill
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing dataset files",
    )
    parser.add_argument(
        "--test_dataset_path", type=str, required=True, help="Path to test dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for results",
    )
    return parser.parse_args()


def plot_prediction_distribution(
    valid_labels,
    valid_preds,
    test_labels,
    test_preds,
    epoch,
    best_threshold_valid,
    best_threshold_test,
    output_dir,
):
    """绘制验证集和测试集的预测分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制验证集分布
    axes[0].hist(
        valid_preds[valid_labels == 0],
        bins=100,
        alpha=0.7,
        color="red",
        label="Negative",
    )
    axes[0].hist(
        valid_preds[valid_labels == 1],
        bins=100,
        alpha=0.7,
        color="green",
        label="Positive",
    )
    axes[0].axvline(
        x=best_threshold_valid,
        color="blue",
        linestyle="--",
        label=f"Best Threshold: {best_threshold_valid:.4f}",
    )
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Validation Set Prediction Distribution (Epoch {epoch})")
    axes[0].legend()
    axes[0].grid(True)

    # 绘制测试集分布
    axes[1].hist(
        test_preds[test_labels == 0], bins=100, alpha=0.7, color="red", label="Negative"
    )
    axes[1].hist(
        test_preds[test_labels == 1],
        bins=100,
        alpha=0.7,
        color="green",
        label="Positive",
    )
    axes[1].axvline(
        x=best_threshold_test,
        color="blue",
        linestyle="--",
        label=f"Best Threshold: {best_threshold_test:.4f}",
    )
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Test Set Prediction Distribution (Epoch {epoch})")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(
        output_dir
        / "valid_test_distributions"
        / f"prediction_distribution_epoch_{epoch}.png"
    )
    plt.close()


def plot_metrics(
    num_epochs,
    train_losses,
    valid_losses,
    test_losses=None,
    variance_train_losses=None,
    variance_valid_losses=None,
    fig_path=None,
    fig_log_path=None,
):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    if test_losses is not None:
        plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    if variance_train_losses is not None:
        plt.plot(
            range(1, num_epochs + 1), variance_train_losses, label="Variance Train Loss"
        )
    if variance_valid_losses is not None:
        plt.plot(
            range(1, num_epochs + 1),
            variance_valid_losses,
            label="Variance Validation Loss",
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training, Validation and Test Loss")
    plt.legend()
    plt.grid()
    plt.savefig(fig_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    if test_losses is not None:
        plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss")
    if variance_train_losses is not None:
        plt.plot(
            range(1, num_epochs + 1), variance_train_losses, label="Variance Train Loss"
        )
    if variance_valid_losses is not None:
        plt.plot(
            range(1, num_epochs + 1),
            variance_valid_losses,
            label="Variance Validation Loss",
        )
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.title("Training, Validation and Test Loss (Log Scale)")
    plt.legend()
    plt.grid()
    plt.savefig(fig_log_path)
    plt.close()

    # 保存数据
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    if test_losses is not None:
        np.savetxt(
            data_dir / f"losses_{num_epochs}",
            np.column_stack((train_losses, valid_losses, test_losses)),
            delimiter=",",
            header="Train Loss,Validation Loss,Test Loss",
            comments="",
        )
    else:
        np.savetxt(
            data_dir / f"losses_{num_epochs}",
            np.column_stack((train_losses, valid_losses)),
            delimiter=",",
            header="Train Loss,Validation Loss",
            comments="",
        )


def plot_roc_curve(fpr, tpr, roc_auc, epoch, dataset_type="valid", output_dir=None):
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (Epoch {epoch}, {dataset_type})")
    plt.legend(loc="lower right")

    if dataset_type == "test":
        plt.savefig(output_dir / "test_rocfigs" / f"roc_epoch_{epoch}.png")
    else:
        plt.savefig(output_dir / "rocfigs" / f"roc_epoch_{epoch}.png")
    plt.close()


def load_data(path):
    with open(path, "rb") as f:
        data_loader = dill.load(f)
    return data_loader


class BgcTransformer(nn.Module):
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
        # 位置编码（最大序列长度设为1024）
        self.pos_encoder = nn.Embedding(max_seq_len, input_dim)

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=False,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 3, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_encoder.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)
        output = self.transformer(x, src_key_padding_mask=mask)
        output = output.permute(1, 0, 2)  # (batch, seq_len, dim)
        return self.classifier(output).squeeze(-1)


def calculate_loss(output: torch.Tensor, label: torch.Tensor, mask: torch.Tensor):
    # output should not be sigmoided
    batch_size = output.shape[0]
    predictions = []
    for i in range(batch_size):
        valid_output = output[i][mask[i]]
        predictions.append(valid_output.mean())
    predictions = torch.stack(predictions)
    loss = F.binary_cross_entropy_with_logits(predictions, label.float())
    return loss


def load_data_dict(path):
    """从文件加载数据字典"""
    return torch.load(path, weights_only=False)


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.items = list(data_dict.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        emb, label = self.items[idx][1]
        return emb, label

    def __add__(self, other):
        new_dict = dict(self.items + other.items)
        return ProteinDataset(new_dict)


def collate_fn(batch):
    """batch: List[(embeddings, labels)]"""
    embeddings, labels = zip(*batch)
    padded_embs = pad_sequence(
        embeddings, batch_first=True, padding_value=0
    )  # (B, L, 1280)
    padded_labs = pad_sequence(labels, batch_first=True, padding_value=-1)  # (B, L)
    mask = (padded_embs.sum(dim=-1) != 0).float()  # (B, L)
    return padded_embs, padded_labs, mask


def find_best_threshold(labels, preds):
    """在0到1之间以0.0001为间隔找出最佳的阈值"""
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


def plot_test_distribution(labels, preds, epoch, output_dir):
    """绘制测试集的数据分布"""
    plt.figure(figsize=(12, 5))

    # 绘制预测概率分布
    plt.subplot(1, 2, 1)
    plt.hist(preds[labels == 0], bins=50, alpha=0.7, label="Negative", color="red")
    plt.hist(preds[labels == 1], bins=50, alpha=0.7, label="Positive", color="green")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title(f"Test Set Prediction Distribution (Epoch {epoch})")
    plt.legend()

    # 绘制混淆矩阵
    plt.subplot(1, 2, 2)
    best_threshold, best_accuracy = find_best_threshold(labels, preds)
    pred_labels = (preds >= best_threshold).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    )
    disp.plot(cmap="Blues")
    plt.title(
        f"Confusion Matrix (Threshold: {best_threshold:.4f}, Acc: {best_accuracy:.4f})"
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "test_distributions" / f"test_distribution_epoch_{epoch}.png"
    )
    plt.close()


def evaluate_model(model, dataloader, device, dataset_type="valid"):
    """评估模型在给定数据集上的性能"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for X, y, mask in dataloader:
            X, y, mask = (
                X.to(device).float(),
                y.to(device).float(),
                mask.to(device).bool(),
            )
            outputs = model(X, mask)

            loss = calculate_loss(outputs, y[:, 0], mask)
            total_loss += loss.item() * X.shape[0]
            total_samples += X.shape[0]

            # 计算每个样本的平均预测值
            batch_size = X.shape[0]
            predictions = []
            for i in range(batch_size):
                valid_output = outputs[i][mask[i]]
                predictions.append(valid_output.mean())
            predictions = torch.stack(predictions)
            predictions = torch.sigmoid(predictions)

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(y[:, 0].cpu().numpy())

    eval_time = time.time() - start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    # 找到最佳阈值
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
        eval_time,
    )


def train_model(config, dataset_dir, test_dataset_path, output_dir):

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("pid: ", os.getpid())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")
    output_dir = Path(output_dir)
    (output_dir / "Wet_rocfigs").mkdir(parents=True, exist_ok=True)
    (output_dir / "test_distributions").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    FIG_PATH = output_dir / "figs.png"
    FIG_LOG_PATH = output_dir / "figs_log.png"
    SAVE_DIR = Path(dataset_dir)

    print("Loading data...")
    start_time = time.time()
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=32) as executor:
        future_train = executor.submit(
            load_data_dict, SAVE_DIR / "train_dataset_esm2.pt"
        )
        future_valid = executor.submit(
            load_data_dict, SAVE_DIR / "valid_dataset_esm2.pt"
        )
        future_test = executor.submit(load_data_dict, test_dataset_path)

        train_dataset = future_train.result()
        valid_dataset = future_valid.result()
        test_dataset = future_test.result()
    print(f"Data loaded in {time.time() - start_time:.2f}s")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print("Creating DataLoaders...")
    start_time = time.time()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Data loaded in {time.time() - start_time:.2f}s")

    # 模型初始化
    model_config = config["model"]
    model = BgcTransformer(
        input_dim=model_config["input_dim"],
        ffn_dim=model_config["ffn_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    model = model.to(device)

    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )

    train_losses, valid_losses, test_losses = [], [], []
    print("Starting training...")
    for epoch in range(train_config["num_epochs"]):
        model.train()
        train_loss = 0.0
        total_samples = 0
        start_time = time.time()
        for X, y, mask in train_dataloader:
            X, y, mask = (
                X.to(device).float(),
                y.to(device).float(),
                mask.to(device).bool(),
            )
            optimizer.zero_grad()
            outputs = model(X, mask)

            loss = calculate_loss(outputs, y[:, 0], mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config["clip_grad"]
            )
            optimizer.step()
            train_loss += loss.item() * X.shape[0]
            total_samples += X.shape[0]

        train_time = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{train_config['num_epochs']}], Training Time: {train_time:.2f} seconds"
        )
        start_time = time.time()
        (
            valid_loss,
            valid_fpr,
            valid_tpr,
            valid_auc,
            valid_preds,
            valid_labels,
            best_threshold_valid,
            best_accuracy_valid,
            valid_time,
        ) = evaluate_model(model, valid_dataloader, device, "valid")

        print(
            f"Epoch [{epoch+1}/{train_config['num_epochs']}], Validation Time: {valid_time:.2f} seconds"
        )
        start_time = time.time()
        (
            test_loss,
            test_fpr,
            test_tpr,
            test_auc,
            test_preds,
            test_labels,
            best_threshold_test,
            best_accuracy_test,
            test_time,
        ) = evaluate_model(model, test_dataloader, device, "test")

        print(
            f"Epoch [{epoch+1}/{train_config['num_epochs']}], Test Time: {test_time:.2f} seconds"
        )
        print(f"Test Prediction Time: {test_time:.4f} seconds")

        avg_train_loss = train_loss / total_samples if total_samples > 0 else 0

        train_losses.append(avg_train_loss)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch [{epoch+1}/{train_config['num_epochs']}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}"
        )

        print(
            f"Validation - Best Threshold: {best_threshold_valid:.4f}, Best Accuracy: {best_accuracy_valid:.4f}"
        )
        print(
            f"Test - Best Threshold: {best_threshold_test:.4f}, Best Accuracy: {best_accuracy_test:.4f}"
        )
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(
            model_to_save.state_dict(),
            output_dir / "models" / f"model_epoch_{epoch+1}.pth",
        )
        print(f"Model saved as {output_dir / 'models' / f'model_epoch_{epoch+1}.pth'}")

        plot_roc_curve(valid_fpr, valid_tpr, valid_auc, epoch + 1, "valid", output_dir)
        plot_roc_curve(test_fpr, test_tpr, test_auc, epoch + 1, "test", output_dir)

        plot_prediction_distribution(
            valid_labels,
            valid_preds,
            test_labels,
            test_preds,
            epoch + 1,
            best_threshold_valid,
            best_threshold_test,
            output_dir,
        )

        plot_metrics(
            epoch + 1,
            train_losses,
            valid_losses,
            test_losses,
            None,
            None,
            FIG_PATH,
            FIG_LOG_PATH,
        )


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    print("Current config:")
    print(json.dumps(config, indent=4))

    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    dataset_dir = (
        Path(args.dataset_dir)
        if os.path.isabs(args.dataset_dir)
        else current_dir / args.dataset_dir
    )
    test_dataset_path = (
        Path(args.test_dataset_path)
        if os.path.isabs(args.test_dataset_path)
        else current_dir / args.test_dataset_path
    )
    output_dir = (
        Path(args.output_dir)
        if os.path.isabs(args.output_dir)
        else current_dir / args.output_dir
    )

    train_model(config, dataset_dir, test_dataset_path, output_dir)
