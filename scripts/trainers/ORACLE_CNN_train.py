import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import esm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
from tqdm import tqdm
from pathlib import Path
from torch import nn
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# Respect externally provided CUDA_VISIBLE_DEVICES first.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
import torch.nn.functional as F


class ClassifierDataset(Dataset):
    def __init__(self, data_dict):
        self.sequences = list(data_dict.values())
        self.indices = list(data_dict.keys())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class CNNClassifierModel(nn.Module):
    """
    采用深度可分离卷积 + SE + 残差结构的 CNN 分类器
    输入：(B, L, D)
    输出：(B, 6)
    """

    def __init__(
        self,
        input_dim: int,  # embedding_size (D)
        ffn_dim: int = 512,  # 1×1 升维维度
        num_heads: int = 8,  # 未使用，留作接口
        num_layers: int = 4,  # 残差块堆叠次数
        dropout: float = 0.3,
        seq_len: int = 512,  # 仅在内部计算位置编码用，可覆盖
    ):
        super().__init__()

        self.input_dim = input_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # 1. 位置编码（可选，但通常有帮助）
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, input_dim) * 0.02)

        # 2. 入口 1×1 升维
        self.input_proj = nn.Conv1d(input_dim, ffn_dim, kernel_size=1)

        # 3. 堆叠残差块
        self.blocks = nn.ModuleList(
            [
                ResidualDSConvSEBlock(ffn_dim, kernel_size=7, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # 4. 全局平均池化 + 分类头
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(ffn_dim, ffn_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim // 2, 7),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        """
        x: (B, L, D)
        mask: (B, L)  0=pad  1=valid  用于后续改进，此处先忽略
        """
        B, L, D = x.size()
        # 统一处理：先截断到最大允许长度，然后填充到固定长度
        x = x[:, : self.seq_len, :]  # 截断超长部分
        pad_len = self.seq_len - x.size(1)  # 计算需要填充的长度
        # 在序列维度(第1维)右侧填充，特征维度(第2维)不填充
        x = F.pad(x, (0, 0, 0, pad_len))  # (B, self.seq_len, D)

        x = x + self.pos_embed  # 加位置编码
        x = x.transpose(1, 2)  # (B, D, L)

        x = self.input_proj(x)  # (B, ffn_dim, L)

        for block in self.blocks:
            x = block(x)

        x = self.pool(x).squeeze(-1)  # (B, ffn_dim)
        logits = self.classifier(x)  # (B, 6)
        return logits


# ----------------- 子模块 -----------------
class ResidualDSConvSEBlock(nn.Module):
    """
    深度可分离卷积 + SE + 残差
    """

    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.3):
        super().__init__()
        self.channels = channels

        # 深度卷积 (groups=channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        # 逐点卷积 1×1
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.Dropout(dropout),
        )

        # SE
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.GELU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

        # 归一化
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        """
        x: (B, C, L)
        """
        residual = x
        # 深度卷积
        out = self.depthwise(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)

        # SE 权重
        se_weight = self.se(out)
        out = out * se_weight

        # 逐点卷积
        out = self.pointwise(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)

        return residual + out


def plot_metrics(
    num_epochs,
    train_losses,
    valid_losses,
    train_accs,  # 新增参数
    valid_accs,  # 新增参数
    output_dir,
):
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "metrics_plot.png"
    fig_log_path = output_dir / "metrics_plot_log.png"
    fig_overall_acc_path = output_dir / "overall_acc.png"
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(fig_path)
    plt.close()

    # 2. 绘制对数损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), valid_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.legend()
    plt.grid()
    plt.savefig(fig_log_path)
    plt.close()

    # 3. 新增：绘制整体准确率曲线
    if train_accs and valid_accs:
        # 计算整体准确率（宏平均）
        overall_train_acc = [sum(acc) / len(acc) for acc in train_accs]
        overall_valid_acc = [sum(acc) / len(acc) for acc in valid_accs]

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), overall_train_acc, label="Train Accuracy")
        plt.plot(
            range(1, num_epochs + 1), overall_valid_acc, label="Validation Accuracy"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)  # 准确率范围0-1
        plt.title("Overall Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(fig_overall_acc_path)
        plt.close()

    # 保存数据
    np.savetxt(
        data_dir / "losses",
        np.column_stack((train_losses, valid_losses)),
        delimiter=",",
        header="Train Loss,Validation Loss",
        comments="",
    )

    # 新增：保存准确率数据
    if train_accs:
        np.savetxt(
            data_dir / "train_accs",
            np.array(train_accs),
            delimiter=",",
            header="Class0,Class1,Class2,Class3,Class4,Class5",
            comments="",
        )
    if valid_accs:
        np.savetxt(
            data_dir / "valid_accs",
            np.array(valid_accs),
            delimiter=",",
            header="Class0,Class1,Class2,Class3,Class4,Class5",
            comments="",
        )


def plot_accs(train_accs, valid_accs, current_epoch, output_dir):
    """绘制每个类别的训练和验证准确率曲线"""
    if not train_accs or not valid_accs:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_accs_path = output_dir / "accs_plot.png"

    num_epochs = current_epoch
    num_classes = len(train_accs[0])
    epochs_range = range(1, num_epochs + 1)

    # 设置颜色和线型
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    line_styles = ["-", "--"]

    plt.figure(figsize=(12, 8))

    # 绘制每个类别的曲线
    for i in range(num_classes):
        # 训练准确率（实线）
        plt.plot(
            epochs_range,
            [acc[i] for acc in train_accs],
            color=colors[i],
            linestyle=line_styles[0],
        )

        # 验证准确率（虚线）
        plt.plot(
            epochs_range,
            [acc[i] for acc in valid_accs],
            color=colors[i],
            linestyle=line_styles[1],
        )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Per-Class Accuracy (Up to Epoch {current_epoch})")
    plt.ylim(0, 1.0)
    plt.grid(True)

    # 优化图例显示 - 使用代理对象
    from matplotlib.lines import Line2D

    legend_elements = []

    # 添加类别颜色图例
    for i in range(num_classes):
        legend_elements.append(
            Line2D([0], [0], color=colors[i], lw=2, label=f"Class {i}")
        )

    # 添加线型图例
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="-", lw=2, label="Train")
    )
    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", lw=2, label="Validation")
    )

    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(fig_accs_path, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--train_dataset", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--valid_dataset", type=str, required=True, help="Path to validation dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for results",
    )
    return parser.parse_args()


def collate_fn(batch):
    embeddings, labels = zip(*batch)
    padded_embs = pad_sequence(embeddings, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    mask = (padded_embs.sum(dim=-1) != 0).float()
    return padded_embs, labels, mask


def load_data(train_path, valid_path, batch_size):
    print("Loading datasets...")
    start_time = time.time()
    train_dataset = torch.load(train_path, weights_only=False)
    valid_dataset = torch.load(valid_path, weights_only=False)
    print("Datasets loaded successfully.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    return train_loader, valid_loader


def train(config, train_dataset_path, valid_dataset_path, output_dir):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("pid: ", os.getpid())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = load_data(
        train_dataset_path, valid_dataset_path, config["training"]["batch_size"]
    )

    model_config = config["model"]
    model = CNNClassifierModel(
        input_dim=model_config["input_dim"],
        ffn_dim=model_config["ffn_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        seq_len=model_config["seq_len"],
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    model = model.to(device)
        
    train_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )

    train_losses, valid_losses = [], []
    train_accs = []
    valid_accs = []
    threshold = 0.5

    for epoch in range(train_config["num_epochs"]):
        model.train()
        train_loss = 0.0
        train_samples = 0
        train_correct = [0] * 7
        start_time = time.time()
        for X, y, mask in train_loader:
            X, y, mask = (
                X.to(device).float(),
                y.to(device).float(),
                mask.to(device).bool(),
            )
            optimizer.zero_grad()
            outputs = model(X, mask)
            loss = F.binary_cross_entropy_with_logits(outputs, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config["clip_grad"]
            )
            optimizer.step()

            train_loss += loss.item() * X.shape[0]
            train_samples += X.shape[0]
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                corrects = (preds == y).float()
                for i in range(7):
                    train_correct[i] += corrects[:, i].sum().item()
        train_epoch_acc = [train_correct[i] / train_samples for i in range(7)]
        train_accs.append(train_epoch_acc)
        print(
            f"Epoch {epoch + 1}/{train_config['num_epochs']}, "
            f"Train Loss: {train_loss / train_samples:.4f}, "
            f"Time: {time.time() - start_time:.2f}s\n"
            f"Train Acc: {train_epoch_acc}\n"
            f"Train correct: {train_correct}"
        )
        start_time = time.time()
        model.eval()
        valid_loss = 0.0
        valid_samples = 0
        valid_correct = [0] * 7
        with torch.no_grad():
            for X, y, mask in valid_loader:
                X, y, mask = (
                    X.to(device).float(),
                    y.to(device).float(),
                    mask.to(device).bool(),
                )
                outputs = model(X, mask)
                loss = F.binary_cross_entropy_with_logits(outputs, y.float())
                valid_loss += loss.item() * X.shape[0]
                valid_samples += X.shape[0]
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                corrects = (preds == y).float()
                for i in range(7):
                    valid_correct[i] += corrects[:, i].sum().item()

        valid_epoch_acc = [valid_correct[i] / valid_samples for i in range(7)]
        valid_accs.append(valid_epoch_acc)
        print(
            f"Validation Loss: {valid_loss / valid_samples:.4f}, "
            f"Time: {time.time() - start_time:.2f}s\n"
            f"Valid Acc: {valid_epoch_acc}"
        )
        avg_train_loss = train_loss / train_samples
        avg_valid_loss = valid_loss / valid_samples
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(
            model_to_save.state_dict(),
            output_dir / f"model_epoch_{epoch + 1}.pt",
        )
        plot_metrics(
            epoch + 1, train_losses, valid_losses, train_accs, valid_accs, output_dir
        )
        plot_accs(train_accs, valid_accs, epoch + 1, output_dir)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    print("Current configuration:")
    print(json.dumps(config, indent=4))
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    train_dataset_path = (
        Path(args.train_dataset)
        if os.path.isabs(args.train_dataset)
        else current_dir / args.train_dataset
    )
    valid_dataset_path = (
        Path(args.valid_dataset)
        if os.path.isabs(args.valid_dataset)
        else current_dir / args.valid_dataset
    )
    output_dir = (
        Path(args.output_dir)
        if os.path.isabs(args.output_dir)
        else current_dir / args.output_dir
    )

    train(config, train_dataset_path, valid_dataset_path, output_dir)
