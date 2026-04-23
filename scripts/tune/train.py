#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

"""Unified training entrypoint (native in-file mode)."""

@dataclass(frozen=True)
class RunContext:
    dry_run: bool
    verbose: bool
    workdir: Path
    env: Dict[str, str]

@dataclass(frozen=True)
class ExecutionResult:
    task: str
    command: List[str]
    return_code: int

def strip_passthrough(args: Sequence[str]) -> List[str]:
    out = list(args)
    if out and out[0] == "--":
        out = out[1:]
    return out

def parse_env_pairs(pairs: Optional[Iterable[str]]) -> Dict[str, str]:
    env = os.environ.copy()
    if not pairs:
        return env
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid --env item: {p}. Use KEY=VALUE")
        k, v = p.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid env key in: {p}")
        env[k] = v
    return env

def run_subprocess(cmd: Sequence[str], ctx: RunContext, task_name: str) -> ExecutionResult:
    if ctx.verbose or ctx.dry_run:
        print(f"[{task_name}] $ {' '.join(cmd)}")
    if ctx.dry_run:
        return ExecutionResult(task=task_name, command=list(cmd), return_code=0)
    cp = subprocess.run(list(cmd), check=False, cwd=str(ctx.workdir), env=ctx.env)
    return ExecutionResult(task=task_name, command=list(cmd), return_code=cp.returncode)

def _run_native_script_block(script_name: str, passthrough: Sequence[str], block: Callable[[], None]) -> int:
    _old_argv = sys.argv[:]
    try:
        sys.argv = [script_name, *list(passthrough)]
        try:
            block()
            return 0
        except SystemExit as e:
            if isinstance(e.code, int):
                return e.code
            return 0 if e.code in (None, False) else 1
    finally:
        sys.argv = _old_argv

def _run_esm2_internal(passthrough: Sequence[str]) -> int:
    def _block() -> None:
        __file__ = str((Path(__file__).resolve().parents[1] / 'archieved' / 'train' / 'esm2_train.py'))
        __name__ = '__main__'
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
    return _run_native_script_block('esm2_train.py', passthrough, _block)

def _run_evo2_internal(passthrough: Sequence[str]) -> int:
    def _block() -> None:
        __file__ = str((Path(__file__).resolve().parents[1] / 'archieved' / 'train' / 'evo2_train.py'))
        __name__ = '__main__'
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
    return _run_native_script_block('evo2_train.py', passthrough, _block)

def _run_ee_internal(passthrough: Sequence[str]) -> int:
    def _block() -> None:
        __file__ = str((Path(__file__).resolve().parents[1] / 'archieved' / 'train' / 'ee_train.py'))
        __name__ = '__main__'
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
    return _run_native_script_block('ee_train.py', passthrough, _block)

def _run_kd_fusion_internal(passthrough: Sequence[str]) -> int:
    def _block() -> None:
        __file__ = str((Path(__file__).resolve().parents[1] / 'archieved' / 'train' / 'kd_fusion_train.py'))
        __name__ = '__main__'
        #!/usr/bin/env python3
        import argparse
        import json
        import random
        from pathlib import Path
        
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.metrics import roc_auc_score, average_precision_score
        from torch.utils.data import DataLoader, TensorDataset
        
        
        class TeacherNet(nn.Module):
            def __init__(self, in_dim: int, hidden: int, dropout: float):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 1),
                )
        
            def forward(self, x):
                return self.net(x).squeeze(-1)
        
        
        class StudentNet(nn.Module):
            def __init__(self, in_dim: int, hidden: int, dropout: float):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
        
            def forward(self, x):
                return self.net(x).squeeze(-1)
        
        
        def set_seed(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        
        def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray):
            probs = 1 / (1 + np.exp(-logits))
            try:
                auc = float(roc_auc_score(y_true, probs))
            except Exception:
                auc = 0.0
            try:
                auprc = float(average_precision_score(y_true, probs))
            except Exception:
                auprc = 0.0
            acc = float(((probs >= 0.5) == y_true).mean())
            return {"auc": auc, "auprc": auprc, "acc": acc}
        
        
        def run_eval(student, loader, device):
            student.eval()
            ys, logits = [], []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    out = student(xb)
                    ys.append(yb.numpy())
                    logits.append(out.cpu().numpy())
            y = np.concatenate(ys)
            lg = np.concatenate(logits)
            return metrics_from_logits(y, lg), y, lg
        
        
        def main():
            ap = argparse.ArgumentParser(description="KD fusion trainer (ESM2 + Evo2 + ProtBERT)")
            ap.add_argument("--config", required=True)
            ap.add_argument("--dataset", required=True)
            ap.add_argument("--outdir", required=True)
            args = ap.parse_args()
        
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        
            mcfg = cfg["model"]
            tcfg = cfg["training"]
        
            set_seed(int(tcfg.get("seed", 42)))
        
            obj = torch.load(args.dataset, map_location="cpu", weights_only=False)
            Xtr, ytr = obj["train"]["X"], obj["train"]["y"]
            Xva, yva = obj["val"]["X"], obj["val"]["y"]
            Xte, yte = obj["test"]["X"], obj["test"]["y"]
        
            in_dim = Xtr.shape[1]
            teacher = TeacherNet(in_dim, int(mcfg["teacher_hidden"]), float(mcfg["dropout"]))
            student = StudentNet(in_dim, int(mcfg["student_hidden"]), float(mcfg["dropout"]))
        
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            teacher = teacher.to(device)
            student = student.to(device)
        
            batch_size = int(tcfg["batch_size"])
            tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
            va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
            te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)
        
            params = list(teacher.parameters()) + list(student.parameters())
            opt = torch.optim.AdamW(
                params,
                lr=float(tcfg["lr"]),
                weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            )
        
            alpha = float(tcfg.get("alpha", 0.5))
            T = float(tcfg.get("temperature", 2.0))
            teacher_w = float(tcfg.get("teacher_loss_weight", 0.5))
            epochs = int(tcfg["num_epochs"])
        
            outdir = Path(args.outdir)
            (outdir / "models").mkdir(parents=True, exist_ok=True)
        
            best_auc = -1.0
            history = []
        
            bce = nn.BCEWithLogitsLoss()
        
            for ep in range(1, epochs + 1):
                teacher.train()
                student.train()
                ep_loss = 0.0
        
                for xb, yb in tr_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
        
                    tlog = teacher(xb)
                    slog = student(xb)
        
                    loss_t = bce(tlog, yb)
                    loss_hard = bce(slog, yb)
        
                    with torch.no_grad():
                        soft_t = torch.sigmoid(tlog / T)
                    loss_kd = bce(slog / T, soft_t) * (T * T)
        
                    loss = teacher_w * loss_t + alpha * loss_hard + (1 - alpha) * loss_kd
        
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    ep_loss += float(loss.item())
        
                val_m, _, _ = run_eval(student, va_loader, device)
                row = {"epoch": ep, "train_loss": ep_loss / max(1, len(tr_loader)), **{f"val_{k}": v for k, v in val_m.items()}}
                history.append(row)
                print(json.dumps(row, ensure_ascii=False))
        
                if val_m["auc"] > best_auc:
                    best_auc = val_m["auc"]
                    torch.save(
                        {
                            "student_state": student.state_dict(),
                            "teacher_state": teacher.state_dict(),
                            "feature_dim": int(in_dim),
                            "config": cfg,
                            "best_val": val_m,
                        },
                        outdir / "models" / "kd_best.pt",
                    )
        
            # final eval on test with best model
            best = torch.load(outdir / "models" / "kd_best.pt", map_location=device, weights_only=False)
            student.load_state_dict(best["student_state"])
            test_m, y, lg = run_eval(student, te_loader, device)
            print("[KD] test:", json.dumps(test_m, ensure_ascii=False))
        
            with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump({"best_val_auc": best_auc, "test": test_m, "history": history}, f, ensure_ascii=False, indent=2)
        
            probs = 1 / (1 + np.exp(-lg))
            with (outdir / "test_predictions.csv").open("w", encoding="utf-8") as f:
                f.write("seq_id,label,prediction\n")
                for sid, yy, pp in zip(obj["test"]["ids"], y.astype(int), probs):
                    f.write(f"{sid},{yy},{pp:.6f}\n")
        
            print(f"[KD] done: {outdir}")
        
        
        if __name__ == "__main__":
            main()
    return _run_native_script_block('kd_fusion_train.py', passthrough, _block)

def _run_oracle_internal(passthrough: Sequence[str]) -> int:
    def _block() -> None:
        __file__ = str((Path(__file__).resolve().parents[1] / 'archieved' / 'train' / 'ORACLE_train.py'))
        __name__ = '__main__'
        import os, time, json, random, logging, argparse
        from typing import List, Tuple, Dict
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModel, set_seed
        
        
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
            def __init__(self, triples: List[Tuple[int, int]], id2seq: dict, max_len: int = 1024):
                self.samples = []
                for pid, tid in triples:
                    seq = id2seq[pid]
                    if len(seq) <= max_len:
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
        
        
        # -----------------------
        # Evaluation (MRR / Hits@K) with soft-negative
        # -----------------------
        @torch.no_grad()
        def evaluate(model, tokenizer, id2seq, pos_triples, batch_size=16, max_len=1024, device="cuda", sample_limit=200):
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
                seqs = [id2seq[pid] for pid in pids]
        
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
            num_types = len(t2id_tr)
            train_ds = KGETripletDataset(pos_tr_triples, id2seq_tr, max_len=args.max_seq_len)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=lambda x: x)
        
            # Tokenizer & Encoder
            logger.info(f"Loading ProtBERT from: {args.protbert_path}")
            tokenizer = AutoTokenizer.from_pretrained(args.protbert_path, local_files_only=True, do_lower_case=False)
            prot_encoder = ProteinEncoder(model_path=args.protbert_path, proj_dim=args.embedding_dim, freeze=not args.unfreeze_encoder)
            if hasattr(args, "freeze_encoder") and args.freeze_encoder:
                logger.info("Freezing ProteinEncoder parameters...")
                for param in prot_encoder.parameters():
                    param.requires_grad = False
        
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
                    seqs = [seq for _, _, seq in batch]
        
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
                    sample_limit=args.eval_sample_limit
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
    return _run_native_script_block('ORACLE_train.py', passthrough, _block)

def _run_oracle_cnn_internal(passthrough: Sequence[str]) -> int:
    def _block() -> None:
        __file__ = str((Path(__file__).resolve().parents[1] / 'archieved' / 'train' / 'ORACLE_CNN_train.py'))
        __name__ = '__main__'
        import os
        import json
        import time
        import argparse
        import pandas as pd
        import numpy as np
        import torch
        import esm
        import matplotlib.pyplot as plt
        from scipy.ndimage import binary_closing
        from tqdm import tqdm
        from pathlib import Path
        from torch import nn
        from Bio import SeqIO
        from torch.utils.data import Dataset, DataLoader
        from torch.nn.utils.rnn import pad_sequence
        from collections import defaultdict
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
    return _run_native_script_block('ORACLE_CNN_train.py', passthrough, _block)

class BaseTask(ABC):
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
    @abstractmethod
    def run(self, passthrough: Sequence[str], ctx: RunContext) -> ExecutionResult:
        raise NotImplementedError

class NativePythonTask(BaseTask):
    def __init__(self, name: str, description: str, runner: Callable[[Sequence[str]], int]) -> None:
        super().__init__(name=name, description=description)
        self.runner = runner
    def run(self, passthrough: Sequence[str], ctx: RunContext) -> ExecutionResult:
        if ctx.verbose or ctx.dry_run:
            print(f"[{self.name}] native python runner")
        if ctx.dry_run:
            return ExecutionResult(task=self.name, command=[self.name, *list(passthrough)], return_code=0)
        old_env = os.environ.copy()
        old_cwd = Path.cwd()
        try:
            os.environ.update(ctx.env)
            os.chdir(ctx.workdir)
            rc = self.runner(passthrough)
        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)
        return ExecutionResult(task=self.name, command=[self.name, *list(passthrough)], return_code=rc)

class ShellTask(BaseTask):
    def __init__(self, name: str, description: str, script_path: Path, fixed_args: Optional[Sequence[str]] = None) -> None:
        super().__init__(name=name, description=description)
        self.script_path = script_path
        self.fixed_args = list(fixed_args or [])
    def run(self, passthrough: Sequence[str], ctx: RunContext) -> ExecutionResult:
        if not self.script_path.exists():
            raise FileNotFoundError(f"Shell script not found: {self.script_path}")
        cmd = ["bash", str(self.script_path), *self.fixed_args, *passthrough]
        return run_subprocess(cmd, ctx, self.name)

class CompositeTask(BaseTask):
    def __init__(self, name: str, description: str, steps: Sequence[str], manager: "TrainManager") -> None:
        super().__init__(name=name, description=description)
        self.steps = list(steps)
        self.manager = manager
    def run(self, passthrough: Sequence[str], ctx: RunContext) -> ExecutionResult:
        final = ExecutionResult(task=self.name, command=[], return_code=0)
        for step in self.steps:
            result = self.manager.run_one(step, passthrough, ctx)
            final = result
            if result.return_code != 0:
                return ExecutionResult(task=self.name, command=result.command, return_code=result.return_code)
        return final

class TrainManager:
    def __init__(self, this_file: Path) -> None:
        self.this_file = this_file.resolve()
        self.this_dir = self.this_file.parent
        self.tasks: Dict[str, BaseTask] = {}
        self._register_default_tasks()
    def _register_default_tasks(self) -> None:
        self.tasks['esm2'] = NativePythonTask('esm2', 'Train ESM2 model', _run_esm2_internal)
        self.tasks['evo2'] = NativePythonTask('evo2', 'Train Evo2 model', _run_evo2_internal)
        self.tasks['ee'] = NativePythonTask('ee', 'Train EE fusion model', _run_ee_internal)
        self.tasks['kd-fusion'] = NativePythonTask('kd-fusion', 'Train KD fusion model', _run_kd_fusion_internal)
        self.tasks['oracle'] = NativePythonTask('oracle', 'Train ORACLE KG model', _run_oracle_internal)
        self.tasks['oracle-cnn'] = NativePythonTask('oracle-cnn', 'Train ORACLE CNN model', _run_oracle_cnn_internal)
        tune_script = self.this_dir / "tune.sh"
        self.tasks['tune-esm2'] = ShellTask('tune-esm2', 'Tune ESM2', tune_script, ['--target', 'esm2'])
        self.tasks['tune-evo2'] = ShellTask('tune-evo2', 'Tune Evo2', tune_script, ['--target', 'evo2'])
        self.tasks['tune-oracle-kg'] = ShellTask('tune-oracle-kg', 'Tune ORACLE KG', tune_script, ['--target', 'oracle-kg'])
        self.tasks['tune-ee'] = ShellTask('tune-ee', 'Tune EE', tune_script, ['--target', 'ee'])
        self.tasks['tune-kd-fusion'] = ShellTask('tune-kd-fusion', 'Tune KD fusion', tune_script, ['--target', 'kd-fusion'])
        self.tasks["combo-core-trainers"] = CompositeTask("combo-core-trainers", "Run core trainers: esm2 -> evo2 -> ee", ["esm2", "evo2", "ee"], self)
        self.tasks["combo-oracle"] = CompositeTask("combo-oracle", "Run oracle stack: oracle -> oracle-cnn", ["oracle", "oracle-cnn"], self)
    def list_tasks(self) -> List[Tuple[str, str]]:
        return sorted((name, task.description) for name, task in self.tasks.items())
    def ensure_task(self, name: str) -> BaseTask:
        if name not in self.tasks:
            raise KeyError(f"Unknown task: {name}")
        return self.tasks[name]
    def run_one(self, name: str, passthrough: Sequence[str], ctx: RunContext) -> ExecutionResult:
        return self.ensure_task(name).run(passthrough, ctx)
    def run_chain(self, names: Sequence[str], passthrough: Sequence[str], ctx: RunContext, continue_on_error: bool = False) -> int:
        overall_rc = 0
        for n in names:
            result = self.run_one(n, passthrough, ctx)
            if result.return_code != 0:
                overall_rc = result.return_code
                if not continue_on_error:
                    return overall_rc
        return overall_rc

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified training entrypoint")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List all available tasks")
    run = sub.add_parser("run", help="Run one task")
    run.add_argument("target", help="Task name")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--workdir", default=".", help="Working directory")
    run.add_argument("--env", action="append", default=[], help="KEY=VALUE (repeatable)")
    run.add_argument("args", nargs=argparse.REMAINDER)
    chain = sub.add_parser("chain", help="Run multiple tasks in order")
    chain.add_argument("--targets", required=True, help="Comma-separated task names")
    chain.add_argument("--continue-on-error", action="store_true")
    chain.add_argument("--dry-run", action="store_true")
    chain.add_argument("--verbose", action="store_true")
    chain.add_argument("--workdir", default=".", help="Working directory")
    chain.add_argument("--env", action="append", default=[], help="KEY=VALUE (repeatable)")
    chain.add_argument("args", nargs=argparse.REMAINDER)
    return p

def main() -> int:
    manager = TrainManager(Path(__file__))
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "list":
        for n, d in manager.list_tasks():
            print(f"{n}\t{d}")
        return 0
    env = parse_env_pairs(getattr(args, "env", []))
    ctx = RunContext(dry_run=bool(getattr(args, "dry_run", False)), verbose=bool(getattr(args, "verbose", False)), workdir=Path(getattr(args, "workdir", ".")).resolve(), env=env)
    passthrough = strip_passthrough(getattr(args, "args", []))
    if args.command == "run":
        return manager.run_one(args.target, passthrough, ctx).return_code
    if args.command == "chain":
        targets = [x.strip() for x in args.targets.split(",") if x.strip()]
        if not targets:
            raise ValueError("--targets is empty")
        return manager.run_chain(targets, passthrough, ctx, continue_on_error=bool(args.continue_on_error))
    raise RuntimeError(f"Unsupported command: {args.command}")

if __name__ == "__main__":
    raise SystemExit(main())

