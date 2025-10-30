# main.py
import numpy as np
import pandas as pd
import torch
from config import MultiTaskConfig

# if __name__ == "__main__":
#     cfg = MultiTaskConfig()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
#
#     cfg.trainer(device = device)
#     print("Training finished. Model saved to outputs/")

import matplotlib.pyplot as plt


def plot_multitask_metrics(save_path="outputs/multitask_metrics.png"):
    """绘制 MMoE 模型在不同数据集的指标对比图（分回归+分类，R²范围调至[-0.5, 1]）"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # ===== 实验结果 =====
    df = pd.DataFrame({
        "MAE": [258.409, 261.749],
        "RMSE": [1031.513, 892.078],
        "R2": [-0.051, -0.082],
        "Accuracy": [0.666, 0.825],
        "F1": [0.799, 0.904],
        "Precision": [0.666, 0.825],
        "Recall": [1.000, 1.000],
    }, index=["django", "opencv"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---------- (1) 回归指标 ----------
    reg_metrics = ["MAE", "RMSE"]
    x = np.arange(len(reg_metrics))
    width = 0.35

    axes[0].bar(x - width/2, df.loc["django", reg_metrics], width, label="django", color="#1f77b4")
    axes[0].bar(x + width/2, df.loc["opencv", reg_metrics], width, label="opencv", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reg_metrics)
    axes[0].set_title("Regression Metrics")
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].legend()

    # ---------- (2) 分类指标（含 R² 调整） ----------
    cls_metrics = ["R2", "Accuracy", "F1", "Precision", "Recall"]
    x2 = np.arange(len(cls_metrics))
    width2 = 0.35

    # 为保持对比一致，这里也包含 R²
    axes[1].bar(x2 - width2/2, df.loc["django", cls_metrics], width2, label="django", color="#1f77b4")
    axes[1].bar(x2 + width2/2, df.loc["opencv", cls_metrics], width2, label="opencv", color="#ff7f0e")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cls_metrics)
    axes[1].set_ylim(-0.5, 1.1)
    axes[1].axhline(0, color='gray', linewidth=0.8)  # 零基线
    axes[1].set_title("Classification Metrics (+ R²)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.suptitle("MMoE Metrics Comparison (django vs opencv)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[Saved] {save_path}")


def plot_classification_comparison(save_path="outputs/classification_comparison.png"):
    """绘制单任务(FeedforwardNN)与多任务(MMoE)分类任务性能对比"""
    import matplotlib.pyplot as plt
    import numpy as np

    models = ["FeedforwardNN (Single-task)", "MMoE (Multi-task)"]

    # 分类结果数据（根据你图中数值）
    # django
    django_acc = [0.76, 0.67]
    django_f1 = [0.83, 0.80]
    django_prec = [0.78, 0.67]
    django_rec = [0.88, 1.00]  # 多任务Recall为1.0

    # opencv
    opencv_acc = [0.60, 0.83]
    opencv_f1 = [0.70, 0.90]
    opencv_prec = [0.91, 0.83]
    opencv_rec = [0.57, 1.00]

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ---------- (1) django ----------
    django_values_single = [django_acc[0], django_prec[0], django_rec[0], django_f1[0]]
    django_values_multi = [django_acc[1], django_prec[1], django_rec[1], django_f1[1]]
    axes[0].bar(x - width/2, django_values_single, width, label="FeedforwardNN", color="#1f77b4")
    axes[0].bar(x + width/2, django_values_multi, width, label="MMoE", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("django Classification Metrics")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].legend()

    # ---------- (2) opencv ----------
    opencv_values_single = [opencv_acc[0], opencv_prec[0], opencv_rec[0], opencv_f1[0]]
    opencv_values_multi = [opencv_acc[1], opencv_prec[1], opencv_rec[1], opencv_f1[1]]
    axes[1].bar(x - width/2, opencv_values_single, width, label="FeedforwardNN", color="#1f77b4")
    axes[1].bar(x + width/2, opencv_values_multi, width, label="MMoE", color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("opencv Classification Metrics")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.suptitle("FeedforwardNN vs MMoE Classification Performance Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[Saved] {save_path}")



# =========================================================
# 6. 使用示例（可放在 main.py 中执行）
# =========================================================
if __name__ == "__main__":

    # --- 指标对比 ---
    plot_classification_comparison()