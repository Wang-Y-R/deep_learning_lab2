# trainers/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.is_merged_model import FeedforwardNN
from utils.utils import plot_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate(model, test_loader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            preds = (outputs >= 0.5).int()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # 指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("=== Evaluation Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc, prec, rec, f1

# utils/utils.py
import matplotlib.pyplot as plt

def plot_metrics(metrics_dict, save_path):
    """
    metrics_dict: dict, {"Accuracy": acc, "Precision": prec, ...}
    save_path: str, 保存路径
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(6,4))
    plt.bar(names, values, color=['skyblue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(cfg, device="cpu"):
    # 1. 加载数据
    train_set = pd.read_csv(cfg.train_path)
    test_set = pd.read_csv(cfg.test_path)
    
    X_train = train_set.drop(columns=[cfg.label_column])
    X_test  = test_set.drop(columns=[cfg.label_column])
    y_train = train_set[cfg.label_column].values
    y_test = test_set[cfg.label_column].values

    # 将 DataFrame 转 numpy 再转 tensor
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test  = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(train_set[cfg.label_column].astype(float).values, dtype=torch.float32).unsqueeze(1)
    y_test  = torch.tensor(test_set[cfg.label_column].astype(float).values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    # 2. 定义模型
    model = FeedforwardNN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # 3. 训练
    os.makedirs(cfg.output_dir, exist_ok=True)
    train_losses = []
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {avg_loss:.4f}")

    # 4. 评估
    acc, prec, rec, f1 = evaluate(model, test_loader, device="cpu")

    # 5. 保存结果
    plot_loss(train_losses, os.path.join(cfg.output_dir, cfg.loss_curve_name + ".png"))
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, cfg.output_model_name + ".pth"))

    # 保存指标图
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1}
    plot_metrics(metrics, os.path.join(cfg.output_dir, cfg.output_result_name + ".png"))

    return model, acc
