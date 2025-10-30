import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.predict_model import WideAndDeepNN, WideOnlyNN, DeepOnlyNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import PredictConfig
import matplotlib.pyplot as plt
from utils.utils import plot_loss

def evaluate_regression(model, test_loader, cfg, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    # 还原log变换
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

    # 只保存评价指标柱状图
    img_output_dir = cfg.train_output_dir
    os.makedirs(img_output_dir, exist_ok=True)
    metrics = [mae, mse, rmse, r2]
    metric_names = ['MAE', 'MSE', 'RMSE', 'R2']
    plt.figure(figsize=(6,4))
    bars = plt.bar(metric_names, metrics, color=['#4e79a7', '#f28e2b', '#76b7b2', '#e15759'])
    for bar, value in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    plt.title(f'Regression Metrics ({cfg.dataset_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(img_output_dir, f'{cfg.dataset_name}_regression_metrics.png'))
    plt.close()

    return mae, mse, rmse, r2


def train_one(cfg, train_data_name, test_data_name, device="cpu"):
    train_path = f"data/{train_data_name}/train.csv"
    test_path = f"data/{train_data_name}/test.csv"
    output_dir = f"outputs/train/predict/{train_data_name}"
    model_type = getattr(cfg, "model_type", "WideAndDeepNN")
    output_model_name = f"train_by_{train_data_name}_{model_type}_model"
    img_output_dir = output_dir
    os.makedirs(img_output_dir, exist_ok=True)

    # 1. 加载数据
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    X_train = train_set.drop(columns=[cfg.label_column])
    X_test = test_set.drop(columns=[cfg.label_column])
    y_train = train_set[cfg.label_column] / 3600  # 秒转小时
    y_test = test_set[cfg.label_column] / 3600

    # 对标签做log1p变换
    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    # 2. 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. 转 tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    # 4. 支持模型组件消融
    model_type = getattr(cfg, "model_type", "WideAndDeepNN")
    if model_type == "WideAndDeepNN":
        model = WideAndDeepNN(input_dim=X_train.shape[1]).to(device)
    elif model_type == "WideOnlyNN":
        model = WideOnlyNN(input_dim=X_train.shape[1]).to(device)
    elif model_type == "DeepOnlyNN":
        model = DeepOnlyNN(input_dim=X_train.shape[1]).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # 5. 训练
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
    plot_loss(train_losses, os.path.join(img_output_dir, f'{train_data_name}_{model_type}_loss_curve.png'))

    # 6. 评估
    class CfgObj:
        pass
    eval_cfg = CfgObj()
    eval_cfg.train_output_dir = output_dir
    eval_cfg.dataset_name = f"{train_data_name}_{model_type}"
    eval_cfg.output_model_name = output_model_name
    mae, mse, rmse, r2 = evaluate_regression(model, test_loader, eval_cfg, device=device)
    # 保存指标为csv
    metrics_df = pd.DataFrame([{"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}])
    metrics_df.to_csv(os.path.join(output_dir, f"{train_data_name}_{model_type}_metrics.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(output_dir, output_model_name + ".pth"))
    return model

if __name__ == "__main__":
    cfg = PredictConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_type in ["WideAndDeepNN", "WideOnlyNN", "DeepOnlyNN"]:
        cfg.model_type = model_type
        for train_name in cfg.train_data_names:
            train_one(cfg, train_name, device)