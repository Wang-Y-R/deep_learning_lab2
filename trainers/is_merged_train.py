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
from utils.utils import plot_loss, evaluate

def train(cfg, device="cpu"):
    # 1. 加载数据
    train_set = pd.read_csv(cfg.train_path)
    test_set = pd.read_csv(cfg.test_path)
    X_train = train_set.drop(columns=[cfg.label_column,"pr_id","created_at","pr_number","time_to_close"]).values
    X_test = test_set.drop(columns=[cfg.label_column,"pr_id","created_at","pr_number","time_to_close"]).values
    y_train = train_set[cfg.label_column].values
    y_test = test_set[cfg.label_column].values

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

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
    plot_loss(train_losses, os.path.join(cfg.output_dir, "loss_curve.png"))
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "model.pth"))

    return model, acc
