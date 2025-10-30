# =========================================================
# trainers/multitask_train.py
# =========================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader

# 1. 数据加载
class PRDataset(Dataset):
    """用于加载单个项目的 train/test.csv，返回特征 + 两个标签"""
    def __init__(self, csv_path, label_columns):
        df = pd.read_csv(csv_path)

        # 把 time_to_close == 0 当成缺失
        df['time_to_close'] = df['time_to_close'].replace(0, np.nan)

        # 1️⃣ 特征部分
        self.X = torch.tensor(df.drop(columns=label_columns).values, dtype=torch.float32)

        # 2️⃣ 时间处理（直接使用小时制）
        ttc_hours = df[label_columns[0]].values # 秒转小时
        self.y_reg = torch.tensor(ttc_hours, dtype=torch.float32).view(-1, 1)

        # 分类标签
        self.y_cls = torch.tensor(df[label_columns[1]].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_cls[idx]


# 2. 多任务模型（MMoE 实现）
class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE)
    - n_experts: 专家数量
    - n_tasks: 任务数量（这里为2）
    """
    def __init__(self, input_dim, n_experts=4, expert_hidden=64, tower_hidden=32):
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = 2  # 任务1=回归，任务2=分类

        # ---- 专家网络 ----
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, expert_hidden),
                nn.ReLU()
            )
            for _ in range(n_experts)
        ])

        # ---- 门控网络（每个任务一个）----
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_experts),
                nn.Softmax(dim=1)
            )
            for _ in range(self.n_tasks)
        ])

        # ---- 每个任务的 tower ----
        self.tower_reg = nn.Sequential(
            nn.Linear(expert_hidden, tower_hidden),
            nn.ReLU(),
            nn.Linear(tower_hidden, 1)
        )
        self.tower_cls = nn.Sequential(
            nn.Linear(expert_hidden, tower_hidden),
            nn.ReLU(),
            nn.Linear(tower_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1️⃣ 每个专家生成输出 (batch, hidden)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # [B, hidden, n_experts]

        # 2️⃣ 每个任务的门控生成权重 (batch, n_experts)
        gate_outputs = [gate(x) for gate in self.gates]  # len=2

        # 3️⃣ 对专家输出加权求和
        task_outputs = []
        for gate in gate_outputs:
            expanded_gate = gate.unsqueeze(1).expand_as(expert_outputs)  # [B, hidden, n_experts]
            weighted_output = torch.sum(expert_outputs * expanded_gate, dim=2)  # [B, hidden]
            task_outputs.append(weighted_output)

        # 4️⃣ 各任务通过自己的 tower 输出
        out_reg = self.tower_reg(task_outputs[0])
        out_cls = self.tower_cls(task_outputs[1])
        return out_reg, out_cls


# 3. 训练与评估函数
def train_single_project(cfg, paths, device):
    """在单个项目（如 django/opencv）上训练多任务模型"""

    print(f"\n=== Training on {paths['train_path']} ===")

    # ---------- 加载数据 ----------
    train_ds = PRDataset(paths["train_path"], cfg.label_columns)
    test_ds = PRDataset(paths["test_path"], cfg.label_columns)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    input_dim = train_ds.X.shape[1]

    # ---------- 初始化模型 ----------
    model = MMoE(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion_reg = nn.L1Loss()      # MAE
    criterion_cls = nn.BCELoss()     # 分类交叉熵

    # ---------- 训练循环 ----------
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for X, y_reg, y_cls in train_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            pred_reg, pred_cls = model(X)

            # ===== 任务掩码处理 =====
            # 把 time_to_close == 0 当成缺失值
            reg_mask = (y_reg > 0) & ~torch.isnan(y_reg)
            cls_mask = ~torch.isnan(y_cls)

            loss = 0.0
            n_tasks = 0

            # 回归任务（仅对有 time_to_close 的样本计算）
            if reg_mask.any():
                loss_reg = criterion_reg(pred_reg[reg_mask], y_reg[reg_mask])
                loss += 0.5 * loss_reg  # 可调权重
                n_tasks += 1

            # 分类任务（通常每个样本都有标签）
            if cls_mask.any():
                loss_cls = criterion_cls(pred_cls[cls_mask], y_cls[cls_mask])
                loss += 1.0 * loss_cls  # 可调权重
                n_tasks += 1

            if n_tasks > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{cfg.epochs}, Loss={total_loss / len(train_loader):.4f}")

    # ---------- 评估 ----------
    model.eval()
    preds_reg, trues_reg = [], []
    preds_cls, trues_cls = [], []

    with torch.no_grad():
        for X, y_reg, y_cls in test_loader:
            X = X.to(device)
            pr, pc = model(X)

            # 转 numpy
            pr = pr.cpu().numpy().squeeze()
            y_reg = y_reg.cpu().numpy().squeeze()
            pc = pc.cpu().numpy().squeeze()
            y_cls = y_cls.cpu().numpy().squeeze()

            # 过滤无效回归标签
            valid_mask = ~np.isnan(y_reg) & (y_reg > 0)
            preds_reg.extend(pr[valid_mask].tolist())
            trues_reg.extend(y_reg[valid_mask].tolist())

            preds_cls.extend((pc > 0.5).astype(int).tolist())
            trues_cls.extend(y_cls.astype(int).tolist())

    # 若所有回归样本都被过滤，避免指标计算报错
    if len(trues_reg) == 0:
        mae = rmse = r2 = float('nan')
    else:
        mae = mean_absolute_error(trues_reg, preds_reg)
        rmse = mean_squared_error(trues_reg, preds_reg, squared=False)
        r2 = r2_score(trues_reg, preds_reg)

    acc = accuracy_score(trues_cls, preds_cls)
    f1 = f1_score(trues_cls, preds_cls)
    precision = precision_score(trues_cls, preds_cls)
    recall = recall_score(trues_cls, preds_cls)

    print("pc:",pc[:10])  # 模型输出概率
    print("y_cls", y_cls[:10])  # 真实标签

    print(f"[{os.path.basename(paths['train_path'])}] MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}, "
          f"ACC={acc:.3f}, F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")

    # ---------- 保存模型与结果 ----------
    os.makedirs(paths["output_dir"], exist_ok=True)
    torch.save(model.state_dict(), paths["output_model"])
    pd.DataFrame({
        "MAE": [mae], "RMSE": [rmse], "R2": [r2],
        "Accuracy": [acc], "F1": [f1],
        "Precision": [precision], "Recall": [recall]
    }).to_csv(paths["output_result"], index=False)


# 4. 主训练入口,由 main.py 调用
def train(cfg, device="cpu"):
    """循环遍历多个项目，独立训练各自的多任务模型"""
    for name in cfg.train_data_names:
        paths = cfg.get_paths(name)
        train_single_project(cfg, paths, device)
    print("\n=== All projects finished training ===")
