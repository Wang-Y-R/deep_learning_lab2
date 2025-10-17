# trainers/train.py
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.is_merged_model import FeedforwardNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from config import Config


def evaluate(model, test_loader, device="cpu"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = torch.sigmoid(model(X)).squeeze()  # 手动 sigmoid
            preds = (outputs >= 0.5).int()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

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



# === 1. 加载数据 ===
cfg = Config()
test_set = pd.read_csv(cfg.test_path)

# evaluate_model_path = "outputs/train/is_merged/" + train_data_name + "_to_" + test_data_name + "/train_by_" + train_data_name + "_model.pth"
# evaluate_data_paths = ["data/" + name + "/test.csv" for name in evaluate_datas_name]
# evaluate_output_dir = [f"outputs/evaluate/is_merged/{train_data_name}_to_{name}" for name in evaluate_datas_name ]

for name,data_path, output_path in zip( cfg.evaluate_datas_name ,cfg.evaluate_data_paths, cfg.evaluate_output_dir):
    model_path = cfg.evaluate_model_path
    print(f"Evaluating model: {model_path} on data: {data_path}, results will be saved to: {output_path}")
    train_set = pd.read_csv(cfg.train_path)
    test_set = pd.read_csv(data_path)

    X_train = train_set.drop(columns=[cfg.label_column])

    X_test = test_set.drop(columns=[cfg.label_column])
    y_test = test_set[cfg.label_column]

    # === 2. 特征标准化 ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    # === 3. 转 tensor ===
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # === 4. 模型、损失函数、优化器 ===
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    input_dim = X_test.shape[1]
    model = FeedforwardNN(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # === 6. 评估 ===
    acc, prec, rec, f1 = evaluate(model, test_loader, device="cpu")

    # === 7. 保存结果 ===
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1}
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plot_metrics(metrics, os.path.join(output_path,"result.png"))
