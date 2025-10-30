# evaluate_predict.py
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.predict_model import WideAndDeepNN, WideOnlyNN, DeepOnlyNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import PredictConfig
import matplotlib.pyplot as plt

def evaluate_regression(model, test_loader, label_scaler=None, device="cpu"):
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
    return mae, mse, rmse, r2

def plot_metrics(metrics_dict, save_path, dataset_name):
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    plt.figure(figsize=(6,4))
    plt.bar(names, values, color=['#4e79a7', '#f28e2b', '#76b7b2', '#e15759'])
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    plt.title(f"Regression Metrics ({dataset_name})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    cfg = PredictConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_types = ["WideAndDeepNN", "WideOnlyNN", "DeepOnlyNN"]
    model_class_map = {
        "WideAndDeepNN": WideAndDeepNN,
        "WideOnlyNN": WideOnlyNN,
        "DeepOnlyNN": DeepOnlyNN
    }
    for model_type in model_types:
        for train_name in cfg.train_data_names:
            for eval_name in cfg.evaluate_datas_name:
                # 训练集fit scaler
                train_path = f"data/{train_name}/train.csv"
                train_set = pd.read_csv(train_path)
                X_train = train_set.drop(columns=[cfg.label_column])
                scaler = StandardScaler()
                scaler.fit(X_train)
                # 加载模型权重
                model_path = f"outputs/train/predict/{train_name}/train_by_{train_name}_{model_type}_model.pth"
                if not os.path.exists(model_path):
                    print(f"[Warning] 权重文件不存在: {model_path}, 跳过该训练集")
                    continue
                data_path = f"data/{eval_name}/test.csv"
                output_path = f"outputs/evaluate/predict/{train_name}_{model_type}_to_{eval_name}"
                print(f"Evaluating model: {model_path} on data: {data_path}, results will be saved to: {output_path}")
                test_set = pd.read_csv(data_path)
                X_test = test_set.drop(columns=[cfg.label_column])
                y_test = test_set[cfg.label_column] / 3600
                y_test = np.log1p(y_test)
                X_test = scaler.transform(X_test)
                X_test = torch.tensor(X_test, dtype=torch.float32)
                y_test = torch.tensor(y_test.values, dtype=torch.float32)
                test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)
                model = model_class_map[model_type](input_dim=X_test.shape[1])
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                mae, mse, rmse, r2 = evaluate_regression(model, test_loader, device=device)
                print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
                metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "train_name": train_name}
                os.makedirs(output_path, exist_ok=True)
                plot_metrics({k: metrics[k] for k in ["MAE", "MSE", "RMSE", "R2"]},
                     os.path.join(output_path, f"{eval_name}_{model_type}_regression_metrics.png"),
                     f"{eval_name}_{model_type}")
                # 保存指标为csv
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(os.path.join(output_path, f"{eval_name}_{model_type}_metrics.csv"), index=False)

if __name__ == "__main__":
    main()
