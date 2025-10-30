import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from trainers.predict_train import train_one, evaluate_regression
from models.predict_model import WideAndDeepNN, WideOnlyNN

from scipy.stats import ttest_ind, mannwhitneyu

# 配置
N_REPEAT = 5
TRAIN_NAME = 'django'
EVAL_NAMES = ['django']  # 可扩展
MODEL_TYPES = ['WideAndDeepNN', 'WideOnlyNN']
SEED_LIST = [42, 43, 44, 45, 46]
SAVE_DIR = 'outputs/ablation_multi_exp'

os.makedirs(SAVE_DIR, exist_ok=True)

results = {m: [] for m in MODEL_TYPES}

from config import PredictConfig
device = "cuda" if torch.cuda.is_available() else "cpu"
for model_type in MODEL_TYPES:
    for i, seed in enumerate(SEED_LIST):
        print(f'[{model_type}] 第{i+1}次实验，random_state={seed}')
        # 配置
        cfg = PredictConfig()
        cfg.model_type = model_type
        # 训练并返回模型
        model = train_one(cfg, TRAIN_NAME, TRAIN_NAME, device=device)
        # 保存权重
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{model_type}_run{i+1}_weights.pth'))
        # 评估
        # 复用predict_train.py的数据处理和评估流程
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        import torch
        test_path = f"data/{TRAIN_NAME}/test.csv"
        test_set = pd.read_csv(test_path)
        label_column = cfg.label_column
        X_test = test_set.drop(columns=[label_column])
        y_test = test_set[label_column] / 3600
        y_test = np.log1p(y_test)
        scaler = StandardScaler()
        train_set = pd.read_csv(f"data/{TRAIN_NAME}/train.csv")
        X_train = train_set.drop(columns=[label_column])
        scaler.fit(X_train)
        X_test = scaler.transform(X_test)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
        from torch.utils.data import DataLoader, TensorDataset
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)
        # 构造评估配置
        class CfgObj:
            pass
        eval_cfg = CfgObj()
        eval_cfg.train_output_dir = SAVE_DIR
        eval_cfg.dataset_name = f"{TRAIN_NAME}_{model_type}_run{i+1}"
        eval_cfg.output_model_name = f"{TRAIN_NAME}_{model_type}_run{i+1}"
        mae, mse, rmse, r2 = evaluate_regression(model, test_loader, eval_cfg, device=device)
        metrics = {'run': i+1, 'random_state': seed, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        results[model_type].append(metrics)

# 保存结果
for model_type in MODEL_TYPES:
    df = pd.DataFrame(results[model_type])
    df.to_csv(os.path.join(SAVE_DIR, f'{model_type}_metrics.csv'), index=False)

# 假设检验
metric_name = 'MAE'  # 可选MAE/MSE/R2
x = pd.DataFrame(results['WideAndDeepNN'])[metric_name]
y = pd.DataFrame(results['WideOnlyNN'])[metric_name]
t_stat, t_p = ttest_ind(x, y, equal_var=False)
u_stat, u_p = mannwhitneyu(x, y, alternative='two-sided')

with open(os.path.join(SAVE_DIR, 'stat_test.txt'), 'w', encoding='utf-8') as f:
    f.write(f'T检验: t={t_stat:.4f}, p={t_p:.4g}\n')
    f.write(f'曼-惠特尼U检验: U={u_stat:.4f}, p={u_p:.4g}\n')
    f.write(f'WideAndDeepNN {metric_name}均值: {x.mean():.4f}\n')
    f.write(f'WideOnlyNN {metric_name}均值: {y.mean():.4f}\n')

print('全部实验完成，结果与假设检验已保存到', SAVE_DIR)
