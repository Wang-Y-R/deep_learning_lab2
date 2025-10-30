import torch
import torch.nn as nn

# Wide&Deep 模型
class WideAndDeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.4):
        super(WideAndDeepNN, self).__init__()
        # Wide部分：线性
        self.wide = nn.Linear(input_dim, 1)
        # Deep部分：更深更宽的MLP
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep_out(self.deep(x))
        out = wide_out + deep_out  # 拼接后回归
        return out.squeeze(-1)

# Wide Only
class WideOnlyNN(nn.Module):
    def __init__(self, input_dim):
        super(WideOnlyNN, self).__init__()
        self.wide = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.wide(x).squeeze(-1)

# Deep Only
class DeepOnlyNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.4):
        super(DeepOnlyNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(hidden_dims[-1], 1)
    def forward(self, x):
        deep_out = self.deep_out(self.deep(x))
        return deep_out.squeeze(-1)
