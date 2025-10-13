# models/model.py
import torch.nn as nn


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64):
        
        # 模型参数
        hidden_dim1 = 256
        hidden_dim2 = 128
        hidden_dim3 = 64
        
        super(FeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim3, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.net(x)

