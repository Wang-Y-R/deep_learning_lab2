# config.py

class Config:
    # 数据集
    train_path = "data/tensorflow/train.csv"
    test_path = 'data/tensorflow/test.csv'  
    label_column = "merged"   # "预测目标列"
    
    #训练器 
    def trainer(self, device="cpu"):
        from trainers.train import train
        return train(self, device=device)
    
    # 训练参数
    batch_size = 32
    lr = 0.001
    epochs = 30
    random_state = 42

    # 输出目录
    output_dir = "outputs/"
