# config.py

class Config:
    # 数据集
    train_data_name = "django"
    test_data_name = "django"
    train_path = "data/" + train_data_name +"/train.csv"
    test_path = "data/" +test_data_name + "/test.csv"  
    label_column = "merged"   # "预测目标列"
    
    #训练器 
    def trainer(self, device="cpu"):
        from trainers.is_merged_train import train
        return train(self, device=device)
    
    # 训练参数
    batch_size = 32
    lr = 0.001
    epochs = 30
    random_state = 42

    # 输出目录
    output_dir = "outputs/is_merged"
    loss_curve_name = "train_by_" + train_data_name + "_loss_curve"
    output_model_name = "train_by_" + train_data_name + "_model"
    output_result_name = "test_by_" + test_data_name + "_result"
