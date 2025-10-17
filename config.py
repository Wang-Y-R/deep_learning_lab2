# config.py

train_data_name = "opencv"
test_data_name = "opencv"

class Config:
    # 数据集
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

    # 单数据集训练输出目录
    train_output_dir = "outputs/train/is_merged/" + train_data_name + "_to_" + test_data_name
    loss_curve_name = "train_by_" + train_data_name + "_loss_curve"
    output_model_name = "train_by_" + train_data_name + "_model"
    output_result_name = "test_by_" + test_data_name + "_result"
    
    # 已有模型泛化评估
    evaluate_datas_name = ["tensorflow", "moby",  "react"]
    evaluate_model_path = "outputs/train/is_merged/" + train_data_name + "_to_" + test_data_name + "/train_by_" + train_data_name + "_model.pth"
    evaluate_data_paths = ["data/" + name + "/test.csv" for name in evaluate_datas_name]
    evaluate_output_dir = [f"outputs/evaluate/is_merged/{train_data_name}_to_{name}" for name in evaluate_datas_name ]
