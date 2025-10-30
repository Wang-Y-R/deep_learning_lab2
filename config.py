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

class PredictConfig:
    # 支持灵活配置
    train_data_names = ["django", "opencv"]
    label_column = "time_to_close"            # 预测目标
    batch_size = 64
    lr = 1e-3
    epochs = 30
    random_state = 42

    # 支持批量评估
    evaluate_datas_name = ["moby", "react", "salt", "scikit-learn"]
    # 下面这些属性在批量训练/评估时由训练器和评估器动态生成
    # train_path, test_path, train_output_dir, output_model_name, dataset_name, evaluate_model_path, evaluate_data_paths, evaluate_output_dir

class MultiTaskConfig:
    # ========= 数据设置 =========
    train_data_names = ["django", "opencv"]
    label_columns = ["time_to_close", "merged"]  # 两个任务标签

    # ========= 模型训练参数 =========
    batch_size = 64
    lr = 1e-3
    epochs = 30
    random_state = 42

    # ========= 训练器入口 =========
    # 对应 trainers/multitask_train.py 中的 train() 函数
    def trainer(self, device="cpu"):
        from trainers.multitask_train import train
        return train(self, device=device)

    # ========= 输出与模型保存 =========
    # 每个仓库都会单独保存训练结果
    def get_paths(self, name):
        base = f"data/{name}/"
        paths = {
            "train_path": base + "train.csv",
            "test_path": base + "test.csv",
            "output_dir": f"outputs/train/multitask/{name}",
            "output_model": f"outputs/train/multitask/{name}/model.pth",
            "output_result": f"outputs/train/multitask/{name}/result.csv"
        }
        return paths