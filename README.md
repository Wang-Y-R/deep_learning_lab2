# 项目结构

- config.py 配置设置
- main.py 模型训练
- evaluate_is_merged.py 二分类任务泛化性评估
- data 数据集
- utils 工具类
- trainers 自定义训练器
- outputs 输出结果
- models 模型定义

# 模型训练和验证

修改config中下述选项，(可为自己的训练器添加其它参数),运行main.py

```py
train_data_name = "django"
test_data_name = "django"

# 请指定自己的训练器 
def trainer(self, device="cpu"):
    from trainers.is_merged_train import train
    return train(self, device=device)

# 训练参数
batch_size = 32
lr = 0.001
epochs = 30
random_state = 42
```

# 二分类任务泛化性评估

修改config.py中下述选项,运行evaluate_is_merged.py
```py
evaluate_model_path = "outputs/train/is_merged/" + train_data_name + "_to_" + test_data_name + "/train_by_" + train_data_name + "_model.pth" #待测试模型
evaluate_datas_name = ["tensorflow", "moby",  "react"] #用来预测的数据集名称
```