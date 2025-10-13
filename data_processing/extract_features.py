import pandas as pd
import os

# 设置存放所有 pr_feature.xlsx 文件的文件夹路径
folder_path = r"E:\codes\MachineLearning\Lab2\all_features\renamed_features\merged_datasets"  # 请使用原始路径，不要修改

# 检查文件夹是否存在
if not os.path.exists(folder_path):
    print(f"文件夹不存在: {folder_path}")
    exit()

# 获取所有文件
all_files = os.listdir(folder_path)
print(f"文件夹中的文件: {all_files}")

# 获取所有 xlsx 文件
xlsx_files = [f for f in all_files if f.endswith('.csv') and 'PR_features' in f]
print(f"找到的PR特征文件: {xlsx_files}")

# 存储所有特征名
all_features = []

for file in xlsx_files:
    file_path = os.path.join(folder_path, file)
    print(f"\n正在读取文件: {file}")

    try:
        # 读取第一行（列名）
        df = pd.read_excel(file_path, nrows=0)
        features = df.columns.tolist()
        all_features.extend(features)
        print(f"成功读取，特征数量: {len(features)}")
        print(f"前5个特征: {features[:5]}")  # 显示前5个特征作为样例
    except Exception as e:
        print(f"读取文件 {file} 时出错: {e}")

# 去重并排序
all_features = sorted(set(all_features))

print(f"\n{'=' * 50}")
print("所有不重复的特征名称如下：")
for i, feature in enumerate(all_features, 1):
    print(f"{i:3d}. {feature}")

print(f"\n总计特征数量: {len(all_features)}")

# 保存到文件以便后续使用
with open('all_features_list.txt', 'w', encoding='utf-8') as f:
    for feature in all_features:
        f.write(feature + '\n')
print("特征列表已保存到: all_features_list.txt")