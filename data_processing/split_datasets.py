import pandas as pd
import os


def split_dataset_for_project(project_name, input_file, output_base_dir, train_ratio=0.8):
    """
    为单个项目划分训练集和测试集
    """
    try:
        print(f"正在处理项目: {project_name}")

        # 读取数据
        data = pd.read_csv(input_file)
        print(f"数据形状: {data.shape}")

        # 设置划分比例
        train_size = int(len(data) * train_ratio)

        # 划分训练集和测试集 ---之前已经按照时间排好序
        train_df = data.iloc[:train_size]
        test_df = data.iloc[train_size:]

        # 创建项目输出文件夹
        project_output_dir = os.path.join(output_base_dir, project_name)
        if not os.path.exists(project_output_dir):
            os.makedirs(project_output_dir)

        # 保存训练集和测试集
        train_output_path = os.path.join(project_output_dir, "train.csv")
        test_output_path = os.path.join(project_output_dir, "test.csv")

        train_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)

        print(f"划分完成:")
        print(f"训练集大小: {len(train_df)} ({len(train_df) / len(data) * 100:.1f}%)")
        print(f"测试集大小: {len(test_df)} ({len(test_df) / len(data) * 100:.1f}%)")
        print(f"保存路径: {project_output_dir}")

        return True

    except Exception as e:
        print(f"处理项目 {project_name} 时出错: {e}")
        return False


def main():
    # 设置路径
    input_folder = r"E:\codes\MachineLearning\Lab2\all_features\renamed_features\merged_datasets\cleaned_data"
    output_base_dir = r"E:\codes\MachineLearning\Lab2\all_features\renamed_features\split_datasets"

    # 创建输出基础文件夹
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # 设置训练集比例
    train_ratio = 0.8

    # 获取所有清洗后的CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.startswith('cleaned_') and f.endswith('.csv')]

    print(f"找到 {len(csv_files)} 个清洗后的数据集文件")
    print(f"训练集比例: {train_ratio * 100}%")
    print("开始批量划分训练集和测试集...\n")

    success_count = 0
    failed_files = []

    # 处理每个文件
    for file in csv_files:
        input_file_path = os.path.join(input_folder, file)

        # 从文件名提取项目名
        # 文件名格式: cleaned_项目名_PR_features.csv
        base_name = file.replace('cleaned_', '').replace('.csv', '')
        if '_PR_features' in base_name:
            project_name = base_name.replace('_PR_features', '')
        else:
            project_name = base_name

        if split_dataset_for_project(project_name, input_file_path, output_base_dir, train_ratio):
            success_count += 1
        else:
            failed_files.append(project_name)

        print()  # 空行分隔

    # 输出总结
    print(f"{'=' * 50}")
    print("批量划分总结:")
    print(f"成功划分: {success_count} 个项目")
    print(f"划分失败: {len(failed_files)} 个项目")
    if failed_files:
        print("失败项目列表:")
        for project in failed_files:
            print(f"   - {project}")
    print(f"划分后的文件保存在: {output_base_dir}")

if __name__ == "__main__":
    main()