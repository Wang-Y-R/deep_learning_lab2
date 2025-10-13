import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def merge_project_files(features_file, info_file, output_dir):
    """
    合并单个项目的 PR_features 和 PR_info 文件
    """
    try:
        # 提取项目名
        project_name = features_file.replace('_PR_features.xlsx', '')
        print(f"正在处理项目: {project_name}")

        # 读取 Excel 文件
        features = pd.read_excel(features_file, dtype={"number": int})
        info = pd.read_excel(info_file, dtype={"number": int})

        print(f"PR_features 形状: {features.shape}")
        print(f"PR_info 形状: {info.shape}")

        # 定义需要从info表中提取的列
        info_cols_to_add = ["created_at", "merged"]

        # 检查必要的列是否存在
        missing_cols = [col for col in info_cols_to_add if col not in info.columns]
        if missing_cols:
            print(f"PR_info 中缺少列: {missing_cols}")
            # 尝试使用其他可能的列名
            if 'merged_at' in info.columns and 'merged' not in info.columns:
                info_cols_to_add = ["created_at", "merged_at"]
                print(f"使用 merged_at 替代 merged")

        # 合并
        merged = pd.merge(
            features,
            info[["number"] + info_cols_to_add],
            on="number",
            how="inner"
        )

        print(f"   合并后形状: {merged.shape}")

        # 转换时间类型并排序
        if "created_at" in merged.columns:
            merged["created_at"] = pd.to_datetime(merged["created_at"], errors='coerce')
            merged = merged.sort_values(by="created_at", ascending=True)

        # 丢掉 created_at 和 embedding 列
        cols_to_drop = ["created_at"]
        embedding_cols = ["title_embedding", "body_embedding", "comment_embedding",
                          "subject_embedding", "message_embedding"]

        # 删除存在的embedding列
        for col in embedding_cols:
            if col in merged.columns:
                cols_to_drop.append(col)

        merged = merged.drop(columns=cols_to_drop, errors='ignore')

        # 替换 inf/-inf 为 NaN
        merged = merged.replace([np.inf, -np.inf], np.nan)

        # 删除所有含 NaN 的行
        initial_rows = len(merged)
        merged = merged.dropna()
        final_rows = len(merged)

        print(f"删除空值后: {final_rows} 行 (删除了 {initial_rows - final_rows} 行)")

        if len(merged) == 0:
            print(f"合并后数据为空，跳过该项目")
            return None

        # 归一化处理
        numeric_cols = merged.select_dtypes(include=['int64', 'float64']).columns

        # 排除不需要归一化的列
        exclude_cols = ['number', 'merged']
        if 'merged_at' in merged.columns:
            exclude_cols.append('merged_at')

        numeric_cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

        if len(numeric_cols_to_scale) > 0:
            scaler = MinMaxScaler()
            merged[numeric_cols_to_scale] = scaler.fit_transform(merged[numeric_cols_to_scale])
            print(f"已对 {len(numeric_cols_to_scale)} 个数值特征进行归一化")

        # 输出为 CSV
        output_file = os.path.join(output_dir, f"{project_name}_PR_features.csv")
        merged.to_csv(output_file, index=False, encoding="utf-8")

        print(f"完成合并: {os.path.basename(output_file)}")
        print(f"最终数据形状: {merged.shape}\n")

        return merged

    except Exception as e:
        print(f"处理项目时出错: {e}")
        return None


def find_matching_project_files(folder_path):
    """
    在目录中查找匹配的 PR_features 和 PR_info 文件
    """
    all_files = os.listdir(folder_path)

    # 找出所有 PR_features 文件
    features_files = [f for f in all_files if f.endswith('_PR_features.xlsx')]

    matching_pairs = []

    for features_file in features_files:
        # 提取项目名
        project_name = features_file.replace('_PR_features.xlsx', '')
        info_file = f"{project_name}_PR_info.xlsx"

        if info_file in all_files:
            matching_pairs.append((features_file, info_file))
        else:
            print(f"找不到匹配的info文件: {info_file}")

    return matching_pairs


def main():
    # 设置路径
    input_folder = r"E:\codes\MachineLearning\Lab2\all_features\renamed_features"
    output_folder = os.path.join(input_folder, "merged_datasets")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("正在查找匹配的项目文件...")

    # 查找匹配的文件对
    matching_pairs = find_matching_project_files(input_folder)

    if not matching_pairs:
        print("未找到匹配的 PR_features 和 PR_info 文件对")
        return

    print(f"找到 {len(matching_pairs)} 对匹配的项目文件\n")

    success_count = 0
    failed_projects = []

    # 处理每个项目
    for features_file, info_file in matching_pairs:
        features_path = os.path.join(input_folder, features_file)
        info_path = os.path.join(input_folder, info_file)

        result = merge_project_files(features_path, info_path, output_folder)
        if result is not None:
            success_count += 1
        else:
            failed_projects.append(features_file.replace('_PR_features.xlsx', ''))

    # 输出总结
    print(f"{'=' * 50}")
    print("批量合并总结:")
    print(f"成功合并: {success_count} 个项目")
    print(f"合并失败: {len(failed_projects)} 个项目")
    if failed_projects:
        print("失败项目列表:")
        for project in failed_projects:
            print(f"   - {project}")
    print(f"合并后的文件保存在: {output_folder}")


if __name__ == "__main__":
    main()