import pandas as pd
import os

# 特征映射字典
feature_mapping = {
    # 代码变更特征映射
    'title_length': 'subject_length',
    'title_readability': 'subject_readability',
    'title_embedding': 'subject_embedding',
    'body_length': 'message_length',
    'body_readability': 'message_readability',
    'body_embedding': 'message_embedding',
    'is_reviewed': 'has_reviewed',
    # 其他特征名称保持不变
}


def rename_features_in_file(file_path, output_dir):
    """重命名单个文件的特征"""
    try:
        # 读取文件
        df = pd.read_excel(file_path)

        # 记录原始特征名
        original_features = df.columns.tolist()

        # 重命名特征
        df_renamed = df.rename(columns=feature_mapping)

        # 获取新文件名
        base_name = os.path.basename(file_path)
        new_file_path = os.path.join(output_dir, f"renamed_{base_name}")

        # 保存重命名后的文件
        df_renamed.to_excel(new_file_path, index=False)

        # 打印重命名信息
        new_features = df_renamed.columns.tolist()
        changed_features = []

        for orig, new in zip(original_features, new_features):
            if orig != new:
                changed_features.append((orig, new))

        print(f"已处理: {base_name}")
        print(f"原始特征数: {len(original_features)}, 新特征数: {len(new_features)}")
        if changed_features:
            print(f"重命名的特征:")
            for orig, new in changed_features:
                print(f"{orig} -> {new}")
        else:
            print(f"无特征需要重命名")

        return True

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def main():
    # 设置路径
    input_folder = r"C:\Users\Minst\Desktop\Lab2\all_features"
    output_folder = os.path.join(input_folder, "renamed_features")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有PR特征文件
    xlsx_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx') and 'PR_features' in f]

    print(f"找到 {len(xlsx_files)} 个PR特征文件")
    print("开始重命名特征...\n")

    success_count = 0
    failed_files = []

    # 处理每个文件
    for file in xlsx_files:
        file_path = os.path.join(input_folder, file)
        if rename_features_in_file(file_path, output_folder):
            success_count += 1
        else:
            failed_files.append(file)

    # 输出总结
    print(f"\n{'=' * 50}")
    print("处理总结:")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {len(failed_files)} 个文件")
    if failed_files:
        print("失败文件列表:")
        for file in failed_files:
            print(f"-{file}")
    print(f"重命名后的文件保存在: {output_folder}")


if __name__ == "__main__":
    main()