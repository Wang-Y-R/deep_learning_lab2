import pandas as pd
import os


def detect_issues_detailed(df: pd.DataFrame, schema_config: dict, report_file: str):
    """
    详细的数据检测，包括每列的实际值范围
    """
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== 详细数据检测报告 ===\n\n")
        f.write(f"原始数据行数: {len(df)}\n")
        f.write(f"原始数据列数: {len(df.columns)}\n\n")

        n_rows = len(df)
        any_issue = False

        for col, cfg in schema_config.items():
            if col not in df.columns:
                f.write(f"{col}: ❌ 列不存在\n\n")
                continue

            series = df[col]
            null_count = series.isna().sum()
            null_ratio = null_count / n_rows

            # 显示列的基本信息
            f.write(f"{col}:\n")
            f.write(f"  非空值数量: {n_rows - null_count}\n")
            f.write(f"  空值比例: {null_ratio:.2%}\n")

            if null_count < n_rows:  # 如果有非空值
                non_null_series = series.dropna()
                f.write(f"  实际值范围: [{non_null_series.min():.6f}, {non_null_series.max():.6f}]\n")
                if len(non_null_series.unique()) <= 10:  # 对于枚举型列
                    f.write(f"  实际唯一值: {sorted(non_null_series.unique())}\n")

            out_of_range_count = 0
            if "range" in cfg and null_count < n_rows:
                low, high = cfg["range"]
                out_of_range_count = (~series.between(low, high)).sum()
                if out_of_range_count > 0:
                    f.write(f"  ⚠️ 超出范围[{low}, {high}]的数量: {out_of_range_count}\n")
            elif "enum" in cfg and null_count < n_rows:
                allowed = set(cfg["enum"])
                actual_values = set(series.dropna().unique())
                unexpected_values = actual_values - allowed
                out_of_range_count = (~series.isin(allowed)).sum()
                if out_of_range_count > 0:
                    f.write(f"  ⚠️ 期望枚举值: {allowed}\n")
                    f.write(f"  ⚠️ 实际枚举值: {actual_values}\n")
                    f.write(f"  ⚠️ 意外值: {unexpected_values}\n")
                    f.write(f"  ⚠️ 非法值数量: {out_of_range_count}\n")

            total_issues = null_count + out_of_range_count
            if total_issues > 0:
                any_issue = True
                f.write(f"  ⚠️ 总问题数量: {total_issues}\n")

            f.write("\n")

        if not any_issue:
            f.write("✅ 全部符合规范\n")


def clean_dataframe_gentle(df: pd.DataFrame, schema_config: dict) -> pd.DataFrame:
    """
    温和的数据清洗：由于数据已经归一化，主要处理缺失值和枚举值
    """
    df_clean = df.copy()

    # 处理枚举值
    for col, cfg in schema_config.items():
        if col not in df_clean.columns:
            continue

        if "enum" in cfg:
            # 对于枚举列，将不在枚举中的值设为默认值或第一个枚举值
            enum_values = cfg["enum"]
            default_value = cfg.get("default", enum_values[0])
            df_clean[col] = df_clean[col].apply(lambda x: x if x in enum_values else default_value)

    # 处理缺失值
    for col, cfg in schema_config.items():
        if col not in df_clean.columns:
            continue

        if df_clean[col].isna().any():
            if "default" in cfg:
                df_clean[col] = df_clean[col].fillna(cfg["default"])
            elif "enum" in cfg:
                # 对于枚举列，用最常见的值填充
                most_common = df_clean[col].mode()
                if len(most_common) > 0:
                    df_clean[col] = df_clean[col].fillna(most_common[0])
            else:
                # 对于数值列，用中位数填充（注意：已经是归一化后的数据）
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 只保留 schema_config 中存在的列
    available_cols = [col for col in schema_config.keys() if col in df_clean.columns]

    return df_clean[available_cols]


def process_single_file(input_file: str, output_dir: str, schema_config: dict):
    """
    处理单个CSV文件 - 移除了归一化步骤
    """
    print(f"🔍 正在处理文件: {os.path.basename(input_file)}")

    try:
        # 读取文件
        df = pd.read_csv(input_file)

        print(f"   原始数据形状: {df.shape}")

        # 检查必要的列
        required_cols = ['number', 'merged']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️ 缺少必要列: {missing_cols}")
            return False

        # 生成详细检测报告
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        report_file = os.path.join(output_dir, f"{base_name}_clean_report.txt")
        detect_issues_detailed(df, schema_config, report_file)

        # 执行温和清洗（主要处理缺失值和枚举值）
        cleaned = clean_dataframe_gentle(df, schema_config)

        print(f"   清洗后数据形状: {cleaned.shape}")

        if len(cleaned) == 0:
            print("   ❌ 清洗后数据为空")
            return False

        # 注意：不再进行归一化，因为数据已经在合表时归一化过了

        # 输出最终结果
        output_file = os.path.join(output_dir, f"cleaned_{base_name}.csv")
        cleaned.to_csv(output_file, index=False)

        print(f"✅ 完成清洗: {os.path.basename(output_file)}")
        print(f"   最终数据形状: {cleaned.shape}")
        return True

    except Exception as e:
        print(f"❌ 处理文件 {input_file} 时出错: {e}")
        import traceback
        print(f"   详细错误: {traceback.format_exc()}")
        return False


def main():
    # 设置路径
    input_folder = r"E:\codes\MachineLearning\Lab2\all_features\renamed_features\merged_datasets"
    output_folder = os.path.join(input_folder, "cleaned_data")

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 调整范围设置以适应归一化后的数据 [0, 1] 范围
    schema_config = {
        # 标识列
        "number": {"type": "int", "required": True},

        # 代码变更特征 - 调整为归一化后的范围
        "directory_num": {"range": (0, 1.1)},  # 稍微超过1以容错
        "language_num": {"range": (0, 1.1)},
        "file_type": {"range": (0, 1.1)},
        "has_test": {"enum": [0, 1], "default": 0},

        # 文本内容特征
        "has_feature": {"enum": [0, 1], "default": 0},
        "has_bug": {"enum": [0, 1], "default": 0},
        "has_document": {"enum": [0, 1], "default": 0},
        "has_improve": {"enum": [0, 1], "default": 0},
        "has_refactor": {"enum": [0, 1], "default": 0},
        "subject_length": {"range": (0, 1.1)},
        "subject_readability": {"range": (0, 1.1)},
        "message_length": {"range": (0, 1.1)},
        "message_readability": {"range": (0, 1.1)},

        # 代码变更规模 - 调整为归一化后的范围
        "lines_added": {"range": (0, 1.1)},
        "lines_deleted": {"range": (0, 1.1)},
        "segs_added": {"range": (0, 1.1)},
        "segs_deleted": {"range": (0, 1.1)},
        "segs_updated": {"range": (0, 1.1)},
        "files_added": {"range": (0, 1.1)},
        "files_deleted": {"range": (0, 1.1)},
        "files_updated": {"range": (0, 1.1)},
        "modify_proportion": {"range": (0, 1.1)},
        "modify_entropy": {"range": (0, 1.1)},
        "test_churn": {"range": (0, 1.1)},
        "non_test_churn": {"range": (0, 1.1)},

        # 评审相关特征
        "reviewer_num": {"range": (0, 1.1)},
        "bot_reviewer_num": {"range": (0, 1.1)},
        "has_reviewed": {"enum": [0, 1], "default": 0},
        "comment_num": {"range": (0, 1.1)},
        "comment_length": {"range": (0, 1.1)},
        "last_comment_mention": {"enum": [0, 1], "default": 0},

        # 目标变量
        "merged": {"enum": [0, 1], "default": 0},
    }

    # 获取所有合并后的CSV文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    print(f"🔍 找到 {len(csv_files)} 个数据集文件")
    print("开始批量清洗数据...\n")

    success_count = 0
    failed_files = []

    # 处理每个文件
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        if process_single_file(file_path, output_folder, schema_config):
            success_count += 1
        else:
            failed_files.append(file)
        print()  # 空行分隔

    # 输出总结
    print(f"{'=' * 50}")
    print("📊 批量处理总结:")
    print(f"✅ 成功处理: {success_count} 个文件")
    print(f"❌ 处理失败: {len(failed_files)} 个文件")
    if failed_files:
        print("失败文件列表:")
        for file in failed_files:
            print(f"   - {file}")
    print(f"📁 清洗后的文件保存在: {output_folder}")


if __name__ == "__main__":
    main()