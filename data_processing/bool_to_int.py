import os
import pandas as pd


def process_csv_simple(file_path):
    """
    简化版本的CSV处理函数
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 自动转换布尔列为整型
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
            else:
                # 尝试将字符串布尔值转换为整型
                if df[col].dtype == 'object':
                    # 将常见的布尔字符串转换为整型
                    bool_map = {
                        'TRUE': 1, 'True': 1, 'true': 1, 'T': 1, 't': 1,
                        'YES': 1, 'Yes': 1, 'yes': 1, 'Y': 1, 'y': 1,
                        'FALSE': 0, 'False': 0, 'false': 0, 'F': 0, 'f': 0,
                        'NO': 0, 'No': 0, 'no': 0, 'N': 0, 'n': 0
                    }
                    df[col] = df[col].map(bool_map).fillna(df[col])

        # 保存文件
        df.to_csv(file_path, index=False)
        print(f"成功处理: {file_path}")
        return True

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def process_all_simple(root_dir='.'):
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            for csv_file in ['test.csv', 'train.csv']:
                csv_path = os.path.join(subdir_path, csv_file)
                if os.path.exists(csv_path):
                    process_csv_simple(csv_path)


# 使用简化版本
if __name__ == "__main__":
    process_all_simple()