import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime
import csv

# 设置序列长度限制
SEQ_LEN = 200


def load_data():
    """加载所有数据文件"""
    print("Loading data files...")

    # 加载数据文件
    submissions = pd.read_csv('../../data/BePKT/submission.csv')
    problems = pd.read_csv('../../data/BePKT/problem_skills.csv')
    code_reason = pd.read_csv('../../data/BePKT/CodeReason.csv')
    code_score = pd.read_csv('../../data/BePKT/CodeScore.csv')

    print(f"Submissions shape: {submissions.shape}")
    print(f"Problems shape: {problems.shape}")
    print(f"CodeReason shape: {code_reason.shape}")
    print(f"CodeScore shape: {code_score.shape}")

    return submissions, problems, code_reason, code_score


def process_timestamp(timestamp_str):
    """处理时间戳，转换为两种格式"""
    try:
        # 解析时间戳
        dt = pd.to_datetime(timestamp_str)
        # ISO格式时间戳
        iso_timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S')
        # Unix时间戳
        unix_timestamp = dt.timestamp()
        return iso_timestamp, unix_timestamp
    except:
        return None, None


def format_skills(skills_str):
    """将Skills格式从 '26,45' 转换为 '[26, 45]'"""
    try:
        if pd.isna(skills_str) or skills_str == '':
            return '[]'

        # 如果已经是方括号格式，直接返回
        if skills_str.startswith('[') and skills_str.endswith(']'):
            return skills_str

        # 分割并转换为整数列表
        skills_list = [int(x.strip()) for x in skills_str.split(',') if x.strip()]
        return str(skills_list)
    except:
        return '[]'


def merge_and_process_data(submissions, problems, code_reason, code_score):
    """合并并处理所有数据"""
    print("Merging and processing data...")

    # 使用submission的id作为CodeStateID进行关联
    # 1. 合并submissions和problems（通过problem_id关联）
    merged = submissions.merge(
        problems[['ProblemID', 'Skills']],
        left_on='problem_id',
        right_on='ProblemID',
        how='left'
    )

    # 2. 合并CodeReason（通过id关联CodeStateID）
    merged = merged.merge(
        code_reason,
        left_on='id',
        right_on='CodeStateID',
        how='left'
    )

    # 3. 合并CodeScore（通过id关联CodeStateID）
    merged = merged.merge(
        code_score,
        left_on='id',
        right_on='CodeStateID',
        how='left',
        suffixes=('', '_score')
    )

    print(f"Merged data shape: {merged.shape}")

    # 处理数据
    MLKT4_data = []

    # 按用户分组以计算时间间隔
    for user_id in tqdm(merged['user_id'].unique(), desc="Processing users"):
        user_data = merged[merged['user_id'] == user_id].copy()

        # 按时间排序
        if 'create_time' in user_data.columns:
            user_data = user_data.sort_values('create_time')

        # 截断到最大序列长度
        user_data = user_data.head(SEQ_LEN)

        # 计算时间间隔
        prev_timestamp = None

        for idx, row in user_data.iterrows():
            # 跳过缺失必要信息的记录
            if pd.isna(row.get('Score')) or pd.isna(row.get('Skills')):
                continue

            # 处理时间戳
            iso_ts, unix_ts = process_timestamp(row['create_time'])
            if iso_ts is None:
                continue

            # 计算时间间隔
            if prev_timestamp is None:
                time_interval = 0.0
            else:
                time_interval = unix_ts - prev_timestamp

            # 标准化分数（假设满分为100，转换到0-1范围）
            label = float(row['Score']) / 50

            # 构建记录
            record = {
                'SubjectID': int(row['user_id']),
                'ProblemID': int(row['problem_id']),
                'Skills': format_skills(row['Skills']),  # 修改：格式化为方括号格式
                'Reason': row['Reason'] if pd.notna(row.get('Reason')) else '[]',
                'Label': round(label, 2),
                'ServerTimestamp': iso_ts,
                'ServerTimestamp2': unix_ts,
                'TimeInterval': time_interval / 60
            }

            MLKT4_data.append(record)
            prev_timestamp = unix_ts

    return pd.DataFrame(MLKT4_data)


def remap_problem_ids(df):
    """对ProblemID按大小排序并重新分配连续编号，从1开始"""
    print("Remapping ProblemIDs...")

    # 获取唯一ProblemID并排序
    unique_problems = sorted(df['ProblemID'].unique())

    # 创建映射字典：旧ID -> 新ID (从1开始)
    problem_map = {old_id: new_id for new_id, old_id in enumerate(unique_problems, start=1)}

    # 替换DataFrame中的ProblemID
    df['ProblemID'] = df['ProblemID'].map(problem_map)

    print(f"Remapped {len(unique_problems)} unique ProblemIDs.")

    return df


def split_and_save_data(df, test_size=0.2, random_state=42):
    """分割数据集并保存"""
    print("Splitting and saving data...")

    # 获取所有唯一用户
    all_users = df['SubjectID'].unique()

    # 分割用户
    train_users, temp_users = train_test_split(
        all_users, test_size=test_size, random_state=random_state
    )
    test_users, valid_users = train_test_split(
        temp_users, test_size=0.5, random_state=random_state
    )

    # 根据用户分割数据
    train_df = df[df['SubjectID'].isin(train_users)]
    test_df = df[df['SubjectID'].isin(test_users)]
    valid_df = df[df['SubjectID'].isin(valid_users)]

    # 创建输出目录
    os.makedirs('../../data/BePKT/MLKT4', exist_ok=True)

    # 保存数据
    train_df.to_csv('../../data/BePKT/MLKT4/q_a_train_detailed.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    test_df.to_csv('../../data/BePKT/MLKT4/q_a_test_detailed.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
    valid_df.to_csv('../../data/BePKT/MLKT4/q_a_valid_detailed.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Train set: {len(train_df)} records from {len(train_users)} users")
    print(f"Test set: {len(test_df)} records from {len(test_users)} users")
    print(f"Valid set: {len(valid_df)} records from {len(valid_users)} users")

    # 打印示例数据
    print("\nSample records from training set:")
    print(train_df.head(3).to_string())

    return train_df, test_df, valid_df


def main():
    """主处理流程"""
    try:
        # 1. 加载数据
        submissions, problems, code_reason, code_score = load_data()

        # 2. 合并和处理数据
        MLKT4_df = merge_and_process_data(
            submissions, problems, code_reason, code_score
        )

        # 3. 重新映射ProblemID
        MLKT4_df = remap_problem_ids(MLKT4_df)

        print(f"\nTotal MLKT4 records: {len(MLKT4_df)}")

        # 4. 分割并保存数据
        train_df, test_df, valid_df = split_and_save_data(MLKT4_df)

        # 5. 打印统计信息
        print(f"\nDataset Statistics:")
        print(f"Total users: {MLKT4_df['SubjectID'].nunique()}")
        print(f"Total problems: {MLKT4_df['ProblemID'].nunique()}")
        print(f"Average label (score): {MLKT4_df['Label'].mean():.3f}")

        print("\nData processing completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please make sure all required CSV files are in the correct directory")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()