import pandas as pd
import numpy as np

def generate_time_intervals_and_write_back(dataset, model="MLKT4"):
    # 处理训练数据
    train_data = pd.read_csv("../../data/"+str(dataset)+"/"+str(model)+"/q_a_train_detailed.csv")
    train_data['ServerTimestamp2'] = pd.to_datetime(train_data['ServerTimestamp']).astype(int) / 10 ** 9
    train_data['TimeInterval'] = np.nan  # 先初始化为 NaN
    train_data = _process_data(train_data)

    # 处理验证数据
    valid_data = pd.read_csv("../../data/"+str(dataset)+"/"+str(model)+"/q_a_valid_detailed.csv")
    valid_data['ServerTimestamp2'] = pd.to_datetime(valid_data['ServerTimestamp']).astype(int) / 10 ** 9
    valid_data['TimeInterval'] = np.nan  # 先初始化为 NaN
    valid_data = _process_data(valid_data)

    # 处理测试数据
    test_data = pd.read_csv("../../data/"+str(dataset)+"/"+str(model)+"/q_a_test_detailed.csv")
    test_data['ServerTimestamp2'] = pd.to_datetime(test_data['ServerTimestamp']).astype(int) / 10 ** 9
    test_data['TimeInterval'] = np.nan  # 先初始化为 NaN
    test_data = _process_data(test_data)

    # 将数据写回原始 CSV 文件
    train_data.to_csv("../../data/"+str(dataset)+"/"+str(model)+"/q_a_train_detailed.csv", index=False)
    valid_data.to_csv("../../data/"+str(dataset)+"/"+str(model)+"/q_a_valid_detailed.csv", index=False)
    test_data.to_csv("../../data/"+str(dataset)+"/"+str(model)+"/q_a_test_detailed.csv", index=False)

def _process_data(data):
    # 按 SubjectID 分组处理
    for subject_id, group in data.groupby('SubjectID'):
        timestamps = group['ServerTimestamp2'].values
        if len(timestamps) > 1:
            # 计算时间间隔（以分钟为单位）
            shft_timestamps = np.concatenate([[timestamps[0]], timestamps[:-1]])
            it = np.maximum(np.minimum((timestamps - shft_timestamps) / 60, 43200), -1)
            # 直接将时间间隔写入数据框
            data.loc[group.index, 'TimeInterval'] = it
        else:
            # 如果只有一个时间戳，时间间隔默认为 1
            data.loc[group.index, 'TimeInterval'] = 1
    return data