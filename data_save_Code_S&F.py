import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description='Prepare Code-S full or 5000-CodeState subset')
parser.add_argument('--subset', choices=['full', '5000'], default='full')
parser.add_argument('--dry_run', action='store_true', help='Validate and count splits without writing CSV files')
args = parser.parse_args()
DATA = Path(__file__).resolve().parents[2] / 'data' / 'Code-S'
suffix = '_5000' if args.subset == '5000' else ''
output_model = 'MLKT4_5000' if args.subset == '5000' else 'MLKT4'
OUTPUT = DATA / output_model
SEQ_LEN = 200
print(f'Subset: {args.subset}; input: {DATA}; output: {OUTPUT}')
def save_split(frame, split):
    if frame.empty:
        raise ValueError(f'{split} split is empty')
    if frame['Label'].isna().any() or not frame['Label'].between(0, 1).all():
        raise ValueError(f'{split} has invalid labels')
    print(f'{split}: {len(frame)} interactions, {frame.SubjectID.nunique()} students')
    if not args.dry_run:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUTPUT / f'q_a_{split}_detailed.csv', index=False)

# Load data
main_table = pd.read_csv(DATA / f'MainTable{suffix}.csv').drop(columns=['Reason'], errors='ignore')

skill_table = pd.read_csv(DATA / 'problem_skills.csv')
code_table = pd.read_csv(DATA / f'CodeStates{suffix}.csv', usecols=['CodeStateID', 'Code'])
score_table = pd.read_csv(DATA / 'V31' / 'CodeScore.csv')
reason_table = pd.read_csv(DATA / 'V31' / 'CodeReason.csv')
if args.subset == '5000':
    ids = set(code_table['CodeStateID'])
    # Existing *_5000 score/reason files select different IDs.
    # Join by the canonical CodeStates_5000 IDs, never by row position.
    score_table = score_table.loc[score_table['CodeStateID'].isin(ids)].copy()
    reason_table = reason_table.loc[reason_table['CodeStateID'].isin(ids)].copy()
    print('Score/Reason: filtered V31 full tables by the selected CodeStateIDs')
    if len(code_table) != 5000 or len(ids) != 5000:
        raise ValueError('Expected exactly 5000 unique CodeStates')
    for name, frame in [('MainTable', main_table), ('CodeScore', score_table), ('CodeReason', reason_table)]:
        if not set(frame['CodeStateID'].dropna()).issubset(ids):
            raise ValueError(f'{name} includes CodeStates outside the 5000 subset')
    for name, frame in [('CodeScore', score_table), ('CodeReason', reason_table)]:
        if len(frame) != 5000 or frame['CodeStateID'].nunique() != 5000:
            raise ValueError(f'{name} must cover exactly the 5000 CodeStates')
    print(f'Checked {len(ids)} unique CodeStates and {len(main_table)} event rows')
# Merge tables
# print(len(code_table))
code_table = code_table.dropna(subset=['Code'])
# print(len(code_table))
main_table = main_table.merge(skill_table, how='left', on='ProblemID')
main_table = main_table.merge(code_table, how='left', on='CodeStateID')



# sys.exit()
# Clean data by dropping rows where 'Score' is NaN and reset index
main_table.dropna(subset=['Score'], inplace=True)

# Step 1: Record CodeStateIDs where Score == 1.0
score_one_code_state_ids = main_table[main_table['Score'] == 1.0]['CodeStateID']
#
main_table = main_table.drop('Score', axis=1)
main_table = main_table.merge(score_table, how='right', on='CodeStateID')
main_table = main_table.merge(reason_table, how='left', on='CodeStateID')

# # Delete rows where 'Reason' is NaN
main_table = main_table.dropna(subset=['Reason'], inplace=False)

column_name = 'Score'
main_table[column_name] = pd.to_numeric(main_table[column_name], errors='coerce')
main_table[column_name] = main_table[column_name] / 50

# Step 2: Manually set new Score to 1.0 for rows where original Score was 1.0
main_table.loc[main_table['CodeStateID'].isin(score_one_code_state_ids), column_name] = 1.0

main_table.reset_index(drop=True, inplace=True)

# Define functions


def genr_new_qid(QuestionId):
    return problem2id[QuestionId]

def genr_label(Score):
    return 1.0 if Score > 0.5 else 0

# Load mappings
with open(DATA / 'user2id', 'r', encoding='utf-8') as fi:
    user2id = eval(fi.read())
with open(DATA / 'problem2id', 'r', encoding='utf-8') as fi:
    problem2id = eval(fi.read())
with open(DATA / 'skill2id', 'r', encoding='utf-8') as fi:
    skill2id = eval(fi.read())

# Apply transformations
main_table = main_table.dropna(subset=['CodeStateID'])
main_table['ProblemID'] = main_table['ProblemID'].apply(genr_new_qid)
# main_table['Score'] = main_table['Score'].apply(genr_label)

# Select relevant columns
columns = ['SubjectID', 'Skills', 'ProblemID', 'Requirement', 'Score', 'ServerTimestamp', 'Code','Reason']
all_data = main_table[columns]

# Prepare for modeling
all_user = np.array(all_data['SubjectID'])
user = sorted(list(set(all_user)))


# Split data
train_all_id, temp_id = train_test_split(user, test_size=0.2, random_state=5)
train_id = np.array(train_all_id)
test_id, valid_id = train_test_split(temp_id, test_size=0.5, random_state=5)

# Initialize training data
q_a_train = []

for item in tqdm(train_id):
    # print(item)
    idx = all_data[all_data['SubjectID'] == item].index.tolist() # 找到对应学生的index
    temp1 = all_data.iloc[idx].sort_values(by='ServerTimestamp') # 找到对应学生的行
    if len(temp1) < 2:
        continue
    for i in range(len(temp1)):
        # print(len(temp1))
        if i >= SEQ_LEN:  # Assuming you want batches of data
            break
        # print(user2id[temp1.iloc[i]['SubjectID']])
        q_a_train.append([
            user2id[temp1.iloc[i]['SubjectID']],
            temp1.iloc[i]['ProblemID'],
            eval(temp1.iloc[i]['Skills']),
            eval(temp1.iloc[i]['Reason']),
            float(temp1.iloc[i]['Score']),
            temp1.iloc[i]['ServerTimestamp']
        ])

# Save to CSV
df = pd.DataFrame(q_a_train, columns=['SubjectID','ProblemID','Skills','Reason','Label','ServerTimestamp'])
save_split(df, 'train')

q_a_test = []

# Process data for each user
for item in tqdm(test_id):  # 使用测试数据集
    idx = all_data[all_data['SubjectID'] == item].index.tolist()
    temp1 = all_data.iloc[idx].sort_values(by='ServerTimestamp')
    if len(temp1) < 2:
        continue
    for i in range(len(temp1)):
        if i >= SEQ_LEN:  # Assuming you want batches of data
            break
        q_a_test.append([
            user2id[temp1.iloc[i]['SubjectID']],
            temp1.iloc[i]['ProblemID'],
            eval(temp1.iloc[i]['Skills']),
            eval(temp1.iloc[i]['Reason']),
            float(temp1.iloc[i]['Score']),
            temp1.iloc[i]['ServerTimestamp']
        ])

# Save to CSV
df_test = pd.DataFrame(q_a_test, columns=['SubjectID','ProblemID','Skills','Reason','Label','ServerTimestamp'])
save_split(df_test, 'test')



q_a_valid = []

# Process data for each user
for item in tqdm(valid_id):  # 使用验证数据集
    idx = all_data[all_data['SubjectID'] == item].index.tolist()
    temp1 = all_data.iloc[idx].sort_values(by='ServerTimestamp')
    if len(temp1) < 2:
        continue
    for i in range(len(temp1)):
        if i >= SEQ_LEN:  # Assuming you want batches of data
            break
        q_a_valid.append([
            user2id[temp1.iloc[i]['SubjectID']],
            temp1.iloc[i]['ProblemID'],
            eval(temp1.iloc[i]['Skills']),
            eval(temp1.iloc[i]['Reason']),
            float(temp1.iloc[i]['Score']),
            temp1.iloc[i]['ServerTimestamp']
        ])

# Save to CSV
df_valid = pd.DataFrame(q_a_valid, columns=['SubjectID','ProblemID','Skills','Reason','Label','ServerTimestamp'])
save_split(df_valid, 'valid')
print('complete')



