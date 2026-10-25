import logging as log
import torch
from torch.utils import data
import pandas as pd

import copy

class KTDataset(data.Dataset):
    def __init__(self, args,problem_number, batch_size, seq_len, mode):

        self.problem_number = problem_number

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.path = args.data_dir
        self.model = args.model
        self.device = args.device




        self.mode = mode
        log.info('Processing data...')
        self.process(self.path, self.mode)
        if self.mode == 'Train':
            self.data_size = len(self.data_dict['train_problem_tensor'])
        elif self.mode == 'Valid':
            self.data_size = len(self.data_dict['valid_problem_tensor'])
        elif self.mode == 'Test':
            self.data_size = len(self.data_dict['test_problem_tensor'])

        log.info('Processing data done!')

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        if self.mode == 'Train':
            # print(len(self.data_dict['train_problem_tensor'][i]))
            # print(len(self.data_dict['train_code_list'][i]))
            return self.data_dict['train_problem_tensor'][i], self.data_dict['train_skills_tensor'][i],\
                self.data_dict['train_label_tensor'][i],self.data_dict['train_note_tensor'][i],\
                self.data_dict['train_reasons_tensor'][i],self.data_dict['train_time_stamp_tensor'][i]
        elif self.mode == 'Valid':

            return self.data_dict['valid_problem_tensor'][i], self.data_dict['valid_skills_tensor'][i],\
                self.data_dict['valid_label_tensor'][i],self.data_dict['valid_note_tensor'][i],\
                self.data_dict['valid_reasons_tensor'][i],self.data_dict['valid_time_stamp_tensor'][i]
        elif self.mode == 'Test':

            return self.data_dict['test_problem_tensor'][i], self.data_dict['test_skills_tensor'][i],\
                self.data_dict['test_label_tensor'][i],self.data_dict['test_note_tensor'][i],\
                self.data_dict['test_reasons_tensor'][i],self.data_dict['test_time_stamp_tensor'][i]

    def pad_and_convert(self, lists, target_length, padding_value=0):
        """
        Pad each sublist in the given list of lists to the target length and convert to a Torch IntTensor.

        Args:
        lists (list of list of int): The list of lists to pad.
        target_length (int): The desired length of each sublist.
        padding_value (int, optional): The value to use for padding. Defaults to 0.

        Returns:
        torch.IntTensor: A tensor of shape (len(lists), target_length) where each sublist is padded.
        """
        padded_lists = []
        for sublist in lists:
            # Truncate the list if it's longer than the target length or pad it if it's shorter
            padded_list = sublist[:target_length] + [padding_value] * (target_length - len(sublist))
            padded_lists.append(padded_list)

        # Convert the list of lists into a tensor of type int
        # print(padded_lists[0][0])
        return torch.tensor(padded_lists, dtype=torch.int32)
    # def pad_and_convert(self, lists, target_length, padding_value=0):
    #     """
    #     Pad each sublist in the given list of lists to the target length and convert to a Torch IntTensor.
    #
    #     Args:
    #     lists (list of list of int): The list of lists to pad.
    #     target_length (int): The desired length of each sublist.
    #     padding_value (int, optional): The value to use for padding. Defaults to 0.
    #
    #     Returns:
    #     torch.IntTensor: A tensor of shape (len(lists), target_length) where each sublist is padded.
    #     """
    #     print("\n" + "=" * 80)
    #     print("🔍 pad_and_convert 调试信息")
    #     print("=" * 80)
    #     print(f"总列表数量: {len(lists)}")
    #     print(f"目标长度: {target_length}")
    #     print(f"填充值: {padding_value}")
    #
    #     padded_lists = []
    #
    #     for idx, sublist in enumerate(lists):
    #         print(f"\n--- 处理第 {idx} 个列表 ---")
    #         print(f"原始 sublist 类型: {type(sublist)}")
    #         print(f"原始 sublist 长度: {len(sublist) if isinstance(sublist, list) else 'N/A'}")
    #         print(f"原始 sublist 内容: {sublist}")
    #
    #         # 检查 sublist 中的每个元素
    #         if isinstance(sublist, list):
    #             for i, element in enumerate(sublist):
    #                 if not isinstance(element, (int, float)):
    #                     print(f"⚠️  警告: 第 {i} 个元素不是数字!")
    #                     print(f"   元素值: {element}")
    #                     print(f"   元素类型: {type(element)}")
    #                     print(f"   这会导致 torch.tensor 报错!")
    #
    #         # Truncate the list if it's longer than the target length or pad it if it's shorter
    #         try:
    #             padded_list = sublist[:target_length] + [padding_value] * (target_length - len(sublist))
    #             print(f"填充后长度: {len(padded_list)}")
    #             print(f"填充后内容: {padded_list}")
    #
    #             # 再次检查填充后的列表
    #             for i, element in enumerate(padded_list):
    #                 if not isinstance(element, (int, float)):
    #                     print(f"❌ 错误: 填充后第 {i} 个元素仍然不是数字!")
    #                     print(f"   元素值: {element}")
    #                     print(f"   元素类型: {type(element)}")
    #
    #             padded_lists.append(padded_list)
    #         except Exception as e:
    #             print(f"❌ 填充过程出错: {e}")
    #             raise
    #
    #     print("\n" + "=" * 80)
    #     print(f"即将转换为 tensor 的数据:")
    #     print(f"总共 {len(padded_lists)} 个列表")
    #
    #     # 检查第一个列表的详细信息
    #     if padded_lists:
    #         print(f"\n第一个列表详情:")
    #         print(f"  长度: {len(padded_lists[0])}")
    #         print(f"  内容: {padded_lists[0]}")
    #         print(f"  第一个元素: {padded_lists[0][0]}, 类型: {type(padded_lists[0][0])}")
    #
    #     # Convert the list of lists into a tensor of type int
    #     try:
    #         print("\n尝试转换为 torch.tensor...")
    #         result = torch.tensor(padded_lists, dtype=torch.int32)
    #         print(f"✅ 转换成功! Tensor shape: {result.shape}")
    #         return result
    #     except TypeError as e:
    #         print(f"\n❌❌❌ torch.tensor 转换失败! ❌❌❌")
    #         print(f"错误信息: {e}")
    #         print(f"\n详细分析:")
    #
    #         # 找出有问题的元素
    #         for i, lst in enumerate(padded_lists):
    #             for j, element in enumerate(lst):
    #                 if isinstance(element, str):
    #                     print(f"  问题位置: 第 {i} 个列表的第 {j} 个元素")
    #                     print(f"  问题元素: '{element}' (类型: {type(element).__name__})")
    #                 elif not isinstance(element, (int, float)):
    #                     print(f"  问题位置: 第 {i} 个列表的第 {j} 个元素")
    #                     print(f"  问题元素: {element} (类型: {type(element).__name__})")
    #
    #         raise

    def gen_list(self, subject_ids, requirements, max_length, column_type=None):

        if column_type == 'Label':
            subject_labels = {}
            for subject_id, requirement in zip(subject_ids, requirements):
                if subject_id not in subject_labels:
                    subject_labels[subject_id] = []

                subject_labels[subject_id].append(requirement)

            result = []
            note = []
            for subject_id in subject_labels.keys():
                requirements = copy.deepcopy(subject_labels[subject_id])  # 使用深拷贝
                temp_note = copy.deepcopy(subject_labels[subject_id])  # 使用深拷贝

                if len(requirements) < max_length:
                    padding_length = max_length - len(requirements)

                    requirements.extend([0] * padding_length)
                    temp_note.extend([-1] * padding_length)
                    # print([-1] * (max_length - len(requirements)))

                result.append(requirements)

                note.append(temp_note)

            return result, note

        elif column_type in ['Problem','ErrorNum']:
            subject_problems = {}

            for subject_id, problems in zip(subject_ids, requirements):
                if subject_id not in subject_problems:
                    subject_problems[subject_id] = []

                subject_problems[subject_id].append(problems)

            result = []
            note = []
            for subject_id in subject_problems.keys():

                requirements = subject_problems[subject_id]

                if len(requirements) < max_length:
                    requirements.extend([0] * (max_length - len(requirements)))
                result.append(requirements)
            return result

        elif column_type == "TimeInterval":
            subject_TimeInterval = {}

            for subject_id, ServerTimestamp in zip(subject_ids, requirements):
                if subject_id not in subject_TimeInterval:
                    subject_TimeInterval[subject_id] = []

                subject_TimeInterval[subject_id].append(ServerTimestamp)

            result = []
            for subject_id in subject_TimeInterval.keys():

                requirements = subject_TimeInterval[subject_id]

                if len(requirements) < max_length:
                    requirements.extend([0] * (max_length - len(requirements)))


                result.append(requirements)

            return result

        elif column_type == 'Code':
            subject_requirements = {}
            # 遍历 SubjectID 和 Requirement 的组合，保持原始顺序
            for subject_id, requirement in zip(subject_ids, requirements):
                # 如果 SubjectID 已经在字典中，添加 Requirement，否则创建新列表
                if subject_id not in subject_requirements:
                    subject_requirements[subject_id] = []

                subject_requirements[subject_id].append(requirement)

            # 维护原始顺序，使用一个列表来存储每个学生的 Requirement 列表
            result = []
            for subject_id in subject_requirements.keys():
                # 获取该 SubjectID 对应的 Requirement 列表
                requirements = subject_requirements[subject_id]

                # 如果 Requirement 数量少于 max_length，则用 "空语句" 填充

                if len(requirements) < max_length:
                    requirements.extend(['public int nullFunction() {}'] * (max_length - len(requirements)))
                    # print(requirements)

                # 将填充后的列表添加到结果中
                result.append(requirements)
            return result
        elif column_type == 'Skills':
            subject_requirements = {}
            for subject_id, requirement in zip(subject_ids, requirements):
                if subject_id not in subject_requirements:
                    subject_requirements[subject_id] = []
                subject_requirements[subject_id].append(requirement)

            # 初始化一个空的三维张量
            result = torch.empty(0, 0, max_length, dtype=torch.int32)

            # 处理每个 SubjectID 的需求并按期望的形状拼接
            for subject_id in subject_requirements.keys():
                requirements = subject_requirements[subject_id]
                requirements_tensor = self.pad_and_convert(requirements, max_length)

                if requirements_tensor.shape[0] < self.seq_len:
                    padding_size = self.seq_len - requirements_tensor.shape[0]
                    # 在第二维度（列）填充0
                    pad_tensor = torch.zeros([padding_size, max_length])

                    requirements_tensor = torch.cat([requirements_tensor, pad_tensor], dim=0)

                # 为了满足形状[num_SubjectID, Seq_len, 6]，需要增加一维
                requirements_tensor = requirements_tensor.unsqueeze(0)  # 这里将形状从[Seq_len, 6]变为[1, Seq_len, 6]
                if result.shape[1:] == (0, max_length):  # 如果result还是初始空状态，设置正确的形状
                    result = requirements_tensor
                else:
                    result = torch.cat([result, requirements_tensor], dim=0)  # 沿着第一个维度（SubjectID维度）拼接
            return result

    def process(self, path, mode='Train'):

        if mode == 'Train':
            print("Training data reading...")

            train_table = pd.read_csv('../../data/' + path + '/'+self.model+'/q_a_train_detailed.csv')
            train_problemid = list(train_table['ProblemID'])
            train_subjectid = list(train_table['SubjectID'])
            train_answer = list(train_table['Label'])
            train_time_stamp = list(train_table['TimeInterval'])
            train_reasons = list(train_table['Reason'].apply(eval))
            train_skills = list(train_table['Skills'].apply(eval))  # 一个问题最多包含6个skill

            train_label_list, train_note_list = self.gen_list(train_subjectid, train_answer, max_length=self.seq_len,
                                                              column_type="Label")  # [num_stu,seq_len]，前者用0填充，后者用1填充
            train_problem_list = self.gen_list(train_subjectid, train_problemid, max_length=self.seq_len,
                                               column_type='Problem')  # [num_stu,seq_len]
            train_time_stamp_list = self.gen_list(train_subjectid,train_time_stamp,max_length=self.seq_len,column_type='TimeInterval')


            train_label_tensor = torch.FloatTensor(train_label_list)  # torch.Size([num_stu,seq_len])
            train_note_tensor = torch.FloatTensor(train_note_list)
            train_problem_tensor = torch.IntTensor(train_problem_list)  # torch.Size([num_stu,seq_len])


            train_skills_tensor = self.gen_list(train_subjectid, train_skills, max_length=8,
                                                column_type='Skills')  # torch.Size([num_stu,seq_len,k])
            train_reasons_tensor = self.gen_list(train_subjectid,train_reasons,max_length=5,
                                                 column_type='Skills')
            train_time_stamp_tensor = torch.FloatTensor(train_time_stamp_list)
            # print('什么罐头我说',train_time_stamp_tensor.shape)
            self.data_dict = {
                'train_problem_tensor': train_problem_tensor,  # [num_stu,seq_len]
                'train_skills_tensor': train_skills_tensor,
                'train_label_tensor': train_label_tensor,  # [num_stu,seq_len]
                'train_note_tensor': train_note_tensor,
                'train_reasons_tensor': train_reasons_tensor,
                'train_time_stamp_tensor':train_time_stamp_tensor

            }
            return self.data_dict

        elif mode == 'Valid':
            print("Validation data reading...")

            valid_table = pd.read_csv('../../data/' + path + '/' + self.model + '/q_a_valid_detailed.csv')
            valid_problemid = list(valid_table['ProblemID'])
            valid_subjectid = list(valid_table['SubjectID'])
            valid_answer = list(valid_table['Label'])
            valid_time_stamp = list(valid_table['TimeInterval'])
            valid_reasons = list(valid_table['Reason'].apply(eval))
            valid_skills = list(valid_table['Skills'].apply(eval))  # 一个问题最多包含6个skill

            valid_label_list, valid_note_list = self.gen_list(valid_subjectid, valid_answer, max_length=self.seq_len,
                                                              column_type="Label")  # [num_stu,seq_len]，前者用0填充，后者用1填充
            valid_problem_list = self.gen_list(valid_subjectid, valid_problemid, max_length=self.seq_len,
                                               column_type='Problem')  # [num_stu,seq_len]
            valid_time_stamp_list = self.gen_list(valid_subjectid, valid_time_stamp, max_length=self.seq_len,
                                                  column_type='TimeInterval')

            valid_label_tensor = torch.FloatTensor(valid_label_list)  # torch.Size([num_stu,seq_len])
            valid_note_tensor = torch.FloatTensor(valid_note_list)
            valid_problem_tensor = torch.IntTensor(valid_problem_list)  # torch.Size([num_stu,seq_len])

            valid_skills_tensor = self.gen_list(valid_subjectid, valid_skills, max_length=8,
                                                column_type='Skills')  # torch.Size([num_stu,seq_len,k])
            valid_reasons_tensor = self.gen_list(valid_subjectid, valid_reasons, max_length=5,
                                                 column_type='Skills')
            valid_time_stamp_tensor = torch.FloatTensor(valid_time_stamp_list)
            # print('什么罐头我说',valid_time_stamp_tensor.shape)
            self.data_dict = {
                'valid_problem_tensor': valid_problem_tensor,  # [num_stu,seq_len]
                'valid_skills_tensor': valid_skills_tensor,
                'valid_label_tensor': valid_label_tensor,  # [num_stu,seq_len]
                'valid_note_tensor': valid_note_tensor,
                'valid_reasons_tensor': valid_reasons_tensor,
                'valid_time_stamp_tensor': valid_time_stamp_tensor
            }
            return self.data_dict
        elif mode == "Test":
            print("Test data reading...")

            test_table = pd.read_csv('../../data/' + path + '/' + self.model + '/q_a_test_detailed.csv')
            test_problemid = list(test_table['ProblemID'])
            test_subjectid = list(test_table['SubjectID'])
            test_answer = list(test_table['Label'])
            test_time_stamp = list(test_table['TimeInterval'])
            test_reasons = list(test_table['Reason'].apply(eval))
            test_skills = list(test_table['Skills'].apply(eval))  # 一个问题最多包含6个skill

            test_label_list, test_note_list = self.gen_list(test_subjectid, test_answer, max_length=self.seq_len,
                                                            column_type="Label")  # [num_stu,seq_len]，前者用0填充，后者用1填充
            test_problem_list = self.gen_list(test_subjectid, test_problemid, max_length=self.seq_len,
                                              column_type='Problem')  # [num_stu,seq_len]
            test_time_stamp_list = self.gen_list(test_subjectid, test_time_stamp, max_length=self.seq_len,
                                                 column_type='TimeInterval')

            test_label_tensor = torch.FloatTensor(test_label_list)  # torch.Size([num_stu,seq_len])
            test_note_tensor = torch.FloatTensor(test_note_list)
            test_problem_tensor = torch.IntTensor(test_problem_list)  # torch.Size([num_stu,seq_len])

            test_skills_tensor = self.gen_list(test_subjectid, test_skills, max_length=8,
                                               column_type='Skills')  # torch.Size([num_stu,seq_len,k])
            test_reasons_tensor = self.gen_list(test_subjectid, test_reasons, max_length=5,
                                                column_type='Skills')
            test_time_stamp_tensor = torch.FloatTensor(test_time_stamp_list)
            # print('什么罐头我说',test_time_stamp_tensor.shape)
            self.data_dict = {
                'test_problem_tensor': test_problem_tensor,  # [num_stu,seq_len]
                'test_skills_tensor': test_skills_tensor,
                'test_label_tensor': test_label_tensor,  # [num_stu,seq_len]
                'test_note_tensor': test_note_tensor,
                'test_reasons_tensor': test_reasons_tensor,
                'test_time_stamp_tensor': test_time_stamp_tensor
            }
            return self.data_dict









