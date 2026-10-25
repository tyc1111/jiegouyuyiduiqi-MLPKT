import torch
import torch.nn as nn
import torch.nn.functional as F


def add_to_zero_tensor(input_emb, indices, NK):
    B, K, D = input_emb.shape
    output_tensor = torch.zeros(B, NK, D, device=input_emb.device)
    indices = indices.unsqueeze(2).expand(B, K, D)
    output_tensor.scatter_add_(1, indices, input_emb)
    return output_tensor


def weighted_concept_embedding(concept_emb, scores):
    weighted_emb = (concept_emb * scores.unsqueeze(3)).sum(dim=2)
    return weighted_emb


def time_based_mlp_change(state, time_interval, learning_mlp=None, forgetting_mlp=None,
                          learning_threshold=2.566666666666667, forgetting_cap=7.516666666666667):
    time_interval = torch.clamp(time_interval, min=0.0)
    time_interval_capped = torch.clamp(time_interval, max=forgetting_cap)
    learning_mask = time_interval < learning_threshold
    forgetting_mask = time_interval >= learning_threshold

    result = torch.zeros_like(state)

    if torch.any(learning_mask) and learning_mlp is not None:
        short_intervals = time_interval[learning_mask]
        short_states = state[learning_mask]
        time_features = short_intervals.unsqueeze(1)
        mlp_input = torch.cat([short_states, time_features], dim=-1)
        delta = learning_mlp(mlp_input)
        result[learning_mask] = short_states + delta

    if torch.any(forgetting_mask) and forgetting_mlp is not None:
        long_intervals = time_interval_capped[forgetting_mask]
        long_states = state[forgetting_mask]
        time_features = long_intervals.unsqueeze(1)
        mlp_input = torch.cat([long_states, time_features], dim=-1)
        result[forgetting_mask] = long_states - long_states * forgetting_mlp(mlp_input)

    if learning_mlp is None:
        result[learning_mask] = state[learning_mask]
    if forgetting_mlp is None:
        result[forgetting_mask] = state[forgetting_mask]

    return result


class StudentKnowledgeModel(nn.Module):
    def __init__(self, num_questions, num_concepts, D, K, device):
        super(StudentKnowledgeModel, self).__init__()
        self.num_concepts = num_concepts + 1
        self.num_questions = num_questions + 1
        self.D = D
        self.K = K
        self.device = device

        # Embeddings
        self.problem_emb = nn.Embedding(self.num_questions, D)
        self.concept_emb = nn.Embedding(self.num_concepts, D, padding_idx=0)
        self.score_emb = nn.Embedding(101, D)

        # Reason embeddings
        self.logic_correct_emb = nn.Parameter(torch.rand(D) * 0.2 + 0.1)
        self.syntax_correct_emb = nn.Parameter(torch.rand(D) * 0.2 + 0.1)
        self.careless_emb = nn.Parameter(torch.zeros(D))
        self.logic_error_emb = nn.Parameter(-torch.rand(D) * 0.2 - 0.1)
        self.proficiency_emb = nn.Parameter(-torch.rand(D) * 0.2 - 0.1)

        # IRT Parameters
        self.problem_difficulty = nn.Parameter(torch.zeros(self.num_questions))
        self.problem_discrimination = nn.Parameter(torch.ones(self.num_questions))
        self.problem_ability_weights = nn.Parameter(torch.ones(self.num_questions, 3) / 3)

        # Ability MLPs
        self.logic_ability_mlp = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, 1)
        )
        self.concept_ability_mlp = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, 1)
        )
        self.problem_ability_mlp = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, 1)
        )

        # Initial states
        self.init_concept_state = nn.Parameter(torch.zeros(self.num_concepts, D))
        self.init_logic_state = nn.Parameter(torch.zeros(D))
        self.init_problem_understanding_state = nn.Parameter(torch.zeros(self.num_questions, D))

        # Time Effect MLPs
        self.logic_learning_mlp = nn.Sequential(
            nn.Linear(D + 1, D),
            nn.ReLU(),
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, D),

        )
        self.logic_forgetting_mlp = nn.Sequential(
            nn.Linear(D + 1, D),
            nn.ReLU(),
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, D),
            nn.Sigmoid()
        )

        self.concept_learning_mlp = nn.Sequential(
            nn.Linear(D + 1, D),
            nn.ReLU(),
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, D)
        )
        self.concept_forgetting_mlp = nn.Sequential(
            nn.Linear(D + 1, D),
            nn.ReLU(),
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, D),
            nn.Sigmoid()
        )

        self.problem_learning_mlp = nn.Sequential(
            nn.Linear(D + 1, D),
            nn.ReLU(),
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, D)
        )
        self.problem_forgetting_mlp = nn.Sequential(
            nn.Linear(D + 1, D),
            nn.ReLU(),
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Linear(D // 2, D),
            nn.Sigmoid()
        )

        # MLPs for ability changes
        self.logic_change_mlp = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
        self.logic_change_mlp_2 = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        self.concept_change_mlp = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
        self.concept_change_mlp_2 = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        self.problem_understanding_mlp = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
        self.problem_understanding_mlp_2 = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        # Prediction MLPs (optional, currently not used)
        self.predict_mlp_logic = nn.Sequential(
            nn.Linear(4 * D, D),
            nn.ReLU(),
            nn.Linear(D, 2)
        )
        self.predict_mlp_concept = nn.Sequential(
            nn.Linear(4 * D, D),
            nn.ReLU(),
            nn.Linear(D, 3)
        )

    def compute_irt_score(self, student_ability, problem_difficulty, problem_discrimination):
        logit = problem_discrimination * (student_ability - problem_difficulty)
        probability = torch.sigmoid(logit)
        return probability

    def compute_student_ability(self, logic_state, concept_state, problem_understanding_state, qc_emb, question_id):
        logic_ability = self.logic_ability_mlp(logic_state).squeeze(-1)
        concept_ability = self.concept_ability_mlp(concept_state).squeeze(-1)
        problem_input = torch.cat([problem_understanding_state, qc_emb], dim=1)
        problem_ability = self.problem_ability_mlp(problem_input).squeeze(-1)

        ability_weights = self.problem_ability_weights[question_id]
        ability_weights = F.softmax(ability_weights, dim=-1)

        student_ability = (ability_weights[:, 0] * logic_ability +
                           ability_weights[:, 1] * concept_ability +
                           ability_weights[:, 2] * problem_ability)

        return student_ability

    def apply_time_effects(self, cur_logic_state, cur_concept_state, cur_problem_understanding_state, time_interval):
        """
        应用时间效应：短时间间隔学习增强，长时间间隔遗忘衰减
        """
        updated_logic_state = time_based_mlp_change(
            cur_logic_state,
            time_interval,
            self.logic_learning_mlp,
            self.logic_forgetting_mlp
        )

        B, NC, D = cur_concept_state.shape
        concept_state_flat = cur_concept_state.view(-1, D)
        time_interval_expanded = time_interval.unsqueeze(1).expand(-1, NC).contiguous().view(-1)
        updated_concept_state_flat = time_based_mlp_change(
            concept_state_flat,
            time_interval_expanded,
            self.concept_learning_mlp,
            self.concept_forgetting_mlp
        )
        updated_concept_state = updated_concept_state_flat.view(B, NC, D)

        B, NQ, D = cur_problem_understanding_state.shape
        problem_state_flat = cur_problem_understanding_state.view(-1, D)
        time_interval_expanded = time_interval.unsqueeze(1).expand(-1, NQ).contiguous().view(-1)
        updated_problem_state_flat = time_based_mlp_change(
            problem_state_flat,
            time_interval_expanded,
            self.problem_learning_mlp,
            self.problem_forgetting_mlp
        )
        updated_problem_understanding_state = updated_problem_state_flat.view(B, NQ, D)

        return updated_logic_state, updated_concept_state, updated_problem_understanding_state

    def forward(self, q_id, c_id, score, err_reason, time_intervals):
        """
        修正后的forward函数，正确的三阶段顺序：
        1. Learning Phase: 从交互t-1中学习
        2. Forgetting/Enhancement Phase: 应用时间间隔Δt(t-1 → t)的影响
        3. Application Phase: 用时间调整后的状态预测时刻t的表现
        """

        q_id = q_id.long().to(self.device)
        c_id = c_id.long().to(self.device)
        score = (score * 100).long().to(self.device)
        err_reason = err_reason.long().to(self.device)
        time_intervals = time_intervals.to(self.device)
        B, S = q_id.shape
        NK = self.num_concepts
        D = self.D
        K = self.K

        # 获取embeddings
        q_id_emb = F.softplus(self.problem_emb(q_id))
        c_id_emb = F.softplus(self.concept_emb(c_id))
        score_emb = F.softplus(self.score_emb(score))

        # 构建reason embeddings
        logic_emb = (err_reason[:, :, 0].unsqueeze(-1) * self.logic_correct_emb +
                     err_reason[:, :, 4].unsqueeze(-1) * self.logic_error_emb)
        concept_emb_reason = (err_reason[:, :, 1].unsqueeze(-1) * self.syntax_correct_emb +
                              err_reason[:, :, 2].unsqueeze(-1) * self.careless_emb +
                              err_reason[:, :, 3].unsqueeze(-1) * self.proficiency_emb)
        reason_emb = torch.stack([logic_emb, concept_emb_reason], dim=2)

        # 计算question-concept embeddings
        raw_scores = (c_id_emb * q_id_emb.unsqueeze(2)).sum(dim=3)
        mask = (c_id != 0).float()
        raw_scores = raw_scores * mask + (1 - mask) * (-1e9)
        related_scores = F.softmax(raw_scores, dim=2)
        qc_emb = weighted_concept_embedding(c_id_emb, related_scores) + q_id_emb

        # 初始化状态
        cur_concept_state = self.init_concept_state.unsqueeze(0).expand(B, -1, -1).clone()
        cur_logic_state = self.init_logic_state.unsqueeze(0).expand(B, -1).clone()
        cur_problem_understanding_state = self.init_problem_understanding_state.unsqueeze(0).expand(B, -1, -1).clone()
        answer_count = torch.zeros(B, self.num_questions, device=self.device)

        # 预测结果
        predict_score = torch.zeros(B, S, device=self.device)
        predict_reason_logic = torch.zeros(B, S, 2, device=self.device)
        predict_reason_concept = torch.zeros(B, S, 3, device=self.device)

        # ============================================
        # 主循环：正确的三阶段顺序
        # ============================================
        for i in range(S - 1):
            # ============================================
            # Phase 1: Learning Phase - 从时刻i的交互中学习
            # ============================================
            score_emb_step = score_emb[:, i, :]
            reason_emb_step = reason_emb[:, i, :, :]
            reason_emb_step_logic = reason_emb_step[:, 0, :]
            reason_emb_step_concept = reason_emb_step[:, 1, :]

            q_id_step = q_id[:, i]
            indices = q_id_step.unsqueeze(1).long()
            count_update = torch.zeros_like(answer_count)
            count_update.scatter_add_(1, indices, torch.ones(B, 1, device=self.device))
            answer_count = answer_count + count_update

            # 1.1 Logical Ability Update
            logic_input = torch.cat([qc_emb[:, i, :], score_emb_step, reason_emb_step_logic], dim=1)
            logic_change = torch.sigmoid(self.logic_change_mlp(logic_input))
            logic_gain = torch.sigmoid(self.logic_change_mlp_2(logic_input))
            logic_update = logic_change * logic_gain * reason_emb_step_logic
            cur_logic_state = cur_logic_state + logic_update

            # 1.2 Conceptual Ability Update
            c_id_emb_step = c_id_emb[:, i, :, :]
            score_emb_step_expanded = score_emb_step.unsqueeze(1).expand(-1, K, -1)
            reason_emb_step_concept_expanded = reason_emb_step_concept.unsqueeze(1).expand(-1, K, -1)
            concept_input = torch.cat([c_id_emb_step, score_emb_step_expanded, reason_emb_step_concept_expanded], dim=2)
            concept_change = torch.sigmoid(self.concept_change_mlp(concept_input))
            concept_gain = torch.sigmoid(self.concept_change_mlp_2(concept_input))
            concept_gainORloss = concept_change * concept_gain * reason_emb_step_concept_expanded
            mask = (c_id[:, i, :] != 0).unsqueeze(2).float()
            concept_gainORloss = concept_gainORloss * mask
            concept_gainORloss = add_to_zero_tensor(concept_gainORloss, c_id[:, i, :], NK)
            cur_concept_state = cur_concept_state + concept_gainORloss

            # 1.3 Problem Understanding Update
            problem_input = torch.cat([q_id_emb[:, i, :], score_emb_step], dim=1)
            problem_change = torch.sigmoid(self.problem_understanding_mlp(problem_input))
            indices_expand = q_id_step.unsqueeze(1).unsqueeze(2).expand(-1, -1, D).long()
            related_cur_problem_state = torch.gather(cur_problem_understanding_state, dim=1,
                                                     index=indices_expand).squeeze(1)
            problem_gain = torch.sigmoid(
                self.problem_understanding_mlp_2(torch.cat([problem_input, related_cur_problem_state], dim=1)))
            count_weight = torch.log1p(answer_count.gather(1, indices))
            problem_gainORloss = problem_change * problem_gain * count_weight
            indices = q_id_step.unsqueeze(1).unsqueeze(2).expand(-1, 1, D)
            update_tensor = torch.zeros_like(cur_problem_understanding_state)
            update_tensor.scatter_add_(1, indices, problem_gainORloss.unsqueeze(1))
            cur_problem_understanding_state = cur_problem_understanding_state + update_tensor

            # ============================================
            # Phase 2: Forgetting/Enhancement Phase
            # 应用从时刻i到时刻i+1的时间间隔影响
            # ============================================
            time_interval = time_intervals[:, i]  # 时间间隔: t_i → t_{i+1}
            cur_logic_state, cur_concept_state, cur_problem_understanding_state = self.apply_time_effects(
                cur_logic_state, cur_concept_state, cur_problem_understanding_state, time_interval
            )
            # 此时的状态已经考虑了时间的影响，表示时刻i+1时学生的真实知识状态

            # ============================================
            # Phase 3: Application Phase
            # 使用时间调整后的状态预测时刻i+1的表现
            # ============================================
            next_qc_emb = qc_emb[:, i + 1, :]
            c_id_next = c_id[:, i + 1, :]

            # 获取与下一个问题相关的概念状态
            indices = c_id_next.unsqueeze(2).expand(-1, -1, D).long()
            rel = torch.gather(cur_concept_state, dim=1, index=indices)
            valid_mask = (c_id_next != 0).unsqueeze(2).float()
            rel = rel * valid_mask
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            related_cur_concept_state = rel.sum(dim=1) / valid_count.squeeze(1)

            # 获取下一个问题的理解状态
            next_q_id = q_id[:, i + 1]
            next_indices = next_q_id.unsqueeze(1).unsqueeze(2).expand(-1, -1, D).long()
            related_cur_problem_state = torch.gather(cur_problem_understanding_state, dim=1,
                                                     index=next_indices).squeeze(1)

            # 计算学生综合能力并预测
            student_ability = self.compute_student_ability(cur_logic_state, related_cur_concept_state,
                                                           related_cur_problem_state, next_qc_emb, next_q_id)
            problem_difficulty = self.problem_difficulty[next_q_id]
            problem_discrimination = torch.abs(self.problem_discrimination[next_q_id]) + 1e-6

            # 使用IRT预测时刻i+1的得分
            predict_score[:, i + 1] = self.compute_irt_score(student_ability, problem_difficulty,
                                                             problem_discrimination)

            # 可选：预测错误原因（当前未使用）
            # predict_input = torch.cat(
            #     [next_qc_emb, cur_logic_state, related_cur_concept_state, related_cur_problem_state], dim=1)
            # predict_reason_logic[:, i + 1, :] = torch.softmax(self.predict_mlp_logic(predict_input), dim=-1)
            # predict_reason_concept[:, i + 1, :] = torch.softmax(self.predict_mlp_concept(predict_input), dim=-1)

        return predict_score, predict_reason_logic, predict_reason_concept, None, None