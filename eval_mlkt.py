import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

def performance(ground_truth, prediction):
    # 将输入展平为一维数组
    ground_truth = ground_truth.flatten().detach().cpu().numpy()  # 移动到 CPU 并转换为 NumPy
    prediction = prediction.flatten().detach().cpu().numpy()  # 移动到 CPU 并转换为 NumPy

    # 过滤掉 ground_truth 中值为 -1 的无效数据
    valid_indices = ground_truth != -1
    if not valid_indices.any():
        return 0.0, 0.0  # 返回默认值，避免空数据错误

    ground_truth = ground_truth[valid_indices]
    prediction = prediction[valid_indices]

    # 计算 MSE
    MSE = mean_squared_error(ground_truth, prediction)

    # 计算 RMSE
    RMSE = np.sqrt(MSE)

    # 计算 MAE
    MAE = np.mean(np.abs(ground_truth - prediction))

    return RMSE, MAE


def classification_performance(ground_truth, prediction, label_threshold=0.5,
                               prediction_threshold=0.5):
    """Binary metrics on valid scores: label > threshold; prediction >= threshold.

    Continuous predictions are used for AUC. Padding (-1) is excluded.
    Empty data or single-class AUC is undefined and returned as NaN.
    This does not change the continuous training targets.
    """
    if not 0 <= label_threshold <= 1 or not 0 <= prediction_threshold <= 1:
        raise ValueError("Metric thresholds must be in [0, 1]")
    if ground_truth.shape != prediction.shape:
        raise ValueError("Labels and predictions must have the same shape")
    labels = ground_truth.detach().flatten().cpu().numpy()
    scores = prediction.detach().flatten().cpu().numpy()
    mask = labels != -1
    labels, scores = labels[mask], scores[mask]
    if labels.size == 0:
        return float("nan"), float("nan")
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise ValueError("Non-finite label or prediction at a valid position")
    if ((labels < 0) | (labels > 1)).any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Valid labels and predicted scores must be in [0, 1]")
    binary_labels = (labels > label_threshold).astype(np.int32)
    acc = accuracy_score(binary_labels, scores >= prediction_threshold)
    auc = (roc_auc_score(binary_labels, scores)
           if np.unique(binary_labels).size == 2 else float("nan"))
    return float(acc), float(auc)

class lossFunc(nn.Module):
    def __init__(self, device):
        super(lossFunc, self).__init__()
        self.device = device
        self.bce_loss = nn.BCELoss(reduction='none')  # 分数预测的二元交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # 原因预测的交叉熵损失

    def forward(self, pred, label, note, reason_tensor=None, pred_reason_logic=None, pred_reason_concept=None,logic_reason_weight=None,concept_reason_weight=None):
        # 移动张量到指定设备
        note = note.to(self.device)
        label = label.to(self.device)

        # 创建掩码，标记有效标签（note != -1 的位置）
        mask = (note != -1).float()  # [B, S-1]
        valid_count = mask.sum() + 1e-8  # 有效位置数量，防止除零

        # 计算分数预测的损失
        score_loss = self.bce_loss(pred, label.float())  # [B, S-1]
        masked_score_loss = score_loss * mask
        final_score_loss = masked_score_loss.sum() / valid_count

        if reason_tensor is None:
            # 仅返回分数损失
            return final_score_loss

        # 移动原因相关张量
        reason_tensor = reason_tensor.to(self.device)
        pred_reason_logic = pred_reason_logic.to(self.device)
        pred_reason_concept = pred_reason_concept.to(self.device)

        # 拆分 reason_tensor
        # err_reason 顺序: [逻辑正确，语法正确，粗心导致的语法错误，不熟练导致的语法错误，逻辑错误]
        logic_reason_tensor = torch.stack([reason_tensor[:, :, 0], reason_tensor[:, :, 4]], dim=-1)  # [B, S-1, 2]
        concept_reason_tensor = reason_tensor[:, :, 1:4]  # [B, S-1, 3]

        # 计算逻辑原因损失
        B, S_minus_1, _ = logic_reason_tensor.shape
        pred_reason_logic_flat = pred_reason_logic.contiguous().view(-1, 2)  # [B*(S-1), 2]
        logic_reason_labels = torch.argmax(logic_reason_tensor.view(-1, 2), dim=-1)  # [B*(S-1)]
        logic_reason_loss = self.ce_loss(pred_reason_logic_flat, logic_reason_labels)  # [B*(S-1)]
        logic_reason_loss = logic_reason_loss.view(B, S_minus_1)  # [B, S-1]
        masked_logic_reason_loss = logic_reason_loss * mask
        final_logic_reason_loss = masked_logic_reason_loss.sum() / valid_count

        # 计算概念原因损失
        pred_reason_concept_flat = pred_reason_concept.contiguous().view(-1, 3)  # [B*(S-1), 3]
        concept_reason_labels = torch.argmax(concept_reason_tensor.view(-1, 3), dim=-1)  # [B*(S-1)]
        concept_reason_loss = self.ce_loss(pred_reason_concept_flat, concept_reason_labels)  # [B*(S-1)]
        concept_reason_loss = concept_reason_loss.view(B, S_minus_1)  # [B, S-1]
        masked_concept_reason_loss = concept_reason_loss * mask
        final_concept_reason_loss = masked_concept_reason_loss.sum() / valid_count
        # print('final_score_loss:',final_score_loss.item())
        # print('final_logic_reason_loss:',final_logic_reason_loss)
        # print('final_concept_reason_loss:',final_concept_reason_loss)
        # 总损失
        total_loss = final_score_loss + 0.01*final_logic_reason_loss + 0.01*final_concept_reason_loss

        return total_loss