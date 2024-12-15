import torch
import torch.nn.functional as F

def calculate_kl_divergence(smodel,u, smodel_base2, data_loader, device):
    # 实例化模型（假设 get_student_model 已定义）
    model_1 = smodel.to(device)

    model_2 =smodel_base2.to(device)
    model_2.load_state_dict(u)

    # 载入模型权重
    # model_1.load_state_dict(torch.load(u, map_location=device))


    # 确保模型处于评估模式
    model_1.eval()
    model_2.eval()

    # 初始化存储KL散度的列表
    kl_divergences = []

    # 遍历数据加载器中的每个样本
    for images, _ in data_loader:  # 假设我们只关心图像数据，不关心标签
        images = images.to(device)  # 确保数据移至正确的设备

        # 进行预测
        output_1 = model_1(images)
        output_2 = model_2(images)

        # 计算概率分布
        # log_prob_1 = F.log_softmax(output_1, dim=1)
        # prob_2 = F.softmax(output_2, dim=1)
        kl_div = F.kl_div(output_1, torch.exp(output_2), reduction='batchmean')
        # 计算KL散度
        # kl_div = F.kl_div(log_prob_1, prob_2, reduction='batchmean')
        kl_divergences.append(kl_div.item())

    # 计算平均KL散度
    average_kl_div = sum(kl_divergences) / len(kl_divergences)
    # print("Average KL Divergence:", average_kl_div)
    # return average_kl_div
    return max(0.0, average_kl_div)

import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

# def calculate_kl_divergence(smodel, u, smodel_base2, data_loader, device):
#     # 实例化模型
#     model_1 = smodel.to(device)
#     model_2 = smodel_base2.to(device)
#     model_2.load_state_dict(u)
#
#     # 确保模型处于评估模式
#     model_1.eval()
#     model_2.eval()
#
#     # 初始化存储KL散度的列表
#     kl_divergences = []
#
#     # 遍历数据加载器中的每个样本
#     for images, _ in data_loader:
#         images = images.to(device)
#
#         # 进行预测，模型输出已经是log probabilities
#         log_prob_1 = model_1(images)  # log_P (真实分布)
#         log_prob_2 = model_2(images)  # log_Q (近似分布)
#
#         # 计算KL散度
#         kl_div = F.kl_div(log_prob_2, log_prob_1, reduction='batchmean', log_target=True)
#         kl_divergences.append(kl_div.item())
#
#     # 计算平均KL散度
#     average_kl_div = sum(kl_divergences) / len(kl_divergences)
#     return average_kl_div