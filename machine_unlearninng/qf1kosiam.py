import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
import os
import medmnist
from mu.net_train_three import get_student_model,get_teacher_model

from medmnist import INFO, Evaluator
from tqdm import tqdm
def aggregate_predictions(models, X_test, weights):
    with torch.no_grad():
        # 使用加权平均合并模型的 logits
        weighted_logits = sum(1 * model.feature_extractor(X_test) for model, weight in zip(models, weights))
        # weighted_logits = sum(weight * model(X_test) for model, weight in zip(models, weights))
        # 计算最终的概率分布
        # probabilities = softmax(weighted_logits, dim=1)
        # print(probabilities)
        # final_prediction = probabilities.max(1)[1]
        # print(weighted_logits)
    return weighted_logits

def set_seed(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(62)
random_seed = 62
# random.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
#
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(random_seed)
def set_seed(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(62)
from torch.utils.data import Dataset, DataLoader, Subset
def analyze_sample_similarity(model,u, device, train_dataset, config):
    set_seed(32)
    sample_index = config['QF_100']['QUERY'][0]
    print(sample_index)
    # 加载模型并设置为评估模式
    model = model.to(device)
    model.load_state_dict(u)
    model.eval()

    # 加载指定索引的单个样本
    sample, label = train_dataset[sample_index]
    # sample_tensor = torch.from_numpy(sample)

    # 将样本转移到设备
    sample = sample.unsqueeze(0).to(device)
    # sample = sample_tensor.unsqueeze(0).to(device)

    # 提取样本的特征向量
    feature = model.feature_extractor(sample)
    feature = feature.view(feature.size(0), -1)

    # 从BASE数据集中找到同类别的样本
    base_indices = config['BASE2']['BASE']
    base_samples = Subset(train_dataset, base_indices)
    base_loader = DataLoader(base_samples, batch_size=len(base_samples), shuffle=False)
    base_data, base_labels = next(iter(base_loader))

    # 过滤出同类别的样本并限制数量至最多30个
    label_value = label.item()  # 提取标签值
    same_class_mask = base_labels == label_value
    same_class_indices = np.where(same_class_mask.numpy())[0]
    if len(same_class_indices) == 0:
        print("No same class samples found in BASE")
        return None
    else:
        # if len(same_class_indices) > 30:

        if len(same_class_indices) > 9000:
            print("Same class samples found in BASE", len(same_class_indices))
            same_class_indices = np.random.choice(same_class_indices,9000, replace=False)
            print("Same class samples found in BASE", same_class_indices)

        # 加载同类别的样本
        same_class_samples = DataLoader(Subset(train_dataset, same_class_indices), batch_size=len(same_class_indices), shuffle=False)
        same_class_data, _ = next(iter(same_class_samples))

        # 转移到设备
        same_class_data = same_class_data.to(device)

        # 提取特征向量
        same_class_features = model.feature_extractor(same_class_data)
        same_class_features = same_class_features.view(same_class_features.size(0), -1)
        mean_same_class_features = torch.mean(same_class_features, dim=0)
        mean_same_class_features = mean_same_class_features.unsqueeze(0)  # 将其形状调整为 (1, feature_dim)

        # 计算距离并求平均
        # 计算两个特征向量之间的欧几里得距离

        # print("qf1_feature shape:", feature.shape)
        # print("same_class_features shape:", same_class_features.shape)

        euclidean_distance = torch.norm(feature - mean_same_class_features, dim=1).item()
        # euclidean_distance = torch.norm(feature - same_class_features, dim=1)
        # average_euclidean = euclidean_distance.mean().item()
        # print("Average Euclidean distance:", average_euclidean)
        # 计算两个特征向量之间的曼哈顿距离
        manhattan_distance = torch.sum(torch.abs(feature - mean_same_class_features), dim=1).item()
        # manhattan_distance = torch.sum(torch.abs(feature - same_class_features), dim=1)
        # average_manhattan = manhattan_distance.mean().item()
        # print("Average Manhattan distance:", average_manhattan)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(feature, mean_same_class_features).item()
        # average_cosine_similarity = cosine_similarity.mean().item()
        # print("Average Cosine Similarity:", average_cosine_similarity)

        return euclidean_distance, manhattan_distance, cosine_similarity
def analyze_sample_similarity_sisa (models,weights, device, train_dataset, config):#多样本遗忘
    set_seed(62)
    sample_index = config['QF_100']['QUERY']
    print(sample_index)
    # 加载模型并设置为评估模式
    models = [model.to(device) for model in models]
    # models = models.to(device)
    # model.load_state_dict(u)
    models = [model.eval() for model in models]
    forget_samples = Subset(train_dataset, sample_index)
    forget_loader = DataLoader(forget_samples, batch_size=len(forget_samples), shuffle=False)
    forget_data,forget_labels = next(iter(forget_loader))
    forget_class_data = forget_data.to(device)

    # 提取特征向量
    forget__class_features = aggregate_predictions(models, forget_class_data, weights)

    # forget__class_features = model.feature_extractor(forget_class_data)
    forget_class_features = forget__class_features.view(forget__class_features.size(0), -1)
    mean_forget_class_features = torch.mean(forget_class_features, dim=0)
    mean_forget_class_features = mean_forget_class_features.unsqueeze(0)  # 将其形状调整为 (1, feature_dim)

    # 加载指定索引的单个样本
    # sample, label = train_dataset[sample_index]
    # sample_tensor = torch.from_numpy(sample)

    # 将样本转移到设备
    # sample = sample.unsqueeze(0).to(device)
    # sample = sample_tensor.unsqueeze(0).to(device)

    # 提取样本的特征向量
    # feature = model.feature_extractor(sample)
    # feature = feature.view(feature.size(0), -1)

    # 从BASE数据集中找到同类别的样本
    base_indices = config['BASE2']['BASE']
    base_samples = Subset(train_dataset, base_indices)
    base_loader = DataLoader(base_samples, batch_size=len(base_samples), shuffle=False)
    base_data, base_labels = next(iter(base_loader))

    # 过滤出同类别的样本并限制数量至最多30个
    # label_value = label.item()  # 提取标签值
    # same_class_mask = base_labels == label_value
    # same_class_indices = np.where(same_class_mask.numpy())[0]
    if len(base_indices) == 0:
        print("No same class samples found in BASE")
        return None
    else:
        # if len(same_class_indices) > 30:
        #
        if len(base_indices) >=399:
            # print("Same class samples found in BASE", len(base_indices))
            same_class_indices = np.random.choice(base_indices,399, replace=False)
            # print("Same class samples found in BASE", same_class_indices)

        # 加载同类别的样本
        same_class_samples = DataLoader(Subset(train_dataset, same_class_indices), batch_size=len(same_class_indices), shuffle=False)
        same_class_data, _ = next(iter(same_class_samples))

        # 转移到设备
        same_class_data = same_class_data.to(device)

        # 提取特征向量
        # same_class_features = aggregate_predictions(models,same_class_data)
        same_class_features = aggregate_predictions(models, same_class_data, weights)
        same_class_features = same_class_features.view(same_class_features.size(0), -1)
        mean_same_class_features = torch.mean(same_class_features, dim=0)
        mean_same_class_features = mean_same_class_features.unsqueeze(0)  # 将其形状调整为 (1, feature_dim)

        # 计算距离并求平均
        # 计算两个特征向量之间的欧几里得距离

        # print("qf1_feature shape:", feature.shape)
        # print("same_class_features shape:", same_class_features.shape)

        euclidean_distance = torch.norm(mean_forget_class_features - mean_same_class_features, dim=1).item()
        # euclidean_distance = torch.norm(feature - same_class_features, dim=1)
        # average_euclidean = euclidean_distance.mean().item()
        # print("Average Euclidean distance:", average_euclidean)
        # 计算两个特征向量之间的曼哈顿距离
        manhattan_distance = torch.sum(torch.abs(mean_forget_class_features - mean_same_class_features), dim=1).item()
        # manhattan_distance = torch.sum(torch.abs(feature - same_class_features), dim=1)
        # average_manhattan = manhattan_distance.mean().item()
        # print("Average Manhattan distance:", average_manhattan)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(mean_forget_class_features, mean_same_class_features).item()
        # average_cosine_similarity = cosine_similarity.mean().item()
        # print("Average Cosine Similarity:", average_cosine_similarity)

        return euclidean_distance, manhattan_distance, cosine_similarity
