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
from torch.utils.data import Dataset, DataLoader, Subset
def analyze_sample_similarity(model,u, device, train_dataset, config):
    sample_index = config['QF_1']['QUERY'][0]
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

    # 从BASE数据集中找到同类别的样本
    base_indices = config['BASE1']['BASE']
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
        if len(same_class_indices) > 30:
            same_class_indices = np.random.choice(same_class_indices, 30, replace=False)

        # 加载同类别的样本
        same_class_samples = DataLoader(Subset(train_dataset, same_class_indices), batch_size=len(same_class_indices), shuffle=False)
        same_class_data, _ = next(iter(same_class_samples))

        # 转移到设备
        same_class_data = same_class_data.to(device)

        # 提取特征向量
        same_class_features = model.feature_extractor(same_class_data)

        # 计算距离并求平均
        # 计算两个特征向量之间的欧几里得距离

        print("qf1_feature shape:", feature.shape)
        print("same_class_features shape:", same_class_features.shape)

        euclidean_distance = torch.norm(feature - same_class_features, dim=1)
        average_euclidean = euclidean_distance.mean().item()
        print("Average Euclidean distance:", average_euclidean)
        # 计算两个特征向量之间的曼哈顿距离
        manhattan_distance = torch.sum(torch.abs(feature - same_class_features), dim=1)
        average_manhattan = manhattan_distance.mean().item()
        print("Average Manhattan distance:", average_manhattan)
        # 计算两个特征向量之间的余弦相似度
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(feature, same_class_features)
        average_cosine_similarity = cosine_similarity.mean().item()
        print("Average Cosine Similarity:", average_cosine_similarity)

        return average_euclidean, average_manhattan, average_cosine_similarity