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

class ExpandToRGB:
    """将单通道Tensor图像扩展为三通道"""
    def __call__(self, tensor):
        # 检查是否为单通道图像（形状为 [1, H, W]）
        if tensor.shape[0] == 1:
            # 重复通道以形成3通道图像
            tensor = tensor.repeat(3, 1, 1)
        return tensor

npz_file_path = '../data/retinamnist.npz'
data = np.load(npz_file_path)

print(list(data.keys()))  # 打印所有在npz文件中的数组名

# ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
images = data['train_images']
images_test =data['val_images']
labels_test =data['val_labels']
labels_cal =data['test_labels']
images_cal = data['test_images']
print(images.shape)
print(images_test.shape)
print(images_cal.shape)
labels = data['train_labels']


# class PathMNISTDataset(Dataset):
#     def __init__(self, images, labels, transform=None):
#         self.images = images
#         self.labels = labels
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, label


class PathMNISTDataset(Dataset):
    def __init__(self, images, labels,transform=None):
        self.original_images = images
        self.original_labels = labels
        self.images = images.copy()
        self.labels = labels.copy()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 返回当前索引下的图像和标签
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def shuffle_data(self, seed=52):
        # 设置随机种子以确保打乱顺序的一致性
        print(f"Shuffling data with seed {seed}") # 这里的seed是一个局部变量 有没有都可以 做的实验是这样的
        np.random.seed(seed)
        indices = np.arange(len(self.original_images))
        np.random.shuffle(indices)
        self.images = self.original_images[indices]
        self.labels = self.original_labels[indices]
        # 如果提供了转换函数，对每个图像进行转换
        # if self.transform:
        #     # 逐个图像应用转换
        #     self.images = np.array([self.transform(image) for image in self.images])


# BASE: 用于训练模型的训练集
# CAL: 用于评估memberhsip的校准数据集
# CALTEST: 校准数据集中的测试部分
# TEST: 模型训练过程中的测试集
# QO: defined query dataset overlapped with base training dataset
# QM: defined query dataset, k=来自训练集的百分比
# QN: query dataset没有和train overlap
# KD: 用于知识蒸馏的训练集，k=原始训练集的百分比, KD不能和query有overlap
# QF: query dataset for forget

CONFIG = {
    'BASE1': {
        'BASE': list(range(0, 1080))
    },
    'BASE2': {
        'BASE': list(range(0, 9900))
    },
    'TEST1': {
        'TEST': list(range(0, 120))
    },
    'CAL_100': {
        'CAL': list(range(100, 200))
    },
    'CAL_1000': {
        'CAL': list(range(40000, 41000))
    },
    'CAL_2000': {
        'CAL': list(range(40000, 42000))
    },
    'CAL_5000': {
        'CAL': list(range(40000, 45000))
    },
    'CALTEST1': {
        'TEST': list(range(0, 100))
    },
    'QO_5000': {
        'QUERY': list(range(0, 5000)),
        'QUERY_MEMBER': [1 for i in range(5000)]
    },
    'QNO_5000': {
        'QUERY': list(range(50000, 55000)),
        'QUERY_MEMBER': [0 for i in range(5000)]
    },
    'QNO_2000': {
        'QUERY': list(range(50000, 52000)),
        'QUERY_MEMBER': [0 for i in range(2000)]
    },
    'QNO_1000': {
        'QUERY': list(range(50000, 51000)),
        'QUERY_MEMBER': [0 for i in range(1000)]
    },
    'QNO_100': {
        'QUERY': list(range(50000, 50100)),
        'QUERY_MEMBER': [0 for i in range(100)]
    },
    'QNO_10': {
        'QUERY': list(range(50000, 50010)),
        'QUERY_MEMBER': [0 for i in range(10)]
    },
    'QNO_1': {
        'QUERY': list(range(50000, 50010)),
        'QUERY_MEMBER': [0 for i in range(10)]
    },
    'QF_100': {
        'QUERY': list(range(980, 1080)),
        'QUERY_MEMBER': [1 for i in range(100)]
    },
    'QF_10': {
        'QUERY': list(range(9990, 10000)),
        'QUERY_MEMBER': [1 for i in range(10)]
    },
    'QF_1': {
        'QUERY': list(range(1079, 1080)),
        'QUERY_MEMBER': [1 for i in range(1)]
    },
    'QF_1000': {
        'QUERY': list(range(80, 1080)),
        'QUERY_MEMBER': [1 for i in range(1000)]
    },
    'KD0.25': {
        'BASE': list(range(0, 270))
    },
    'KD0.01': {
        'BASE': list(range(0, 100))
    },

    'KD0.5': {
        'BASE': list(range(0, 540))
    },
    'KD0.75': {
        'BASE': list(range(0, 810))
    },
}

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

transform_mnist = transforms.Compose([

    transforms.ToTensor(),
    ExpandToRGB(),  # 确保这个转换在ToTensor之后

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform = transforms.Compose([

    transforms.ToTensor(),
    ExpandToRGB(),  # 确保这个转换在ToTensor之后

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# preprocessing
data_transform = transforms.Compose([

    transforms.ToTensor(),
    ExpandToRGB(),  # 确保这个转换在ToTensor之后

    transforms.Normalize(mean=[.5], std=[.5])
])


# csv_file = '/root/autodl-tmp/wangbin/yiwang/data/final.csv'
train_dataset = PathMNISTDataset(images, labels, transform=transform)
train_dataset.shuffle_data()  # 使用固定种子打乱数据

# train_dataset_no = PathMNISTDataset(images, labels, transform=transform_no_salt_pepper)
# train_dataset_no.shuffle_data()
random_seed = 32
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
# qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,
#                             generator=torch.Generator().manual_seed(random_seed))
# cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_50']['CAL']), batch_size=64, shuffle=False,
#                              generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")

modelmlp = get_student_model()
modelmlp = modelmlp.to(device)

# 参数设置
modelmlp.load_state_dict(torch.load('/root/autodl-tmp/wangbin/yiwang/best_model_retinanet_base2.pth'))  # 需要使用改后的"a_mlp_train_three.py"训练得到
# modelmlp.load_state_dict(torch.load('/mnt/f/share/AFS-main/template/PathMNIST/models/EXP1/KD0.5_pure_student/best_model.pth'))  # 需要使用改后的"a_mlp_train_three.py"训练得到
# modelmlp.load_state_dict(torch.load('/mnt/f/share/AFS-main/template/PathMNIST/models/EXPchongxun/BASE2_pure_student/best_model.pth'))  # 需要使用改后的"a_mlp_train_three.py"训练得到


model = modelmlp.to(device)
model.eval()

# 加载 qf1 中的单个样本
qf1_index = CONFIG['QF_1']['QUERY'][0]  # qf1中只有一个样本
qf1_sample, qf1_label = train_dataset[qf1_index]

# 将样本转移到设备
qf1_sample = qf1_sample.unsqueeze(0).to(device)

# 提取 qf1 的特征向量
qf1_feature = model.feature_extractor(qf1_sample)

# 从 BASE 数据集中找到同类别的样本
base_indices = CONFIG['BASE1']['BASE']
# print(base_indices)
base_samples = Subset(train_dataset, base_indices)
base_loader = DataLoader(base_samples, batch_size=len(base_samples), shuffle=False)
base_data, base_labels = next(iter(base_loader))
print(base_labels.size())


# 过滤出同类别的样本并限制数量至最多30个
# 假设 qf1_label 是一个Tensor，并且我们知道它只有一个元素
qf1_label_value = qf1_label.item()  # 提取标签值
same_class_mask = base_labels == qf1_label_value  # 比较
print(same_class_mask)
same_class_indices = np.where(same_class_mask)[0]
if len(same_class_indices) == 0:
    print("No same class samples found in BASE")
else:
    if len(same_class_indices) > 30:
        same_class_indices = np.random.choice(same_class_indices, 30, replace=False)  # 选择最多30个样本

    # 加载同类别的样本
    same_class_samples = DataLoader(Subset(train_dataset, same_class_indices), batch_size=len(same_class_indices), shuffle=False)
    same_class_data, _ = next(iter(same_class_samples))

    # 转移到设备
    same_class_data = same_class_data.to(device)

    # 提取特征向量
    same_class_features = model.feature_extractor(same_class_data)

    # 计算距离并求平均
    euclidean_distance = torch.norm(qf1_feature - same_class_features, dim=1)
    average_euclidean = euclidean_distance.mean().item()
    print("Average Euclidean distance:", average_euclidean)
    # 计算两个特征向量之间的曼哈顿距离
    manhattan_distance = torch.sum(torch.abs(qf1_feature - same_class_features), dim=1)
    average_manhattan = manhattan_distance.mean().item()
    print("Average Manhattan distance:", average_manhattan)
    # 计算两个特征向量之间的余弦相似度
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarity = cos(qf1_feature, same_class_features)
    average_cosine_similarity = cosine_similarity.mean().item()
    print("Average Cosine Similarity:", average_cosine_similarity)
