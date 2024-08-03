
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 如果需要加载图像数据或者进行图像预处理
from torchvision import transforms
from PIL import Image

# 随机数生成和操作
import numpy as np

# 对于文件操作，如加载模型权重
import os
from mu.net_train_three import get_student_model


import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from unlearning_random_afs_rand import train_student_model_random
from Domainadaptation_net_three import  domainadaptation
from test_model_path import test_model
from puri import api
import random
from mu.net_train_three import get_student_model,get_teacher_model
from torch.utils.data import Dataset, DataLoader, Subset
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from plot_with_new_data import update_plot_with_new_data
from torch.utils.data import DataLoader, Subset,Dataset
from torchvision import datasets, transforms
# from tsne_mnist_samedata import tsne
from tsne_mnist import tsne
# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs

# 加载npz文件
import torch.nn.init as init

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

npz_file_path = './data/pathmnist.npz'
data = np.load(npz_file_path)

print(list(data.keys()))  # 打印所有在npz文件中的数组名

# ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
images = data['train_images']
labels = data['train_labels']

# 创建自定义数据集
class PathMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label




# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs


CONFIG = {
'BASE1': {
    'BASE' : list(range(0,10000))
},
'BASE2': {
    'BASE' : list(range(0,9900))
},
'TEST1': {
    'TEST' : list(range(30000,35000))
},
'CAL_100': {
    'CAL' : list(range(40000,40100))
},
'CAL_1000': {
    'CAL' : list(range(40000,41000))
},
'CAL_2000': {
    'CAL' : list(range(40000,42000))
},
'CAL_5000': {
    'CAL' : list(range(40000,45000))
},
'CALTEST1': {
    'TEST' : list(range(46000,47000))
},
'QO_5000': {
    'QUERY': list(range(0,5000)),
    'QUERY_MEMBER': [1 for i in range(5000)]
    },
'QNO_5000': {
    'QUERY': list(range(50000,55000)),
    'QUERY_MEMBER': [0 for i in range(5000)]
    },
'QNO_2000': {
    'QUERY': list(range(50000,52000)),
    'QUERY_MEMBER': [0 for i in range(2000)]
    },
'QNO_1000': {
    'QUERY': list(range(50000,51000)),
    'QUERY_MEMBER': [0 for i in range(1000)]
    },
'QNO_100': {
    'QUERY': list(range(50000,50100)),
    'QUERY_MEMBER': [0 for i in range(100)]
    },
'QNO_10': {
    'QUERY': list(range(50000,50010)),
    'QUERY_MEMBER': [0 for i in range(10)]
    },
'QNO_1': {
    'QUERY': list(range(50000,50010)),
    'QUERY_MEMBER': [0 for i in range(10)]
    },
'QF_100':{
    'QUERY': list(range(9900,10000)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QF_10':{
    'QUERY': list(range(9990,10000)),
    'QUERY_MEMBER': [1 for i in range(10)]
},
'QF_1':{
    'QUERY': list(range(9999,10000)),
    'QUERY_MEMBER': [1 for i in range(1)]
},
'QF_1000':{
    'QUERY': list(range(9000,10000)),
    'QUERY_MEMBER': [1 for i in range(1000)]
},
'KD0.25': {
    'BASE': list(range(0,2500))
},
'KD0.5': {
    'BASE': list(range(0,5000))
},
'KD0.75': {
    'BASE': list(range(0,7500))
},
}


# random_seed = 10
# random_seed = 23
random_seed = 32
# random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
def add_salt_and_pepper_noise(img):
    """
    向图像添加盐椒噪声
    img: Tensor图像
    """
    # 设定噪声比例
    amount = 0.1  # 噪声占图像比例
    # 0.005
    salt_vs_pepper = 0.5  # 盐和椒的比例
    num_salt = np.ceil(amount * img.numel() * salt_vs_pepper)
    num_pepper = np.ceil(amount * img.numel() * (1.0 - salt_vs_pepper))

    # 添加盐噪声
    indices = torch.randperm(img.numel())[:int(num_salt)]
    img.view(-1)[indices] = 1

    # 添加椒噪声
    indices = torch.randperm(img.numel())[:int(num_pepper)]
    img.view(-1)[indices] = 0

    return img


# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")
# 设置数据转换，这里仅进行基础的转换和标准化
transform_no_salt_pepper = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(add_salt_and_pepper_noise),
    # transforms.Lambda(add_speckle_noise),
    # transforms.Lambda(add_gaussian_noise)

])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = PathMNISTDataset(images, labels, transform=transform)
train_dataset_no = PathMNISTDataset(images, labels, transform=transform_no_salt_pepper)


base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_1000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_2000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_2000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_5000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_5000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qo_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QO_5000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qno_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_5000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qno_2000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_2000']['QUERY']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qno_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1000']['QUERY']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qno_100_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_100']['QUERY']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))

qno_1_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1']['QUERY']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qno_10_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_10']['QUERY']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))

qf_100_loader_noise = DataLoader(Subset(train_dataset_no, CONFIG['QF_100']['QUERY']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qf_1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qf_10_loader = DataLoader(Subset(train_dataset, CONFIG['QF_10']['QUERY']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))

qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
# 参数设置


# 实例化两个模型

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 实例化模型
model_1 = get_student_model().to(device)  # 将模型移动到指定的设备
model_2 = get_student_model().to(device)  # 将模型移动到指定的设备

# 载入权重

# model_1.load_state_dict(torch.load('./saved_models/model_epoch_21.pth', map_location=device))
model_1.load_state_dict(torch.load('/root/autodl-tmp/wangbin/yiwang/saved_modelrandomoct52qf1/model_epoch_46.pth', map_location=device))
model_2.load_state_dict(torch.load('best_model_octnetrandom52qf1.pth', map_location=device))

model_1.eval()
model_2.eval()
#
# model_1.eval()
# model_2.eval()
# # 假设样本数据已经是Tensor格式，适合模型输入
# # 示例中直接使用一个随机Tensor代表样本数据
# sample = torch.randn(1, 3, 28, 28)  # 假设输入数据是32x32的RGB图像
# # 对同一个样本进行预测
# output_1 = model_1(sample)
# output_2 = model_2(sample)

# 将输出通过softmax转换为概率分布
# prob_1 = F.softmax(output_1, dim=1)
# prob_2 = F.softmax(output_2, dim=1)

# print(model_2{})
# 对模型输出应用log_softmax和softmax以获得对数概率和概率
# log_prob_1 = F.log_softmax(output_1, dim=1)
# prob_2 = F.softmax(output_2, dim=1)
#
# # 使用对数概率和概率计算KL散度
# kl_div = F.kl_div(log_prob_1, prob_2, reduction='batchmean')
#
# print(f'KL散度: {kl_div.item()}')


# # 计算两个概率分布之间的KL散度
# kl_div = F.kl_div(output_1, prob_2, reduction='batchmean')
#
# print(f'KL散度: {kl_div.item()}')
# 这两个输出事相同的
'''
for epoch
'''

# 确保模型处于评估模式
model_1.eval()
model_2.eval()

# 初始化存储KL散度的列表
kl_divergences = []

# 遍历数据加载器中的每个样本
for images, _ in qf_1_loader:  # 假设我们只关心图像数据，不关心标签
    images = images.to(device)  # 确保数据移至正确的设备

    # 进行预测
    output_1 = model_1(images)
    output_2 = model_2(images)

    # 计算概率分布
    log_prob_1 = F.log_softmax(output_1, dim=1)
    prob_2 = F.softmax(output_2, dim=1)

    # 计算KL散度
    kl_div = F.kl_div(log_prob_1, prob_2, reduction='batchmean')
    kl_divergences.append(kl_div.item())

# 计算平均KL散度
average_kl_div = sum(kl_divergences) / len(kl_divergences)
print(f'平均KL散度: {average_kl_div}')



