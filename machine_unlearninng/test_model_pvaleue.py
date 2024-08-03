import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import sys

sys.path.append('/root/autodl-tmp/wangbin/yiwang')
from torch.utils.data import ConcatDataset, DataLoader, Subset

from unlearning_random_afs_rand import train_student_model_random
from Domainadaptation_net_three import domainadaptation
from test_model_path import test_model
from puri import api
import random
from mu.net_train_three import get_student_model, get_teacher_model
from torch.utils.data import Dataset, DataLoader, Subset
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from plot_with_new_data import update_plot_with_new_data
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
# from tsne_mnist_samedata import tsne
# from tsne_mnist_tuo import tsne
from tsne_mnist_guding1 import tsnet
from tsne_mnist_guding2 import tsnes
# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs

# 加载npz文件
import torch.nn.init as init


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


npz_file_path = '/root/autodl-tmp/wangbin/yiwang/data/pathmnist.npz'
data = np.load(npz_file_path)

print(list(data.keys()))  # 打印所有在npz文件中的数组名

# ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
images = data['train_images']
labels = data['train_labels']
images_cal = data['val_images']
labels_cal = data['val_labels']
images_test = data['test_images']
labels_test = data['test_labels']



class PathMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
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
        print(f"Shuffling data with seed {seed}")  # 这里的seed是一个局部变量 有没有都可以 做的实验是这样的
        np.random.seed(seed)
        indices = np.arange(len(self.original_images))
        np.random.shuffle(indices)
        self.images = self.original_images[indices]
        self.labels = self.original_labels[indices]
        # 如果提供了转换函数，对每个图像进行转换
        # if self.transform:
        #     # 逐个图像应用转换
        #     self.images = np.array([self.transform(image) for image in self.images])


# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs
# random_seed = 10
# random_seed = 23
# random_seed = 52
# random_seed = 32
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

CONFIG = {
    'BASE1': {
        'BASE': list(range(0, 10000))
    },
    'BASE2': {
        'BASE': list(range(0, 9900))
    },
    'TEST1': {
        'TEST': list(range(0, 7180))
    },
    'CAL_100': {
        'CAL': list(range(40000, 40100))
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
        'TEST': list(range(46000, 47000))
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
        'QUERY': list(range(9900, 10000)),
        'QUERY_MEMBER': [1 for i in range(100)]
    },
    'QF_10': {
        'QUERY': list(range(9990, 10000)),
        'QUERY_MEMBER': [1 for i in range(10)]
    },
    'QF_1': {
        'QUERY': list(range(9999, 10000)),
        'QUERY_MEMBER': [1 for i in range(1)]
    },
    'QF_1000': {
        'QUERY': list(range(9000, 10000)),
        'QUERY_MEMBER': [1 for i in range(1000)]
    },
    'KD0.25': {
        'BASE': list(range(0, 2500))
    },
    'KD0.01': {
        'BASE': list(range(0, 100))
    },

    'KD0.5': {
        'BASE': list(range(0, 5000))
    },
    'KD0.75': {
        'BASE': list(range(0, 7500))
    },
}


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

'''
数据转换

'''

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
train_dataset.shuffle_data()  # 使用固定种子打乱数据
test_dataset = PathMNISTDataset(images_cal, labels_cal, transform=transform)
test_dataset.shuffle_data()  # 使用固定种子打乱数据

train_dataset_no = PathMNISTDataset(images, labels, transform=transform_no_salt_pepper)
train_dataset_no.shuffle_data()

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(test_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=64, shuffle=False,
                            generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_1000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
cal_2000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_2000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
cal_5000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_5000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
qo_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QO_5000']['QUERY']), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
qno_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_5000']['QUERY']), batch_size=64, shuffle=True,
                             generator=torch.Generator().manual_seed(random_seed))
qno_2000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_2000']['QUERY']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
qno_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1000']['QUERY']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
qno_100_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_100']['QUERY']), batch_size=32, shuffle=False,
                            generator=torch.Generator().manual_seed(random_seed))

qno_1_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1']['QUERY']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
qno_10_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_10']['QUERY']), batch_size=32, shuffle=False,
                           generator=torch.Generator().manual_seed(random_seed))

qf_100_loader_noise = DataLoader(Subset(train_dataset_no, CONFIG['QF_100']['QUERY']), batch_size=32, shuffle=True,
                                 generator=torch.Generator().manual_seed(random_seed))
# 创建十个 Subset 的副本
subsets = [Subset(train_dataset, CONFIG['QF_100']['QUERY']) for _ in range(10)]
concat_dataset = ConcatDataset(subsets)

# 使用 ConcatDataset 创建 DataLoader
qf_100_loader10 = DataLoader(
    concat_dataset,
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(random_seed)
)
qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=16, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))

qf_1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=32, shuffle=True,
                         generator=torch.Generator().manual_seed(random_seed))
qf_10_loader = DataLoader(Subset(train_dataset, CONFIG['QF_10']['QUERY']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))

qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
qf_1000_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,
                               generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                             generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(random_seed))
kd0_01_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.01']['BASE']), batch_size=32, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
# 参数设置
smodelmlp = get_student_model()
smodelmlp = smodelmlp.to(device)
tmodelmlp = get_student_model()
tmodelmlp = tmodelmlp.to(device)

tmodelmlp.apply(init_weights)
smodelmlp.apply(init_weights)
# u = smodelmlp.state_dict()
# f_u = smodelmlp.feature_extractor.state_dict()
# T = 4
modelmlp = get_teacher_model()
modelmlp = modelmlp.to(device)

# 加载整个封装字典
# checkpoint = torch.load('/root/autodl-tmp/wangbin/yiwang/weight/shardchard1qf100.pt')
# 仅提取状态字典部分
# state_dict = checkpoint['state_dict']
# 加载到模型
# modelmlp.load_state_dict(state_dict)


modelmlp.load_state_dict(torch.load('/root/autodl-tmp/wangbin/yiwang/weight/shard1460.pt'))
u = modelmlp.state_dict()
f_u = modelmlp.feature_extractor.state_dict()

member_gt = [1 for i in range(100)]




current_accuracy, accuracy1 = test_model(test1_loader, qf_100_loader, kd0_25_loader, device, modelmlp, modelmlp,tmodelmlp, u, f_u)


_t, pv, EMA_res, risk_score = api(device, modelmlp, u, f_u, qf_100_loader, member_gt, cal_1000_loader,caltest1_loader)


print(f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')








