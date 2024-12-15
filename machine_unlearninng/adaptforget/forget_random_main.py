"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate random forgetting.
Seperate file to allow for easy reuse.
"""

import random
import os
import wandb

# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model.net_train_three import get_student_model
import models
from unlearn import *
from utils import *
import forget_random_strategies
import datasets
import models
import conf
from training_utils import *


"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, required=True, help="net type")
parser.add_argument(
    "-weight_path",
    type=str,
    required=True,
    help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
    "-dataset",
    type=str,
    required=True,
    nargs="?",
    choices=["Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition"],
    help="dataset to train on",
)
parser.add_argument("-classes", type=int, required=True, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=False, help="use gpu or not")
parser.add_argument("-b", type=int, default=128, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument(
    "-method",
    type=str,
    required=True,
    nargs="?",
    choices=[
        "baseline",
        "retrain",
        "finetune",
        "blindspot",
        "amnesiac",
        "FisherForgetting",
        "ssd_tuning",
        "FisherForgetting",
    ],
    help="select unlearning method from choice set",
)
parser.add_argument(
    "-forget_perc", type=float, required=True, help="Percentage of trainset to forget"
)
parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")
args = parser.parse_args()
# 相比上一个  更改了 数据集的划分方法
# random_seed = 10
# random_seed = 23
# random_seed = 52
# random_seed = 32
random_seed = 62
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

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
        # print(f"Image shape at index {idx}: {image.shape}")

        if self.transform:
            image = self.transform(image)
        return image, label

    def shuffle_data(self, seed=52):
        # 设置随机种子以确保打乱顺序的一致性
        print(f"Shuffling data with seed {seed}")  # 这里的seed是一个局部变量 有没有都可以 做的实验是这样的
        # logger.info(f"Shuffling data with seed {seed}")

        np.random.seed(seed)
        indices = np.arange(len(self.original_images))
        np.random.shuffle(indices)
        self.images = self.original_images[indices]
        self.labels = self.original_labels[indices]
        # 如果提供了转换函数，对每个图像进行转换
        # if self.transform:
        #     # 逐个图像应用转换
        #     self.images = np.array([self.transform(image) for image in self.images])



CONFIG = {
    'BASE1': {
        'BASE': list(range(0, 10000))
    },
    'BASE2': {
        'BASE': list(range(0, 9900))
    },

    'BASE4': {
        'BASE': list(range(0, 9999))
    },
    'BASE3': {
        'BASE': list(range(0, 9000))
    },


    'TEST1': {
        'TEST': list(range(30000, 35000))
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

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")


class ExpandToRGB:
    """将单通道Tensor图像扩展为三通道"""

    def __call__(self, tensor):
        # 检查是否为单通道图像（形状为 [1, H, W]）
        if tensor.shape[0] == 1:
            # 重复通道以形成3通道图像
            tensor = tensor.repeat(3, 1, 1)
        return tensor


transform = transforms.Compose([
    transforms.ToTensor(),
    ExpandToRGB(),  # 确保这个转换在ToTensor之后

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = PathMNISTDataset(images, labels, transform=transform)
train_dataset.shuffle_data()  # 使用固定种子打乱数据

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
base2_indices = CONFIG['BASE2']['BASE']
base2_dataset = Subset(train_dataset, base2_indices)
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
# # 根据配置索引创建 base3 数据集
# base3_indices = CONFIG['BASE3']['BASE']
# base3_dataset = Subset(train_dataset, base3_indices)
# # 创建 base3 数据集的数据加载器
# base3_loader = DataLoader(Subset(train_dataset, CONFIG['BASE3']['BASE']), batch_size=32, shuffle=True,
#                           generator=torch.Generator().manual_seed(random_seed))

# 根据配置索引创建 base_1000 数据集
base3_indices = CONFIG['BASE3']['BASE']
base3_dataset = Subset(train_dataset, base3_indices)
# 创建 base3 数据集的数据加载器
base3_loader = DataLoader(Subset(train_dataset, CONFIG['BASE3']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))


base4_indices = CONFIG['BASE4']['BASE']
base4_dataset = Subset(train_dataset, base4_indices)
# 创建 base3 数据集的数据加载器
base4_loader = DataLoader(Subset(train_dataset, CONFIG['BASE4']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))


test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,
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
qf_100_indices = CONFIG['QF_100']['QUERY']
qf_100_dataset = Subset(train_dataset, qf_100_indices)
qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=16, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
qf_1_indices = CONFIG['QF_1']['QUERY']
qf_1_dataset = Subset(train_dataset, qf_100_indices)
qf_1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=16, shuffle=True,
                         generator=torch.Generator().manual_seed(random_seed))
qf_10_loader = DataLoader(Subset(train_dataset, CONFIG['QF_10']['QUERY']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
qf_1000_indices = CONFIG['QF_1000']['QUERY']
qf_1000_dataset = Subset(train_dataset, qf_1000_indices)
qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
kd0_01_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.01']['BASE']), batch_size=32, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


batch_size = args.b

# get network
# net = getattr(models, args.net)(num_classes=args.classes)
net = get_student_model()
net.load_state_dict(torch.load(args.weight_path))

unlearning_teacher = get_student_model()
# unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)

if args.gpu:
    net = net.cuda()
    unlearning_teacher = unlearning_teacher.cuda()

#读取数据的root
# root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

# 照片的尺寸

img_size = 224 if args.net == "ViT" else 32

#数据集 以及数据集加载
# trainset = getattr(datasets, args.dataset)(
#     root=root, download=True, train=True, unlearning=True, img_size=img_size
# )
# validset = getattr(datasets, args.dataset)(
#     root=root, download=True, train=False, unlearning=True, img_size=img_size
# )

trainloader = base1_loader
# trainloader = DataLoader(trainset, num_workers=4, batch_size=args.b, shuffle=True)
validloader = test1_loader
# validloader = DataLoader(validset, num_workers=4, batch_size=args.b, shuffle=False)
# 数据集划分
# print("Dataset length:", len(trainset))
# dataset_length=len(trainset)
# split1 = int(dataset_length * 0.002)  # 80% 用于训练
# split2 = dataset_length - split1    # 剩余的 20% 用于验证
# forget_train, retain_train = torch.utils.data.random_split(trainset, [split1, split2])
# forget_train, retain_train = torch.utils.data.random_split(
#     trainset, [args.forget_perc, 1 - args.forget_perc]
# )

#数据集的加载
# forget_train_dl = DataLoader(list(forget_train), batch_size=128)

#修改1000的时候这里需要修改
forget_train_dl = qf_1_loader
# forget_train_dl = qf_100_loader
# 修改1000的时候这里需要修改
retain_train_dl = base4_loader
# retain_train_dl = base2_loader
# retain_train_dl = DataLoader(list(retain_train), batch_size=128, shuffle=True)
forget_valid_dl = forget_train_dl
retain_valid_dl = test1_loader
# retain_valid_dl = validloader



# Change alpha here as described in the paper
model_size_scaler = 1
if args.net == "ViT":
    model_size_scaler = 1
else:
    model_size_scaler = 1


# full_train_dl = DataLoader(
#     ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
#     batch_size=batch_size,
# )
full_train_dl = base1_loader
#修改1000的时候这里需要修改
kwargs = {
    "model": net,
    "unlearning_teacher": unlearning_teacher,
    "retain_train_dl": retain_train_dl,
    "retain_valid_dl": retain_valid_dl,
    "forget_train_dl": forget_train_dl,
    "forget_valid_dl": forget_valid_dl,
    "full_train_dl": full_train_dl,
    "valid_dl": validloader,
    "dampening_constant": 1,
    "selection_weighting": 10 * model_size_scaler,
    "num_classes": args.classes,
    "dataset_name": args.dataset,
    "device": "cuda" if args.gpu else "cpu",
    "model_name": args.net,
    "cal_1000_loader": cal_1000_loader,
    "caltest1_loader": caltest1_loader,
    # "base3_loader": base_1000_loader,
    "qf_100_dataset": qf_1_dataset,
    "base2_dataset":base4_dataset,
    # "base2_dataset":base2_dataset,
    "method": args.method,
}

wandb.init(
    project=f"R1_{args.net}_{args.dataset}_random_{args.forget_perc}perc",
    name=f"{args.method}",
)


# wandb.init(project=f"{args.dataset}_forget_random_{args.forget_perc}", name=f'{args.method}')

import time

start = time.time()

testacc, retainacc, zrf, mia, d_f = getattr(forget_random_strategies, args.method)(
    **kwargs
)
end = time.time()
time_elapsed = end - start

print(testacc, retainacc, zrf, mia)
wandb.log(
    {
        "TestAcc": testacc,
        "RetainTestAcc": retainacc,
        "ZRF": zrf,
        "MIA": mia,
        "Df": d_f,
        "model_scaler": model_size_scaler,
        "MethodTime": time_elapsed,  # do not forget to deduct baseline time from it to remove results calc (acc, MIA, ...)
    }
)

wandb.finish()
