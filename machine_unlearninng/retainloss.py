
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from resnet18 import ResNet18
# 固定随机种子以确保每次运行结果一致
# torch.manual_seed(0)
# np.random.seed(0)
#
# # 检查GPU是否可用
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
#
#
# # 设置数据转换，这里仅进行基础的转换和标准化
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # 下载MNIST数据集
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#
# # 创建DataLoader
# # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
# num_forget = 10000
# indices = list(range(len(train_dataset)))
# np.random.shuffle(indices)
#
# forget_indices = indices[:num_forget]
# retain_indices = indices[num_forget:]
#
# forget_set = Subset(train_dataset, forget_indices)
# retain_set = Subset(train_dataset, retain_indices)
#
# train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
# retain_loader = DataLoader(retain_set, batch_size=64, shuffle=True)
# forget_loader = DataLoader(forget_set, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
# # 加载 MNIST 数据集
#
#
# # 加载 ResNet18 模型
# # 初始化 ResNet18 模型
# resnet18 = ResNet18()
# print(device)
# # 加载您的预训练权重
# resnet18.load_state_dict(torch.load('resnet18_mnist.pth'))
# resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)  # 适应 MNIST 的 10 个类别
# resnet18 =  resnet18.to(device)
# # 设置损失函数
# criterion = nn.CrossEntropyLoss()
#
# # 禁用梯度
# with torch.no_grad():
#     for data, target in retain_loader:
#         # 预测
#         print(device)
#         data, target = data.to(device), target.to(device)
#         output = resnet18(data)
#
#         # 计算损失
#         loss = criterion(output, target)
#
#         # 输出损失
#         print("损失：", loss.item())
#         break  # 示例中只处理一个批次
#



def evaluate_model(weights, model, retain_loader):
    # 检查GPU是否可用
    print("evaluate model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载模型权重
    # model.load_state_dict(torch.load(weights))
    model.load_state_dict(weights)

    model = model.to(device)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()

    # 禁用梯度计算
    with torch.no_grad():
        for data, target in retain_loader:
            # 将数据移至设备
            data, target = data.to(device), target.to(device)

            # 进行预测
            output = model(data)

            # 计算损失
            loss = criterion(output, target)


            # 返回损失值
            print("loss1:",loss.item())

            return loss

# 示例用法
# weights = 'resnet18_mnist.pth'
# model = ResNet18()
# model.fc = nn.Linear(model.fc.in_features, 10)  # 适应 MNIST 的 10 个类别
#
# # 假设 retain_loader 已经定义
# loss = evaluate_model(weights, model, retain_loader)
# print("损失：", loss)
