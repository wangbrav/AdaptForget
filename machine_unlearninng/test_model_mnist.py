import torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet18 import ResNet18
# from resnet34 import ResNet34
from mu.net_train import get_model

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from KDloss import SoftTarget
from tqdm import tqdm

# 假设 retain_loader 和 forget_loader 已经被定义
# 假设 device, model34, smodel18, tmodel18, criterionCls, criterionKD, criterion 已经被定义

def test_model(test_loader,retain_loader, forget_loader, device, model34, smodel18, tmodel18,u,f_u):
    # model34 = get_model()
    # model34=model34.to(device)
    # model34.load_state_dict(torch.load('best_modelnet2.pth'))

    smodel18.load_state_dict(u)
    # smodel18.features.load_state_dict(f_u)
    smodel18.feature_extractor.load_state_dict(f_u)
    smodel18.eval()  # 设置为评估模式
    tmodel18.eval()  # 设置为评估模式
    model34.eval()   # 设置为评估模式

    correctt = 0
    totalt = 0
    correctu = 0
    totalu = 0
    correcttt = 0
    totaltt = 0


    with torch.no_grad():  # 确保不更新梯度
        # for data in retain_loader:
        #     inputs, labels = data[0].to(device), data[1].to(device)
        #     outputs18s = smodel18(inputs)
        #     # outputs34 = model34(inputs)
        #
        #     labels=   labels.squeeze().long()
        #     _, predicted = torch.max(outputs18s.data, 1)
        #     totalt += labels.size(0)
        #     correctt += (predicted == labels).sum().item()

        # for data in forget_loader:
        #     inputs, labels = data[0].to(device), data[1].to(device)
        #     outputs18s = smodel18(inputs)
        #     # outputs18t = tmodel18(inputs)
        #     _, predicted = torch.max(outputs18s.data, 1)
        #     totalu += labels.size(0)
        #     correctu += (predicted == labels).sum().item()

        for data in test_loader:
            data[1] = data[1].long()
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs18s = smodel18(inputs)
            # outputs18t = tmodel18(inputs)
            pred = outputs18s.data.max(1)[1]
            # _, predicted = torch.max(outputs18s.data, 1)
            correcttt += pred.eq(labels.view(-1)).sum().item()

            # print(labels)
            # print(predicted)
            # print(labels.size(0))
            totaltt += labels.size(0)
            # labels = labels.squeeze()  #  在path中使用
            # print((predicted == labels).sum().item())
            # correcttt = (predicted == labels).sum().item()
            # correcttt += (predicted == labels).sum().item()  #  计算 mnist 数据集的时候就没有问题


    # print(f'Accuracy on retain data: {100 * correctt / totalt}%')
    print(f'Accuracy on test data: {100 * correcttt / totaltt}%')
    # print(f'Accuracy on forget data: {100 * correctu / totalu}%')

# 在实际使用中，您需要传递适当的参数来调用 test_model 函数。
# test_model(retain_loader, forget_loader, device, model34, smodel18, tmodel18)
