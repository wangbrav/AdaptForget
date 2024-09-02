import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from res.resnet18 import ResNet18
# from res.resnet34 import ResNet34
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
sys.path.append('/root/autodl-tmp/wangbin/yiwang/')
from KDloss import SoftTarget
from tqdm import tqdm
import torch.nn.functional as F

from utils.Metric import AverageMeter, accuracy, Performance
#TODO xiangbi shangyige zengjia  le  afs



def train_student_model_random(forget_loader, retain_loader, tmodel18, model34, smodel18, u,f_u):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    risk_losses = AverageMeter()
    total_losses = AverageMeter()
    print("train student model")

    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)



    # 设置模型到设备
    # tmodel18 = tmodel18.to(device)
    # model34 = model34.to(device)
    # smodel18 = smodel18.to(device)
    # model34 = ResNet34().to(device)
    # model34.load_state_dict(torch.load('best_modelnet2.pth'))

    smodel18.load_state_dict(u)
    smodel18.feature_extractor.load_state_dict(f_u)

    # 定义超参数
    T = 4
    # lambda_kd = 1
    lambda_kd = 1.0

    # lambda_risk = 0.00095  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.051  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.081  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 1 # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.011  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.23 #  qf 1 kd0.5
    # lambda_risk = 0.0  # qf 100 bijaio hao de kd0.5
    # lambda_risk = 0.2  # qf 100 bijaio hao de kd0.5
    lambda_risk = 0.09  # qf 100 bijaio hao de kd0.5
    # lambda_risk = 0.4  # qf 100 kd0.5  还不知道好不好
    # lambda_risk = 0.008  # qf 1000 bijaio hao de
    # lambda_feature = 0.1


    #
    #
    num_epochs = 1  # 训练轮数
    print(f'lambda_risk: {lambda_risk}')
    print(f'lambda_kd: {lambda_kd}')
    print(f'T: {T}')
    print(f'num_epochs: {num_epochs}')
    #
    #定义损失函数和优化器
    # criterionCls = nn.CrossEntropyLoss().to(device)
    criterionKD = SoftTarget(T)
    criterionCls = nn.CrossEntropyLoss().to(device)
    # 定义MSE损失函数
    feature_loss_fn = nn.MSELoss()

    # criterionCls = nn.CrossEntropyLoss().to(device)
    #
    optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)
    # optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)
    #
    #训练过程
    # num_epochs = 1  # 训练轮数
    # 现在这里应该在加上 跟着afs一样的操作 然后
    for epoch in range(num_epochs):
        model34.eval()
        tmodel18.eval()
        smodel18.train()
        for X, Y in tqdm(retain_loader):
            print(X.size())
            X = X.to(device)
            Y = Y.to(device)
            # print(Y.shape)
            Y = Y.squeeze().long() # 是path的
            # print(Y.shape)



            snet_pred = smodel18(X)
            tnet_pred = model34(X)

            cls_loss = criterionCls(snet_pred, Y)
            kd_loss = criterionKD(snet_pred, tnet_pred.detach())
            cls_losses.update(cls_loss.item(), X.size(0))
            kd_losses.update(kd_loss.item(), X.size(0))

            loss = cls_loss + kd_loss * lambda_kd
            risk_loss = torch.tensor(0.0).to(device)
            # 在循环之前初始化累加器和批次计数器
            feature_loss_sum = 0.0
            risk_loss_sum = 0.0
            batch_count = 0
            epochs = 1
            for epoch in range(epochs):

                for _X, _Y in forget_loader:
                    _X = _X.to(device)
                    _Y = _Y.to(device)
                    # num_classes = 2 # 假设 tmodel18 的最后一层是全连接层fc
                    num_classes = tmodel18.classifier.num_classes  # 假设 tmodel18 的最后一层是全连接层fc
                    # print(tmodel18.dim_out)
                    # feature_extractort = tmodel18.feature_extractor(_X)
                    # feature_extractors = smodel18.feature_extractor(_X)
                    # feature_loss = feature_loss_fn(feature_extractors, feature_extractort)
                    # feature_loss_sum += feature_loss.item()
                    # _X  =  _X.unsqueeze(0)
                    random_probs = torch.rand(_X.size(0), num_classes).to(device)
                    teacher_probs = F.softmax(random_probs, dim=-1)  # 转换为概率分布
                    # print(random_probs)
                    # print(_X.size())
                    # print(smodel18(_X))
                    # _X = _X.unsqueeze(0)
                    student_probs = F.softmax(smodel18(_X), dim=-1)
                    # print(smodel18(_X))
                    risk_loss = F.kl_div(teacher_probs, student_probs, reduction='batchmean')
                    risk_losses.update(risk_loss.item(), _X.size(0))
                    risk_loss_sum += risk_loss.item()
                    batch_count += 1

                    # loss = loss
                    #
                    loss = loss + risk_loss * torch.tensor(lambda_risk).to(device)  # 原本的计算方法
                    # loss = loss + feature_loss*torch.tensor(lambda_feature).to(device) + risk_loss * torch.tensor(lambda_risk).to(device)  # 原本的计算方法
            # 所有批次结束后，计算 feature_loss 和 risk_loss 的平均值
            # average_feature_loss = feature_loss_sum / batch_count
            # average_risk_loss = risk_loss_sum / batch_count

            # 计算最终的 loss，包括平均 feature_loss 和平均 risk_loss
            # loss = loss + average_feature_loss * torch.tensor(lambda_feature).to(device) + average_risk_loss * torch.tensor(lambda_risk).to(device)
            total_losses.update(loss.item(), X.size(0))
            optimizer18s.zero_grad()

            loss.backward()
            optimizer18s.step()

    # 返回更新后的学生模型的参数
    return smodel18.feature_extractor.state_dict(),smodel18.state_dict(),smodel18.classifier.state_dict()



