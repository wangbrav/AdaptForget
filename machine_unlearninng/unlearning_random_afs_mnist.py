import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet18 import ResNet18
from resnet34 import ResNet34
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
    smodel18.features.load_state_dict(f_u)

    # 定义超参数
    T = 4
    lambda_kd = 1
    # lambda_risk = 2
    lambda_risk = 1.15
    # num_epochs = 10  # 训练轮数

    # 定义损失函数和优化器
    # criterionCls = nn.CrossEntropyLoss().to(device)
    criterionKD = SoftTarget(T)
    # criterionCls = nn.NLLLoss().to(device)  mnist shuju ji
    criterionCls = nn.CrossEntropyLoss().to(device)

    optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)

    # 训练过程
    num_epochs = 1  # 训练轮数
    # 现在这里应该在加上 跟着afs一样的操作 然后
    for epoch in range(num_epochs):
        model34.eval()
        tmodel18.eval()
        smodel18.train()
        for X, Y in tqdm(retain_loader):
            X = X.to(device)
            Y = Y.to(device)
            Y = Y.squeeze().long()



            snet_pred = smodel18(X)
            tnet_pred = model34(X)

            cls_loss = criterionCls(snet_pred, Y)
            kd_loss = criterionKD(snet_pred, tnet_pred.detach())
            cls_losses.update(cls_loss.item(), X.size(0))
            kd_losses.update(kd_loss.item(), X.size(0))

            loss = cls_loss + kd_loss * lambda_kd
            risk_loss = torch.tensor(0.0).to(device)
            for _X, _Y in forget_loader:
                _X = _X.to(device)
                _Y = _Y.to(device)
                outputs18t = tmodel18(_X)
                outputs18s = smodel18(_X)
                print(outputs18s)
                teacher_probs = torch.exp(outputs18t)
                student_probs = torch.exp(outputs18s)
                print(student_probs)
                risk_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
                risk_losses.update(risk_loss.item(), _X.size(0))
            # loss = loss
                loss = loss + risk_loss * torch.tensor(lambda_risk).to(device)
            total_losses.update(loss.item(), X.size(0))
            optimizer18s.zero_grad()

            loss.backward()
            optimizer18s.step()

    # 返回更新后的学生模型的参数
    return smodel18.features.state_dict(),smodel18.state_dict()



