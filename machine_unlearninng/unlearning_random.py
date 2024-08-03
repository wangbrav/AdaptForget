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

# # 固定随机种子以确保每次运行结果一致
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
# # model = ResNet18()
#
# smodel18 = ResNet18().to(device)
# tmodel18 = ResNet18().to(device)
# tmodel18.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
#
# model34 = ResNet34().to(device)
# model34.load_state_dict(torch.load('resnet34_mnist.pth'))
# T = 4
# lambda_kd = 1
#
# lambda_risk = 10
# criterion = nn.CrossEntropyLoss()
# criterionKD = SoftTarget(T)
# criterionCls = torch.nn.NLLLoss().to(device)
# optimizer34 = optim.Adam(model34.parameters(), lr=0.001)
# optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)
# 训练原始模型
# num_epochs = 50  # 训练轮数
# for epoch in range(num_epochs):
#     model34.train()
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         # 将数据和标签移至GPU
#         inputs, labels = data[0].to(device), data[1].to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model34(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
#
# torch.save(model34.state_dict(), 'resnet34_mnist.pth')


# 进行一个循环的学习  也有一个对抗性
# num_epochs = 10  # 训练轮数
#
# for epoch in range(num_epochs):
#     model34.eval()
#     tmodel18.eval()
#     smodel18.train()
#     running_loss = 0.0
#     totalt = 0
#     correctt = 0
#     totalu = 0
#     correctu = 0
#     for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#         inputs, labels = data[0].to(device), data[1].to(device)
#         optimizer18s.zero_grad()
#         outputs34 = model34(inputs)
#         outputs18s = smodel18(inputs)
#         cls_loss = criterionCls(outputs18s, labels)
#         kd_loss = criterionKD(outputs18s, outputs34.detach())
#         loss = cls_loss + kd_loss * lambda_kd
#         _, predicted = torch.max(outputs18s.data, 1)
#         totalt += labels.size(0)
#         correctt += (predicted == labels).sum().item()
#
#
#         for i, data in enumerate(forget_loader, 0):
#             inputs, labels = data[0].to(device), data[1].to(device)
#
#             outputs18t = tmodel18(inputs)
#             outputs18s = smodel18(inputs)
#             risk_loss = criterion(outputs18s, outputs18t)
#             _, predicted = torch.max(outputs18s.data, 1)
#             totalu += labels.size(0)
#             correctu += (predicted == labels).sum().item()
#
#         loss = loss + risk_loss * torch.tensor(lambda_risk).to(device)
#         loss.backward()
#         optimizer18s.step()
#     accuracyt = 100 * correctt / totalt
#     accuracyu = 100 * correctu / totalu
#     print(f'Accuracy of the network on the test images: {accuracyt}%')
#     print(f'Accuracy of the network on the test images: {accuracyu}%')



# # 打印数据集大小
# print("训练集大小:", len(train_dataset))
#
# print("保持数据集大小:", len(retain_set))
# print("遗忘集大小:", len(forget_set))
# print("测试集大小:", len(test_dataset))''

def train_student_model_random(forget_loader, retain_loader, tmodel18, model34, smodel18, u,f_u):
    # 固定随机种子以确保每次运行结果一致
    # torch.manual_seed(0)
    # np.random.seed(0)
    print("train student model")

    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #数据处理
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    # 设置模型到设备
    # tmodel18 = tmodel18.to(device)
    # model34 = model34.to(device)
    # smodel18 = smodel18.to(device)
    # model34 = ResNet34().to(device)
    model34.load_state_dict(torch.load('best_modelmlp1.pth'))

    smodel18.load_state_dict(u)
    smodel18.features.load_state_dict(f_u)

    # 定义超参数
    T = 4
    lambda_kd = 1
    lambda_risk = 10
    # num_epochs = 10  # 训练轮数

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    criterionKD = SoftTarget(T)
    criterionCls = nn.NLLLoss().to(device)
    optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)

    # 训练过程
    num_epochs = 50  # 训练轮数
    num_epochs2 = 2
    # 现在这里应该在加上 跟着afs一样的操作 然后
    for epoch in range(num_epochs):
        model34.eval()
        tmodel18.eval()
        smodel18.train()
        running_loss = 0.0
        totalt = 0
        correctt = 0

        for data in tqdm(retain_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer18s.zero_grad()
            outputs34 = model34(inputs)
            outputs18s = smodel18(inputs)
            cls_loss = criterionCls(outputs18s, labels)
            kd_loss = criterionKD(outputs18s, outputs34.detach())
            loss = cls_loss + kd_loss * lambda_kd
            _, predicted = torch.max(outputs18s.data, 1)
            totalt += labels.size(0)
            correctt += (predicted == labels).sum().item()
            totalu = 0
            correctu = 0
            total_loss = 0.0  # 初始化总损失
            total_count = 0  # 初始化样本总数
            for i, data in enumerate(forget_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs18t = tmodel18(inputs)
                outputs18s = smodel18(inputs)
                teacher_probs = torch.exp(outputs18t)
                student_probs = torch.exp(outputs18s)
                risk_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')

                # risk_loss = criterion(outputs18s, outputs18t)
                total_loss += risk_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs18s.data, 1)
                total_count += inputs.size(0)
                totalu += labels.size(0)
                correctu += (predicted == labels).sum().item()
            accuracyu = 100 * correctu / totalu
            average_loss = total_loss / total_count
            # print(f'Accuracy of the network on the forget images: {accuracyu}%')

            # loss = loss
            loss = loss + average_loss * torch.tensor(lambda_risk).to(device)
            loss.backward()
            optimizer18s.step()
        accuracyt = 100 * correctt / totalt
        # accuracyu = 100 * correctu / totalu
        print(f'Accuracy of the network on the retain  images: {accuracyt}%')
        print(f'Accuracy of the network on the forget images: {accuracyu}%')

    # 返回更新后的学生模型的参数
    return smodel18.features.state_dict(),smodel18.state_dict()




    # for epoch in range(num_epochs):
    #     model34.eval()
    #     tmodel18.eval()
    #     smodel18.train()
    #     running_loss = 0.0
    #     totalt = 0
    #     correctt = 0
    #     totalu = 0
    #     correctu = 0
    #     for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         optimizer18s.zero_grad()
    #         outputs34 = model34(inputs)
    #         outputs18s = smodel18(inputs)
    #         cls_loss = criterionCls(outputs18s, labels)
    #         kd_loss = criterionKD(outputs18s, outputs34.detach())
    #         loss = cls_loss + kd_loss * lambda_kd
    #         _, predicted = torch.max(outputs18s.data, 1)
    #         totalt += labels.size(0)
    #         correctt += (predicted == labels).sum().item()
    #
    #         for i, data in enumerate(forget_loader, 0):
    #             inputs, labels = data[0].to(device), data[1].to(device)
    #
    #             outputs18t = tmodel18(inputs)
    #             outputs18s = smodel18(inputs)
    #             risk_loss = criterion(outputs18s, outputs18t)
    #             _, predicted = torch.max(outputs18s.data, 1)
    #             totalu += labels.size(0)
    #             correctu += (predicted == labels).sum().item()
    #
    #         # loss = loss
    #         loss = loss + risk_loss * torch.tensor(lambda_risk).to(device)
    #         loss.backward()
    #         optimizer18s.step()
    #     accuracyt = 100 * correctt / totalt
    #     accuracyu = 100 * correctu / totalu
    #     print(f'Accuracy of the network on the test images: {accuracyt}%')
    #     print(f'Accuracy of the network on the test images: {accuracyu}%')

  #第二种方法 使用的是 只进行遗忘 分别进行

  # for epoch in range(num_epochs2):
    #     model34.eval()
    #     tmodel18.eval()
    #     smodel18.train()
    #     running_loss = 0.0
    #     totalt = 0
    #     correctt = 0
    #
    #     for data in tqdm(retain_loader, desc=f"Epoch {epoch + 1}/{num_epochs2}"):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         optimizer18s.zero_grad()
    #         outputs34 = model34(inputs)
    #         outputs18s = smodel18(inputs)
    #         cls_loss = criterionCls(outputs18s, labels)
    #         kd_loss = criterionKD(outputs18s, outputs34.detach())
    #         loss = cls_loss + kd_loss * lambda_kd
    #         _, predicted = torch.max(outputs18s.data, 1)
    #         totalt += labels.size(0)
    #         correctt += (predicted == labels).sum().item()
    #         loss.backward()
    #         optimizer18s.step()
    #     accuracyt = 100 * correctt / totalt
    #     print(f'Accuracy of the network on the train images: {accuracyt}%')

    # for epoch in range(num_epochs):
    #     model34.eval()
    #     tmodel18.eval()
    #     smodel18.train()
    #     running_loss = 0.0
    #     # totalt = 0
    #     # correctt = 0
    #     totalu = 0
    #     correctu = 0
    #
    #     total_loss = 0.0  # 初始化总损失
    #     total_count = 0  # 初始化样本总数
    #
    #     for i, data in enumerate(forget_loader, 0):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #
    #         outputs18t = tmodel18(inputs)
    #         outputs18s = smodel18(inputs)
    #         risk_loss = criterion(outputs18s, outputs18t)
    #         total_loss += risk_loss.item() * inputs.size(0)  # 累加损失，乘以批次中的样本数
    #         total_count += inputs.size(0)  # 累加样本数
    #
    #         _, predicted = torch.max(outputs18s.data, 1)
    #         totalu += labels.size(0)
    #         correctu += (predicted == labels).sum().item()
    #
    #         # average_loss = total_loss / total_count  # 计算平均损失
    #
    #         # loss = loss
    #         loss =0.00000001+ risk_loss * torch.tensor(lambda_risk).to(device)
    #         loss.backward()
    #         optimizer18s.step()
    #     # accuracyt = 100 * correctt / totalt
    #     accuracyu = 100 * correctu / totalu
    #     # print(f'Accuracy of the network on the test images: {accuracyt}%')
    #     print(f'Accuracy of the network on the test images: {accuracyu}%')

