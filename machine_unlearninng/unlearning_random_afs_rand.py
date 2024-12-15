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
import logging

from utils.Metric import AverageMeter, accuracy, Performance
logging.basicConfig(filename='./tc/training_log_qf1circulate_asdv7.log', level=logging.INFO, format='%(asctime)s %(message)s')
# logging.basicConfig(filename='./tc/pathablation.log', level=logging.INFO, format='%(asctime)s %(message)s')
# logging.basicConfig(filename='./tc/pathtsne.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()
#TODO xiangbi shangyige zengjia  le  afs
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for X, Y in tqdm(test_loader):
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)
                # _, predicted = torch.max(outputs.data, 1)
                  # 调整Y的形状和数据类型

                pred = outputs.data.max(1)[1]
                # print(predicted.shape)
                # print(Y.shape)
                total += Y.size(0)

                # Y = Y.squeeze(1).long()  #因为表格数据集 将其注销掉
                correct += pred.eq(Y.view(-1)).sum().item()
                # print(f'Batch correct: {(predicted == Y).sum().item()}, Batch total: {Y.size(0)}')  # 新增的打印语句
    print(f"Correct: {correct}, Total: {total}")
    accuracy = 100 * correct / total
    return accuracy



def train_student_model_random(lambda_risk,lambda_kd,forget_loader, retain_loader,test1loader, tmodel18, model34, smodel18, u,f_u):
# def train_student_model_random(forget_loader, retain_loader,test1loader, tmodel18, model34, smodel18, u,f_u):

    # def train_student_model_random(forget_loader, retain_loader, tmodel18, model34, smodel18, u,f_u,lambda_risk,lambda_kd):
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
    T = 6
    # T = 8
    # lambda_kd = 1 #path derma o
    # lambda_kd = 3 #path derma o
    # lambda_kd = 3 #path derma o
    lambda_kd = lambda_kd
    # lambda_kd = 0.9

    # lambda_risk = 0.0000095  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.051  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.081  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 1 # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.015  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.0071  #path 多样本遗忘 pathmnist kd0.75 和kd0.5不错  derma也不错
    # lambda_risk = 0.0061  #path 多样本遗忘 pathmnist kd0.75 和kd0.5不错  derma也不错
    # lambda_risk = 0.0051  #path 多样本遗忘 pathmnist kd0.75 和kd0.5不错  derma也不错
    # lambda_risk = 0.0091  # qf 100 bijaio hao de kd0.75
    # lambda_risk = 0.23 #  qf 1 kd0.5
    # lambda_risk = 0.0  # qf 100 bijaio hao de kd0.5
    # lambda_risk = 0.2  # qf 100 bijaio hao de kd0.5
    lambda_risk =lambda_risk # qf 100 bijaio hao de kd0.5
    # lambda_risk = 0.09  # qf 100 bijaio hao de kd0.5
    # lambda_risk = 0.4  # qf 100 kd0.5  还不知道好不好
    # lambda_risk = 0.008  # qf 1000 bijaio hao de
    # lambda_feature = 0.1


    #
    #
    best_accuracy = 0

    num_epochs = 1 # 训练轮数
    # print(f'lambda_risk: {lambda_risk}')
    # print(f'lambda_kd: {lambda_kd}')
    print(f'T: {T}')
    print(f'num_epochs: {num_epochs}')
    #
    #定义损失函数和优化器
    # criterionCls = nn.CrossEntropyLoss().to(device)
    criterionKD = SoftTarget(T)
    criterionKD2 = SoftTarget(9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterionCls = nn.CrossEntropyLoss().to(device)
    # 定义MSE损失函数
    feature_loss_fn = nn.MSELoss()

    # criterionCls = nn.CrossEntropyLoss().to(device)
    #
    optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)
    # optimizer34 = torch.optim.Adam(model34.parameters(), lr=0.001)
    # optimizer18s = torch.optim.Adam(smodel18.parameters(), lr=0.001)

    # optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)
    #
    #训练过程
    # num_epochs = 1  # 训练轮数
    # 现在这里应该在加上 跟着afs一样的操作 然后

# for epoch in range(epochs):
#     model.train()
#     correct = 0
#     train_loss = 0
#
#     with tqdm(total=len(base1_loader)) as t:
#         for X, Y in tqdm(base1_loader):
#             X = X.to(device)
#             Y = Y.to(device)
#             # Training pass
#             optimizer.zero_grad()
#             output = model(X)
#             # print(output)
#             # output = torch.argmax(output, dim=1)  # 假设Y是独热编码，转换成类别索引
#             # print(output.shape)  # 检查Y的形状
#             # print(output)
#             # print(Y.shape)
#             # print(Y)
#             # Y = Y.squeeze(1) # 调整Y的形状和数据类型
#             # print(Yield.shape)
#             # print(Y)
#             Y = Y.squeeze(1).long()  # 调整Y的形状和数据类型
#             # print("Model output shape:", output.shape)
#             # print("Target labels:", Y.unique())
#             loss = criterion(output, Y)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
#             pred = output.data.max(1)[1]
#             correct += pred.eq(Y.view(-1)).sum().item()
#
#     train_loss /= len(base1_loader.dataset)
#     train_accuracy = 100 * correct / len(base1_loader.dataset)
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#
#     # 测试模型
#     test_accuracy = test(model, test1_loader, device)
#     print(f"Test Accuracy: {test_accuracy:.2f}%")
#
#     # 保存最佳模型
#     if test_accuracy > best_accuracy:
#         best_accuracy = test_accuracy
#         logger.info(f"Saving best model with accuracy {best_accuracy}")
#         best_model_state_trained = model.state_dict().copy()
#         save_dir = "./quanzhong/"
#
#         # 如果文件夹不存在，则创建该文件夹
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         # 定义保存权重的文件路径
#         save_path_trained = os.path.join(save_dir, "best_trained.pth")
#         # 保存模型权重到文件
#         torch.save(best_model_state_trained, save_path_trained)
#         logger.info(f"Model weights saved successfully to {save_path_trained}.")
#
    for epoch in range(num_epochs):
        # model34.train()
        model34.eval()
        tmodel18.eval()
        smodel18.train()
        correct = 0
        train_loss = 0
        # for X, Y in tqdm(retain_loader):
        #     # print(X.size())
        #     X = X.to(device)
        #     Y = Y.to(device)
        #     # print(Y.shape)
        #     Y = Y.squeeze().long() # 是path的
        #     # print(Y.shape)
        #
        #
        #
        #     snet_pred = smodel18(X)
        #     tnet_pred = model34(X)
        #
        #     cls_loss = criterionCls(snet_pred, Y)
        #     kd_loss = criterionKD(snet_pred, tnet_pred.detach())
        #     cls_losses.update(cls_loss.item(), X.size(0))
        #     kd_losses.update(kd_loss.item(), X.size(0))
        #
        #     loss = cls_loss*1 + kd_loss * 0
        #     # loss = cls_loss + kd_loss * lambda_kd
        #     risk_loss = torch.tensor(0.0).to(device)
        #     # 在循环之前初始化累加器和批次计数器
        #     feature_loss_sum = 0.0
        #     risk_loss_sum = 0.0
        #     batch_count = 0
        #     epochs = 1
        #     for epoch in range(epochs):
        #
        #         for _X, _Y in forget_loader:
        #             _X = _X.to(device)
        #             _Y = _Y.to(device)
        #             # num_classes = 2 # 假设 tmodel18 的最后一层是全连接层fc
        #             num_classes = tmodel18.classifier.num_classes  # 假设 tmodel18 的最后一层是全连接层fc
        #             # print(tmodel18.dim_out)
        #             # feature_extractort = tmodel18.feature_extractor(_X)
        #             # feature_extractors = smodel18.feature_extractor(_X)
        #             # feature_loss = feature_loss_fn(feature_extractors, feature_extractort)
        #             # feature_loss_sum += feature_loss.item()
        #             # _X  =  _X.unsqueeze(0)
        #             random_probs = torch.rand(_X.size(0), num_classes).to(device)
        #             teacher_probs = F.softmax(random_probs, dim=-1)  # 转换为概率分布
        #             # print(random_probs)
        #             # print(_X.size())
        #             # print(smodel18(_X))
        #             # _X = _X.unsqueeze(0)
        #             # student_probs = smodel18(_X)
        #             student_probs = F.softmax(smodel18(_X), dim=-1)
        #             # print(smodel18(_X))
        #             risk_loss = F.kl_div(teacher_probs, student_probs, reduction='batchmean')
        #             risk_losses.update(risk_loss.item(), _X.size(0))
        #             risk_loss_sum += risk_loss.item()
        #             batch_count += 1
        #
        #             loss = loss
        #
        #             # loss = loss + risk_loss * torch.tensor(lambda_risk).to(device)  # 原本的计算方法
        #             # loss = loss + feature_loss*torch.tensor(lambda_feature).to(device) + risk_loss * torch.tensor(lambda_risk).to(device)  # 原本的计算方法
        #     # 所有批次结束后，计算 feature_loss 和 risk_loss 的平均值
        #     # average_feature_loss = feature_loss_sum / batch_count
        #     # average_risk_loss = risk_loss_sum / batch_count
        #
        #     # 计算最终的 loss，包括平均 feature_loss 和平均 risk_loss
        #     # loss = loss + average_feature_loss * torch.tensor(lambda_feature).to(device) + average_risk_loss * torch.tensor(lambda_risk).to(device)
        #     total_losses.update(loss.item(), X.size(0))
        #     optimizer18s.zero_grad()
        #
        #     loss.backward()
        #     optimizer18s.step()
        with tqdm(total=len(retain_loader)) as t:
            for X, Y in tqdm(retain_loader):
                X = X.to(device)
                Y = Y.to(device)
                # Training pass
                optimizer18s.zero_grad()
                # optimizer34.zero_grad()
                # output = smodel18(X)
                snet_pred = smodel18(X)
                # snet_pred = smodel18(X)
                tnet_pred = model34(X)
                # print(output)
                # output = torch.argmax(output, dim=1)  # 假设Y是独热编码，转换成类别索引
                # print(output.shape)  # 检查Y的形状
                # print(output)
                # print(Y.shape)
                # print(Y)
                # Y = Y.squeeze(1) # 调整Y的形状和数据类型
                # print(Yield.shape)
                # print(Y)

                Y = Y.squeeze().long()  # 调整Y的形状和数据类型
                # Y = Y.squeeze(1).long()  # 调整Y的形状和数据类型  # 是path的
                # print("Model output shape:", output.shape)
                # print("Target labels:", Y.unique())
                cls_loss = criterionCls(snet_pred, Y)
                # loss1 = criterionCls(snet_pred, Y)
                kd_loss = criterionKD(snet_pred, tnet_pred.detach())
                # loss = cls_loss
                loss = cls_loss + kd_loss * lambda_kd

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

                    # teacher_probs = F.softmax(random_probs, dim=-1)  # 转换为概率分布
                    student_probs = smodel18(_X)
                    # student_probs = F.softmax(smodel18(_X), dim=-1)
                    teacher_probs = tmodel18(_X)
                    # teacher_probs = F.softmax(tmodel18(_X), dim=-1)
                    risk_loss = criterionKD2(student_probs, teacher_probs.detach())

                    # risk_loss = F.kl_div(teacher_probs, student_probs, reduction='batchmean')
                    # loss = loss
                    # loss = loss + feature_loss * torch.tensor(lambda_risk).to(device)  # 原本的计算方法
                    loss = loss + risk_loss * torch.tensor(lambda_risk).to(device)  # 原本的计算方法
                    # loss = loss + risk_loss   # 原本的计算方法
                # print('risk',risk_loss)
                # print('kd',kd_loss)
                # print('cls',loss1)
                # loss = loss1
                # loss = loss1  + kd_loss
                # loss = loss1  + kd_loss+ risk_loss    #tupainxuyao
                # loss = loss1  + kd_loss+ risk_loss   #tupainxuyao
                # loss = loss1  + kd_loss+ risk_loss* torch.tensor(lambda_risk).to(device)
                loss.backward()
                train_loss += loss.item()
                optimizer18s.step()
                # optimizer18s.step()
                pred = snet_pred.data.max(1)[1]
                correct += pred.eq(Y.view(-1)).sum().item()
        train_loss /= len(retain_loader.dataset)
        train_accuracy = 100 * correct / len(retain_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        # 测试模型
        test_accuracy = test(smodel18, test1loader, device)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"Saving best model with accuracy {best_accuracy}")

            # logger.info(f"Saving best model with accuracy {best_accuracy}")
        #     best_model_state_mu_u = smodel18.state_dict().copy()
        #     best_model_state_mu_f_u=smodel18.feature_extractor.state_dict().copy()

            # save_dir = "./quanzhong/"

            # 如果文件夹不存在，则创建该文件夹
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # 定义保存权重的文件路径
            # save_path_trained = os.path.join(save_dir, "best_trained.pth")
            # 保存模型权重到文件
            # torch.save(best_model_state_trained, save_path_trained)
            # logger.info(f"Model weights saved successfully to {save_path_trained}.")

        # 测试模型


    # 返回更新后的学生模型的参数
    # return best_model_state_mu_f_u,best_model_state_mu_u,smodel18.classifier.state_dict()
    return smodel18.feature_extractor.state_dict(),smodel18.state_dict(),smodel18.classifier.state_dict()
    # return None



