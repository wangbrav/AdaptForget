
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

# 转换all_labels和all_preds为NumPy数组以便使用它们
import numpy as np


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
# from KDloss import SoftTarget
from tqdm import tqdm

# 假设 retain_loader 和 forget_loader 已经被定义
# 假设 device, model34, smodel18, tmodel18, criterionCls, criterionKD, criterion 已经被定义

def test_model(test_loader, device, smodel18):
    # model34 = get_model()
    # model34=model34.to(device)
    # model34.load_state_dict(torch.load('best_model_mlp0111.pth'))

    # smodel18.load_state_dict(u)
    # smodel18.feature_extractor.load_state_dict(f_u)
    smodel18.eval()  # 设置为评估模式
    # tmodel18.eval()  # 设置为评估模式
    # model34.eval()   # 设置为评估模式

    correctt = 0
    totalt = 0
    correctu = 0
    totalu = 0
    correcttt = 0
    totaltt = 0
    all_labels = []
    all_preds = []

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
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs18s = smodel18(inputs)
            # outputs18t = tmodel18(inputs)
            _, predicted = torch.max(outputs18s.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            # print(labels)
            # print(predicted)
            # print(labels.size(0))
            totaltt += labels.size(0)
            labels = labels.squeeze()  #  在path中使用
            # print((predicted == labels).sum().item())
            # correcttt = (predicted == labels).sum().item()
            correcttt += (predicted == labels).sum().item()  #  计算 mnist 数据集的时候就没有问题

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels_np, all_preds_np)
    TP = conf_matrix[1, 1]  # 真正例
    FP = conf_matrix[0, 1]  # 假正例
    FN = conf_matrix[1, 0]  # 假负例
    f1 = 2 * TP / (2 * TP + FP + FN)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # print(f'Accuracy on retain data: {100 * correctt / totalt}%')
    print(f'Accuracy on test data: {100 * correcttt / totaltt}%')
    # print(f'Accuracy on forget data: {100 * correctu / totalu}%')
    return 100 * correcttt / totaltt,correcttt/totaltt
# 在实际使用中，您需要传递适当的参数来调用 test_model 函数。
# test_model(retain_loader, forget_loader, device, model34, smodel18, tmodel18)
