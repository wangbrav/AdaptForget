import torch
import torch.nn.functional as F
import numpy as np
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
from KDloss import SoftTarget
from tqdm import tqdm
import torch.nn.functional as F
import os

from utils.Metric import AverageMeter, accuracy, Performance

def returnresultqf100(forget_loader, smodel18, u,f_u,num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    predictions = []
    save_dir = './pathmnist/qf100/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 确保模型在评估模式下，这对于某些模型来说很重要，比如使用了Dropout或BatchNorm的模型
    smodel18.eval()
    smodel18.load_state_dict(u)
    smodel18.feature_extractor.load_state_dict(f_u)

    for X, Y in tqdm(forget_loader):
        X = X.to(device)
        Y = Y.to(device)

        Y = Y.squeeze().long()  # 是path的

        snet_pred = smodel18(X)
        predictions.append(snet_pred.cpu().numpy())

    import numpy as np
    import matplotlib.pyplot as plt

    # 假设`predictions`是上一步中收集的预测结果列表
    # 我们将其转换为一个大的Numpy数组，方便处理
    predictions_np = np.concatenate(predictions, axis=0)

    # 从对数概率恢复到概率
    probabilities_np = np.exp(predictions_np)

    # 找到每个样本最可能的类别
    predicted_classes = np.argmax(probabilities_np, axis=1)

    # 统计每个类别的频率
    unique, counts = np.unique(predicted_classes, return_counts=True)
    frequency = dict(zip(unique, counts))

    # 绘制频率图
    plt.figure(figsize=(10, 6))
    plt.bar(frequency.keys(), frequency.values())
    plt.xlabel('类别')
    plt.ylabel('频率')
    plt.title('模型预测的类别频率图')
    plt.show()
    image_path = os.path.join(save_dir, f'tsne_plot_{num + 1}.png')
    plt.savefig(image_path)






