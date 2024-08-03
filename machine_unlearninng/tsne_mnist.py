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

from utils.Metric import AverageMeter, accuracy, Performance


import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
# 其他需要的库...
import os

# 创建目录


def tsne(forget_loader, retain_loader, qno_loader, tmodel18, model34, smodel18, u, f_u,num):
    print("tsne")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    smodel18.load_state_dict(u)
    smodel18.feature_extractor.load_state_dict(f_u)
    save_dir = './pathmnist/exp1noise/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 提取特征的函数
    # def extract_features(loader):
    #     features = []
    #     labels = []
    #     for X, Y in loader:
    #         X = X.to(device)
    #         with torch.no_grad():
    #             X = X.view(-1, 28 * 28 * 3)
    #             output = smodel18.features(X)
    #         features.append(output.cpu().numpy())
    #         labels.append(Y.cpu().numpy())
    #     return np.concatenate(features), np.concatenate(labels)

    def extract_features(loader):
         # Set the model to evaluation mode
        features = []
        labels = []
        for X, Y in loader:
            X = X.to(device)
            with torch.no_grad():
                output = smodel18(X)
            features.append(output.cpu().numpy())
            labels.append(Y.cpu().numpy())
        return np.concatenate(features),np.concatenate(labels)

    # 对每个数据集提取特征
    forget_features, forget_labels = extract_features(forget_loader)
    retain_features, retain_labels = extract_features(retain_loader)
    qno_features, qno_labels = extract_features(qno_loader)

    # 应用 t-SNE 进行降维
    all_features = np.concatenate([forget_features, retain_features, qno_features])
    all_labels = np.concatenate([forget_labels, retain_labels, qno_labels])
    tsne = TSNE(n_components=2, random_state=42)
    all_features_tsne = tsne.fit_transform(all_features)

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.scatter(all_features_tsne[:len(forget_features), 0], all_features_tsne[:len(forget_features), 1], c='red', alpha=0.5, label='Forget Data')
    plt.scatter(all_features_tsne[len(forget_features):len(forget_features)+len(retain_features), 0], all_features_tsne[len(forget_features):len(forget_features)+len(retain_features), 1], c='green', alpha=0.5, label='Retain Data')
    plt.scatter(all_features_tsne[-len(qno_features):, 0], all_features_tsne[-len(qno_features):, 1], c='blue', alpha=0.3, label='QNO Data')
    plt.legend()
    plt.title('t-SNE Visualization of Different Datasets')
    # plt.show()
    image_path = os.path.join(save_dir, f'tsne_plot_{num + 1}.png')
    plt.savefig(image_path)
    plt.close()
    return None
