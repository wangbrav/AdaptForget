import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def tsne(forget_loader, retain_loader, tmodel18, model34, smodel18, u, f_u, num):
    print("tsne")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    smodel18.load_state_dict(u)
    smodel18.feature_extractor.load_state_dict(f_u)
    save_dir = './pathmnist/exp1zong/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def extract_features(loader):
        features = []
        labels = []
        for X, Y in loader:
            X = X.to(device)
            with torch.no_grad():
                output = model34(X)
            features.append(output.cpu().numpy())
            labels.append(Y.cpu().numpy())
        return np.concatenate(features), np.concatenate(labels)

    # 对每个数据集提取特征
    forget_features, forget_labels = extract_features(forget_loader)
    retain_features, retain_labels = extract_features(retain_loader)

    # 应用 t-SNE 进行降维，去除了QNO数据集的处理
    all_features = np.concatenate([forget_features, retain_features])
    all_labels = np.concatenate([forget_labels, retain_labels])
    tsne = TSNE(n_components=2, random_state=42)
    all_features_tsne = tsne.fit_transform(all_features)

    # 可视化，去除了蓝色的数据集即QNO数据集的可视化
    plt.figure(figsize=(12, 8))
    plt.scatter(all_features_tsne[:len(forget_features), 0], all_features_tsne[:len(forget_features), 1], c='red', alpha=0.5, label='Forget Data')
    plt.scatter(all_features_tsne[len(forget_features):, 0], all_features_tsne[len(forget_features):, 1], c='grey', alpha=0.5, label='Retain Data')
    plt.legend()
    plt.title('t-SNE Visualization of Forget and Retain Datasets')
    image_path = os.path.join(save_dir, f'tsne_plot_{num + 1}.png')
    plt.savefig(image_path)
    plt.close()
    return None
