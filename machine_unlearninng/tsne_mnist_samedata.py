import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def tsne(loader,smodel,zmodel, device, save_dir, num,u,f_u):
    print("tsne")
    print("Using device:", device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    smodel.load_state_dict(u)
    smodel.feature_extractor.load_state_dict(f_u)
    zmodel.load_state_dict(torch.load('best_smodelcnntsne.pth'))


    def extract_features(model, loader):
        model.eval()  # Set the model to evaluation mode
        features = []
        for X, _ in loader:
            X = X.to(device)
            with torch.no_grad():
                output = model(X)
            features.append(output.cpu().numpy())
        return np.concatenate(features)

    # 对每个模型和数据集提取特征
    features_a = extract_features(smodel, loader)
    features_b = extract_features(zmodel, loader)

    # 应用 t-SNE 进行降维
    all_features = np.concatenate([features_a, features_b])
    tsne = TSNE(n_components=2, random_state=42)
    all_features_tsne = tsne.fit_transform(all_features)

    # 可视化
    plt.figure(figsize=(12, 8))
    plt.scatter(all_features_tsne[:len(features_a), 0], all_features_tsne[:len(features_a), 1], c='red', alpha=0.5, label='Model A')
    plt.scatter(all_features_tsne[len(features_a):, 0], all_features_tsne[len(features_a):, 1], c='blue', alpha=0.5, label='Model B')
    plt.legend()
    plt.title('t-SNE Visualization of Features Extracted by Two Models')
    image_path = os.path.join(save_dir, f'tsne_plot_{num + 1}.png')
    plt.savefig(image_path)
    plt.close()





