import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax
from puri import api
# 定义简单的神经网络
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import sys
sys.path.append('/root/autodl-tmp/wangbin/yiwang')
from torch.utils.data import ConcatDataset, DataLoader, Subset

from unlearning_random_afs_rand import train_student_model_random
from Domainadaptation_net_three import domainadaptation
from test_model_path import test_model
from puri import api
import random
# from mu.net_train_three import get_student_model, get_teacher_model
from torch.utils.data import Dataset, DataLoader, Subset
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from plot_with_new_data import update_plot_with_new_data
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
# from tsne_mnist_samedata import tsne
# from tsne_mnist_tuo import tsne
from tsne_mnist_guding1 import tsnet
from tsne_mnist_guding2 import tsnes
# 相比上一个  更改了 数据集的划分方法
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
import os
from torch.utils.data import Dataset
import pandas as pd

from functions import ReverseLayerF
import torch.nn as nn
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score as calculate_f12_score

class FeatureExtractor(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),  # 直接使用8个特征作为输入
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.network(x)

class Classifier(nn.Module):
    def __init__(self, dim_hidden, dim_out,):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.LogSoftmax(dim=-1)         # 使用LogSoftmax进行分类
        )

    def forward(self, x):
        return self.network(x)


class CombinedModel(nn.Module):
    def __init__(self, feature_extractor, classifier,num_classes=2):
        super(CombinedModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.classifier.num_classes = num_classes

    def forward(self, x):
        features = self.feature_extractor(x)  # 直接提取特征
        output = self.classifier(features)
        return output
def get_model():
    # return MLP(57, 128, 2) # death
    # return MLP(21, 128, 3) # diabetes
    feature_extractortt = FeatureExtractor(dim_in=21, dim_hidden=256)
    classifierstt = Classifier(dim_hidden=256, dim_out=2)

    # 创建组合模型实例

    return CombinedModel(feature_extractortt, classifierstt)# autism


# TODO xiangbi shangyige  zengjia le  afs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")

# 加载npz文件
import torch.nn.init as init
class AutismDataset(Dataset):
    def __init__(self):
        # 读取 CSV 文件
        self.df = pd.read_csv('/root/autodl-tmp/wangbin/yiwang/data/diabetes_binary_train.csv')

        # 提取特征和标签
        self.original_X = np.array(self.df.iloc[:, :-1])
        self.original_Y = np.array(self.df.iloc[:, -1])

        # 复制到当前使用的数据
        self.X = self.original_X.copy()
        self.Y = self.original_Y.copy()
        # 检查数据集是否为空
        if self.X is None or self.Y is None:
            raise ValueError("Dataset initialization failed!")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return np.float32(self.X[index]), self.Y[index]

    def shuffle_data(self, seed= 52 ):
        # 打印信息
        print(f"Shuffling data with seed {seed}")

        # 设置随机种子以确保一致性
        np.random.seed(seed)

        # 生成索引并随机打乱
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        if indices is None:
            raise ValueError("Shuffle failed!")

        # 根据打乱的索引重新排序数据
        self.X = self.original_X[indices]
        self.Y = self.original_Y[indices]
# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs
# random_seed = 10
# random_seed = 23
# random_seed = 52
# random_seed = 32
random_seed = 32
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


# npz_file_path = '/root/autodl-tmp/wangbin/yiwang/data/modified_raw_balanced_dataset.csv'
# data = np.load(npz_file_path)

# print(list(data.keys()))  # 打印所有在npz文件中的数组名
#
# # ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
# images = data['train_images']
# labels = data['train_labels']
# images_cal = data['val_images']
# labels_cal = data['val_labels']
# images_test = data['test_images']
# labels_test = data['test_labels']
# 定义简单的神经网络

'''
新版config
'''
CONFIG = {
'BASE1': {
    'BASE' : list(range(0,20000))
},
'TEST1': {
    'TEST' : list(range(20000,25000))
},
'CAL_100': {
    'CAL' : list(range(40000,40100))
},
'CAL_1000': {
    'CAL' : list(range(25000,26000))
},
'CAL_2000': {
    'CAL' : list(range(40000,42000))
},
'CAL_5000': {
    'CAL' : list(range(40000,45000))
},
'CALTEST1': {
    'TEST' : list(range(26000,27000))
},
'QO_5000': {
    'QUERY': list(range(0,5000)),
    'QUERY_MEMBER': [1 for i in range(5000)]
    },
'QNO_5000': {
    'QUERY': list(range(50000,55000)),
    'QUERY_MEMBER': [0 for i in range(5000)]
    },
'QF_100':{
    'QUERY': list(range(5900,6000)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QF_1000':{
    'QUERY': list(range(9000,10000)),
    'QUERY_MEMBER': [1 for i in range(1000)]
},
'KD0.25': {
    'BASE': list(range(0,5000))
},
'KD0.5': {
    'BASE': list(range(0,10000))
},
'KD0.75': {
    'BASE': list(range(0,15000))
},
}




def calculate_tp_fp_fn(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, fp, fn

def calculate_f1_score(y_true, y_pred):
    tp, fp, fn = calculate_tp_fp_fn(y_true, y_pred)
    if 2 * tp + fp + fn == 0:
        return 0  # 避免除以0的错误
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")

'''
数据转换

'''
train_dataset = AutismDataset()
train_dataset.shuffle_data()


base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))

test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=64, shuffle=False,
                            generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_1000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
cal_2000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_2000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
cal_5000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_5000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
qo_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QO_5000']['QUERY']), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
qno_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_5000']['QUERY']), batch_size=64, shuffle=True,
                             generator=torch.Generator().manual_seed(random_seed))





qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=16, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))



qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))


import numpy as np

def adjust_shards(shards):
    # 检查每个分片
    for i in range(len(shards)):
        if len(shards[i]) % 32 == 1:
            # 如果当前分片大小不符合要求，尝试调整
            if i < len(shards) - 1:  # 确保不是最后一个分片
                # 从下一个分片中借一个元素
                shards[i] = np.append(shards[i], shards[i+1][0])
                shards[i+1] = shards[i+1][1:]
            else:
                # 如果是最后一个分片，从前一个分片借
                shards[i-1] = np.append(shards[i-1], shards[i][0])
                shards[i] = shards[i][1:]
    return shards

# 首先生成随机打乱的索引
indices = np.random.permutation(len(Subset(train_dataset, CONFIG['BASE1']['BASE'])))

# 分片
shards = np.array_split(indices, 10)

# 调整分片以确保每个分片的大小不满足 len(shard) % 32 == 1
shards = adjust_shards(shards)

# 划分数据为三个片段
# shards = np.array_split(np.random.permutation(len(Subset(train_dataset, CONFIG['BASE1']['BASE']))), 10)
print(len(Subset(train_dataset, CONFIG['BASE1']['BASE']))   )
models = []
accuracies = []
for i, shard in enumerate(shards):
    print(f"Length of shard {i+1}: {len(shard)}")
# 训练模型并计算准确率
for shard in shards:
    # X_shard = X[shard]
    # y_shard = y[shard]
    subset = Subset(Subset(train_dataset, CONFIG['BASE1']['BASE']), shard)
    # dataset = TensorDataset(X_shard, y_shard)
    # loader = DataLoader(dataset, batch_size=5, shuffle=True)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(random_seed))

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    for epoch in range(100):
        for data, target in dataloader:
            data = data.to(device)
            target= target.squeeze().long().to(device)  # 是path的
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            # print(target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 计算准确率
    correcttt = 0
    totaltt = 0
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():  # 确保不更新梯度
        for data in test1_loader:
            # inputs, labels = data[0], data[1]
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs18s = model(inputs)
            # outputs18t = tmodel18(inputs)
            _, predicted = torch.max(outputs18s.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            # print(labels)
            # print(predicted)
            # print(labels.size(0))
            totaltt += labels.size(0)
            labels = labels.squeeze()  # 在path中使用
            # print((predicted == labels).sum().item())
            # correcttt = (predicted == labels).sum().item()
            correcttt += (predicted == labels).sum().item()  # 计算 mnist 数据集的时候就没有问题

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
    accuracies.append(correcttt / totaltt)

    models.append(model)

# 根据准确率计算权重
weights = torch.tensor(accuracies) / sum(accuracies)
# print(weights)

def aggregate_predictions(models, X_test, weights):
    with torch.no_grad():
        # 使用加权平均合并模型的 logits
        weighted_logits = sum(weight * model(X_test) for model, weight in zip(models, weights))
        # 计算最终的概率分布
        probabilities = softmax(weighted_logits, dim=1)
        final_prediction = probabilities.max(1)[1]
    return final_prediction

# 对测试数据进行预测
# X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.3)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# predicted = aggregate_predictions(models, X_test, weights)
# print("Initial Accuracy:", accuracy_score(y_test, predicted.numpy()))

# 假设我们需要遗忘第一个片段的数据
# indices_to_forget = shards[0][:10]
# print(shards[0])
member_gt = [1 for i in range(100)]
_t, pv, EMA_res, risk_score = api(device, models, weights, qf_100_loader, member_gt, cal_1000_loader, caltest1_loader)
print("EMA_res:", EMA_res, "risk_score:", risk_score, "pvalue:", pv)

# 更新shards[0]，排除需要遗忘的数据
# shards[0] = np.setdiff1d(shards[0], indices_to_forget)
forget_set = set(CONFIG['QF_100']['QUERY'])
updated_shards = []
for shard in shards:
    shard_set = set(shard)
    updated_shard = list(shard_set - forget_set)
    updated_shards.append(updated_shard)
updated_shards = adjust_shards(updated_shards)
# 使用更新后的shards[0]重新训练模型
for i, shard in enumerate(updated_shards):
    print(f"Length of shard {i+1}: {len(shard)}")
# dataset_forgotten = TensorDataset(X_train_forgotten, y_train_forgotten)
# loader_forgotten = DataLoader(dataset_forgotten, batch_size=5, shuffle=True)

# model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
for i, shard in enumerate(updated_shards):
    # X_shard = X[shard]
    # y_shard = y[shard]
    subset = Subset(Subset(train_dataset, CONFIG['BASE1']['BASE']), shard)
    # dataset = TensorDataset(X_shard, y_shard)
    # loader = DataLoader(dataset, batch_size=5, shuffle=True)
    dataloader = DataLoader(subset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(random_seed))

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    for epoch in range(100):
        for data, target in dataloader:
            data  = data.to(device)
            target= target.squeeze().long().to(device)  # 是path的
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            # print(target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 计算准确率
    correcttt = 0
    totaltt = 0
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():  # 确保不更新梯度
        for data in test1_loader:
            # inputs, labels = data[0], data[1]
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs18s = model(inputs)
            # outputs18t = tmodel18(inputs)
            _, predicted = torch.max(outputs18s.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            # print(labels)
            # print(predicted)
            # print(labels.size(0))
            totaltt += labels.size(0)
            labels = labels.squeeze()  # 在path中使用
            # print((predicted == labels).sum().item())
            # correcttt = (predicted == labels).sum().item()
            correcttt += (predicted == labels).sum().item()  # 计算 mnist 数据集的时候就没有问题

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
    accuracies.append(correcttt / totaltt)

    models[i-1] = model

# 根据准确率计算权重
weights = torch.tensor(accuracies) / sum(accuracies)



# 重新评估第一个模型的准确率
# model.eval()
# with torch.no_grad():
#     predictions = model(X_train_forgotten).max(1)[1]
#     accuracy = (predictions == y_train_forgotten).float().mean().item()
# accuracies[0] = accuracy
# weights = torch.tensor(accuracies) / sum(accuracies)

# 重新进行聚合预测
predicted = []
zY = []
with torch.no_grad():
    for X, Y in test1_loader:
        X = X.to(device)
        Y = Y.to(device)
        out_Y = aggregate_predictions(models, X, weights)
        predicted.append(out_Y)
        Y = Y.squeeze().long()  # pathmnist
        zY.append(Y)
predicted = torch.cat(predicted).cpu().numpy()
zY = torch.cat(zY).cpu().numpy()

# predicted = aggregate_predictions(models, X_test, weights)
print("Updated Accuracy:", accuracy_score(zY, predicted))
print("F1 Score (Macro):", calculate_f12_score(zY, predicted, average='macro'))

f1_score = calculate_f1_score(zY, predicted)
print("F1 Score:", f1_score)

# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.int64)
# dataset_test = TensorDataset(X_test,y_test)
# loader_test = DataLoader(dataset_test, batch_size=5, shuffle=True)
member_gt = [1 for i in range(100)]
_t, pv, EMA_res, risk_score = api(device, models, weights, qf_100_loader, member_gt, cal_1000_loader, caltest1_loader)
print("EMA_res:", EMA_res, "risk_score:", risk_score, "pvalue:", pv)

# 其余部分不变，这里省略
