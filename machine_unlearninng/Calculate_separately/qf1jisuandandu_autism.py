import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
import sys
sys.path.append('/root/autodl-tmp/wangbin/yiwang')
from mlp_three_csv_wu import CombinedModel,FeatureExtractor,Classifier
# from mlp_three_csv_wu import CombinedModel,FeatureExtractor,Classifier
import torch.nn as nn
from tqdm import tqdm
# from mu.net_train import get_model
import torch.optim as optim
from torch.utils.data import  DataLoader, Subset
from torchvision import transforms
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

CONFIG = {
'BASE1': {
    'BASE' : list(range(0,500))
},
'TEST1': {
    'TEST' : list(range(500,600))
},
'CAL_100': {
    'CAL' : list(range(500,600))
},
'CALTEST1': {
    'TEST' : list(range(600,700))
},
'QO_300': {
    'QUERY': list(range(0,400)),
    'QUERY_MEMBER': [1 for i in range(300)]
    },
'QNO_300': {
    'QUERY': list(range(700,1000)),
    'QUERY_MEMBER': [0 for i in range(300)]
    },
'QF_50':{
    'QUERY': list(range(450,500)),
    'QUERY_MEMBER': [1 for i in range(50)]
},

'QF_1':{
    'QUERY': list(range(499,500)),
    'QUERY_MEMBER': [1 for i in range(1)]
},
'QF_100':{
    'QUERY': list(range(400,500)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QS':{
    'QUERY': None,
    'QUERY_MEMBER': None
},
'KD0.25': {
    'BASE': list(range(0,125))
},
'KD0.5': {
    'BASE': list(range(0,250))
},
'KD0.75': {
    'BASE': list(range(0,375))
},
}

best_accuracy=0

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")



# 定义标准化转换
def transform(features):
    # 定义添加噪声的转换
    return (features - features.mean()) / features.std()
def transform_no(features):
    # 在原始数据的每一个特征值上随意赋值0-10的随机数值
    noise = np.random.randint(0, 11, size=features.shape)
    return features + noise

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 目前仅进行简单的归一化处理，可以根据需要进行更复杂的处理(标准化等)



class TableDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.data_frame.iloc[idx, 1:-1].astype(np.float32).values  # 假设所有特征都是数值型(8个特征从第二列开始算)
        labels = self.data_frame.iloc[idx, -1].astype(np.int64)

        if self.transform:
            features = self.transform(features)

        return torch.tensor(features), torch.tensor(labels)
# class AutismDataset(Dataset):
#     def __init__(self,transform=None):
#         # 读取 CSV 文件
#         # self.df = pd.read_csv('/mnt/f/share/AFS-main/template/Autism/final.csv')
#
#         # 提取特征和标签
#         self.original_X = np.array(self.df.iloc[:, :-1])
#         self.original_Y = np.array(self.df.iloc[:, -1])
#
#         # 复制到当前使用的数据
#         self.X = self.original_X.copy()
#         self.Y = self.original_Y.copy()
#         # 检查数据集是否为空
#         if self.X is None or self.Y is None:
#             raise ValueError("Dataset initialization failed!")
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, index):
#         return np.float32(self.X[index]), self.Y[index]
#
#     def shuffle_data(self, seed=32 ):
#         # 打印信息
#         print(f"Shuffling data with seed {seed}")
#
#         # 设置随机种子以确保一致性
#         np.random.seed(seed)
#
#         # 生成索引并随机打乱
#         indices = np.arange(len(self.X))
#         np.random.shuffle(indices)
#         if indices is None:
#             raise ValueError("Shuffle failed!")
#
#         # 根据打乱的索引重新排序数据
#         self.X = self.original_X[indices]
#         self.Y = self.original_Y[indices]
# 加载数据集

csv_file = '/root/autodl-tmp/wangbin/yiwang/data/final.csv'
train_dataset = TableDataset(csv_file=csv_file,transform=transform)
# train_dataset_no = TableDataset(csv_file=csv_file, transform=transform_no)
# test_dataset = TableDataset(csv_file=csv_file, transform=transform)

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_50']['QUERY']), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
# cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_50']['CAL']), batch_size=64, shuffle=False,
#                              generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))

# 初始化模型
feature_extractor = FeatureExtractor(dim_in=24, dim_hidden=64)
classifier = Classifier(dim_hidden=64, dim_out=2)
remodelmlp = CombinedModel(feature_extractor, classifier)
remodelmlp = remodelmlp.to(device)
state_dict = torch.load('../best_model_mlp0330_wu_Remove_batch_qf1chongxun.pth')
print(state_dict.keys())

model_state_dict = remodelmlp.state_dict()
print('zhende',model_state_dict.keys())

# 参数设置
remodelmlp.load_state_dict(torch.load('../best_model_mlp0330_wu_Remove_batch_qf1chongxun.pth'))  # 需要使用改后的"a_mlp_train_three.py"训练得到

import torch
from torch.utils.data import DataLoader, Subset

# 假设我们已经有了模型和数据集
# modelmlp已经定义并加载了相应的权重
remodelmlp.eval()
# 加载 qf1 数据点
qf1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=1, shuffle=False)
qf1_data, qf1_label = next(iter(qf1_loader))  # 获取 qf1 中的数据和标签

# 提取 qf1 的特征向量
qf1_data = qf1_data.to(device)
qf1_feature = remodelmlp.feature_extractor(qf1_data)

# 从 BASE 数据集中找到同类别的样本
base_indices = CONFIG['BASE1']['BASE']
base_loader = DataLoader(Subset(train_dataset, base_indices), batch_size=len(base_indices), shuffle=False)
base_data, base_labels = next(iter(base_loader))

# 过滤出同类别的样本
same_class_indices = base_labels == qf1_label
same_class_data = base_data[same_class_indices]

max_samples = 30
if len(same_class_data) > max_samples:
    # 如果超过30个，随机选择30个样本
    selected_indices = torch.randperm(len(same_class_data))[:max_samples]
    same_class_data = same_class_data[selected_indices]
elif len(same_class_data) > 0:
    # 少于30个但非零，使用所有同类样本
    same_class_data = same_class_data
else:
    # 没有找到同类样本
    print("No same class samples found in BASE")
    same_class_data = None

# 如果有符合条件的同类样本，则提取特征并计算距离
if same_class_data is not None and len(same_class_data) > 0:
    same_class_data = same_class_data.to(device)
    same_class_features = remodelmlp.feature_extractor(same_class_data)



    euclidean_distance = torch.norm(qf1_feature - same_class_features, dim=1)
    average_euclidean = euclidean_distance.mean().item()
    print("Average Euclidean distance:", average_euclidean)
    # 计算两个特征向量之间的曼哈顿距离
    manhattan_distance = torch.sum(torch.abs(qf1_feature - same_class_features), dim=1)
    average_manhattan = manhattan_distance.mean().item()
    print("Average Manhattan distance:", average_manhattan)
    # 计算两个特征向量之间的余弦相似度
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarity = cos(qf1_feature, same_class_features)
    average_cosine_similarity = cosine_similarity.mean().item()
    print("Average Cosine Similarity:", average_cosine_similarity)