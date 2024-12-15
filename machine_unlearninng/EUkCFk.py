import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# 定义模型类
class FeatureExtractor(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),  # 直接使用24个特征作为输入
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.network(x)


class Classifier(nn.Module):
    def __init__(self, dim_hidden, dim_out):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.LogSoftmax(dim=-1)  # 使用LogSoftmax进行分类
        )

    def forward(self, x):
        return self.network(x)


class CombinedModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(CombinedModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x)  # 直接提取特征
        output = self.classifier(features)
        return output

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
# 数据集类
class TableDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.data_frame.iloc[idx, 1:-1].astype(np.float32).values  # 假设所有特征都是数值型
        labels = self.data_frame.iloc[idx, -1].astype(np.int64)

        if self.transform:
            features = self.transform(features)

        return torch.tensor(features), torch.tensor(labels)


# 标准化转换
def transform(features):
    return (features - features.mean()) / features.std()


# 加载数据集
csv_file = '/root/autodl-tmp/wangbin/yiwang/data/final.csv'
train_dataset = TableDataset(csv_file=csv_file, transform=transform)
test1_loader = DataLoader(Subset(train_dataset, list(range(0, 500))), batch_size=32, shuffle=False)
test1_loader = DataLoader(Subset(train_dataset, list(range(500, 600))), batch_size=32, shuffle=False)
cal_50_loader = DataLoader(Subset(train_dataset, list(range(192, 242))), batch_size=64, shuffle=False)

# 初始化模型
feature_extractor = FeatureExtractor(dim_in=24, dim_hidden=256)
classifier = Classifier(dim_hidden=256, dim_out=2)
modelmlp = CombinedModel(feature_extractor, classifier).to(device)

# 加载预训练模型
modelmlp.load_state_dict(torch.load('best_model_mlp0330_wu.pth'))

# 重新初始化最后一层分类器
modelmlp.classifier.network[0] = nn.Linear(256, 2).to(device)
modelmlp.classifier.network[1] = nn.LogSoftmax(dim=-1).to(device)

# 冻结特征提取器的所有层
for param in modelmlp.feature_extractor.parameters():
    param.requires_grad = False

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(modelmlp.classifier.parameters(), lr=0.001)

# 训练最后一层
modelmlp.train()
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(cal_50_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = modelmlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(cal_50_loader)}')


# 测试重新训练最后一层的模型
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


accuracy_retrained = test(modelmlp, test1_loader, device)
print(f'Retrained Classifier Test Accuracy: {accuracy_retrained}%')

# 微调最后一层
# 这里我们仅微调最后一层，所以不解冻特征提取器层
optimizer = optim.Adam(modelmlp.classifier.parameters(), lr=0.0001)

modelmlp.train()
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(cal_50_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = modelmlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(cal_50_loader)}')

# 测试微调后的模型
accuracy_finetuned = test(modelmlp, test1_loader, device)
print(f'Finetuned Classifier Test Accuracy: {accuracy_finetuned}%')
