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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")

# 定义简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(10, 3)  # 隐藏层到输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 加载数据
data = load_iris()
X = data.data
y = data.target

# 将numpy数组转换为torch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

# 划分数据为三个片段
shards = np.array_split(np.random.permutation(len(X)), 3)
models = []
accuracies = []

# 训练模型并计算准确率
for shard in shards:
    X_shard = X[shard]
    y_shard = y[shard]
    dataset = TensorDataset(X_shard, y_shard)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    model.train()
    for epoch in range(50):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 计算准确率
    model.eval()
    with torch.no_grad():
        predictions = model(X_shard).max(1)[1]
        accuracy = (predictions == y_shard).float().mean().item()
        accuracies.append(accuracy)

    models.append(model)

# 根据准确率计算权重
weights = torch.tensor(accuracies) / sum(accuracies)

def aggregate_predictions(models, X_test, weights):
    with torch.no_grad():
        # 使用加权平均合并模型的 logits
        weighted_logits = sum(weight * model(X_test) for model, weight in zip(models, weights))
        # 计算最终的概率分布
        probabilities = softmax(weighted_logits, dim=1)
        final_prediction = probabilities.max(1)[1]
    return final_prediction

# 对测试数据进行预测
X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.3)
X_test = torch.tensor(X_test, dtype=torch.float32)
predicted = aggregate_predictions(models, X_test, weights)
print("Initial Accuracy:", accuracy_score(y_test, predicted.numpy()))

# 假设我们需要遗忘第一个片段的数据
indices_to_forget = shards[0][:10]
print(shards[0])

# 更新shards[0]，排除需要遗忘的数据
shards[0] = np.setdiff1d(shards[0], indices_to_forget)

# 使用更新后的shards[0]重新训练模型
X_train_forgotten = X[shards[0]]
print(X_train_forgotten)
y_train_forgotten = y[shards[0]]
print(y_train_forgotten)
dataset_forgotten = TensorDataset(X_train_forgotten, y_train_forgotten)
loader_forgotten = DataLoader(dataset_forgotten, batch_size=5, shuffle=True)

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(50):
    for data, target in loader_forgotten:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

models[0] = model

# 重新评估第一个模型的准确率
model.eval()
with torch.no_grad():
    predictions = model(X_train_forgotten).max(1)[1]
    accuracy = (predictions == y_train_forgotten).float().mean().item()
accuracies[0] = accuracy
weights = torch.tensor(accuracies) / sum(accuracies)

# 重新进行聚合预测
predicted = aggregate_predictions(models, X_test, weights)
print("Updated Accuracy:", accuracy_score(y_test, predicted.numpy()))
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int64)
dataset_test = TensorDataset(X_test,y_test)
loader_test = DataLoader(dataset_test, batch_size=5, shuffle=True)
member_gt = [1 for i in range(10)]
_t, pv, EMA_res, risk_score = api(device, models, weights, loader_forgotten, member_gt, loader_test, loader_test)
print("EMA_res:", EMA_res, "risk_score:", risk_score, "pvalue:", pv)

# 其余部分不变，这里省略
