import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
from mlp import get_teacher_model
# 定义MLP模型

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
    'BASE' : list(range(0,10000))
},
'BASE2': {
    'BASE' : list(range(0,9900))
},
'BASE3': {
    'BASE' : list(range(0,9000))
},

'TEST1': {
    'TEST' : list(range(30000,40000))
},
'CAL_100': {
    'CAL' : list(range(10000,10100))
},
'CAL_1000': {
    'CAL' : list(range(10000,11000))
},
'CAL_2000': {
    'CAL' : list(range(10000,12000))
},
'CAL_5000': {
    'CAL' : list(range(10000,15000))
},
'CALTEST1': {
    'TEST' : list(range(30000,31000))
},
'QO_5000': {
    'QUERY': list(range(0,5000)),
    'QUERY_MEMBER': [1 for i in range(5000)]
    },
'QNO_2000': {
    'QUERY': list(range(20000,22000)),
    'QUERY_MEMBER': [0 for i in range(2000)]
    },
'QNO_1000': {
    'QUERY': list(range(20000,21000)),
    'QUERY_MEMBER': [0 for i in range(1000)]
    },
'QF_100':{
    'QUERY': list(range(9900,10000)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QF_1000':{
    'QUERY': list(range(9000,10000)),
    'QUERY_MEMBER': [1 for i in range(1000)]
},
'KD0.25': {
    'BASE': list(range(0,2500))
},
'KD0.5': {
    'BASE': list(range(0,5000))
},
'KD0.75': {
    'BASE': list(range(0,7500))
},
}


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
# 设置数据转换，这里仅进行基础的转换和标准化

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 下载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
base3_loader = DataLoader(Subset(train_dataset, CONFIG['BASE3']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_1000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_2000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_2000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_5000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_5000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qo_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QO_5000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qno_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_2000']['QUERY']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qno_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1000']['QUERY']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))

qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))


# 参数设置
epochs = 20
learning_rate = 0.001

# 数据加载和预处理
best_accuracy=0
best_model_state = None

# 初始化模型
model = get_teacher_model()
model = model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 训模型
for epoch in range(epochs):
    model.train()
    correct = 0
    train_loss = 0

    with tqdm(total=len(base3_loader)) as t:
        for X, Y in tqdm(base3_loader):
            X = X.to(device)
            Y = Y.to(device)

            # Training pass
            optimizer.zero_grad()

            output = model(X)
            loss = criterion(output, Y)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            pred = output.data.max(1)[1]
            correct += pred.eq(Y.view(-1)).sum().item()

    train_loss /= len(base3_loader.dataset)
    train_accuracy = 100 * correct / len(base3_loader.dataset)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # 测试模型
    test_accuracy = test(model, test1_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # 保存最佳模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_state = model.state_dict().copy()

# 保存最佳模型权重
# torch.save(best_model_state, 'best_model_mlp0111.pth')
print(f"Best Test Accuracy: {best_accuracy:.2f}%")
