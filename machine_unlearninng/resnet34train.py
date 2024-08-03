import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet18 import ResNet18
from resnet34 import ResNet34
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
import  random

from  resnet34 import ResNet34


CONFIG = {
'BASE1': {
    'BASE' : list(range(0,10000))
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
'QNO_100': {
    'QUERY': list(range(20000,20100)),
    'QUERY_MEMBER': [0 for i in range(100)]
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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_1000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_2000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_2000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_5000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_5000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qo_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QO_5000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qno_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_2000']['QUERY']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qno_100_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_100']['QUERY']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))

qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))

model = ResNet34().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    for data, target in base1_loader:
        data, target = data.to(device), target.to(device)  # 将数据和目标转移到设备上
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test1_loader:
            data, target = data.to(device), target.to(device)  # 将数据和目标转移到设备上

            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch}, Accuracy: {accuracy}%')

    # 保存最佳模型权重
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model34.pth')
