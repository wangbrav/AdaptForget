import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random
from mu.net_train_three import get_student_model,get_teacher_model

import torch.nn.init as init

from torch.utils.data import Dataset, DataLoader, Subset
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for X, Y in tqdm(test_loader):
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)
                # _, predicted = torch.max(outputs.data, 1)
                  # 调整Y的形状和数据类型

                pred = outputs.data.max(1)[1]
                # print(predicted.shape)
                # print(Y.shape)
                total += Y.size(0)

                Y = Y.squeeze(1).long()
                correct += pred.eq(Y.view(-1)).sum().item()
                # print(f'Batch correct: {(predicted == Y).sum().item()}, Batch total: {Y.size(0)}')  # 新增的打印语句
    print(f"Correct: {correct}, Total: {total}")
    accuracy = 100 * correct / total
    return accuracy



def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

npz_file_path = './data/retinamnist.npz'
data = np.load(npz_file_path)

print(list(data.keys()))  # 打印所有在npz文件中的数组名

# ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
images = data['train_images']
images_test =data['val_images']
labels_test =data['val_labels']
labels_cal =data['test_labels']
image_cal = data['test_images']
print(images.shape)
print(images_test.shape)
print(image_cal.shape)
labels = data['train_labels']

# 创建自定义数据集
class PathMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label




# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs


CONFIG = {
'BASE1': {
    'BASE' : list(range(0,1080))
},
'BASE2': {
    'BASE' : list(range(0,1079))
},
'BASE3': {
    'BASE' : list(range(0,1079))
},
'TEST1': {
    'TEST' : list(range(0,120))
},
'CAL_100': {
    'CAL' : list(range(40000,40100))
},
'CAL_1000': {
    'CAL' : list(range(40000,41000))
},
'CAL_2000': {
    'CAL' : list(range(40000,42000))
},
'CAL_5000': {
    'CAL' : list(range(40000,45000))
},
'CALTEST1': {
    'TEST' : list(range(46000,47000))
},
'QO_5000': {
    'QUERY': list(range(0,5000)),
    'QUERY_MEMBER': [1 for i in range(5000)]
    },
'QNO_5000': {
    'QUERY': list(range(50000,55000)),
    'QUERY_MEMBER': [0 for i in range(5000)]
    },
'QNO_2000': {
    'QUERY': list(range(50000,52000)),
    'QUERY_MEMBER': [0 for i in range(2000)]
    },
'QNO_1000': {
    'QUERY': list(range(50000,51000)),
    'QUERY_MEMBER': [0 for i in range(1000)]
    },
'QNO_100': {
    'QUERY': list(range(50000,50100)),
    'QUERY_MEMBER': [0 for i in range(100)]
    },
'QNO_10': {
    'QUERY': list(range(50000,50010)),
    'QUERY_MEMBER': [0 for i in range(10)]
    },
'QNO_1': {
    'QUERY': list(range(50000,50010)),
    'QUERY_MEMBER': [0 for i in range(10)]
    },
'QF_100':{
    'QUERY': list(range(9900,10000)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QF_10':{
    'QUERY': list(range(9990,10000)),
    'QUERY_MEMBER': [1 for i in range(10)]
},
'QF_1':{
    'QUERY': list(range(999,1000)),
    'QUERY_MEMBER': [1 for i in range(1)]
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
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = PathMNISTDataset(images, labels, transform=transform)
test_dataset = PathMNISTDataset(images_test, labels_test, transform=transform)

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
base3_loader = DataLoader(Subset(train_dataset, CONFIG['BASE3']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(test_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_1000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_2000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_2000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_5000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_5000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qo_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QO_5000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qno_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_5000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qno_2000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_2000']['QUERY']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qno_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1000']['QUERY']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qno_100_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_100']['QUERY']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))

qno_1_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1']['QUERY']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qno_10_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_10']['QUERY']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))

qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qf_1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
qf_10_loader = DataLoader(Subset(train_dataset, CONFIG['QF_10']['QUERY']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))

qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
# 参数设置
# 初始化FeatureExtractor和Classifier
# feature_extractor = FeatureExtractor(dim_in=28, dim_hidden=256)  # 注意这里假设输入尺寸为28*28*3
# classifier = Classifier(dim_hidden=256, dim_out=10)

# 创建组合模型实例
# model =get_teacher_model().to(device)
model =get_student_model().to(device)

# 其余的代码保持不变

# 参数设置
epochs = 40
learning_rate = 0.001

# 数据加载和预处理

best_accuracy=0
best_model_state = None

# 初始化模型


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 训模型
for epoch in range(epochs):
    model.train()
    correct = 0
    train_loss = 0

    with tqdm(total=len(base2_loader)) as t:
        for X, Y in tqdm(base2_loader):
            X = X.to(device)
            Y = Y.to(device)

            # Training pass
            optimizer.zero_grad()

            output = model(X)
            # print(output)
            # output = torch.argmax(output, dim=1)  # 假设Y是独热编码，转换成类别索引
            # print(output.shape)  # 检查Y的形状
            # print(Y.shape)
            # print(Y)
            Y = Y.squeeze(1).long()  # 调整Y的形状和数据类型
            # print(output)
            # print(Y)
            loss = criterion(output, Y)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            pred = output.data.max(1)[1]
            correct += pred.eq(Y.view(-1)).sum().item()

    train_loss /= len(base2_loader.dataset)
    train_accuracy = 100 * correct / len(base2_loader.dataset)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # 测试模型
    test_accuracy = test(model, test1_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # 保存最佳模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_state = model.state_dict().copy()

# 保存最佳模型权重
torch.save(best_model_state, 'best_model_retinanet_base2.pth')
print(f"Best Test Accuracy: {best_accuracy:.2f}%")
