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
from sklearn.metrics import accuracy_score, f1_score as calculate_f12_score

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


from functions import ReverseLayerF
import torch.nn as nn
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden, num_classes=9):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),  #后续添加的
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)





class Net(nn.Module):
    def __init__(self, in_channels, hidden, num_classes=9):
        super(Net, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels)
        self.classifier = Classifier(hidden, num_classes)


    def forward(self, x, alpha=None):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)

        return class_output

def get_model():
    return Net(3,256,5)
    # return Net(3,128,9)

def get_teacher_model():
    return Net(3,256,5)

def get_student_model():
    return Net(3,32,5)
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

# TODO xiangbi shangyige  zengjia le  afs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")

# 加载npz文件
import torch.nn.init as init
class PathMNISTDataset(Dataset):
    def __init__(self, images, labels,transform=None):
        self.original_images = images
        self.original_labels = labels
        self.images = images.copy()
        self.labels = labels.copy()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 返回当前索引下的图像和标签
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def shuffle_data(self, seed=42):
        # 设置随机种子以确保打乱顺序的一致性
        print(f"Shuffling data with seed {seed}") # 这里的seed是一个局部变量 有没有都可以 做的实验是这样的
        np.random.seed(seed)
        indices = np.arange(len(self.original_images))
        np.random.shuffle(indices)
        self.images = self.original_images[indices]
        self.labels = self.original_labels[indices]
        # 如果提供了转换函数，对每个图像进行转换
        # if self.transform:
        #     # 逐个图像应用转换
        #     self.images = np.array([self.transf  orm(image) for image in self.images])
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


npz_file_path = '/root/autodl-tmp/wangbin/yiwang/data/retinamnist.npz'
data = np.load(npz_file_path)

print(list(data.keys()))  # 打印所有在npz文件中的数组名

# ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
images = data['train_images']
labels = data['train_labels']
images_cal = data['val_images']
labels_cal = data['val_labels']
images_test = data['test_images']
labels_test = data['test_labels']
# 定义简单的神经网络

CONFIG = {
    'BASE1': {
        'BASE': list(range(0, 1080))
    },
    'BASE2': {
        'BASE': list(range(0, 9900))
    },
    'TEST1': {
        'TEST': list(range(0, 120))
    },
    'CAL_100': {
        'CAL': list(range(40000, 40100))
    },
    'CAL_200': {
        'CAL': list(range(0, 200))
    },
    'CAL_1000': {
        # 2005
        'CAL': list(range(0, 1000))
    },
    'CAL_2000': {
        'CAL': list(range(40000, 42000))
    },
    'CAL_5000': {
        'CAL': list(range(40000, 45000))
    },
    'CALTEST1': {
        'TEST': list(range(200, 400))
    },
    'QO_5000': {
        'QUERY': list(range(0, 5000)),
        'QUERY_MEMBER': [1 for i in range(5000)]
    },
    'QNO_5000': {
        'QUERY': list(range(50000, 55000)),
        'QUERY_MEMBER': [0 for i in range(5000)]
    },
    'QNO_2000': {
        'QUERY': list(range(50000, 52000)),
        'QUERY_MEMBER': [0 for i in range(2000)]
    },
    'QNO_1000': {
        'QUERY': list(range(50000, 51000)),
        'QUERY_MEMBER': [0 for i in range(1000)]
    },
    'QNO_100': {
        'QUERY': list(range(50000, 50100)),
        'QUERY_MEMBER': [0 for i in range(100)]
    },
    'QNO_10': {
        'QUERY': list(range(50000, 50010)),
        'QUERY_MEMBER': [0 for i in range(10)]
    },
    'QNO_1': {
        'QUERY': list(range(50000, 50010)),
        'QUERY_MEMBER': [0 for i in range(10)]
    },
    'QF_100': {
        'QUERY': list(range(980, 1080)),
        'QUERY_MEMBER': [1 for i in range(100)]
    },
    'QF_10': {
        'QUERY': list(range(9990, 10000)),
        'QUERY_MEMBER': [1 for i in range(10)]
    },
    'QF_1': {
        'QUERY': list(range(7006, 7007)),
        'QUERY_MEMBER': [1 for i in range(1)]
    },
    'QF_1000': {
        'QUERY': list(range(80, 1080)),
        'QUERY_MEMBER': [1 for i in range(1000)]
    },
    'KD0.25': {
        'BASE': list(range(0, 270))
    },
    'KD0.01': {
        'BASE': list(range(0, 100))
    },

    'KD0.5': {
        'BASE': list(range(0, 540))
    },
    'KD0.75': {
        'BASE': list(range(0, 810))
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


def add_salt_and_pepper_noise(img):
    """
    向图像添加盐椒噪声
    img: Tensor图像
    """
    # 设定噪声比例
    amount = 0.1  # 噪声占图像比例
    # 0.005
    salt_vs_pepper = 0.5  # 盐和椒的比例
    num_salt = np.ceil(amount * img.numel() * salt_vs_pepper)
    num_pepper = np.ceil(amount * img.numel() * (1.0 - salt_vs_pepper))

    # 添加盐噪声
    indices = torch.randperm(img.numel())[:int(num_salt)]
    img.view(-1)[indices] = 1

    # 添加椒噪声
    indices = torch.randperm(img.numel())[:int(num_pepper)]
    img.view(-1)[indices] = 0

    return img


# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")

'''
数据转换

'''

# 设置数据转换，这里仅进行基础的转换和标准化
transform_no_salt_pepper = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(add_salt_and_pepper_noise),
    # transforms.Lambda(add_speckle_noise),
    # transforms.Lambda(add_gaussian_noise)

])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = PathMNISTDataset(images, labels, transform=transform)
train_dataset.shuffle_data()  # 使用固定种子打乱数据

train_dataset_no = PathMNISTDataset(images, labels, transform=transform_no_salt_pepper)
train_dataset_no.shuffle_data()

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=32, shuffle=True,
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
qno_2000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_2000']['QUERY']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
qno_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1000']['QUERY']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
qno_100_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_100']['QUERY']), batch_size=32, shuffle=False,
                            generator=torch.Generator().manual_seed(random_seed))

qno_1_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_1']['QUERY']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
qno_10_loader = DataLoader(Subset(train_dataset, CONFIG['QNO_10']['QUERY']), batch_size=32, shuffle=False,
                           generator=torch.Generator().manual_seed(random_seed))

qf_100_loader_noise = DataLoader(Subset(train_dataset_no, CONFIG['QF_100']['QUERY']), batch_size=32, shuffle=True,
                                 generator=torch.Generator().manual_seed(random_seed))
# 创建十个 Subset 的副本
subsets = [Subset(train_dataset, CONFIG['QF_100']['QUERY']) for _ in range(10)]
concat_dataset = ConcatDataset(subsets)

# 使用 ConcatDataset 创建 DataLoader
qf_100_loader10 = DataLoader(
    concat_dataset,
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(random_seed)
)
qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=16, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))

qf_1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=32, shuffle=True,
                         generator=torch.Generator().manual_seed(random_seed))
qf_10_loader = DataLoader(Subset(train_dataset, CONFIG['QF_10']['QUERY']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))

qf_1000_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
qf_1000_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['QF_1000']['QUERY']), batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['KD0.25']['BASE']), batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                             generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['KD0.75']['BASE']), batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))
kd0_01_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.01']['BASE']), batch_size=32, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))


# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
#         self.relu = nn.ReLU()  # 激活函数
#         self.fc2 = nn.Linear(10, 3)  # 隐藏层到输出层
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# 加载数据
# data = load_iris()
# X = data.data
# y = data.target

# 将numpy数组转换为torch张量
# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.int64)

# 划分数据为三个片段
# shards = np.array_split(np.random.permutation(len(Subset(train_dataset, CONFIG['BASE1']['BASE']))), 10)
# print(len(Subset(train_dataset, CONFIG['BASE1']['BASE']))   )
indices = np.random.permutation(len(Subset(train_dataset, CONFIG['BASE1']['BASE'])))

# 分片
shards = np.array_split(indices, 10)

# 调整分片以确保每个分片的大小不满足 len(shard) % 32 == 1
shards = adjust_shards(shards)
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

    model = get_student_model().to(device)
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

    model = get_student_model().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    for epoch in range(100):
        for data, target in dataloader:
            data = data.to(device)
            target= target.squeeze().long().to(device) # 是path的
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
