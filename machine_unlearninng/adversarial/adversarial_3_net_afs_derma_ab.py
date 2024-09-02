import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from unlearning_random_afs_rand import train_student_model_random
from Domainadaptation_net_three import domainadaptation
from test_model_path import test_model
from puri import api
import random
from mu.net_train_three import get_student_model, get_teacher_model
from torch.utils.data import Dataset, DataLoader, Subset
import os
from qf1kosiam import analyze_sample_similarity

from calculate_kl_divergence import calculate_kl_divergence
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from plot_with_new_data import update_plot_with_new_data
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
# from tsne_mnist_samedata import tsne
# from tsne_mnist_tuo import tsne
from tsne_mnist_guding1 import tsnet
from tsne_mnist_guding2 import tsnes
# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs

# 加载npz文件
import torch.nn.init as init


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


npz_file_path = '/root/autodl-tmp/wangbin/yiwang/data/dermamnist.npz'
data = np.load(npz_file_path)

print(list(data.keys()))  # 打印所有在npz文件中的数组名

# ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
images = data['train_images']
images_test =data['val_images']
labels_test =data['val_labels']
images_cal = data['test_images']
labels_cal =data['test_labels']
print(images.shape)
print(images_test.shape)
print(images_cal.shape)
labels = data['train_labels']


# 创建自定义数据集
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

    def shuffle_data(self, seed=52):
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
        #     self.images = np.array([self.transform(image) for image in self.images])

# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs

CONFIG = {
    'BASE1': {
        'BASE': list(range(0, 7007))
    },
    'BASE2': {
        'BASE': list(range(0, 9900))
    },
    'TEST1': {
        'TEST': list(range(0, 1003))
    },
    'CAL_100': {
        'CAL': list(range(40000, 40100))
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
        'TEST': list(range(1000, 2000))
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
        'QUERY': list(range(6907, 7007)),
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
        'QUERY': list(range(6007, 7007)),
        'QUERY_MEMBER': [1 for i in range(1000)]
    },
    'KD0.25': {
        'BASE': list(range(0, 1751))
    },
    'KD0.01': {
        'BASE': list(range(0, 100))
    },

    'KD0.5': {
        'BASE': list(range(0, 3503))
    },
    'KD0.75': {
        'BASE': list(range(0, 5255))
    },
}

# random_seed = 10
# random_seed = 23
random_seed = 52
# random_seed = 32
# random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


def add_salt_and_pepper_noise(img):
    """
    向图像添加盐椒噪声
    img: Tensor图像
    """
    # 设定噪声比例
    amount = 0.1 # 噪声占图像比例
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
test_dataset = PathMNISTDataset(images_test, labels_test, transform=transform)
cal_dataset = PathMNISTDataset(images_cal, labels_cal, transform=transform)
train_dataset_no = PathMNISTDataset(images, labels, transform=transform_no_salt_pepper)

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(test_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(cal_dataset, CONFIG['CAL_100']['CAL']), batch_size=64, shuffle=False,
                            generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader = DataLoader(Subset(cal_dataset, CONFIG['CAL_1000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
cal_2000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_2000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
cal_5000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_5000']['CAL']), batch_size=64, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(cal_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,
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
qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=32, shuffle=True,
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
# 参数设置
smodelmlp = get_student_model()
smodelmlp_base2 = get_student_model()
smodelmlp = smodelmlp.to(device)
smodelmlp_base2  = smodelmlp_base2 .to(device)
tmodelmlp = get_student_model()
tmodelmlp = tmodelmlp.to(device)
# smodel18 = ResNet18().to(device)
# tmodel18 = ResNet18().to(device)
# tmodel18.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
# smodel18.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
tmodelmlp.apply(init_weights)
smodelmlp.apply(init_weights)
u = smodelmlp.state_dict()
f_u = smodelmlp.feature_extractor.state_dict()
# T = 4
modelmlp = get_teacher_model()
modelmlp = modelmlp.to(device)
# model34 = ResNet34().to(device)

smodelmlp_base2 .load_state_dict(torch.load('/root/autodl-tmp/wangbin/yiwang/best_model_dermanet_base2.pth'))

modelmlp.load_state_dict(torch.load('/root/autodl-tmp/wangbin/yiwang/best_model_derma_615.pth'))
# criterion = nn.CrossEntropyLoss()
# criterionKD = SoftTarget(T)
# criterionCls = nn.NLLLoss().to(device)
# optimizer18s = optim.Adam(smodelmlp.parameters(), lr=0.001)
# num_epochs2 = 1
# lambda_kd = 1
# for epoch in range(num_epochs2):
#     model34.eval()
#     tmodel18.eval()
#     smodel18.train()
#     running_loss = 0.0
#     totalt = 0
#     correctt = 0
#     # 这里的数据集不应该是 retain而是 train数据集（）有蒸馏损失
#     for data in tqdm(base1_loader, desc=f"Epoch {epoch + 1}/{num_epochs2}"):
#         inputs, labels = data[0].to(device), data[1].to(device)
#         optimizer18s.zero_grad()
#         outputs34 = model34(inputs)
#         outputs18s = smodel18(inputs)
#         cls_loss = criterionCls(outputs18s, labels)
#         kd_loss = criterionKD(outputs18s, outputs34.detach())
#         loss = cls_loss + kd_loss * lambda_kd
#         _, predicted = torch.max(outputs18s.data, 1)
#         totalt += labels.size(0)
#         correctt += (predicted == labels).sum().item()
#         loss.backward()
#         optimizer18s.step()
#     accuracyt = 100 * correctt / totalt
#     print(f'Accuracy of the network on the train images: {accuracyt}%')

num_epochsall = 100
# optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)
member_gt = [1 for i in range(1)]
num = 0
best_accuracy = 0.0
best_f12 = 1
sj1, sj2, sw1, sw2 = 0.0,0.0,0.0,0.0
tj1, tj2, tw1, tw2 = 0.0,0.0,0.0,0.0
for epoch in range(num_epochsall):

    '''
    
    进行 machine unlearning
    
    '''
    print(epoch)
    print("test accuracy")
    # current_accuracy = test_model(test1_loader, qf_100_loader, kd0_5_loader, device, modelmlp, smodelmlp, tmodelmlp, u,f_u)

    f_u, u, c_u = train_student_model_random(qf_1_loader, kd0_5_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u)
    current_accuracy, accuracy1 = test_model(test1_loader, qf_1_loader, kd0_5_loader, device, modelmlp, smodelmlp,tmodelmlp, u, f_u)

    analyze_sample_similarity(smodelmlp,u,device,train_dataset,CONFIG)
    calculate_kl_divergence(smodelmlp,u,smodelmlp_base2, qf_1_loader, device)

    _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_1_loader, member_gt, cal_1000_loader,caltest1_loader)
    print(f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')


    '''
    
    怎么去取得最佳权重
    
    '''
    # f11 = 0.2 * accuracy1 * 0.8 * pv / (0.2 * accuracy1 + 0.8 * pv)
    # f12 = (0.2 * accuracy1 + 0.8 * pv) / 0.2 * accuracy1 * 0.8 * pv
    # print(f'f11 : {f11}')
    # print(f'f12 : {f12}')
    # if f12 < best_f12:
    #     best_f12 = f12
    # else:
    #     pass
    # print(f'best_f12:{best_f12}')

    # if not os.path.exists('./saved_modelrandomderma32qf1'):
    #     os.makedirs('./saved_modelrandomderma32qf1')
    # #
    # model_save_path = f'./saved_modelrandomderma32qf1/model_epoch_{epoch}.pth'
    # # 保存模型权重
    # torch.save(smodelmlp.state_dict(), model_save_path)
    #
    # print(f"模型权重已保存到 {model_save_path}")

    # update_plot_with_new_data(epoch, pv, current_accuracy)

    # if current_accuracy > best_accuracy:
    #     best_accuracy = current_accuracy
    #     best_model_state_f_u = f_u
    #     best_model_state_u = u

    # 可以在这里添加代码保存模型状态到文件，例如使用torch.save
    # torch.save(smodelmlp.state_dict(), 'best_smodelcnntsne.pth')
    # save_dirtsne = './visualizationzhengliu+domain+honghui/'
    # tsne(qf_100_loader, smodelmlp, smodelmlp, device,save_dirtsne,num,u,f_u)
    # tsne(qf_100_loader, kd0_5_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u, num)
    # tj11, tj22, tw11, tw22 = tsnet(qf_100_loader, kd0_5_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u, epoch)
    # tsnet(qf_100_loader, kd0_5_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u, epoch)
    # tsnes(qf_100_loader, kd0_5_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u, epoch)
    # sj11, sj22, sw11, sw22 = tsnes(qf_100_loader, kd0_5_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u, epoch)
    # sj1+=sj11
    # sj2+=sj22
    # sw1+=sw11
    # sw2+=sw22
    # tj1+=tj11
    # tj2+=tj22
    # tw1+=tw11
    # tw2+=tw22

    # update_plot_with_new_data(epoch, pv, current_accuracy) # shiyongde shi zheg
    """
    对抗部分的代码
    """

    f_u = domainadaptation(f_u, c_u, qf_1_loader, kd0_5_loader_no)
    _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_1_loader, member_gt, cal_1000_loader,caltest1_loader)
    current_accuracy = test_model(test1_loader, qf_1_loader, kd0_5_loader, device, modelmlp, smodelmlp, tmodelmlp, u,f_u)

    analyze_sample_similarity(smodelmlp,u,device,train_dataset,CONFIG)
    calculate_kl_divergence(smodelmlp,u,smodelmlp_base2, qf_1_loader, device)
    num = num + 1
    print(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')

    # if not os.path.exists('./saved_modelrandomderma52qf1'):
    #     os.makedirs('./saved_modelrandomderma52qf1')
    # #
    # model_save_path = f'./saved_modelrandomderma52qf1/model_epoch_{epoch}.pth'
    # # 保存模型权重
    # torch.save(smodelmlp.state_dict(), model_save_path)
    #
    # print(f"模型权重已保存到 {model_save_path}")

    '''
    这是废弃的代码
    '''
    # extract_features_and_save(u,forget_loader)
    # extract_features_and_savet(u,test_loader)
    # merge_csv_files()
    # loss1 = evaluate_model(u,smodel18,retain_loader)
    # loss2 = train_model_and_compute_loss()

    # loss1_tensor = torch.tensor(loss1)
    # loss2_tensor = torch.tensor(loss2)
    # total_loss = loss1_tensor + loss2_tensor
    # total_loss = loss2 + loss1
    # compute_and_update_weights(total_loss,smodel18,u)

    # u = updateu(retain_loader,u,smodel18)


"""
输出对应的距离
"""
# print(f'total sj1{sj1}')
# print(f'total sj2{sj2}')
# print(f'total sw1{sw1}')
# print(f'total sw2{sw2}')
# print(f'total tj1{tj1}')
# print(f'total tj2{tj2}')
# print(f'total tw1{tw1}')
# print(f'total tw2{tw2}')
