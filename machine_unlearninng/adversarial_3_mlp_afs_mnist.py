import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from KDloss import SoftTarget
from unlearning_random_afs import train_student_model_random
from Domainadaptation_mnist import  domainadaptation
from test_model_mnist import test_model
from puri import api
import random
from mlp import get_student_model,get_teacher_model

# 相比上一个  更改了 数据集的划分方法
# TODO xiangbi shangyige  zengjia le  afs

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
'QNO_1000': {
    'QUERY': list(range(20000,21000)),
    'QUERY_MEMBER': [0 for i in range(1000)]
    },
'QNO_100': {
    'QUERY': list(range(20000,20100)),
    'QUERY_MEMBER': [0 for i in range(100)]
    },
'QNO_1': {
    'QUERY': list(range(20000,20001)),
    'QUERY_MEMBER': [0 for i in range(1)]
    },
'QNO_10': {
    'QUERY': list(range(20000,20010)),
    'QUERY_MEMBER': [0 for i in range(10)]
    },
    
'QF_100':{
    'QUERY': list(range(9900,10000)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QF_1':{
    'QUERY': list(range(9999,10000)),
    'QUERY_MEMBER': [1 for i in range(1)]
},
'QF_10':{
    'QUERY': list(range(9990,10000)),
    'QUERY_MEMBER': [1 for i in range(10)]
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
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 下载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_1000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_2000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_2000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
cal_5000_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_5000']['CAL']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=64, shuffle=False,generator=torch.Generator().manual_seed(random_seed))
qo_5000_loader = DataLoader(Subset(train_dataset, CONFIG['QO_5000']['QUERY']), batch_size=64, shuffle=True,generator=torch.Generator().manual_seed(random_seed))
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

smodelmlp = get_student_model()
smodelmlp = smodelmlp.to(device)
tmodelmlp = get_student_model()
tmodelmlp = tmodelmlp.to(device)
# smodel18 = ResNet18().to(device)
# tmodel18 = ResNet18().to(device)
# tmodel18.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
# smodel18.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
tmodelmlp.apply(weights_init)
smodelmlp.apply(weights_init)
u = smodelmlp.state_dict()
f_u = smodelmlp.features.state_dict()
T = 4
modelmlp =get_teacher_model()
modelmlp = modelmlp.to(device)
# model34 = ResNet34().to(device)
modelmlp.load_state_dict(torch.load('best_model_mlp0111.pth'))
criterion = nn.CrossEntropyLoss()
criterionKD = SoftTarget(T)
criterionCls = nn.NLLLoss().to(device)
optimizer18s = optim.Adam(smodelmlp.parameters(), lr=0.001)
num_epochs2 = 1
lambda_kd = 1
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

num_epochsall = 75
# optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)

for epoch in range(num_epochsall):
    print(epoch)
    print("test accuracy")

    f_u,u = train_student_model_random(qf_100_loader,kd0_5_loader,tmodelmlp,modelmlp,smodelmlp,u,f_u)
    print("machine unlearning 以后")
    # test_model(test1_loader,qf_1000_loader,kd0_5_loader,device,modelmlp,smodelmlp,tmodelmlp,u,f_u)

    member_gt =[1 for i in range(100)]
    #数据集的划分还存在问题，需要划分出校准数据集  遗忘数据集  是有了 但是没有两个 校准数据集  得划分
    _t, pv, EMA_res, risk_score = api(device,smodelmlp, u,f_u,qf_100_loader, member_gt, cal_1000_loader,caltest1_loader)
    print(f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')

    f_u = domainadaptation(f_u,qf_100_loader,qno_100_loader)

    _t, pv, EMA_res, risk_score = api(device,smodelmlp, u,f_u,qf_100_loader, member_gt, cal_1000_loader,caltest1_loader)
    print("ad  以后")
    test_model(test1_loader,qf_100_loader,kd0_5_loader,device,modelmlp,smodelmlp,tmodelmlp,u,f_u)

    print(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')




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



