import torch.nn as nn
import torch
from torch.utils.data import  Subset
import sys
sys.path.append('/autodl-tmp/wangbin/yiwang')
import numpy as np
# from KDloss import SoftTarget
from unlearning_random_afs_rand import train_student_model_random
from Domainadaptation_csv_three_wu import domainadaptation
from test_model_path import test_model
from puri import api
import random
from mlp_three_csv_wu import FeatureExtractor, Classifier, CombinedModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from qf1kosiam import  analyze_sample_similarity
from qf1kosiam import analyze_sample_similarity

from calculate_kl_divergence import calculate_kl_divergence
global epochs  # Epoch编号
p_values = []  # 对应的p-value
accuracies = []  # 对应的准确率

# 重新划分的表格数据集(划分逻辑与之前相似)
'''
原版的config
'''
# CONFIG = {
#     'BASE1': {
#         'BASE': list(range(0, 192))
#     },
#     'TEST1': {
#         'TEST': list(range(512, 576))
#     },
#     'CAL_25': {
#         'CAL': list(range(192, 217))
#     },
#     'CAL_50': {
#         'CAL': list(range(192, 242))
#     },
#     'CAL_100': {
#         'CAL': list(range(192, 292))
#     },
#     'CAL_150': {
#         'CAL': list(range(192, 342))
#     },
#     'CALTEST1': {
#         'TEST': list(range(576, 626))
#     },
#     'QO_150': {
#         'QUERY': list(range(0, 150)),
#         'QUERY_MEMBER': [1 for _ in range(150)]
#     },
#     'QNO_50': {
#         'QUERY': list(range(342, 392)),
#         'QUERY_MEMBER': [0 for _ in range(50)]
#     },
#     'QNO_25': {
#         'QUERY': list(range(342, 367)),
#         'QUERY_MEMBER': [0 for _ in range(25)]
#     },
#     'QNO_10': {
#         'QUERY': list(range(342, 352)),
#         'QUERY_MEMBER': [0 for _ in range(10)]
#     },
#     'QF_25': {
#         'QUERY': list(range(217, 242)),
#         'QUERY_MEMBER': [1 for _ in range(25)]
#     },
#     'QF_50': {
#         'QUERY': list(range(217, 267)),
#         'QUERY_MEMBER': [1 for _ in range(50)]
#     },
#     'KD0.25': {
#         'BASE': list(range(0, 48))
#     },
#     'KD0.5': {
#         'BASE': list(range(0, 96))
#     },
#     'KD0.75': {
#         'BASE': list(range(0, 144))
#     },
# }

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
# 添加随机数
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
    # 目前这里就简单的归一化处理，可以改
    return (features - features.mean()) / features.std()

# 定义添加噪声的转换
def transform_no(features):
    # 在原始数据的每一个特征值上随意赋值0-10的随机数值
    noise = np.random.randint(0, 11, size=features.shape)
    return features + noise

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# class TableDataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         features = self.data_frame.iloc[idx, 1:-1].astype(np.float32).values  # 假设所有特征都是数值型(特征从第二列开始算，第一列为id)
#         labels = self.data_frame.iloc[idx, -1].astype(np.int64)
#
#         if self.transform:
#             features = self.transform(features)
#
#         # return torch.tensor(features), torch.tensor(labels)
#         return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TableDataset(Dataset):
    def __init__(self, csv_file, transform=None, shuffle=True, seed=32):
        self.data_frame = pd.read_csv(csv_file)

        # 如果启用随机化
        if shuffle:
            # 设置随机种子
            np.random.seed(seed)
            # 打乱数据框的索引
            shuffled_indices = np.random.permutation(self.data_frame.index)
            self.data_frame = self.data_frame.loc[shuffled_indices].reset_index(drop=True)

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

        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

# 加载数据集
csv_file = '/root/autodl-tmp/wangbin/yiwang/data/final.csv' # 数据集路径
train_dataset = TableDataset(csv_file=csv_file, transform=transform)
train_dataset_no = TableDataset(csv_file=csv_file, transform=transform_no)
test_dataset = TableDataset(csv_file=csv_file, transform=transform)


# 先测试这几个qf_50_loader, kd0_5_loader, test1_loader, cal_50_loader, caltest1_loader
base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
qf_50_loader = DataLoader(Subset(train_dataset, CONFIG['QF_50']['QUERY']), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))

qf_1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=1, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=32, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
caltest1_loader = DataLoader(Subset(train_dataset, CONFIG['CALTEST1']['TEST']), batch_size=32, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
kd0_75_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.75']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
kd0_25_loader = DataLoader(Subset(train_dataset, CONFIG['KD0.25']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
kd0_5_loader_no = DataLoader(Subset(train_dataset_no, CONFIG['KD0.5']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))

feature_extractort = FeatureExtractor(dim_in=24, dim_hidden=256)  # 注意这里的要调整为适应于表格数据的dim_in=8，dim_out=2，可以改
classifiert = Classifier(dim_hidden=256, dim_out=2)

# 创建组合模型实例
tmodelmlp = CombinedModel(feature_extractort, classifiert).to(device)
feature_extractors = FeatureExtractor(dim_in=24, dim_hidden=64)
classifiers = Classifier(dim_hidden=64, dim_out=2)
# 创建组合模型实例
smodelmlp = CombinedModel(feature_extractors, classifiers).to(device)
# smodelmlp = smodelmlp.to(device)

smodelmlp_base2 = CombinedModel(feature_extractors, classifiers).to(device)
smodelmlp = smodelmlp.to(device)
smodelmlp_base2  = smodelmlp_base2 .to(device)
tmodelmlp = tmodelmlp.to(device)

tmodelmlp.apply(weights_init)
smodelmlp.apply(weights_init)
u = smodelmlp.state_dict()
f_u = smodelmlp.feature_extractor.state_dict()
T = 4
feature_extractortt = FeatureExtractor(dim_in=24, dim_hidden=256)
classifierstt = Classifier(dim_hidden=256, dim_out=2)
smodelmlp_base2 .load_state_dict(torch.load('/root/autodl-tmp/wangbin/yiwang/best_model_mlp0330_wu_Remove_batch_qf1chongxun.pth'))

# 创建组合模型实例
modelmlp = CombinedModel(feature_extractortt, classifierstt).to(device)
modelmlp = modelmlp.to(device)

modelmlp.load_state_dict(torch.load('best_model_mlp0330_wu_Remove_batch.pth'))  # 需要使用改后的"a_mlp_train_three.py"训练得到
criterion = nn.CrossEntropyLoss()
# criterionKD = SoftTarget(T)
criterionCls = nn.NLLLoss().to(device)

num_epochsall = 75
# member_gt = [1 for i in range(100)]  # (当数据量没有这么多时，就要适当的减少，目的是要和api中的member_pred相等)
member_gt = [1 for i in range(50)]
best_accuracy = 0
best_model_state_f_u = None

best_model_state_u = None
num = 0

for epoch in range(num_epochsall):
    print(epoch)

    f_u, u , c_u = train_student_model_random(qf_50_loader, kd0_75_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u)
    print("machine unlearning 以后")
    # analyze_sample_similarity(smodelmlp,u,device,train_dataset,CONFIG)

    current_accuracy = test_model(test1_loader, qf_50_loader, kd0_75_loader, device, modelmlp, smodelmlp, tmodelmlp, u,
                                  f_u)
    # analyze_sample_similarity(smodelmlp,u,device,train_dataset,CONFIG)
    # calculate_kl_divergence(smodelmlp,u,smodelmlp_base2, qf_1_loader, device)
    print("currentmu_accuracy")
    print(current_accuracy)
    _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_50_loader, member_gt, cal_100_loader,caltest1_loader)

    print(f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')


    f_u = domainadaptation(f_u, u, qf_50_loader, kd0_5_loader_no)
    current_accuracy = test_model(test1_loader, qf_50_loader, kd0_75_loader, device, modelmlp, smodelmlp, tmodelmlp, u,
                                  f_u)
    print("currentad_accuracy")
    print(current_accuracy)

    num = num + 1
    _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_50_loader, member_gt, cal_100_loader,
                                      caltest1_loader)
    print("ad  以后")
    print(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
    # update_plot_with_new_data(epoch, pv, current_accuracy)