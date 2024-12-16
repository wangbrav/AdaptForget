from __future__ import print_function
#首先是数据集的加载 加载包括   有一个文件的加载  dataset的嘞也需要 还有一个是dataloader的加载  还有他的config  以及他的颜色扩充通道  以及其的加噪函数  transform  种子的设置
# import sys
# sys.path.append('/root/autodl-tmp/wangbin/yiwang')
import copy
import torch.nn as nn
import torch
from torch.utils.data import  Subset
import sys
sys.path.append('/root/autodl-tmp/wangbin/yiwang')
import numpy as np
# from KDloss import SoftTarget
from unlearning_random_afs_rand import train_student_model_random
from Domainadaptation_csv_three_wu import domainadaptation
from test_model_path import test_model
from puri import api
import random
from mlp_three_csv_wu import FeatureExtractor, Classifier, CombinedModel,get_student_model, get_teacher_model
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from qf1kosiam import  analyze_sample_similarity
from qf1kosiam import analyze_sample_similarity

from calculate_kl_divergence import calculate_kl_divergence
global epochs  # Epoch编号
# import utils.Purification
import argparse
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import ConcatDataset, DataLoader, Subset
import logging
from tqdm import tqdm
import torch.nn.functional as F
from unlearning_random_afs_rand import train_student_model_random
from Domainadaptation_csv_three_wu import domainadaptation
from test_model_path import test_model
# from unlearning_random_afs_rand import train_student_model_random
# from Domainadaptation_net_three import domainadaptation
# from test_model_path import test_model
# from puri import api
import random
from puri import api
import itertools
from advertorch.attacks import L2PGDAttack
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from itertools import cycle
# from mu.net_train_three import get_student_model, get_teacher_model, get_student_model_t
from torch.utils.data import Dataset, DataLoader, Subset
import os
from utilsinstance import JointDataset, NormalizeLayer, naive_train, train, adv_attack, testins, estimate_parameter_importance

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
from qf1kosiam import analyze_sample_similarity

from calculate_kl_divergence import calculate_kl_divergence
# 加载npz文件
import torch.nn.init as init
#   afs
from copy import deepcopy

import matplotlib.pyplot as plt
import argparse
import sys
import torch
from utils.KDloss import SoftTarget
from utils.Metric import AverageMeter, accuracy, Performance
import os
import numpy as np
from tqdm import tqdm
from utils.Log import log_creater
import utils.Audit as Audit
import torch.nn as nn
import time
import random
import os
# import wandb

# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import models
from unlearn import *
from utils4 import *
import forget_random_strategies
import datasets
# import models
import conf
from training_utils import *
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from utils_w import *
from utils_ww import *
from mlp_three_csv_wu import FeatureExtractor, Classifier, CombinedModel,get_student_model, get_teacher_model
# from Dataset import DataModule, CONFIG
# from Model import get_teacher_model, get_student_model
# 设置日志记录 循环
logging.basicConfig(filename='./tc/training_log_qf1circulate_asdv7.log', level=logging.INFO, format='%(asctime)s %(message)s')
# logging.basicConfig(filename='./tc/training_log_qf1circulate_diabetesfinalv5.log', level=logging.INFO, format='%(asctime)s %(message)s')
# log_file_path = './tc/training_log_qf1circulate_diabetesfinalv4.log'
# absolute_path = os.path.abspath(log_file_path)
# print(f"The log file is saved at: {absolute_path}")
logger = logging.getLogger()
logger.info('This is an info message')
# class TableDataset(Dataset):
#     def __init__(self, csv_file, transform=None, shuffle=True, seed=32):
#         self.data_frame = pd.read_csv(csv_file)
#
#         # 如果启用随机化
#         if shuffle:
#             # 设置随机种子
#             np.random.seed(seed)
#             # 打乱数据框的索引
#             shuffled_indices = np.random.permutation(self.data_frame.index)
#             self.data_frame = self.data_frame.loc[shuffled_indices].reset_index(drop=True)
#
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         features = self.data_frame.iloc[idx, 1:-1].astype(np.float32).values  # 假设所有特征都是数值型
#         labels = self.data_frame.iloc[idx, -1].astype(np.int64)
#
#         if self.transform:
#             features = self.transform(features)
#
#         return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

#这里需要更改gt的大小

# 定义标准化转换
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

    def get_labels(self, indices):
        # 直接从数据框中提取指定索引的标签
        return self.data_frame.iloc[indices, -1].astype(np.int64).tolist()

def transform(features):
    # 目前这里就简单的归一化处理，可以改
    # print((features - features.mean()) / features.std())
    return (features - features.mean()) / features.std()
# print()
# 定义添加噪声的转换
def transform_no(features):
    # 在原始数据的每一个特征值上随意赋值0-10的随机数值
    # noise = np.random.randint(20, 200, size=features.shape)
    # noise = np.random.randint(0, 11, size=features.shape)
    return 0.3*(features  - features.mean()) / features.std()
    # return features + noise
# def transform_no(features):
#     # 将features中的所有数值替换为0
#     return np.zeros_like(features)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
# def afs(args,best_model_state_trained,best_model_state_retrained,base1_loader,base2_loader,test1_loader,cal_1000_loader,caltest1_loader,qf_100_loader,device):
#     sys.path.append(args.root)
#     best_model_state_afs = None
#
#     # init model
#     tnet = get_teacher_model().to(device)
#     snet = get_student_model().to(device)
#     # snet_base2 = get_student_model().to(device)
#     tnet.load_state_dict(best_model_state_trained)
#     snet_base2 = get_student_model().to(device)
#     snet_base2.load_state_dict(best_model_state_retrained)
#     tnet.eval()
#     # if 'MNIST' in args.root:
#     #     criterionCls = torch.nn.NLLLoss().to(args.device)
#     # elif 'PathMNIST' in args.root:
#     criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
#     # elif 'COVIDx' in args.root:
#     #     criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
#     # else:
#         # criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
#     criterionKD = SoftTarget(args.T)
#
#     optimizer_afs = torch.optim.Adam(snet.parameters(),
#                                 lr=args.lr)
#     # args.mode = 'base'
#     # args.base_label = args.KD_label
#     # baseDataset = DataModule(dir=args.root, args = args, batch_size = args.train_batch_size, num_workers=args.num_workers)
#     # baseTrainDataLoader = baseDataset.train_dataloader()
#     # baseTestDataLoader = baseDataset.test_dataloader()
#     # args.mode = 'query'
#     # queryDataset = DataModule(dir=args.root, args=args, batch_size = args.KP_infer_batch_size, num_workers=args.num_workers)
#     # member_gt = CONFIG[args.query_label]['QUERY_MEMBER']
#     # args.mode = 'cal'
#     # calDataset = DataModule(dir=args.root, args=args, batch_size = args.KP_infer_batch_size, num_workers=args.num_workers)
#
#     # logger.info('>> start KP')
#     save_metric_best = 0
#     # teacher_model = MLP(28, 64, 10).to(args.device)
#     member_gt = [1 for i in range(1)]
#
#     # teacher_model.apply(initialize_weights)
#     for epoch in range(args.epochs):
#         train(snet, tnet, criterionCls, criterionKD, base1_loader, optimizer_afs, args, qf_100_loader)
#         test_acc, stat = afstest(snet, tnet, criterionCls, criterionKD, test1_loader, args)
#         # logger.info(f'>> snet test acc: {test_acc}')
#         # logger.info(f">> snet test stat: {','.join([str(_) for _ in stat])}")
#         f_u,u = snet.feature_extractor.state_dict(), snet.state_dict()
#         # logger.info(f'>> evaluate membership attack on snet after training')
#         _t, pv, EMA_res, risk_score = api(device, snet, u, f_u, qf_100_loader, member_gt, cal_1000_loader,
#                                           caltest1_loader)
#         print(f'>> test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#
#         # save model
#         if save_metric_best < test_acc:
#             save_metric_best = test_acc
#             print(f'>> saving best snet model')
#             logger.info(f'>> saving best snet model{test_acc}')
#             _t, pv, EMA_res, risk_score = api(device, snet, u, f_u, qf_100_loader, member_gt, cal_1000_loader,
#                                               caltest1_loader)
#             print(f'>>best test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#             logger.info(f'>>afs best test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#             average_euclidean_afs, average_manhattan_afs, average_cosine_similarity_afs=analyze_sample_similarity(snet, u, device, train_dataset, CONFIG)
#             print(f'>>average_euclidean_afs: {average_euclidean_afs}, average_manhattan_afs: {average_manhattan_afs}, average_cosine_similarity_afs: {average_cosine_similarity_afs}')
#             logger.info(f'>>average_euclidean_afs: {average_euclidean_afs}, average_manhattan_afs: {average_manhattan_afs}, average_cosine_similarity_afs: {average_cosine_similarity_afs}')
#             average_kl_div_afs=calculate_kl_divergence(snet, best_model_state_retrained, snet_base2, qf_1_loader, device)
#             print(f'>>average_kl_div_afs: {average_kl_div_afs}')
#             logger.info(f'>>average_kl_div_afs: {average_kl_div_afs}')
#             # save_best_ckpt(snet, args)
#             best_model_state_afs = snet.state_dict().copy()
#
#         # save_last_ckpt(snet, args)

# def parser():
#     """
#     :return: args
#     """
#     parser = argparse.ArgumentParser(prog='AFS')
#     subparsers = parser.add_subparsers(help='sub-command help')
#     #
#     # parser_audit = subparsers.add_parser('audit')
#     # parser_audit.add_argument('--root',
#     #                     default='../template/MNIST',
#     #                     help='root dir to the project')
#     # parser_audit.add_argument('--query_label',
#     #                     default='EXP1',
#     #                     help='label of the query data, defined in Dataset.py/Config')
#     # parser_audit.add_argument('--cal_label',
#     #                     default='CAL1',
#     #                     help='label of the calibration data, defined in Dataset.py/Config')
#     # parser_audit.add_argument('--cal_test_label',
#     #                     default='CALTEST1',
#     #                     help='label of the calibration test data, defined in Dataset.py/Config')
#     # parser_audit.add_argument('--test_label',
#     #                     default='TEST1',
#     #                     help='label of the test data, defined in Dataset.py/Config')
#     # parser_audit.add_argument('--model2audit',
#     #                     default='./models/base/best_model.pth',
#     #                     help='relative path of model to be auditted to the root')
#     # parser_audit.add_argument('--model2cal',
#     #                     default='./models/cal/best_model.pth',
#     #                     help='relative path of the calibration model to the root')
#     # parser_audit.add_argument('--device',
#     #                     default='cuda:0')
#     # parser_audit.add_argument('--KP_infer_batch_size',
#     #                     type=int,
#     #                     default=1024,
#     #                     help='batch size for inference during membership attack')
#     # parser_audit.add_argument('--nclass',
#     #                     type=int,
#     #                     default=10,
#     #                     help='number of classes')
#     # parser_audit.add_argument('--num_workers',
#     #                     type=int,
#     #                     default=5,
#     #                     help='number of num_workers')
#     # parser_audit.add_argument('--command_class',
#     #                           default=0,
#     #                           type=int,
#     #                           help='for internal use only, no change')
#
#     parser_forget = subparsers.add_parser('forget')
#     parser_forget.add_argument('--root',
#                         default='../template/MNIST',
#                         help='root dir to the project')
#     parser_forget.add_argument('--expname',
#                         default='EXP1',
#                         help='name of exp, will affect the path and dataset splitting')
#     parser_forget.add_argument('--teacher_model',
#                         default='./models/EXP1/base/best_model.pth',
#                         help='relative path of model to be distilled to the root')
#     parser_forget.add_argument('--KD_label',
#                         default='KD0.25',
#                         help='the name of base dataset used for KD, should be defined in CONFIG')
#     parser_forget.add_argument('--test_label',
#                         default='TEST1',
#                         help='label of the test data, defined in Dataset.py/Config')
#     parser_forget.add_argument('--cal_label',
#                         default='CAL1',
#                         help='label of the calibration data, defined in Dataset.py/Config')
#     parser_forget.add_argument('--cal_test_label',
#                         default='CALTEST1',
#                         help='label of the calibration test data, defined in Dataset.py/Config')
#     parser_forget.add_argument('--query_label',
#                         default='QO1',
#                         help='label of the query data, defined in Dataset.py/Config, here the query dataset should overlap with training dataset')
#     parser_forget.add_argument('--add_risk_loss',
#                         type=int,
#                         default=1,
#                         help='1: will add risk loss when running KP, 0: same as pure KD')
#     parser_forget.add_argument('--nclass',
#                         type=int,
#                         default=9,
#                         help='number of classes')
#     parser_forget.add_argument('--train_batch_size',
#                         type=int,
#                         default=32)
#     parser_forget.add_argument('--KP_infer_batch_size',
#                         type=int,
#                         default=128,
#                         help='batch size for inference during membership attack')
#     parser_forget.add_argument('--device',
#                         default='cuda:0')
#     parser_forget.add_argument('--epochs',
#                         type=int,
#                         default=20,
#                         help='number of epochs')
#     parser_forget.add_argument('--T',
#                         type=float,
#                         default=4.0,
#                         help='temperature for ST')
#     parser_forget.add_argument('--lr',
#                         type=float,
#                         default=0.1,
#                         help='initial learning rate')
#     parser_forget.add_argument('--lambda_kd',
#                         type=float,
#                         default=1,
#                         help='trade-off parameter for kd loss')
#     parser_forget.add_argument('--lambda_risk',
#                         type=float,
#                         default=10,
#                         help='trade-off parameter for risk loss')
#     parser_forget.add_argument('--num_workers',
#                         type=int,
#                         default=5,
#                         help='number of num_workers')
#     parser_forget.add_argument('--command_class',
#                               default=1,
#                               type=int,
#                               help='for internal use only, no change')
#     parser = argparse.ArgumentParser(prog='Knowledge Purification (KP)')
#     parser.add_argument('--root',
#                           default='../template/MNIST',
#                           help='root dir to the project')
#     parser.add_argument('--expname',
#                         default='EXP1',
#                         help='name of exp, will affect the path and dataset splitting')
#     parser.add_argument('--teacher_model',
#                         default='./models/EXP1/base/best_model.pth',
#                         help='relative path of model to be distilled to the root')
#     parser.add_argument('--KD_label',
#                         default='KD0.25',
#                         help='the name of base dataset used for KD, should be defined in CONFIG')
#     parser.add_argument('--test_label',
#                         default='TEST1',
#                         help='label of the test data, defined in Dataset.py/Config')
#     parser.add_argument('--cal_label',
#                         default='CAL1',
#                         help='label of the calibration data, defined in Dataset.py/Config')
#     parser.add_argument('--cal_test_label',
#                         default='CALTEST1',
#                         help='label of the calibration test data, defined in Dataset.py/Config')
#     parser.add_argument('--query_label',
#                         default='QO1',
#                         help='label of the query data, defined in Dataset.py/Config, here the query dataset should overlap with training dataset')
#     parser.add_argument('--add_risk_loss',
#                         type=int,
#                         default=1,
#                         help='1: will add risk loss when running KP, 0: same as pure KD')
#     parser.add_argument('--nclass',
#                         type=int,
#                         default=9,
#                         help='number of classes')
#     parser.add_argument('--train_batch_size',
#                         type=int,
#                         default=32)
#     parser.add_argument('--KP_infer_batch_size',
#                         type=int,
#                         default=128,
#                         help='batch size for inference during membership attack')
#     parser.add_argument('--device',
#                         default='cuda:0')
#     parser.add_argument('--epochs',
#                         type=int,
#                         default=20,
#                         help='number of epochs')
#     parser.add_argument('--T',
#                         type=float,
#                         default=4.0,
#                         help='temperature for ST')
#     parser.add_argument('--lr',
#                         type=float,
#                         default=0.001,
#                         help='initial learning rate')
#     parser.add_argument('--lambda_kd',
#                         type=float,
#                         default=1,
#                         help='trade-off parameter for kd loss')
#     parser.add_argument('--lambda_risk',
#                         type=float,
#                         default=10,
#                         help='trade-off parameter for risk loss')
#     parser.add_argument('--num_workers',
#                         type=int,
#                         default=5,
#                         help='number of num_workers')
#     args = parser.parse_args()
#     return args
#     # args = parser_forget.parse_args()
# args = parser()

# args = parser(args.root)
# def init_weights(m):
#     if type(m) == nn.Conv2d or type(m) == nn.Linear:
#         init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             init.zeros_(m.bias)
# def zero_gradients(x):
#     if isinstance(x, torch.Tensor):
#         if x.grad is not None:
#             x.grad.detach_()
#             x.grad.zero_()
#     elif isinstance(x, collections.abc.Iterable):
#         for elem in x:
#             zero_gradients(elem)

# def test(model, test_loader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         with tqdm(total=len(test_loader)) as t:
#             for X, Y in tqdm(test_loader):
#                 X, Y = X.to(device), Y.to(device)
#                 outputs = model(X)
#                 # _, predicted = torch.max(outputs.data, 1)
#                   # 调整Y的形状和数据类型
#
#                 pred = outputs.data.max(1)[1]
#                 # print(predicted.shape)
#                 # print(Y.shape)
#                 total += Y.size(0)
#
#                 # Y = Y.squeeze(1).long()
#                 correct += pred.eq(Y.view(-1)).sum().item()
#                 # print(f'Batch correct: {(predicted == Y).sum().item()}, Batch total: {Y.size(0)}')  # 新增的打印语句
#     print(f"Correct: {correct}, Total: {total}")
#     accuracy = 100 * correct / total
#     return accuracy
# def train(snet, tnet, criterionCls, criterionKD, trainDataLoader, optimizer, args, queryDataset):
#     cls_losses = AverageMeter()
#     kd_losses = AverageMeter()
#     risk_losses = AverageMeter()
#     total_losses = AverageMeter()
#     snet.train()
#     tnet.eval()
#     with tqdm(total=len(trainDataLoader)) as t:
#         for X, Y in tqdm(trainDataLoader):
#             X = X.to(args.device)
#             Y = Y.to(args.device)
#
#             # if 'PathMNIST' in args.root:
#             Y = Y.squeeze().long()
#
#             #print(X)
#             snet_pred = snet(X)
#             tnet_pred = tnet(X)
#
#             #print(Y, tnet_pred, snet_pred)
#
#             cls_loss = criterionCls(snet_pred, Y)
#             kd_loss = criterionKD(snet_pred, tnet_pred.detach())
#             cls_losses.update(cls_loss.item(), X.size(0))
#             kd_losses.update(kd_loss.item(), X.size(0))
#
#             loss = cls_loss + kd_loss* args.lambda_kd
#
#             #if loss < 1:
#             #    args.add_risk_loss = True
#             #    print(f'>> start using risk loss')
#             #else:
#             #    args.add_risk_loss = False
#
#
#             if args.add_risk_loss == 1:
#                 risk_loss = torch.tensor(0.0).to(args.device)
#                 queryTestDataLoader = queryDataset
#                 for _X, _Y in queryTestDataLoader:
#                     _X = _X.to(args.device)
#                     _Y = _Y.to(args.device)
#                     # if 'PathMNIST' in args.root:
#                     _Y = _Y.squeeze().long()
#                     if _Y.dim() == 0:
#                         _Y = _Y.unsqueeze(0)
#                     out_Y = snet(_X)
#                     partial_risk_loss = torch.nn.CrossEntropyLoss().to(args.device)(out_Y, _Y)
#                     risk_loss += partial_risk_loss
#
#                 ## using Audit.api requires more time for training
#                 #t, pv, EMA_res, risk_score = Audit.api(args, model, queryDataset, member_gt, calDataset)
#                 #risk_loss = risk_score
#
#                 risk_loss = torch.tensor(1.0).to(args.device) / risk_loss # same performance, strongly correlated
#
#                 #risk_loss = risk_loss
#                 risk_losses.update(risk_loss.item(), _X.size(0))
#                 loss = loss + risk_loss*torch.tensor(args.lambda_risk).to(args.device)
#
#             total_losses.update(loss.item(), X.size(0))
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             t.set_postfix(
#                 cls_losses='{:05.8f}'.format(cls_losses.avg),
#                 kd_losses='{:05.8f}'.format(kd_losses.avg),
#                 risk_losses='{:05.8f}'.format(risk_losses.avg),
#                 total_losses='{:05.8f}'.format(total_losses.avg),
#             )
#             t.update()
# def afstest(snet, tnet, criterionCls, criterionKD, testDataLoader, args):
#     cls_losses = AverageMeter()
#     kd_losses = AverageMeter()
#     correct = 0
#
#     snet.eval()
#     tnet.eval()
#
#     total_pred = []
#     total_y = []
#
#     with torch.no_grad():
#         with tqdm(total=len(testDataLoader)) as t:
#             for X, Y in tqdm(testDataLoader):
#                 X = X.to(args.device)
#                 Y = Y.to(args.device)
#
#                 # if 'PathMNIST' in args.root:
#                 Y = Y.squeeze().long()
#
#                 snet_pred = snet(X)
#                 tnet_pred = tnet(X)
#
#                 cls_loss = criterionCls(snet_pred, Y)
#                 kd_loss = criterionKD(snet_pred, tnet_pred.detach()) * args.lambda_kd
#
#                 pred = snet_pred.data.max(1)[1]
#                 correct += pred.eq(Y.view(-1)).sum().item()
#
#                 cls_losses.update(cls_loss.item(), X.size(0))
#                 kd_losses.update(kd_loss.item(), X.size(0))
#
#                 t.set_postfix(
#                     cls_losses='{:05.8f}'.format(cls_losses.avg),
#                     kd_losses='{:05.8f}'.format(kd_losses.avg),
#                 )
#                 t.update()
#
#                 total_pred += pred
#                 total_y += Y.view(-1)
#
#     test_acc = correct / len(testDataLoader.dataset)
#     stat = Performance(total_pred, total_y)
#
#     return test_acc, stat
# # 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("main")

random_seed = 32
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

CONFIG = {
'BASE1': {
    'BASE' : list(range(0,20000))
},

'BASE2': {
    'BASE' : list(range(0,19900))
},
'BASE4': {
    'BASE' : list(range(0,7000))
},

'TEST1': {
    'TEST' : list(range(20000,25000))
},
'CAL_100': {
    'CAL' : list(range(40000,40100))
},
'CAL_1000': {
    'CAL' : list(range(25000,26000))
},
'CAL_2000': {
    'CAL' : list(range(40000,42000))
},
'CAL_5000': {
    'CAL' : list(range(40000,45000))
},
'CALTEST1': {
    'TEST' : list(range(26000,27000))
},
'QO_5000': {
    'QUERY': list(range(0,5000)),
    'QUERY_MEMBER': [1 for i in range(5000)]
    },
'QNO_5000': {
    'QUERY': list(range(50000,55000)),
    'QUERY_MEMBER': [0 for i in range(5000)]
    },
'QF_100':{
    'QUERY': list(range(5900,6000)),
    'QUERY_MEMBER': [1 for i in range(100)]
},

'QF_1':{
    'QUERY': list(range(5999,6000)),
    'QUERY_MEMBER': [1 for i in range(100)]
},
'QF_1000':{
    'QUERY': list(range(9000,10000)),
    'QUERY_MEMBER': [1 for i in range(1000)]
},
'KD0.25': {
    'BASE': list(range(0,5000))
},
'KD0.5': {
    'BASE': list(range(0,10000))
},
'KD0.75': {
    'BASE': list(range(0,15000))
},
}

# class ExpandToRGB:
#     """将单通道Tensor图像扩展为三通道"""
#     def __call__(self, tensor):
#         # 检查是否为单通道图像（形状为 [1, H, W]）
#         if tensor.shape[0] == 1:
#             # 重复通道以形成3通道图像
#             tensor = tensor.repeat(3, 1, 1)
#         return tensor








csv_file = '/root/autodl-tmp/wangbin/yiwang/data/diabetes_binary_train.csv' # 数据集路径

# csv_file = '/mnt/f/BaiduNetdiskDownload/yiwang/data/final.csv' # 数据集路径
train_dataset = TableDataset(csv_file=csv_file, transform=transform)
train_dataset_no = TableDataset(csv_file=csv_file, transform=transform_no)
test_dataset = TableDataset(csv_file=csv_file, transform=transform)
# train_dataset.shuffle_data()  # 
# train_dataset_no.shuffle_data()

base1_indices = CONFIG['BASE1']['BASE']
base1_dataset = Subset(train_dataset, base1_indices)
base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
base2_indices = CONFIG['BASE2']['BASE']
base2_dataset = Subset(train_dataset, base2_indices)
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))

test1_loader = DataLoader(Subset(train_dataset, CONFIG['TEST1']['TEST']), batch_size=32, shuffle=False,
                          generator=torch.Generator().manual_seed(random_seed))
# qf_50_loader = DataLoader(Subset(train_dataset, CONFIG['QF_50']['QUERY']), batch_size=32, shuffle=True,
#                             generator=torch.Generator().manual_seed(random_seed))

qf_1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=1, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(random_seed))
cal_100_loader = DataLoader(Subset(train_dataset, CONFIG['CAL_100']['CAL']), batch_size=32, shuffle=False,
                             generator=torch.Generator().manual_seed(random_seed))
cal_1000_loader= DataLoader(Subset(train_dataset, CONFIG['CAL_1000']['CAL']), batch_size=32, shuffle=False,
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

# feature_extractorteacherv1 = FeatureExtractor(dim_in=24, dim_hidden=256)  # 注意这里的要调整为适应于表格数据的dim_in=8，dim_out=2，可以改
# classifierteacherv1 = Classifier(dim_hidden=256, dim_out=2)
#
# # 创建组合模型实例
# tmodelmlp = CombinedModel(feature_extractorteacherv1, classifierteacherv1).to(device)
# feature_extractorstudentv1 = FeatureExtractor(dim_in=24, dim_hidden=64)
# classifierstudentv1 = Classifier(dim_hidden=64, dim_out=2)
# feature_extractorstudentv2 = FeatureExtractor(dim_in=24, dim_hidden=64)
# classifierstudentv2 = Classifier(dim_hidden=64, dim_out=2)
# smodelmlp = CombinedModel(feature_extractorstudentv1, classifierstudentv1).to(device)
# # smodelmlp = smodelmlp.to(device)
#
# smodelmlp_base2 = CombinedModel(feature_extractorstudentv2, classifierstudentv2).to(device)
# smodelmlp = smodelmlp.to(device)
# smodelmlp_base2  = smodelmlp_base2 .to(device)
# tmodelmlp = tmodelmlp.to(device)
#
# tmodelmlp.apply(weights_init)
# smodelmlp.apply(weights_init)
model =get_teacher_model().to(device)
model_strained =get_student_model().to(device)
model_s =get_student_model().to(device)





epochs =30
learning_rate = 0.001

# 数据加载和预处理

best_accuracy=0
best_accuracy_strained=0
best_accuracy_s=0



best_model_state_retrained = None
best_model_state_trained = None
best_model_state_strained = None


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer_strained = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_strained = torch.optim.Adam(model_strained.parameters(), lr=learning_rate)
optimizer_s = torch.optim.Adam(model_s.parameters(), lr=learning_rate)


for qf1_start in range(19910, 19999):
    # best_model_state_retrained =torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_retrained_di.pth')
    # best_model_state_retrained =torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_retrained.pth')
    best_model_state_retrained =torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/diabetes/best_retrained_test1.pth')
    # best_model_state_strained=torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_strained_di.pth')
    # best_model_state_strained=torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_strained.pth')
    best_model_state_strained=torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/diabetes/best_strained_test_zui1.pth')
    # best_model_state_trained = torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_trained_di.pth')
    # best_model_state_trained = torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_trained.pth')
    best_model_state_trained = torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/diabetes/best_trained_test1.pth')
    u11=best_model_state_retrained
    logger.info(f'>>qf1_start: {qf1_start}')
    qf1_end = qf1_start + 1  
    CONFIG['QF1'] = {
        'QUERY': list(range(qf1_start, qf1_end)),
        'QUERY_MEMBER': [1 for _ in range(qf1_end - qf1_start)]
    }
    qf_1_indices = CONFIG['QF1']['QUERY']
    qf_1_dataset = Subset(train_dataset, qf_1_indices)
    qf1_loader = DataLoader(Subset(train_dataset, CONFIG['QF1']['QUERY']), batch_size=32, shuffle=False)
    average_euclidean_retrained, average_manhattan_retrained, average_cosine_similarity_retrained = analyze_sample_similarity(
        model_s, u11,
        device,
        train_dataset,
        CONFIG)
    print(
        f'>>average_euclidean_retrained: {average_euclidean_retrained}, average_manhattan_retrained: {average_manhattan_retrained}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained}')
    logger.info(
        f'>>average_euclidean_retrained: {average_euclidean_retrained}, average_manhattan_retrained: {average_manhattan_retrained}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained}')

    network = 'resnet18'
    dataset = 'cifar10'
    method = 'ssd_tuning'
    classes = 2
    train_and_forget(network,kd0_5_loader,test1_loader, best_model_state_strained, best_model_state_retrained, dataset, classes, qf_1_dataset,qf1_loader, base2_dataset, method, gpu=True,train_dataset=train_dataset,CONFIG=CONFIG,cal_1000_loader=cal_100_loader,caltest1_loader=caltest1_loader)
    method = 'amnesiac'

    train_and_forget(network,kd0_5_loader,test1_loader, best_model_state_strained, best_model_state_retrained, dataset, classes, qf_1_dataset,qf1_loader, base2_dataset, method, gpu=True,train_dataset=train_dataset,CONFIG=CONFIG,cal_1000_loader=cal_100_loader,caltest1_loader=caltest1_loader)
    #
    method = 'blindspot'
    train_and_forget(network, kd0_5_loader,test1_loader,best_model_state_strained, best_model_state_retrained, dataset, classes, qf_1_dataset,qf1_loader, base2_dataset, method, gpu=True,train_dataset=train_dataset,CONFIG=CONFIG,cal_1000_loader=cal_100_loader,caltest1_loader=caltest1_loader)

    method = 'FisherForgetting'
    train_and_forget(network, kd0_5_loader,test1_loader,best_model_state_strained, best_model_state_retrained, dataset, classes, qf_1_dataset,qf1_loader, base2_dataset, method, gpu=True,train_dataset=train_dataset,CONFIG=CONFIG,cal_1000_loader=cal_100_loader,caltest1_loader=caltest1_loader)


    subset_indices = base1_indices  
    train_labels = TableDataset.get_labels(train_dataset, subset_indices)
    instance(base1_dataset, base1_indices, test1_loader, best_model_state_retrained, best_model_state_strained,cal_100_loader, caltest1_loader, qf_1_indices,CONFIG,train_labels=train_labels,train_dataset=train_dataset)

    start_time = time.time()
    adaptforget(
        lambda_domain=1,
        lambda_risk=1,
        lambda_kd=1,
        train_dataset=train_dataset,
        num_epochsall=60,
        device=device,
        qf_100_loader=qf1_loader,
        kd=kd0_5_loader,
        test1_loader=test1_loader,
        cal_1000_loader=cal_100_loader,
        caltest1_loader=caltest1_loader,
        best_model_state_retrained=best_model_state_retrained,
        best_model_state_trained=best_model_state_trained,
        CONFIG=CONFIG,
        kd0_5_loader_no=kd0_5_loader_no,
        base2_loader=base2_loader,
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(f"The adaptforget function completed in {total_time:.2f} seconds.")
    # args = parser()
    # afs(args,best_model_state_trained,best_model_state_retrained,kd0_5_loader,base2_loader,test1_loader,cal_100_loader,caltest1_loader,qf1_loader,device,train_dataset,CONFIG)

# adaptforget(
#     num_epochsall=50,
#     device=device,
#     qf_100_loader=qf_1_loader,
#     kd0_5_loader=kd0_5_loader,
#     test1_loader=test1_loader,
#     cal_1000_loader=cal_1000_loader,
#     caltest1_loader=caltest1_loader,
#     best_model_state_retrained=best_model_state_retrained,
#     best_model_state_trained=best_model_state_trained
# )

# afs(args,best_model_state_trained)
# afs(args,best_model_state_trained,base1_loader,base2_loader,test1_loader,cal_1000_loader,caltest1_loader,qf_100_loader,device)
