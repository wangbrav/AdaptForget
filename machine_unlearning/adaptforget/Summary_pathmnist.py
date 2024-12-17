from __future__ import print_function
import sys
sys.path.append('/root/autodl-tmp/wangbin/yiwang')
import copy

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
from Domainadaptation_net_three import domainadaptation
from test_model_path import test_model
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
from mu.net_train_three import get_student_model, get_teacher_model, get_student_model_t
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
# TODO xiangbi shangyige  zengjia le  afs
from qf1kosiam import analyze_sample_similarity

from calculate_kl_divergence import calculate_kl_divergence
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
# from Dataset import DataModule, CONFIG
# from Model import get_teacher_model, get_student_model
from utils_w import *
from utils_ww import *
logging.basicConfig(filename='./tc/path.log', level=logging.INFO, format='%(asctime)s %(message)s')
# logging.basicConfig(filename='./tc/training_log_qf1circulate_pathmnistfinalv23.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

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
#                                 lr=0.001)
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
#     epochs= 20
#     # teacher_model.apply(initialize_weights)
#     for epoch in range(epochs):
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
#             logger.info(f'>>stat{stat}')
#             _t, pv, EMA_res, risk_score = api(device, snet, u, f_u, qf_100_loader, member_gt, cal_1000_loader,
#                                               caltest1_loader)
#             print(f'>>best test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#             logger.info(f'>>afs best test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#             average_euclidean_afs, average_manhattan_afs, average_cosine_similarity_afs=analyze_sample_similarity(snet, u, device, train_dataset, CONFIG)
#             print(f'>>average_euclidean_afs: {average_euclidean_afs}, average_manhattan_afs: {average_manhattan_afs}, average_cosine_similarity_afs: {average_cosine_similarity_afs}')
#             logger.info(f'>>average_euclidean_afs: {average_euclidean_afs}, average_manhattan_afs: {average_manhattan_afs}, average_cosine_similarity_afs: {average_cosine_similarity_afs}')
#             average_kl_div_afs=calculate_kl_divergence(snet, best_model_state_retrained, snet_base2, qf_100_loader, device)
#             print(f'>>average_kl_div_afs: {average_kl_div_afs}')
#             logger.info(f'>>average_kl_div_afs: {average_kl_div_afs}')
#             # save_best_ckpt(snet, args)
#             # best_model_state_afs = snet.state_dict().copy()
#
#         # save_last_ckpt(snet, args)
#
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
#     # parser.add_argument('add_risk_loss',
#     #                     type=int,
#     #                     default=1,
#     #                     help='')
#
#     args = parser.parse_args()
#     return args
#     # args = parser_forget.parse_args()
# args = parser()
#
#     # args = parser(args.root)
# def init_weights(m):
#     if type(m) == nn.Conv2d or type(m) == nn.Linear:
#         init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             init.zeros_(m.bias)
# # def zero_gradients(x):
# #     if isinstance(x, torch.Tensor):
# #         if x.grad is not None:
# #             x.grad.detach_()
# #             x.grad.zero_()
# #     elif isinstance(x, collections.abc.Iterable):
# #         for elem in x:
# #             zero_gradients(elem)
#
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
#
#                 pred = outputs.data.max(1)[1]
#                 # print(predicted.shape)
#                 # print(Y.shape)
#                 total += Y.size(0)
#
#                 Y = Y.squeeze(1).long()
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
#
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
# def adaptforget(num_epochsall, device, qf_100_loader, kd, test1_loader, cal_1000_loader,
#                                caltest1_loader, best_model_state_retrained, best_model_state_trained):
#     smodelmlp = get_student_model().to(device)
#     smodelmlp_base2 = get_student_model().to(device)
#     tmodelmlp = get_student_model().to(device)
#     tmodelmlp.apply(init_weights)
#     smodelmlp.apply(init_weights)
#     u = smodelmlp.state_dict()
#     f_u = smodelmlp.feature_extractor.state_dict()
#     modelmlp = get_teacher_model().to(device)
#
#     smodelmlp_base2.load_state_dict(best_model_state_retrained)
#     modelmlp.load_state_dict(best_model_state_trained)
#
#     member_gt = [1 for _ in range(1)]
#     num = 0
#     best_accuracy = 0.0
#
#     for epoch in range(num_epochsall):
#         # print(f"Epoch {epoch} starting...")
#         print(f"Epoch {epoch} starting...")
#         logger.info(f"Epoch {epoch} starting...")
#
#         #  machine unlearning
#         f_u, u, c_u = train_student_model_random(qf_100_loader, kd,test1_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u)
#         current_accuracy, accuracy1 = test_model(test1_loader, qf_100_loader, kd0_5_loader, device, modelmlp,
#                                                  smodelmlp, tmodelmlp, u, f_u)
#         logger.info(f'>>current_accuracy: {current_accuracy}, accuracy1: {accuracy1}')
#         average_euclidean_adapt, average_manhattan_adapt, average_cosine_similarity_adapt = analyze_sample_similarity(smodelmlp, u,
#                                                                                                                 device,
#                                                                                                                 train_dataset,
#                                                                                                                 CONFIG)
#         # logger
#         print(f'>>average_euclidean_adapt: {average_euclidean_adapt}, average_manhattan_adapt: {average_manhattan_adapt}, average_cosine_similarity_adapt: {average_cosine_similarity_adapt}')
#         logger.info(f'>>average_euclidean_adapt: {average_euclidean_adapt}, average_manhattan_adapt: {average_manhattan_adapt}, average_cosine_similarity_adapt: {average_cosine_similarity_adapt}')
#
#
#         average_kl_div_adapt=calculate_kl_divergence(smodelmlp,best_model_state_retrained,smodelmlp_base2, qf_100_loader, device)
#         print(f'>>average_kl_div_adapt: {average_kl_div_adapt}')
#         logger.info(f'>>average_kl_div_adapt: {average_kl_div_adapt}')
#         _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_100_loader, member_gt, cal_1000_loader,
#                                           caltest1_loader)
#         logger.info(f'Test value: {_t}, p-value: {pv}, EMA: {EMA_res}, Risk score: {risk_score}')
#         # print(f'Test value: {_t}, p-value: {pv}, EMA: {EMA_res}, Risk score: {risk_score}')
#
#
#         # f_u = domainadaptation(f_u, c_u, qf_100_loader, kd0_5_loader_no)
#         analyze_sample_similarity(smodelmlp,u,device,train_dataset,CONFIG)
#         calculate_kl_divergence(smodelmlp, best_model_state_retrained,smodelmlp_base2, qf_100_loader, device)
#         _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_100_loader, member_gt, cal_1000_loader, caltest1_loader)
#         current_accuracy = test_model(test1_loader, qf_100_loader, kd0_5_loader, device, modelmlp, smodelmlp, tmodelmlp, u,f_u)
#         num += 1
#         print(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#         # print(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#         logger.info(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("main")

npz_file_path = '/root/autodl-tmp/wangbin/yiwang/data/pathmnist.npz'

data = np.load(npz_file_path)


images = data['train_images']
labels = data['train_labels']

images_cal =data['val_images']
labels_cal =data['val_labels']
images_test = data['test_images']
labels_test =data['test_labels']

random_seed = 32
# random.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
#
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(random_seed)
def set_seed(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(32)
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
        image = self.images[idx]
        label = self.labels[idx]
        # print(f"Image shape at index {idx}: {image.shape}")

        if self.transform:
            image = self.transform(image)
        return image, label

    def shuffle_data(self, seed=32):
        print(f"Shuffling data with seed {seed}")

        np.random.seed(seed)
        indices = np.arange(len(self.original_images))
        np.random.shuffle(indices)
        self.images = self.original_images[indices]
        self.labels = self.original_labels[indices]
        # if self.transform:
        #     self.images = np.array([self.transform(image) for image in self.images])

    def get_labels(self, indices):
            return [self.labels[i] for i in indices]
CONFIG = {
    'BASE1': {
        'BASE': list(range(0, 10000))
    },
    'BASE2': {
        'BASE': list(range(0, 9900))
    },
    'TEST2': {
        'TEST': list(range(0, 5000))
    },
    'TEST1': {
        'TEST': list(range(30000, 35000))
    },
    'CAL_100': {
        'CAL': list(range(40000, 40100))
    },
    'CAL_1000': {
        'CAL': list(range(40000, 41000))
    },
    'CAL_2000': {
        'CAL': list(range(40000, 42000))
    },
    'CAL_5000': {
        'CAL': list(range(40000, 45000))
    },
    'CALTEST1': {
        'TEST': list(range(46000, 47000))
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
        'QUERY': list(range(9900, 10000)),
        'QUERY_MEMBER': [1 for i in range(100)]
    },
    'QF_10': {
        'QUERY': list(range(9990, 10000)),
        'QUERY_MEMBER': [1 for i in range(10)]
    },
    # 'QF_1': {
    #     'QUERY': list(range(9999, 10000)),
    #     'QUERY_MEMBER': [1 for i in range(1)]
    # },
    'QF_1000': {
        'QUERY': list(range(9000, 10000)),
        'QUERY_MEMBER': [1 for i in range(1000)]
    },
    'KD0.25': {
        'BASE': list(range(0, 2500))
    },
    'KD0.01': {
        'BASE': list(range(0, 100))
    },

    'KD0.5': {
        'BASE': list(range(0, 5000))
    },
    'KD0.75': {
        'BASE': list(range(0, 7500))
    },
}
class ExpandToRGB:
    def __call__(self, tensor):
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor
def add_salt_and_pepper_noise(img):

    amount = 0.1
    salt_vs_pepper = 0.5
    num_salt = np.ceil(amount * img.numel() * salt_vs_pepper)
    num_pepper = np.ceil(amount * img.numel() * (1.0 - salt_vs_pepper))
    indices = torch.randperm(img.numel())[:int(num_salt)]
    img.view(-1)[indices] = 1

    indices = torch.randperm(img.numel())[:int(num_pepper)]
    img.view(-1)[indices] = 0
    return img
transform = transforms.Compose([
    transforms.ToTensor(),
    ExpandToRGB(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_no_salt_pepper = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(add_salt_and_pepper_noise),
    # transforms.Lambda(add_speckle_noise),
    # transforms.Lambda(add_gaussian_noise)

])



train_dataset = PathMNISTDataset(images, labels, transform=transform)
test_dataset = PathMNISTDataset(images_test, labels_test, transform=transform)
# plot_images_per_class(train_dataset, 9)

train_dataset.shuffle_data()

train_dataset_no = PathMNISTDataset(images, labels, transform=transform_no_salt_pepper)
import os
import matplotlib.pyplot as plt

output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

num_images_to_save = 5
for i in range(num_images_to_save):
    image, label = train_dataset_no[i]
    image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.savefig(f'{output_dir}/image_{i}.png')
    plt.close()


train_dataset_no.shuffle_data()
base1_indices = CONFIG['BASE1']['BASE']
base1_dataset = Subset(train_dataset, base1_indices)
base1_loader = DataLoader(Subset(train_dataset, CONFIG['BASE1']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
base2_indices = CONFIG['BASE2']['BASE']
base2_dataset = Subset(train_dataset, base2_indices)
base2_loader = DataLoader(Subset(train_dataset, CONFIG['BASE2']['BASE']), batch_size=32, shuffle=True,
                          generator=torch.Generator().manual_seed(random_seed))
test1_loader = DataLoader(Subset(test_dataset, CONFIG['TEST2']['TEST']), batch_size=32, shuffle=False,
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
subsets = [Subset(train_dataset, CONFIG['QF_100']['QUERY']) for _ in range(10)]
concat_dataset = ConcatDataset(subsets)
qf_100_loader10 = DataLoader(
    concat_dataset,
    batch_size=64,
    shuffle=True,
    generator=torch.Generator().manual_seed(random_seed)
)
qf_100_loader = DataLoader(Subset(train_dataset, CONFIG['QF_100']['QUERY']), batch_size=16, shuffle=True,
                           generator=torch.Generator().manual_seed(random_seed))

# qf_1_loader = DataLoader(Subset(train_dataset, CONFIG['QF_1']['QUERY']), batch_size=32, shuffle=True,
#                          generator=torch.Generator().manual_seed(random_seed))
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
model =get_teacher_model().to(device)
model_strained =get_student_model().to(device)
model_s =get_student_model().to(device)


# epochs =30
# learning_rate = 0.001
# #
# #
# # best_accuracy=0
# best_accuracy_strained=0
# # best_accuracy_s=0
# #
# # best_model_state_retrained = None
# # best_model_state_trained = None
# best_model_state_strained = None
# #
# #
# criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # # optimizer_strained = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer_strained = torch.optim.Adam(model_strained.parameters(), lr=learning_rate)
# # optimizer_s = torch.optim.Adam(model_s.parameters(), lr=learning_rate)
#
# for epoch in range(epochs):
#     model_strained.train()
#     correct = 0
#     train_loss = 0
#
#     with tqdm(total=len(base1_loader)) as t:
#         for X, Y in tqdm(base1_loader):
#             X = X.to(device)
#             Y = Y.to(device)
#             # Training pass
#             optimizer_strained.zero_grad()
#             output = model_strained(X)
#             # print(output)
#             # output = torch.argmax(output, dim=1)
#             # print(output.shape)
#             # print(output)
#             # print(Y.shape)
#             # print(Y)
#             # Y = Y.squeeze(1)
#             # print(Yield.shape)
#             # print(Y)
#             Y = Y.squeeze(1).long()
#             # print("Model output shape:", output.shape)
#             # print("Target labels:", Y.unique())
#             loss = criterion(output, Y)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer_strained.step()
#             pred = output.data.max(1)[1]
#             correct += pred.eq(Y.view(-1)).sum().item()
#
#     train_loss /= len(base1_loader.dataset)
#     train_accuracy = 100 * correct / len(base1_loader.dataset)
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#     best_model_state_strained_zui = model_strained.state_dict().copy()
#     save_dir = "./quanzhong/"
#     save_path_strained_zui = os.path.join(save_dir, "best_strained_test_zui.pth")
#     torch.save(best_model_state_strained_zui, save_path_strained_zui)
#
#     test_accuracy = test(model_strained, test1_loader, device)
#     print(f"Test Accuracy: {test_accuracy:.2f}%")

    # if test_accuracy > best_accuracy_strained:
    #     best_accuracy_strained = test_accuracy
    #     logger.info(f"Saving best model with accuracy {best_accuracy_strained}")
    #     best_model_state_strained = model_strained.state_dict().copy()
    #     save_dir = "./quanzhong/"
    #
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #
    #     save_path_strained = os.path.join(save_dir, "best_strained_test.pth")
    #
    #     torch.save(best_model_state_strained, save_path_strained)
    #     logger.info(f"Model weights saved successfully to {save_path_strained}.")

# for epoch in range(epochs):
#     model.train()
#     correct = 0
#     train_loss = 0
#
#     with tqdm(total=len(base1_loader)) as t:
#         for X, Y in tqdm(base1_loader):
#             X = X.to(device)
#             Y = Y.to(device)
#             # Training pass
#             optimizer.zero_grad()
#             output = model(X)
#             # print(output)
#             # output = torch.argmax(output, dim=1)
#             # print(output.shape)
#             # print(output)
#             # print(Y.shape)
#             # print(Y)
#             # Y = Y.squeeze(1)
#             # print(Yield.shape)
#             # print(Y)
#             Y = Y.squeeze(1).long()
#             # print("Model output shape:", output.shape)
#             # print("Target labels:", Y.unique())
#             loss = criterion(output, Y)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
#             pred = output.data.max(1)[1]
#             correct += pred.eq(Y.view(-1)).sum().item()
#
#     train_loss /= len(base1_loader.dataset)
#     train_accuracy = 100 * correct / len(base1_loader.dataset)
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#
#     test_accuracy = test(model, test1_loader, device)
#     print(f"Test Accuracy: {test_accuracy:.2f}%")
#
#     if test_accuracy > best_accuracy:
#         best_accuracy = test_accuracy
#         logger.info(f"Saving best model with accuracy {best_accuracy}")
#         best_model_state_trained = model.state_dict().copy()
#         save_dir = "./quanzhong/"
#
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         save_path_trained = os.path.join(save_dir, "best_trained_test.pth")
#         torch.save(best_model_state_trained, save_path_trained)
#         logger.info(f"Model weights saved successfully to {save_path_trained}.")
#
# for epoch in range(epochs):
#     model_s.train()
#     correct = 0
#     train_loss = 0
#
#     with tqdm(total=len(base2_loader)) as t:
#         for X, Y in tqdm(base2_loader):
#             X = X.to(device)
#             Y = Y.to(device)
#             # Training pass
#             optimizer_s.zero_grad()
#             output = model_s(X)
#             # print(output)
#             # output = torch.argmax(output, dim=1)
#             # print(output.shape)
#             # print(output)
#             # print(Y.shape)
#             # print(Y)
#             # Y = Y.squeeze(1)
#             # print(Yield.shape)
#             # print(Y)
#             Y = Y.squeeze(1).long()
#             # print("Model output shape:", output.shape)
#             # print("Target labels:", Y.unique())
#             loss = criterion(output, Y)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer_s.step()
#             pred = output.data.max(1)[1]
#             correct += pred.eq(Y.view(-1)).sum().item()
#
#     train_loss /= len(base2_loader.dataset)
#     train_accuracy = 100 * correct / len(base2_loader.dataset)
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#
#     test_accuracy = test(model_s, test1_loader, device)
#     print(f"Test Accuracy: {test_accuracy:.2f}%")
#
#     if test_accuracy > best_accuracy_s:
#         best_accuracy_s = test_accuracy
#         logger.info(f"Saving retrained best model with accuracy {best_accuracy_s}")
#         best_model_state_retrained = model_s.state_dict().copy()
#         u = best_model_state_retrained
#         save_dir = "./quanzhong/"
#
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         save_path_retrained = os.path.join(save_dir, "best_retrained_test.pth")
#
#         torch.save(best_model_state_retrained, save_path_retrained)
#         logger.info(f"Model weights saved successfully to {save_path_retrained}.")
#
#         # average_euclidean_retrained, average_manhattan_retrained, average_cosine_similarity_retrained = analyze_sample_similarity(model_s, u,
#         #                                                                                                         device,
#         #                                                                                                         train_dataset,
#         #                                                                                                         CONFIG)
#         # print(f'>>average_euclidean_retrained: {average_euclidean_retrained}, average_manhattan_retrained: {average_manhattan_retrained}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained}')
#         # logger.info(f'>>average_euclidean_retrained: {average_euclidean_retrained}, average_manhattan_retrained: {average_manhattan_retrained}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained}')
#         #

# def adaptforget(num_epochsall, device, qf_100_loader, kd0_5_loader, test1_loader, cal_1000_loader,
#                                caltest1_loader, best_model_state_retrained, best_model_state_trained):
#     smodelmlp = get_student_model().to(device)
#     smodelmlp_base2 = get_student_model().to(device)
#     tmodelmlp = get_student_model().to(device)
#     tmodelmlp.apply(init_weights)
#     smodelmlp.apply(init_weights)
#     u = smodelmlp.state_dict()
#     f_u = smodelmlp.feature_extractor.state_dict()
#     modelmlp = get_teacher_model().to(device)
#
#     smodelmlp_base2.load_state_dict(best_model_state_retrained)
#     modelmlp.load_state_dict(best_model_state_trained)
#
#     member_gt = [1 for _ in range(1)]
#     num = 0
#     best_accuracy = 0.0
#
#     for epoch in range(num_epochsall):
#         # print(f"Epoch {epoch} starting...")
#         print(f"Epoch {epoch} starting...")
#         logger.info(f"Epoch {epoch} starting...")
#
#         # 进行 machine unlearning
#         f_u, u, c_u = train_student_model_random(qf_100_loader, base1_loader,test1_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u)
#         current_accuracy, accuracy1 = test_model(test1_loader, qf_100_loader, kd0_5_loader, device, modelmlp,
#                                                  smodelmlp, tmodelmlp, u, f_u)
#         logger.info(f'>>current_accuracy: {current_accuracy}, accuracy1: {accuracy1}')
#         average_euclidean_adapt, average_manhattan_adapt, average_cosine_similarity_adapt = analyze_sample_similarity(smodelmlp, u,
#                                                                                                                 device,
#                                                                                                                 train_dataset,
#                                                                                                                 CONFIG)
#         # logger
#         print(f'>>average_euclidean_adapt: {average_euclidean_adapt}, average_manhattan_adapt: {average_manhattan_adapt}, average_cosine_similarity_adapt: {average_cosine_similarity_adapt}')
#         logger.info(f'>>average_euclidean_adapt: {average_euclidean_adapt}, average_manhattan_adapt: {average_manhattan_adapt}, average_cosine_similarity_adapt: {average_cosine_similarity_adapt}')
#
#
#         average_kl_div_adapt=calculate_kl_divergence(smodelmlp,best_model_state_retrained,smodelmlp_base2, qf_100_loader, device)
#         print(f'>>average_kl_div_adapt: {average_kl_div_adapt}')
#         logger.info(f'>>average_kl_div_adapt: {average_kl_div_adapt}')
#         _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_100_loader, member_gt, cal_1000_loader,
#                                           caltest1_loader)
#         logger.info(f'Test value: {_t}, p-value: {pv}, EMA: {EMA_res}, Risk score: {risk_score}')
#         # print(f'Test value: {_t}, p-value: {pv}, EMA: {EMA_res}, Risk score: {risk_score}')
#

#         # f_u = domainadaptation(f_u, c_u, qf_100_loader, kd0_5_loader_no)
#         analyze_sample_similarity(smodelmlp,u,device,train_dataset,CONFIG)
#         calculate_kl_divergence(smodelmlp, best_model_state_retrained,smodelmlp_base2, qf_100_loader, device)
#         _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_100_loader, member_gt, cal_1000_loader, caltest1_loader)
#         current_accuracy = test_model(test1_loader, qf_100_loader, kd0_5_loader, device, modelmlp, smodelmlp, tmodelmlp, u,f_u)
#         num += 1
#         print(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#         # print(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#         logger.info(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')

# def instance(base1_dataset, base1_indices,test1_loader,best_model_state_retrained,best_model_state_strained,cal_1000_loader,caltest1_loader,qf_1_indices,CONFIG):
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=15, metavar='N',
#                         help='number of epochs to train (default: 14)')
#     parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--dry-run', action='store_true', default=False,
#                         help='quickly check a single pass')
#     parser.add_argument('--seed', type=int, default=0, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--save-model', action='store_true', default=False,
#                         help='For Saving the current Model')
#     parser.add_argument('--pgd-eps', type=float, default=2.0, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--pgd-alpha', type=float, default=0.1, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--pgd-iter', type=int, default=100, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#
#     parser.add_argument('--unlearn-label', type=int, default=9, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--unlearn-k', type=int, default=10, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--unlearn-lr', type=float, default=0.01, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--num-adv-images', type=int, default=None, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--reg-lamb', type=float, default=10.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     eps = args.pgd_eps
#     iters = args.pgd_iter
#     alpha = args.pgd_alpha
#     # k_arr = qf_100_indices
#     # k_arr = qf_100_indices
#     # print(qf_100_indices)
#     k_arr = [100]
#     #   k_arr = [1, 16, 64, 128, 256]
#     D_r_acc = []
#     D_f_acc = []
#     D_test_acc = []
#     case1_D_r = []
#     case2_D_r = []
#     case3_D_r = []
#     case1_D_f = []
#     case2_D_f = []
#     case3_D_f = []
#     case1_D_test = []
#     case2_D_test = []
#     case3_D_test = []
#
#     train_kwargs = {'batch_size': 256}
#     test_kwargs = {'batch_size': 256}
#     # test_kwargs = {'batch_size': 1024}
#
#     naiive_unlearn_kwargs = {'batch_size': 32}
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#
#     # dataset1 = datasets.CIFAR10('../data', train=True, download=True,
#     #                    transform=transform)
#
#     dataset1 = base1_dataset
#     # 一个 是训练数据集 一个 是测试数据集
#     # dataset2 = datasets.CIFAR10('../data', train=False,
#     #                    transform=transform)
#
#     if use_cuda:
#         cuda_kwargs = {'num_workers': 0,
#                        'pin_memory': True,
#                        'shuffle': False}
#         train_kwargs.update(cuda_kwargs)
#         test_kwargs.update(cuda_kwargs)
#
#     for unlearn_k in k_arr:
#
#         # torch.manual_seed(args.seed)
#         # torch.cuda.manual_seed(args.seed)
#         # torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         # np.random.seed(args.seed)
#         # random.seed(args.seed)
#
#         unlearn_label = args.unlearn_label
#         subset_indices = base1_indices  # 获取 Subset 的索引
#         train_labels = PathMNISTDataset.get_labels(train_dataset, subset_indices)
#
#         # train_labels = dataset1.labels
#         # train_labels = dataset1.targets
#
#         train_labels = torch.from_numpy(np.array(train_labels))
#
#         # indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]
#         # indices_k_unlearn = torch.tensor(qf_1_indices)
#         indices_k_unlearn = torch.tensor(qf_1_indices)
#
#         # indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]
#         # print(indices_k_unlearn)
#         # indices_k_unlearn = qf_100_indices
#         # print ('indices_k_unlearn : ', indices_k_unlearn)
#
#         copy_train_labels = train_labels.clone()
#         copy_train_labels[indices_k_unlearn] = -10
#         print('len(copy_train_labels) : ', len(copy_train_labels))
#
#         indices_other_data = (copy_train_labels != -10).nonzero(as_tuple=False)
#         indices_other_data = torch.unique(indices_other_data)
#         # indices_other_data = torch.unique(indices_other_data)
#
#         print('len(indices_other_data) : ', len(indices_other_data))
#         unlearn_dataset = Subset(dataset1, indices_k_unlearn.view(-1, ))
#         # print("indices_k_unlearn : ",indices_k_unlearn.view(-1,))
#         unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset, **naiive_unlearn_kwargs)
#
#         other_dataset = Subset(dataset1, indices_other_data.contiguous().view(-1, ))
#         print(len(Subset(dataset1, indices_other_data)))
#         print(indices_other_data.contiguous().view(-1, ).shape)
#         print('len(other_dataset) : ', len(other_dataset))
#         other_loader = torch.utils.data.DataLoader(other_dataset, **test_kwargs)
#         print('len(other_loader) : ', len(other_loader))
#         # print(indices_other_data.contiguous().view(-1, ).shape)
#
#         # cifar_test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
#
#         print('len(unlearn_dataset) : ', len(unlearn_dataset), ' len(other_dataset) : ', len(other_dataset))
#         criterion = nn.CrossEntropyLoss()
#
#         # optimizer = optim.Adam(modelmlp.parameters(), lr=learning_rate)
#         model = get_student_model().to(device)
#         # normalize_layer = NormalizeLayer((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         # model = torch.nn.Sequential(normalize_layer, model)
#
#         optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-3)
#
#         # model = resnet18().to(device)
#         # smodelmlp_base2.load_state_dict(
#         #     torch.load('/root/autodl-tmp/wangbin/yiwang/qf1model/best_model_path_qf1_base2.pth'))
#         # def train_model(model, trainloader, validloader, optimizer, criterion, device, epochs=10):
#         #     model.to(device)
#         #     best_accuracy = 0
#         #     best_model_state = None
#         #
#         #     for epoch in range(epochs):
#         #         model.train()
#         #         for inputs, labels in trainloader:
#         #             inputs, labels = inputs.to(device), labels.to(device)
#         #             optimizer.zero_grad()
#         #             outputs = model(inputs)
#         #             labels = labels.squeeze().long()
#         #             loss = criterion(outputs, labels)
#         #             loss.backward()
#         #             optimizer.step()
#         #
#         #         model.eval()
#         #         correct = 0
#         #         total = 0
#         #         with torch.no_grad():
#         #             for inputs, labels in validloader:
#         #                 inputs, labels = inputs.to(device), labels.to(device)
#         #                 outputs = model(inputs)
#         #                 _, predicted = torch.max(outputs.data, 1)
#         #                 total += labels.size(0)
#         #                 correct += (predicted == labels).sum().item()
#         #
#         #         accuracy = 100 * correct / total
#         #         print(f'Epoch {epoch + 1}: Validation Accuracy = {accuracy:.2f}%')
#         #
#         #         if accuracy > best_accuracy:
#         #             best_accuracy = accuracy
#         #             best_model_state = model.state_dict().copy()  # 深拷贝模型状态
#         #
#         #     return best_model_state, best_accuracy
#
#         # def train_model(model, trainloader, validloader, optimizer, criterion, device, epochs=10,
#         #                 model_path='/root/autodl-tmp/wangbin/L2UL-main/weights/best_model.pth'):
#         #     model.to(device)
#         #     best_accuracy = 0
#         #     best_model_state = None
#         #
#         #     for epoch in range(epochs):
#         #         model.train()
#         #         for inputs, labels in trainloader:
#         #             inputs, labels = inputs.to(device), labels.to(device)
#         #             optimizer.zero_grad()
#         #             outputs = model(inputs)
#         #             labels = labels.squeeze().long()
#         #             loss = criterion(outputs, labels)
#         #             loss.backward()
#         #             optimizer.step()
#         #
#         #         model.eval()
#         #         correct = 0
#         #         total = 0
#         #         with torch.no_grad():
#         #             for inputs, labels in validloader:
#         #                 inputs, labels = inputs.to(device), labels.to(device)
#         #                 outputs = model(inputs)
#         #                 _, predicted = torch.max(outputs.data, 1)
#         #                 total += labels.size(0)
#         #                 correct += (predicted == labels).sum().item()
#         #
#         #         accuracy = 100 * correct / total
#         #         print(f'Epoch {epoch + 1}: Validation Accuracy = {accuracy:.2f}%')
#         #
#         #         if accuracy > best_accuracy:
#         #             best_accuracy = accuracy
#         #             best_model_state = model.state_dict().copy()  # 深拷贝模型状态
#         #             torch.save(best_model_state, model_path)
#         #             print(f"Saved new best model with accuracy: {best_accuracy:.2f}%")
#         #
#         #     return best_model_state, best_accuracy
#         #
#         # def train_modelqf1(model, trainloader, validloader, optimizer, criterion, device, epochs=10,
#         #                    model_path='/root/autodl-tmp/wangbin/L2UL-main/weights/best_modelqf1.pth'):
#         #     model.to(device)
#         #     best_accuracy = 0
#         #     best_model_state = None
#         #
#         #     for epoch in range(epochs):
#         #         model.train()
#         #         for inputs, labels in trainloader:
#         #             inputs, labels = inputs.to(device), labels.to(device)
#         #             optimizer.zero_grad()
#         #             outputs = model(inputs)
#         #             labels = labels.squeeze().long()
#         #             loss = criterion(outputs, labels)
#         #             loss.backward()
#         #             optimizer.step()
#         #
#         #         model.eval()
#         #         correct = 0
#         #         total = 0
#         #         with torch.no_grad():
#         #             for inputs, labels in validloader:
#         #                 inputs, labels = inputs.to(device), labels.to(device)
#         #                 outputs = model(inputs)
#         #                 _, predicted = torch.max(outputs.data, 1)
#         #                 total += labels.size(0)
#         #                 correct += (predicted == labels).sum().item()
#         #
#         #         accuracy = 100 * correct / total
#         #         print(f'Epoch {epoch + 1}: Validation Accuracy = {accuracy:.2f}%')
#         #
#         #         if accuracy > best_accuracy:
#         #             best_accuracy = accuracy
#         #             best_model_state = model.state_dict().copy()
#         #             torch.save(best_model_state, model_path)
#         #             print(f"Saved new best model with accuracy: {best_accuracy:.2f}%")
#         #
#         #     return best_model_state, best_accuracy
#
#         best_model_statet = best_model_state_strained
#         best_model_stateqf1=best_model_state_retrained
#         print(f'Best Validation Accuracy: {best_accuracy:.2f}%')
#         # model.load_state_dict(best_model_state)
#
#         model.load_state_dict(best_model_statet)
#         # model.load_state_dict(torch.load('/root/autodl-tmp/wangbin/L2UL-main/weights/best_model.pth'))
#
#         # normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         # model = torch.nn.Sequential(normalize_layer, model)
#         # criterion = nn.CrossEntropyLoss()
#
#         # optimizer = optim.Adam(modelmlp.parameters(), lr=learning_rate)
#         optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
#         # optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-3)
#
#         model.eval()
#         print('Baseline 1: Naiive Appraoch - finetuning with D_forget (maximizing CE loss)')
#
#         # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
#         other_loss, other_acc, other_f1 = testins(model, device, other_loader)
#         # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
#         unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
#         # test_loss, test_acc,test_f1 = test(model, device, test1_loader)
#         test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
#         # test_loss, test_acc,test_f1 = test(model, device, cifar_test_loader)
#
#         str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
#             unlearn_acc) + ', D_test acc : ' + str(test_acc)
#
#         D_test_acc.append(test_acc)
#         D_r_acc.append(other_acc)
#         D_f_acc.append(unlearn_acc)
#
#         print(str_list)
#         print('before  D_test - D_forget f1:', other_f1, 'D_forget_f1:', unlearn_f1, 'D_test_f1:', test_f1)
#         unlearn_acc = 100
#         max_iter = 1000
#
#         j = 0
#
#         while unlearn_acc != 0:
#
#             naive_train(args, model, device, unlearn_loader, optimizer, 0)
#             # naive_train(args, model, device, unlearn_loader, optimizer, 0)
#             model.eval()
#             # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
#             unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
#
#             j += 1
#             # print(1)
#             if max_iter < j:
#                 break
#
#         model.eval()
#         # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
#         other_loss, other_acc, other_f1 = testins(model, device, other_loader)
#         test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
#         # test_loss, test_acc,test_f1 = test(model, device, cifar_test_loader)
#
#         str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
#             unlearn_acc) + ', D_test acc : ' + str(test_acc)
#         print(str_list)
#         print('after  D_test - D_forget f1:', other_f1, 'D_forget_f1:', test_f1, 'D_test_f1:', test_f1)
#         member_gt = [1 for i in range(1)]
#
#         qf1_model = deepcopy(model)
#         qf1_model.load_state_dict(best_model_stateqf1)
#
#         # average_euclidean0, average_manhattan0, average_cosine_similarity0 = analyze_sample_similarity(qf1_model,
#         #                                                                                                device,
#         #                                                                                                train_dataset,
#         #                                                                                                CONFIG)
#         # average_euclidean1, average_manhattan1, average_cosine_similarity1 = analyze_sample_similarity(model, device,
#         #                                                                                                train_dataset,
#         #                                                                                                CONFIG)
#         # average_kl_div = calculate_kl_divergence(model, qf1_model, unlearn_loader, device)
#         # _t, pv, EMA_res, risk_score = api(device, model, unlearn_loader, member_gt, cal_1000_loader, caltest1_loader)
#         # print(
#         #     f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#         # print(
#         #     f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
#         # print(
#         #     f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
#         # print(f'average_kl_div: {average_kl_div}')
#         case1_D_test.append(test_acc)
#         case1_D_r.append(other_acc)
#         case1_D_f.append(unlearn_acc)
#
#         # model = resnet18().to(device)
#         # model.load_state_dict(torch.load('./cifar10_pretrained_models/resnet18.pt'))
#         # normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         # model = torch.nn.Sequential(normalize_layer, model)
#         model = get_student_model().to(device)
#         # normalize_layer = NormalizeLayer((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         # model = torch.nn.Sequential(normalize_layer, model)
#         # model = resnet18().to(device)
#         model.load_state_dict(best_model_statet)
#
#         # model.load_state_dict(torch.load("/root/autodl-tmp/wangbin/L2UL-main/weights/best_model_path_base1.pth"))
#         # best_model_state, best_accuracy = train_model(model, base1_loader, test1_loader, optimizer, criterion, device)
#         # (print
#         #  (f'Best Validation Accuracy: {best_accuracy:.2f}%'))
#         # model.load_state_dict(best_model_state)
#
#         optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-3)
#
#         origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
#
#         print()
#         print('\n Baseline 2: Our Appraoch - using adversarial examples only')
#
#         unlearn_acc = 100
#         alpha = 0.0
#
#         model.eval()
#         # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
#         other_loss, other_acc, other_f1 = testins(model, device, other_loader)
#         # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
#         unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
#         test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
#         # test_loss, test_acc,test_f1 = test(model, device, cifar_test_loader)
#
#         str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
#             unlearn_acc) + ', D_test acc : ' + str(test_acc)
#         print(str_list)
#         print('before D_test - D_forget_f1:', other_f1, 'D_forget_f1:', unlearn_f1, 'D_test_f1:', test_f1)
#
#         adversary = L2PGDAttack(model, eps=args.pgd_eps, eps_iter=args.pgd_alpha, nb_iter=args.pgd_iter,
#                                 rand_init=True, targeted=True)
#
#         # adv_images, target_labels = adv_attack(args, model, device, qf_100_loader, adversary, unlearn_k, args.num_adv_images)
#         adv_images, target_labels = adv_attack(args, model, device, unlearn_loader, adversary, unlearn_k,
#                                                args.num_adv_images)
#
#         adv_dataset = JointDataset(adv_images, target_labels)
#         adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)
#
#         j = 0
#
#         # unlearn_loader_cycle = cycle(qf_100_loader)
#         unlearn_loader_cycle = cycle(unlearn_loader)
#         CE = nn.CrossEntropyLoss()
#
#         while unlearn_acc != 0:
#             model.train()
#             for i, data in enumerate(zip(adv_loader, unlearn_loader_cycle)):
#
#                 model.train()
#
#                 (adv_data, adv_target), (data, target) = data
#
#                 optimizer.zero_grad()
#
#                 output_adv = model(adv_data.to(device))
#                 output = model(data.to(device))
#                 # target = target.squeeze().long()  # 是path的
#                 # adv_target = adv_target.squeeze().long()  # 是path的
#                 tensor11 = torch.randn(1, 1)
#                 # print("target.size():", target.size())
#                 if target.size() == tensor11.size():
#                     target = target.squeeze(0).long()
#                 else:
#                     target = target.squeeze().long()
#                 if adv_target.size() == tensor11.size():
#                     adv_target = adv_target.squeeze(0).long()
#                 else:
#                     adv_target = adv_target.squeeze().long()
#                 loss_unlearn = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
#                 loss_adv = CE(output_adv, adv_target.to(device)) * (
#                             adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))
#
#                 loss = loss_unlearn + loss_adv
#
#                 loss.backward()
#                 optimizer.step()
#
#                 model.eval()
#                 unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
#                 # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
#
#                 if unlearn_acc == 0:
#                     print('unlearn_acc == 0, Break at j = ', j, ' i = ', i)
#                     break
#
#             j += 1
#
#             if max_iter < j:
#                 break
#
#         model.eval()
#         unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
#         # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
#         other_loss, other_acc, other_f1 = testins(model, device, other_loader)
#         # other_loss, other_acc,other_f1= test(model, device, base2_loader)
#         test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
#         str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
#             unlearn_acc) + ', D_test acc : ' + str(test_acc)
#         print(str_list)
#         print(' after D_test - D_forget f1:', other_f1, 'D_forget f1:', unlearn_f1, 'D_test f1:', test_f1)
#         member_gt = [1 for i in range(1)]
#         qf1_model = deepcopy(model)
#         qf1_model.load_state_dict(best_model_stateqf1)
#         #
#         # average_euclidean0, average_manhattan0, average_cosine_similarity0 = analyze_sample_similarity(qf1_model,
#         #                                                                                                device,
#         #                                                                                                train_dataset,
#         #                                                                                                CONFIG)
#         # average_euclidean1, average_manhattan1, average_cosine_similarity1 = analyze_sample_similarity(model, device,
#         #                                                                                                train_dataset,
#         #                                                                                                CONFIG)
#         # average_kl_div = calculate_kl_divergence(model, qf1_model, unlearn_loader, device)
#         # _t, pv, EMA_res, risk_score = api(device, model, unlearn_loader, member_gt, cal_1000_loader, caltest1_loader)
#         # print(
#         #     f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#         # print(
#         #     f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
#         # print(
#         #     f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
#         # print(f'average_kl_div: {average_kl_div}')
#
#         case2_D_test.append(test_acc)
#         case2_D_r.append(other_acc)
#         case2_D_f.append(unlearn_acc)
#
#         print()
#         print('\n Baseline 3: Our Appraoch - using both adversarial examples and weight importance')
#
#         # model = resnet18().to(device)
#         # model.load_state_dict(torch.load('./cifar10_pretrained_models/resnet18.pt'))
#         # normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         # model = torch.nn.Sequential(normalize_layer, model)
#         model = get_student_model().to(device)
#         # normalize_layer = NormalizeLayer((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         # normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         # model = torch.nn.Sequential(normalize_layer, model)
#         # model = resnet18().to(device)
#         model.load_state_dict(best_model_statet)
#
#         # best_model_state, best_accuracy = train_model(model, base1_loader, test1_loader, optimizer, criterion, device)
#         # (print
#         #  (f'Best Validation Accuracy: {best_accuracy:.2f}%'))
#         # model.load_state_dict(best_model_state)
#
#         # model.load_state_dict(torch.load("/root/autodl-tmp/wangbin/L2UL-main/weights/best_model_path_base1.pth"))
#         optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-3)
#
#         origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
#
#         model_for_importance = copy.deepcopy(model)
#         num_samples = len(unlearn_loader)
#         importance = estimate_parameter_importance(unlearn_loader, model_for_importance, device, num_samples, optimizer)
#         # importance = estimate_parameter_importance(unlearn_loader, model_for_importance, device, num_samples, optimizer)
#
#         for keys in importance.keys():
#             importance[keys] = (importance[keys] - importance[keys].min()) / (
#                         importance[keys].max() - importance[keys].min())
#             importance[keys] = (1 - importance[keys])
#
#         CE = nn.CrossEntropyLoss()
#
#         unlearn_acc = 100
#
#         model.eval()
#         other_loss, other_acc, other_f1 = testins(model, device, other_loader)
#         # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
#         unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
#         # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
#         test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
#
#         str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
#             unlearn_acc) + ', D_test acc : ' + str(test_acc)
#         print(str_list)
#         print('before D_test - D_forget f1:', other_f1, 'D_forget f1:', unlearn_f1, 'D_test f1:', test_f1)
#         adv_dataset = JointDataset(adv_images, target_labels)
#         adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)
#
#         j = 0
#         unlearn_loader_cycle = cycle(unlearn_loader)
#
#         while unlearn_acc != 0:
#             # while unlearn_acc > 50:
#             for i, data in enumerate(zip(adv_loader, unlearn_loader_cycle)):
#
#                 model.train()
#
#                 (adv_data, adv_target), (data, target) = data
#
#                 optimizer.zero_grad()
#
#                 output_adv = model(adv_data.to(device))
#                 output = model(data.to(device))
#                 # target = target.squeeze().long()
#                 # adv_target = adv_target.squeeze().long()
#                 tensor11 = torch.randn(1, 1)
#                 # print("target.size():", target.size())
#                 if target.size() == tensor11.size():
#                     target = target.squeeze(0).long()
#                 else:
#                     target = target.squeeze().long()
#
#                 if adv_target.size() == tensor11.size():
#                     adv_target = adv_target.squeeze(0).long()
#                 else:
#                     adv_target = adv_target.squeeze().long()
#                 loss_unlearn = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
#                 loss_adv = CE(output_adv, adv_target.to(device)) * (
#                             adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))
#
#                 loss_reg = 0
#
#                 for n, p in model.named_parameters():
#                     if n in importance.keys():
#                         loss_reg += torch.sum(importance[n] * (p - origin_params[n]).pow(2)) / 2
#
#                 loss = loss_unlearn + loss_adv + loss_reg * args.reg_lamb
#
#                 loss.backward()
#                 optimizer.step()
#
#                 model.eval()
#                 unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
#
#                 if unlearn_acc == 0:
#                     # if unlearn_acc <= 50:
#                     #     print ('unlearn_acc <=50 , Break at j = ', j, ' i = ', i)
#                     print('unlearn_acc == 0, Break at j = ', j, ' i = ', i)
#                     break
#
#             j += 1
#
#             if max_iter < j:
#                 break
#
#         model.eval()
#         unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
#         # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
#         other_loss, other_acc, other_f1 = testins(model, device, other_loader)
#         # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
#         test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
#         # test_loss, test_acc,test_f1 = test(model, device, test1_loader)
#
#         str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
#             unlearn_acc) + ', D_test acc : ' + str(test_acc)
#         logger.info(str_list)
#         # logger.info(' after D_test - D_forget f1:', other_f1, 'D_forget_f1:', unlearn_f1, 'D_test_f1:', test_f1)
#         print(' after D_test - D_forget f1:', other_f1, 'D_forget_f1:', unlearn_f1, 'D_test_f1:', test_f1)
#         member_gt = [1 for i in range(1)]
#         qf1_model = deepcopy(model)
#         qf1_model.load_state_dict(best_model_stateqf1)
#         u1=model.state_dict()
#         average_euclidean0, average_manhattan0, average_cosine_similarity0 = analyze_sample_similarity(qf1_model,best_model_stateqf1,
#                                                                                                        device,
#                                                                                                        train_dataset,
#                                                                                                        CONFIG)
#         average_euclidean1, average_manhattan1, average_cosine_similarity1 = analyze_sample_similarity(model,u1, device,
#                                                                                                        train_dataset,
#                                                                                                        CONFIG)
#         average_kl_div = calculate_kl_divergence(model,best_model_state_retrained ,qf1_model, unlearn_loader, device)
#         f_u,u=model.feature_extractor.state_dict(),model.state_dict()
#         model1=get_student_model().to(device)
#         _t, pv, EMA_res, risk_score = api(device,model1,u, f_u ,unlearn_loader, member_gt, cal_1000_loader, caltest1_loader)
#         print(
#             f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#         print(
#             f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
#         print(
#             f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
#         print(f'average_kl_div: {average_kl_div}')
#         logger.info(
#             f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
#         logger.info(
#             f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
#         logger.info(
#             f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
#         logger.info(f'average_kl_div: {average_kl_div}')
#         case3_D_test.append(test_acc)
#         case3_D_r.append(other_acc)
#         case3_D_f.append(unlearn_acc)

# def train_and_forget(network,base2_loader,test1_loader ,best_model_state_strained,best_model_state_retrained, dataset, classes,qf_1_dataset, qf_1_loader,base2_dataset , method,gpu,train_dataset,CONFIG):
#     # batch_size = args.b
#
#     # get network
#     # net = getattr(models, args.net)(num_classes=args.classes)
#     net = get_student_model()
#     net.load_state_dict(best_model_state_strained)
#
#     unlearning_teacher = get_student_model()
#     # unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)
#
#     if gpu:
#         net = net.cuda()
#         unlearning_teacher = unlearning_teacher.cuda()
#
#     # root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"
#
#
#     img_size = 224 if network == "ViT" else 32
#
#     # trainset = getattr(datasets, args.dataset)(
#     #     root=root, download=True, train=True, unlearning=True, img_size=img_size
#     # )
#     # validset = getattr(datasets, args.dataset)(
#     #     root=root, download=True, train=False, unlearning=True, img_size=img_size
#     # )
#     trainloader = base1_loader
#     # trainloader = DataLoader(trainset, num_workers=4, batch_size=args.b, shuffle=True)
#     validloader = test1_loader
#     # validloader = DataLoader(validset, num_workers=4, batch_size=args.b, shuffle=False)
#     # print("Dataset length:", len(trainset))
#     # dataset_length=len(trainset)
#     # split1 = int(dataset_length * 0.002)
#     # split2 = dataset_length - split1
#     # forget_train, retain_train = torch.utils.data.random_split(trainset, [split1, split2])
#     # forget_train, retain_train = torch.utils.data.random_split(
#     #     trainset, [args.forget_perc, 1 - args.forget_perc]
#     # )
#     # forget_train_dl = DataLoader(list(forget_train), batch_size=128)
#     forget_train_dl = qf_1_loader
#     # forget_train_dl = qf_100_loader
#     retain_train_dl = base2_loader
#     # retain_train_dl = base2_loader
#     # retain_train_dl = DataLoader(list(retain_train), batch_size=128, shuffle=True)
#     forget_valid_dl = forget_train_dl
#     retain_valid_dl = base2_loader
#     # retain_valid_dl = validloader
#     model_size_scaler = 1
#     if net == "ViT":
#         model_size_scaler = 1
#     else:
#         model_size_scaler = 1
#
#     # full_train_dl = DataLoader(
#     #     ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
#     #     batch_size=batch_size,
#     # )
#     full_train_dl = base1_loader
#     kwargs = {
#         "model": net,
#         "unlearning_teacher": unlearning_teacher,
#         "retain_train_dl": retain_train_dl,
#         "retain_valid_dl": retain_valid_dl,
#         "forget_train_dl": forget_train_dl,
#         "forget_valid_dl": forget_valid_dl,
#         "full_train_dl": full_train_dl,
#         "valid_dl": validloader,
#         "dampening_constant": 1,
#         "selection_weighting": 10 * model_size_scaler,
#         "num_classes": classes,
#         "dataset_name": dataset,
#         "device": "cuda" if gpu else "cpu",
#         "model_name": network,
#         "cal_1000_loader": cal_1000_loader,
#         "caltest1_loader": caltest1_loader,
#         # "base3_loader": base_1000_loader,
#         "qf_100_dataset": qf_1_dataset,
#         "base2_dataset": base2_dataset,
#         # "base2_dataset":base2_dataset,
#         "method": method,
#         "best_model_state_retrained": best_model_state_retrained,
#         "train_dataset": train_dataset,
#         "CONFIG": CONFIG
#     }
#     # wandb.init(project=f"{args.dataset}_forget_random_{args.forget_perc}", name=f'{args.method}')
#
#     import time
#
#     start = time.time()
#
#     testacc, retainacc, zrf, mia, d_f,curent = getattr(forget_random_strategies, method)(
#         **kwargs
#     )
#     end = time.time()
#     time_elapsed = end - start
#
#     print(testacc, retainacc, zrf, mia,curent)
#     logger.info(f"TestAcc: {testacc}, RetainTestAcc: {retainacc}, ZRF: {zrf}, MIA: {mia}, Df: {d_f},current:{curent}")
#     # wandb.log(
#     #     {
#     #         "TestAcc": testacc,
#     #         "RetainTestAcc": retainacc,
#     #         "ZRF": zrf,
#     #         "MIA": mia,
#     #         "Df": d_f,
#     #         "model_scaler": model_size_scaler,
#     #         "MethodTime": time_elapsed,
#     #         # do not forget to deduct baseline time from it to remove results calc (acc, MIA, ...)
#     #     }
#     # )
for qf1_start in range(9005, 9999):
    best_model_state_retrained =torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_retrained_test.pth')

    best_model_state_strained=torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_strained_test_zui.pth')

    best_model_state_trained = torch.load('/root/autodl-tmp/wangbin/yiwang/afsandadapt/quanzhong/best_trained_test.pth')
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
    # average_euclidean_retrained, average_manhattan_retrained, average_cosine_similarity_retrained = analyze_sample_similarity(
    #     model_s, u11,
    #     device,
    #     train_dataset,
    #     CONFIG)
    # print(
    #     f'>>average_euclidean_retrained: {average_euclidean_retrained}, average_manhattan_retrained: {average_manhattan_retrained}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained}')
    # logger.info(
    #     f'>>average_euclidean_retrained: {average_euclidean_retrained}, average_manhattan_retrained: {average_manhattan_retrained}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained}')
    #
    # network = 'resnet18'
    dataset = 'cifar10'
    method = 'ssd_tuning'

    classes = 9
    train_and_forget(network,kd0_5_loader,test1_loader, best_model_state_strained, best_model_state_retrained, dataset, classes, qf_1_dataset,qf1_loader, base2_dataset, method, gpu=True,train_dataset=train_dataset,CONFIG=CONFIG,cal_1000_loader=cal_1000_loader,caltest1_loader=caltest1_loader)
    method = 'amnesiac'

    train_and_forget(network,kd0_5_loader,test1_loader, best_model_state_strained, best_model_state_retrained, dataset, classes, qf_1_dataset,qf1_loader, base2_dataset, method, gpu=True,train_dataset=train_dataset,CONFIG=CONFIG,cal_1000_loader=cal_1000_loader,caltest1_loader=caltest1_loader)
    #
    method = 'blindspot'
    train_and_forget(network, kd0_5_loader,test1_loader,best_model_state_strained, best_model_state_retrained, dataset, classes, qf_1_dataset,qf1_loader, base2_dataset, method, gpu=True,train_dataset=train_dataset,CONFIG=CONFIG,cal_1000_loader=cal_1000_loader,caltest1_loader=caltest1_loader)

    method = 'FisherForgetting'
    train_and_forget(network, kd0_5_loader,test1_loader,best_model_state_strained, best_model_state_retrained, dataset, classes, qf_1_dataset,qf1_loader, base2_dataset, method, gpu=True,train_dataset=train_dataset,CONFIG=CONFIG,cal_1000_loader=cal_1000_loader,caltest1_loader=caltest1_loader)


    subset_indices = base1_indices
    train_labels = PathMNISTDataset.get_labels(train_dataset, subset_indices)
    instance(base1_dataset, base1_indices, test1_loader, best_model_state_retrained, best_model_state_strained,cal_1000_loader, caltest1_loader, qf_1_indices,CONFIG,train_labels=train_labels,train_dataset=train_dataset)
    adaptforget(
        train_dataset=train_dataset,
        num_epochsall=60,
        device=device,
        qf_100_loader=qf1_loader,
        kd=kd0_5_loader,
        test1_loader=test1_loader,
        cal_1000_loader=cal_1000_loader,
        caltest1_loader=caltest1_loader,
        best_model_state_retrained=best_model_state_retrained,
        best_model_state_trained=best_model_state_trained,
        CONFIG=CONFIG,
        kd0_5_loader_no=kd0_5_loader_no,
        base2_loader=base2_loader,
    )

    args = parser()
    afs(args,best_model_state_trained,best_model_state_retrained,base2_loader,base2_loader,test1_loader,cal_1000_loader,caltest1_loader,qf1_loader,device,train_dataset,CONFIG)
