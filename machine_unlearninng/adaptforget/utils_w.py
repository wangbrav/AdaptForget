from __future__ import print_function
#首先是数据集的加载 加载包括   有一个文件的加载  dataset的嘞也需要 还有一个是dataloader的加载  还有他的config  以及他的颜色扩充通道  以及其的加噪函数  transform  种子的设置
import sys

# from adversarial_3_net_afs_path_ab import base2_loader

sys.path.append('/root/autodl-tmp/wangbin/yiwang')
import copy
from sklearn.metrics import f1_score
import torch
from tqdm import tqdm
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
from torch.nn.functional import softmax
from qf1kosiam import analyze_sample_similarity,analyze_sample_similarity_sisa
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
from unlearning_random_afs_rand import train_student_model_random
from Domainadaptation_net_three import domainadaptation
from test_model_path import test_model
# from puri import api
import random
from puri import api
from afsandadapt.puri_sisa import  api_sisa
import itertools
from advertorch.attacks import L2PGDAttack
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from itertools import cycle
# from mu.net_train_three import get_student_model, get_teacher_model, get_student_model_t
from mlp_three_csv_wu import FeatureExtractor, Classifier, CombinedModel,get_student_model, get_teacher_model

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
# from tsne_mnist_guding2 import tsnes
from tsne_mnist_guding2_tsne import tsnes
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
from sklearn.metrics import accuracy_score, f1_score as calculate_f12_score
from tsne_mnist_guding2_tsne import tsnes
from tsne_mnist_guding2_tsne import tsnessisa

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
from torch.utils.data import DataLoader, Dataset
from utils.KDloss import SoftTarget
from utils.Metric import AverageMeter, accuracy, Performance
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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from training_utils import *
logging.basicConfig(filename='./tc/training_log_qf1circulate_asdv7.log', level=logging.INFO, format='%(asctime)s %(message)s')
# logging.basicConfig(filename='./tc/pathablation.log', level=logging.INFO, format='%(asctime)s %(message)s')
# logging.basicConfig(filename='./tc/pathtsne.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

def parser():
    """
    :return: args
    """
    parser = argparse.ArgumentParser(prog='AFS')
    subparsers = parser.add_subparsers(help='sub-command help')
    #
    # parser_audit = subparsers.add_parser('audit')
    # parser_audit.add_argument('--root',
    #                     default='../template/MNIST',
    #                     help='root dir to the project')
    # parser_audit.add_argument('--query_label',
    #                     default='EXP1',
    #                     help='label of the query data, defined in Dataset.py/Config')
    # parser_audit.add_argument('--cal_label',
    #                     default='CAL1',
    #                     help='label of the calibration data, defined in Dataset.py/Config')
    # parser_audit.add_argument('--cal_test_label',
    #                     default='CALTEST1',
    #                     help='label of the calibration test data, defined in Dataset.py/Config')
    # parser_audit.add_argument('--test_label',
    #                     default='TEST1',
    #                     help='label of the test data, defined in Dataset.py/Config')
    # parser_audit.add_argument('--model2audit',
    #                     default='./models/base/best_model.pth',
    #                     help='relative path of model to be auditted to the root')
    # parser_audit.add_argument('--model2cal',
    #                     default='./models/cal/best_model.pth',
    #                     help='relative path of the calibration model to the root')
    # parser_audit.add_argument('--device',
    #                     default='cuda:0')
    # parser_audit.add_argument('--KP_infer_batch_size',
    #                     type=int,
    #                     default=1024,
    #                     help='batch size for inference during membership attack')
    # parser_audit.add_argument('--nclass',
    #                     type=int,
    #                     default=10,
    #                     help='number of classes')
    # parser_audit.add_argument('--num_workers',
    #                     type=int,
    #                     default=5,
    #                     help='number of num_workers')
    # parser_audit.add_argument('--command_class',
    #                           default=0,
    #                           type=int,
    #                           help='for internal use only, no change')

    parser_forget = subparsers.add_parser('forget')
    parser_forget.add_argument('--root',
                        default='../template/MNIST',
                        help='root dir to the project')
    parser_forget.add_argument('--expname',
                        default='EXP1',
                        help='name of exp, will affect the path and dataset splitting')
    parser_forget.add_argument('--teacher_model',
                        default='./models/EXP1/base/best_model.pth',
                        help='relative path of model to be distilled to the root')
    parser_forget.add_argument('--KD_label',
                        default='KD0.25',
                        help='the name of base dataset used for KD, should be defined in CONFIG')
    parser_forget.add_argument('--test_label',
                        default='TEST1',
                        help='label of the test data, defined in Dataset.py/Config')
    parser_forget.add_argument('--cal_label',
                        default='CAL1',
                        help='label of the calibration data, defined in Dataset.py/Config')
    parser_forget.add_argument('--cal_test_label',
                        default='CALTEST1',
                        help='label of the calibration test data, defined in Dataset.py/Config')
    parser_forget.add_argument('--query_label',
                        default='QO1',
                        help='label of the query data, defined in Dataset.py/Config, here the query dataset should overlap with training dataset')
    parser_forget.add_argument('--add_risk_loss',
                        type=int,
                        default=1,
                        help='1: will add risk loss when running KP, 0: same as pure KD')
    parser_forget.add_argument('--nclass',
                        type=int,
                        default=9,
                        help='number of classes')
    parser_forget.add_argument('--train_batch_size',
                        type=int,
                        default=32)
    parser_forget.add_argument('--KP_infer_batch_size',
                        type=int,
                        default=128,
                        help='batch size for inference during membership attack')
    parser_forget.add_argument('--device',
                        default='cuda:0')
    parser_forget.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='number of epochs')
    parser_forget.add_argument('--T',
                        type=float,
                        default=4.0,
                        help='temperature for ST')
    parser_forget.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='initial learning rate')
    parser_forget.add_argument('--lambda_kd',
                        type=float,
                        default=1,
                        help='trade-off parameter for kd loss')
    parser_forget.add_argument('--lambda_risk',
                        type=float,
                        default=10,
                        help='trade-off parameter for risk loss')
    parser_forget.add_argument('--num_workers',
                        type=int,
                        default=5,
                        help='number of num_workers')
    parser_forget.add_argument('--command_class',
                              default=1,
                              type=int,
                              help='for internal use only, no change')
    parser = argparse.ArgumentParser(prog='Knowledge Purification (KP)')
    parser.add_argument('--root',
                          default='../template/MNIST',
                          help='root dir to the project')
    parser.add_argument('--expname',
                        default='EXP1',
                        help='name of exp, will affect the path and dataset splitting')
    parser.add_argument('--teacher_model',
                        default='./models/EXP1/base/best_model.pth',
                        help='relative path of model to be distilled to the root')
    parser.add_argument('--KD_label',
                        default='KD0.25',
                        help='the name of base dataset used for KD, should be defined in CONFIG')
    parser.add_argument('--test_label',
                        default='TEST1',
                        help='label of the test data, defined in Dataset.py/Config')
    parser.add_argument('--cal_label',
                        default='CAL1',
                        help='label of the calibration data, defined in Dataset.py/Config')
    parser.add_argument('--cal_test_label',
                        default='CALTEST1',
                        help='label of the calibration test data, defined in Dataset.py/Config')
    parser.add_argument('--query_label',
                        default='QO1',
                        help='label of the query data, defined in Dataset.py/Config, here the query dataset should overlap with training dataset')
    parser.add_argument('--add_risk_loss',
                        type=int,
                        default=1,
                        help='1: will add risk loss when running KP, 0: same as pure KD')
    parser.add_argument('--nclass',
                        type=int,
                        default=9,
                        help='number of classes')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--KP_infer_batch_size',
                        type=int,
                        default=128,
                        help='batch size for inference during membership attack')
    parser.add_argument('--device',
                        default='cuda:0')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='number of epochs')
    parser.add_argument('--T',
                        type=float,
                        default=4.0,
                        help='temperature for ST')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='initial learning rate')
    parser.add_argument('--lambda_kd',
                        type=float,
                        default=1,
                        help='trade-off parameter for kd loss')
    parser.add_argument('--lambda_risk',
                        type=float,
                        default=1,
                        help='trade-off parameter for risk loss')
    parser.add_argument('--num_workers',
                        type=int,
                        default=5,
                        help='number of num_workers')
    # parser.add_argument('add_risk_loss',
    #                     type=int,
    #                     default=1,
    #                     help='')

    args = parser.parse_args()
    return args
    # args = parser_forget.parse_args()

def afs(args,best_model_state_trained,best_model_state_retrained,base1_loader,base2_loader,kd0_5_loader_no,test1_loader,cal_1000_loader,caltest1_loader,qf_100_loader,device,train_dataset, CONFIG):
    sys.path.append(args.root)
    best_model_state_afs = None

    # init model
    tnet = get_teacher_model().to(device)
    snet = get_student_model().to(device)
    # snet_base2 = get_student_model().to(device)
    tnet.load_state_dict(best_model_state_trained)
    snet_base2 = get_student_model().to(device)
    snet_base2.load_state_dict(best_model_state_retrained)
    tnet.eval()
    # if 'MNIST' in args.root:
    #     criterionCls = torch.nn.NLLLoss().to(args.device)
    # elif 'PathMNIST' in args.root:
    criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    # elif 'COVIDx' in args.root:
    #     criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    # else:
        # criterionCls = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    criterionKD = SoftTarget(args.T)

    optimizer_afs = torch.optim.Adam(snet.parameters(),
                                lr=0.001)
    # args.mode = 'base'
    # args.base_label = args.KD_label
    # baseDataset = DataModule(dir=args.root, args = args, batch_size = args.train_batch_size, num_workers=args.num_workers)
    # baseTrainDataLoader = baseDataset.train_dataloader()
    # baseTestDataLoader = baseDataset.test_dataloader()
    # args.mode = 'query'
    # queryDataset = DataModule(dir=args.root, args=args, batch_size = args.KP_infer_batch_size, num_workers=args.num_workers)
    # member_gt = CONFIG[args.query_label]['QUERY_MEMBER']
    # args.mode = 'cal'
    # calDataset = DataModule(dir=args.root, args=args, batch_size = args.KP_infer_batch_size, num_workers=args.num_workers)

    # logger.info('>> start KP')
    save_metric_best = 0
    # teacher_model = MLP(28, 64, 10).to(args.device)
    member_gt = [1 for i in range(100)]
    epochs= 10
    # teacher_model.apply(initialize_weights)
    for epoch in range(epochs):
        train(snet, tnet, criterionCls, criterionKD, base1_loader, optimizer_afs, args, qf_100_loader)
        test_acc,f1 = afstest(snet, tnet, criterionCls, criterionKD, test1_loader, args)
        # test_acc, stat,f1 = afstest(snet, tnet, criterionCls, criterionKD, test1_loader, args)
        # logger.info(f'>> snet test acc: {test_acc}')'
        f_u, u = snet.feature_extractor.state_dict(), snet.state_dict()

        # logger.info(f">> snet test stat: {','.join([str(_) for _ in stat])}")
        # logger.info(f'>> evaluate membership attack on snet after training')
        _t, pv, EMA_res, risk_score = api(device, snet, u, f_u, qf_100_loader, member_gt, cal_1000_loader,
                                          caltest1_loader)
        print(f'>> test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')

        # save model
        if save_metric_best < test_acc:
            save_metric_best = test_acc
            print(f'>> saving best snet model')
            logger.info(f'>> saving best snet model{test_acc}')
            # logger.info(f'>>stat{stat}')
            logger.info(f'>>f1{f1}')
            f_u, u = snet.feature_extractor.state_dict(), snet.state_dict()

            _t, pv, EMA_res, risk_score = api(device, snet, u, f_u, qf_100_loader, member_gt, cal_1000_loader,
                                              caltest1_loader)
            print(f'>>best test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
            logger.info(f'>>afs best test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
            average_euclidean_afs, average_manhattan_afs, average_cosine_similarity_afs=analyze_sample_similarity(snet, u, device, train_dataset, CONFIG)
            print(f'>>average_euclidean_afs: {average_euclidean_afs}, average_manhattan_afs: {average_manhattan_afs}, average_cosine_similarity_afs: {average_cosine_similarity_afs}')
            logger.info(f'>>average_euclidean_afs: {average_euclidean_afs}, average_manhattan_afs: {average_manhattan_afs}, average_cosine_similarity_afs: {average_cosine_similarity_afs}')
            average_kl_div_afs=calculate_kl_divergence(snet, best_model_state_retrained, snet_base2, qf_100_loader, device)
            print(f'>>average_kl_div_afs: {average_kl_div_afs}')
            logger.info(f'>>average_kl_div_afs: {average_kl_div_afs}')
            num = 0
            tsnes(qf_100_loader, base2_loader, kd0_5_loader_no, snet, snet, snet, u, f_u, num)

            # save_best_ckpt(snet, args)
            # best_model_state_afs = snet.state_dict().copy()

        # save_last_ckpt(snet, args)



    # args = parser(args.root)

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
#                 # Y = Y.squeeze(1).long()    #表格数据集要注释 图片数据集要用的
#                 correct += pred.eq(Y.view(-1)).sum().item()
#                 # print(f'Batch correct: {(predicted == Y).sum().item()}, Batch total: {Y.size(0)}')  # 新增的打印语句
#     print(f"Correct: {correct}, Total: {total}")
#     accuracy = 100 * correct / total
#     return accuracy


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []  # To store all predictions
    all_labels = []  # To store all true labels

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for X, Y in tqdm(test_loader):
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)

                pred = outputs.data.max(1)[1]
                total += Y.size(0)
                correct += pred.eq(Y.view(-1)).sum().item()

                # Collect predictions and labels for F1 score calculation
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(Y.view(-1).cpu().numpy())

    print(f"Correct: {correct}, Total: {total}")
    accuracy = 100 * correct / total

    # Calculate F1 Score
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Use 'weighted' for class imbalance

    print(f"Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")
    return accuracy, f1
def train(snet, tnet, criterionCls, criterionKD, trainDataLoader, optimizer, args, queryDataset):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    risk_losses = AverageMeter()
    total_losses = AverageMeter()
    snet.train()
    tnet.eval()
    args.lambda_kd= 1
    args.lambda_risk = 10
    args.add_risk_loss = 1

    print(args.lambda_kd)
    with tqdm(total=len(trainDataLoader)) as t:
        for X, Y in tqdm(trainDataLoader):
            X = X.to(args.device)
            Y = Y.to(args.device)

            # if 'PathMNIST' in args.root:
            Y = Y.squeeze().long()

            #print(X)
            snet_pred = snet(X)
            tnet_pred = tnet(X)

            #print(Y, tnet_pred, snet_pred)

            cls_loss = criterionCls(snet_pred, Y)
            kd_loss = criterionKD(snet_pred, tnet_pred.detach())
            cls_losses.update(cls_loss.item(), X.size(0))
            kd_losses.update(kd_loss.item(), X.size(0))

            loss = cls_loss + kd_loss* args.lambda_kd

            #if loss < 1:
            #    args.add_risk_loss = True
            #    print(f'>> start using risk loss')
            #else:
            #    args.add_risk_loss = False

            if args.add_risk_loss == 1:
                risk_loss = torch.tensor(0.0).to(args.device)
                queryTestDataLoader = queryDataset
                for _X, _Y in queryTestDataLoader:
                    _X = _X.to(args.device)
                    _Y = _Y.to(args.device)
                    # if 'PathMNIST' in args.root:
                    _Y = _Y.squeeze().long()
                    if _Y.dim() == 0:
                        _Y = _Y.unsqueeze(0)
                    out_Y = snet(_X)
                    partial_risk_loss = torch.nn.CrossEntropyLoss().to(args.device)(out_Y, _Y)
                    risk_loss += partial_risk_loss

                ## using Audit.api requires more time for training
                #t, pv, EMA_res, risk_score = Audit.api(args, model, queryDataset, member_gt, calDataset)
                #risk_loss = risk_score

                risk_loss = torch.tensor(1.0).to(args.device) / risk_loss # same performance, strongly correlated

                #risk_loss = risk_loss
                risk_losses.update(risk_loss.item(), _X.size(0))
                loss = loss +  cls_loss + kd_loss* args.lambda_kd + risk_loss*torch.tensor(args.lambda_risk).to(args.device)
                # print("risk_loss",risk_loss)
                # print("risk_loss*torch.tensor(args.lambda_risk)",risk_loss*torch.tensor(args.lambda_risk))
                # loss = loss + risk_loss*torch.tensor(args.lambda_risk).to(args.device)
            # print(risk_loss)
            total_losses.update(loss.item(), X.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(
                cls_losses='{:05.8f}'.format(cls_losses.avg),
                kd_losses='{:05.8f}'.format(kd_losses.avg),
                risk_losses='{:05.8f}'.format(risk_losses.avg),
                total_losses='{:05.8f}'.format(total_losses.avg),
            )
            t.update()

def afstest(snet, tnet, criterionCls, criterionKD, testDataLoader, args):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    correct = 0

    snet.eval()
    tnet.eval()

    total_pred = []
    total_y = []

    with torch.no_grad():
        with tqdm(total=len(testDataLoader)) as t:
            for X, Y in tqdm(testDataLoader):
                X = X.to(args.device)
                Y = Y.to(args.device)

                # if 'PathMNIST' in args.root:
                Y = Y.squeeze().long()

                snet_pred = snet(X)
                tnet_pred = tnet(X)

                cls_loss = criterionCls(snet_pred, Y)
                kd_loss = criterionKD(snet_pred, tnet_pred.detach()) * args.lambda_kd

                pred = snet_pred.data.max(1)[1]
                correct += pred.eq(Y.view(-1)).sum().item()

                cls_losses.update(cls_loss.item(), X.size(0))
                kd_losses.update(kd_loss.item(), X.size(0))

                t.set_postfix(
                    cls_losses='{:05.8f}'.format(cls_losses.avg),
                    kd_losses='{:05.8f}'.format(kd_losses.avg),
                )
                t.update()

                # total_pred += pred
                # total_y += Y.view(-1)
                total_pred.extend(pred.cpu().numpy())
                total_y.extend(Y.view(-1).cpu().numpy())

    test_acc = correct / len(testDataLoader.dataset)
    # stat = Performance(total_pred, total_y)
    f1_macro = f1_score(total_y, total_pred, average='macro')

    # return test_acc, stat,f1_macro
    return test_acc,f1_macro
#
def instance(base1_dataset, base1_indices,test1_loader,best_model_state_retrained,best_model_state_strained,cal_1000_loader,caltest1_loader,qf_1_indices,CONFIG,train_labels,train_dataset,kd0_5_loader_no):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pgd-eps', type=float, default=2.0, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--pgd-alpha', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--pgd-iter', type=int, default=100, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--unlearn-label', type=int, default=9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--unlearn-k', type=int, default=10, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--unlearn-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--num-adv-images', type=int, default=None, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--reg-lamb', type=float, default=10.0, metavar='LR',
                        help='learning rate (default: 1.0)')

    args = parser.parse_args()








    # 使用的gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    # 不明白
    eps = args.pgd_eps
    iters = args.pgd_iter
    alpha = args.pgd_alpha

    # k_arr = qf_100_indices
    # k_arr = qf_100_indices
    # print(qf_100_indices)
    k_arr = [100]
    #   k_arr = [1, 16, 64, 128, 256]

    D_r_acc = []
    D_f_acc = []
    D_test_acc = []
    # 一个是  用于其他的数据集  的准确性   一个用于数据集的忘却的准确性 一个是 用于测试集的准确性
    # case1_D_r  case2_D_r  case3_D_r 是 三个方法
    # 1 一种简单的方法 其中模型在为学习的数据上微调 2 一种使用对抗样本的方法  3 一种使用对抗样本和权重重要性的方法
    case1_D_r = []
    case2_D_r = []
    case3_D_r = []

    case1_D_f = []
    case2_D_f = []
    case3_D_f = []

    case1_D_test = []
    case2_D_test = []
    case3_D_test = []

    train_kwargs = {'batch_size': 256}
    test_kwargs = {'batch_size': 256}
    # test_kwargs = {'batch_size': 1024}

    naiive_unlearn_kwargs = {'batch_size': 32}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset1 = datasets.CIFAR10('../data', train=True, download=True,
    #                    transform=transform)

    dataset1 = base1_dataset
    # 一个 是训练数据集 一个 是测试数据集
    # dataset2 = datasets.CIFAR10('../data', train=False,
    #                    transform=transform)

    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    for unlearn_k in k_arr:

        # torch.manual_seed(args.seed)
        # torch.cuda.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # np.random.seed(args.seed)
        # random.seed(args.seed)

        unlearn_label = args.unlearn_label
        # 假设 dataset1 是通过 Subset 创建的，原始数据集是 pathmnist_dataset
        subset_indices = base1_indices  # 获取 Subset 的索引
        # train_labels = PathMNISTDataset.get_labels(train_dataset, subset_indices)

        # train_labels = dataset1.labels
        # train_labels = dataset1.targets

        train_labels = torch.from_numpy(np.array(train_labels))

        # indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]
        # indices_k_unlearn = torch.tensor(qf_1_indices)
        indices_k_unlearn = torch.tensor(qf_1_indices)

        # indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]
        # print(indices_k_unlearn)
        # indices_k_unlearn = qf_100_indices
        # print ('indices_k_unlearn : ', indices_k_unlearn)

        copy_train_labels = train_labels.clone()
        copy_train_labels[indices_k_unlearn] = -10
        print('len(copy_train_labels) : ', len(copy_train_labels))

        indices_other_data = (copy_train_labels != -10).nonzero(as_tuple=False)
        indices_other_data = torch.unique(indices_other_data)
        # indices_other_data = torch.unique(indices_other_data)

        print('len(indices_other_data) : ', len(indices_other_data))
        unlearn_dataset = Subset(dataset1, indices_k_unlearn.view(-1, ))
        # print("indices_k_unlearn : ",indices_k_unlearn.view(-1,))
        unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset, **naiive_unlearn_kwargs)

        other_dataset = Subset(dataset1, indices_other_data.contiguous().view(-1, ))
        print(len(Subset(dataset1, indices_other_data)))
        print(indices_other_data.contiguous().view(-1, ).shape)
        print('len(other_dataset) : ', len(other_dataset))
        other_loader = torch.utils.data.DataLoader(other_dataset, **test_kwargs)
        print('len(other_loader) : ', len(other_loader))
        # print(indices_other_data.contiguous().view(-1, ).shape)

        # cifar_test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        print('len(unlearn_dataset) : ', len(unlearn_dataset), ' len(other_dataset) : ', len(other_dataset))
        criterion = nn.CrossEntropyLoss()

        # optimizer = optim.Adam(modelmlp.parameters(), lr=learning_rate)
        model = get_student_model().to(device)
        # normalize_layer = NormalizeLayer((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # model = torch.nn.Sequential(normalize_layer, model)

        optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-3)

        # model = resnet18().to(device)
        # smodelmlp_base2.load_state_dict(
        #     torch.load('/root/autodl-tmp/wangbin/yiwang/qf1model/best_model_path_qf1_base2.pth'))
        # def train_model(model, trainloader, validloader, optimizer, criterion, device, epochs=10):
        #     model.to(device)
        #     best_accuracy = 0
        #     best_model_state = None  # 用于保存最佳模型的状态
        #
        #     for epoch in range(epochs):
        #         model.train()
        #         for inputs, labels in trainloader:
        #             inputs, labels = inputs.to(device), labels.to(device)
        #             optimizer.zero_grad()
        #             outputs = model(inputs)
        #             labels = labels.squeeze().long()
        #             loss = criterion(outputs, labels)
        #             loss.backward()
        #             optimizer.step()
        #
        #         # 在每个epoch结束后在验证集上评估模型
        #         model.eval()
        #         correct = 0
        #         total = 0
        #         with torch.no_grad():
        #             for inputs, labels in validloader:
        #                 inputs, labels = inputs.to(device), labels.to(device)
        #                 outputs = model(inputs)
        #                 _, predicted = torch.max(outputs.data, 1)
        #                 total += labels.size(0)
        #                 correct += (predicted == labels).sum().item()
        #
        #         accuracy = 100 * correct / total
        #         print(f'Epoch {epoch + 1}: Validation Accuracy = {accuracy:.2f}%')
        #
        #         # 如果这个epoch的准确率高于之前的最高准确率，更新最佳模型状态
        #         if accuracy > best_accuracy:
        #             best_accuracy = accuracy
        #             best_model_state = model.state_dict().copy()  # 深拷贝模型状态
        #
        #     return best_model_state, best_accuracy

        # # 训练模型并获得最佳模型状态
        # def train_model(model, trainloader, validloader, optimizer, criterion, device, epochs=10,
        #                 model_path='/root/autodl-tmp/wangbin/L2UL-main/weights/best_model.pth'):
        #     model.to(device)
        #     best_accuracy = 0
        #     best_model_state = None  # 用于保存最佳模型的状态
        #
        #     for epoch in range(epochs):
        #         model.train()
        #         for inputs, labels in trainloader:
        #             inputs, labels = inputs.to(device), labels.to(device)
        #             optimizer.zero_grad()
        #             outputs = model(inputs)
        #             labels = labels.squeeze().long()
        #             loss = criterion(outputs, labels)
        #             loss.backward()
        #             optimizer.step()
        #
        #         # 在每个epoch结束后在验证集上评估模型
        #         model.eval()
        #         correct = 0
        #         total = 0
        #         with torch.no_grad():
        #             for inputs, labels in validloader:
        #                 inputs, labels = inputs.to(device), labels.to(device)
        #                 outputs = model(inputs)
        #                 _, predicted = torch.max(outputs.data, 1)
        #                 total += labels.size(0)
        #                 correct += (predicted == labels).sum().item()
        #
        #         accuracy = 100 * correct / total
        #         print(f'Epoch {epoch + 1}: Validation Accuracy = {accuracy:.2f}%')
        #
        #         # 如果这个epoch的准确率高于之前的最高准确率，更新最佳模型状态
        #         if accuracy > best_accuracy:
        #             best_accuracy = accuracy
        #             best_model_state = model.state_dict().copy()  # 深拷贝模型状态
        #             # 保存最佳模型的状态
        #             torch.save(best_model_state, model_path)
        #             print(f"Saved new best model with accuracy: {best_accuracy:.2f}%")
        #
        #     return best_model_state, best_accuracy
        #
        # def train_modelqf1(model, trainloader, validloader, optimizer, criterion, device, epochs=10,
        #                    model_path='/root/autodl-tmp/wangbin/L2UL-main/weights/best_modelqf1.pth'):
        #     model.to(device)
        #     best_accuracy = 0
        #     best_model_state = None  # 用于保存最佳模型的状态
        #
        #     for epoch in range(epochs):
        #         model.train()
        #         for inputs, labels in trainloader:
        #             inputs, labels = inputs.to(device), labels.to(device)
        #             optimizer.zero_grad()
        #             outputs = model(inputs)
        #             labels = labels.squeeze().long()
        #             loss = criterion(outputs, labels)
        #             loss.backward()
        #             optimizer.step()
        #
        #         # 在每个epoch结束后在验证集上评估模型
        #         model.eval()
        #         correct = 0
        #         total = 0
        #         with torch.no_grad():
        #             for inputs, labels in validloader:
        #                 inputs, labels = inputs.to(device), labels.to(device)
        #                 outputs = model(inputs)
        #                 _, predicted = torch.max(outputs.data, 1)
        #                 total += labels.size(0)
        #                 correct += (predicted == labels).sum().item()
        #
        #         accuracy = 100 * correct / total
        #         print(f'Epoch {epoch + 1}: Validation Accuracy = {accuracy:.2f}%')
        #
        #         # 如果这个epoch的准确率高于之前的最高准确率，更新最佳模型状态
        #         if accuracy > best_accuracy:
        #             best_accuracy = accuracy
        #             best_model_state = model.state_dict().copy()  # 深拷贝模型状态
        #             # 保存最佳模型的状态
        #             torch.save(best_model_state, model_path)
        #             print(f"Saved new best model with accuracy: {best_accuracy:.2f}%")
        #
        #     return best_model_state, best_accuracy

        best_model_statet = best_model_state_strained
        best_model_stateqf1=best_model_state_retrained
        # print(f'Best Validation Accuracy: {best_accuracy:.2f}%')
        # model.load_state_dict(best_model_state)

        model.load_state_dict(best_model_statet)
        # model.load_state_dict(torch.load('/root/autodl-tmp/wangbin/L2UL-main/weights/best_model.pth'))
        #   输出模型的结构   为什么要输出模型的结构

        # normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # model = torch.nn.Sequential(normalize_layer, model)
        # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()

        # optimizer = optim.Adam(modelmlp.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
        # optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-3)

        model.eval()
        print('Baseline 1: Naiive Appraoch - finetuning with D_forget (maximizing CE loss)')

        # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
        other_loss, other_acc, other_f1 = testins(model, device, other_loader)
        # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
        unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
        # test_loss, test_acc,test_f1 = test(model, device, test1_loader)
        test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
        # test_loss, test_acc,test_f1 = test(model, device, cifar_test_loader)

        str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
            unlearn_acc) + ', D_test acc : ' + str(test_acc)

        D_test_acc.append(test_acc)
        D_r_acc.append(other_acc)
        D_f_acc.append(unlearn_acc)

        print(str_list)
        print('before  D_test - D_forget f1:', other_f1, 'D_forget_f1:', unlearn_f1, 'D_test_f1:', test_f1)
        unlearn_acc = 100
        max_iter = 1000

        j = 0

        while unlearn_acc != 0:

            naive_train(args, model, device, unlearn_loader, optimizer, 0)
            # naive_train(args, model, device, unlearn_loader, optimizer, 0)
            model.eval()
            # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
            unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)

            j += 1
            # print(1)
            if max_iter < j:
                break

        model.eval()
        # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
        other_loss, other_acc, other_f1 = testins(model, device, other_loader)
        test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
        # test_loss, test_acc,test_f1 = test(model, device, cifar_test_loader)

        str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
            unlearn_acc) + ', D_test acc : ' + str(test_acc)
        print(str_list)
        print('after  D_test - D_forget f1:', other_f1, 'D_forget_f1:', test_f1, 'D_test_f1:', test_f1)
        member_gt = [1 for i in range(1)]

        qf1_model = deepcopy(model)
        qf1_model.load_state_dict(best_model_stateqf1)

        # average_euclidean0, average_manhattan0, average_cosine_similarity0 = analyze_sample_similarity(qf1_model,
        #                                                                                                device,
        #                                                                                                train_dataset,
        #                                                                                                CONFIG)
        # average_euclidean1, average_manhattan1, average_cosine_similarity1 = analyze_sample_similarity(model, device,
        #                                                                                                train_dataset,
        #                                                                                                CONFIG)
        # average_kl_div = calculate_kl_divergence(model, qf1_model, unlearn_loader, device)
        # _t, pv, EMA_res, risk_score = api(device, model, unlearn_loader, member_gt, cal_1000_loader, caltest1_loader)
        # print(
        #     f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
        # print(
        #     f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
        # print(
        #     f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
        # print(f'average_kl_div: {average_kl_div}')
        case1_D_test.append(test_acc)
        case1_D_r.append(other_acc)
        case1_D_f.append(unlearn_acc)

        # model = resnet18().to(device)
        # model.load_state_dict(torch.load('./cifar10_pretrained_models/resnet18.pt'))
        # normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # model = torch.nn.Sequential(normalize_layer, model)
        model = get_student_model().to(device)
        # normalize_layer = NormalizeLayer((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # model = torch.nn.Sequential(normalize_layer, model)
        # model = resnet18().to(device)
        model.load_state_dict(best_model_statet)

        # model.load_state_dict(torch.load("/root/autodl-tmp/wangbin/L2UL-main/weights/best_model_path_base1.pth"))
        # best_model_state, best_accuracy = train_model(model, base1_loader, test1_loader, optimizer, criterion, device)
        # (print
        #  (f'Best Validation Accuracy: {best_accuracy:.2f}%'))
        # model.load_state_dict(best_model_state)

        optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-3)

        origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        print()
        print('\n Baseline 2: Our Appraoch - using adversarial examples only')

        unlearn_acc = 100
        alpha = 0.0

        model.eval()
        # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
        other_loss, other_acc, other_f1 = testins(model, device, other_loader)
        # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
        unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
        test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
        # test_loss, test_acc,test_f1 = test(model, device, cifar_test_loader)

        str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
            unlearn_acc) + ', D_test acc : ' + str(test_acc)
        print(str_list)
        print('before D_test - D_forget_f1:', other_f1, 'D_forget_f1:', unlearn_f1, 'D_test_f1:', test_f1)

        adversary = L2PGDAttack(model, eps=args.pgd_eps, eps_iter=args.pgd_alpha, nb_iter=args.pgd_iter,
                                rand_init=True, targeted=True)

        # adv_images, target_labels = adv_attack(args, model, device, qf_100_loader, adversary, unlearn_k, args.num_adv_images)
        adv_images, target_labels = adv_attack(args, model, device, unlearn_loader, adversary, unlearn_k,
                                               args.num_adv_images)

        adv_dataset = JointDataset(adv_images, target_labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)

        j = 0

        # unlearn_loader_cycle = cycle(qf_100_loader)
        unlearn_loader_cycle = cycle(unlearn_loader)
        CE = nn.CrossEntropyLoss()

        while unlearn_acc != 0:
            model.train()
            for i, data in enumerate(zip(adv_loader, unlearn_loader_cycle)):

                model.train()

                (adv_data, adv_target), (data, target) = data

                optimizer.zero_grad()

                output_adv = model(adv_data.to(device))
                output = model(data.to(device))
                # target = target.squeeze().long()  # 是path的
                # adv_target = adv_target.squeeze().long()  # 是path的
                tensor11 = torch.randn(1, 1)
                # print("target.size():", target.size())
                if target.size() == tensor11.size():
                    # 如果是标量，增加一个维度并转换为 long 类型
                    target = target.squeeze(0).long()
                else:
                    target = target.squeeze().long()  # 注意，这里的 squeeze 实际上不会改变形状

                if adv_target.size() == tensor11.size():
                    # 如果是标量，增加一个维度并转换为 long 类型
                    adv_target = adv_target.squeeze(0).long()
                else:
                    # 如果不是标量，仅转换类型为 long
                    adv_target = adv_target.squeeze().long()
                loss_unlearn = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                loss_adv = CE(output_adv, adv_target.to(device)) * (
                            adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))

                loss = loss_unlearn + loss_adv

                loss.backward()
                optimizer.step()

                model.eval()
                unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
                # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)

                if unlearn_acc == 0:
                    print('unlearn_acc == 0, Break at j = ', j, ' i = ', i)
                    break

            j += 1

            if max_iter < j:
                break

        model.eval()
        unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
        # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
        other_loss, other_acc, other_f1 = testins(model, device, other_loader)
        # other_loss, other_acc,other_f1= test(model, device, base2_loader)
        test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
        str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
            unlearn_acc) + ', D_test acc : ' + str(test_acc)
        print(str_list)
        print(' after D_test - D_forget f1:', other_f1, 'D_forget f1:', unlearn_f1, 'D_test f1:', test_f1)
        member_gt = [1 for i in range(1)]
        qf1_model = deepcopy(model)
        qf1_model.load_state_dict(best_model_stateqf1)
        #
        # average_euclidean0, average_manhattan0, average_cosine_similarity0 = analyze_sample_similarity(qf1_model,
        #                                                                                                device,
        #                                                                                                train_dataset,
        #                                                                                                CONFIG)
        # average_euclidean1, average_manhattan1, average_cosine_similarity1 = analyze_sample_similarity(model, device,
        #                                                                                                train_dataset,
        #                                                                                                CONFIG)
        # average_kl_div = calculate_kl_divergence(model, qf1_model, unlearn_loader, device)
        # _t, pv, EMA_res, risk_score = api(device, model, unlearn_loader, member_gt, cal_1000_loader, caltest1_loader)
        # print(
        #     f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
        # print(
        #     f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
        # print(
        #     f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
        # print(f'average_kl_div: {average_kl_div}')

        case2_D_test.append(test_acc)
        case2_D_r.append(other_acc)
        case2_D_f.append(unlearn_acc)

        print()
        print('\n Baseline 3: Our Appraoch - using both adversarial examples and weight importance')

        # model = resnet18().to(device)
        # model.load_state_dict(torch.load('./cifar10_pretrained_models/resnet18.pt'))
        # normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # model = torch.nn.Sequential(normalize_layer, model)
        model = get_student_model().to(device)
        # normalize_layer = NormalizeLayer((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # normalize_layer = NormalizeLayer((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # model = torch.nn.Sequential(normalize_layer, model)
        # model = resnet18().to(device)
        model.load_state_dict(best_model_statet)

        # best_model_state, best_accuracy = train_model(model, base1_loader, test1_loader, optimizer, criterion, device)
        # (print
        #  (f'Best Validation Accuracy: {best_accuracy:.2f}%'))
        # model.load_state_dict(best_model_state)

        # model.load_state_dict(torch.load("/root/autodl-tmp/wangbin/L2UL-main/weights/best_model_path_base1.pth"))
        # optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=1e-3)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)

        origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        model_for_importance = copy.deepcopy(model)
        num_samples = len(unlearn_loader)
        importance = estimate_parameter_importance(unlearn_loader, model_for_importance, device, num_samples, optimizer)
        # importance = estimate_parameter_importance(unlearn_loader, model_for_importance, device, num_samples, optimizer)

        for keys in importance.keys():
            importance[keys] = (importance[keys] - importance[keys].min()) / (
                        importance[keys].max() - importance[keys].min())
            importance[keys] = (1 - importance[keys])

        CE = nn.CrossEntropyLoss()

        unlearn_acc = 100

        model.eval()
        other_loss, other_acc, other_f1 = testins(model, device, other_loader)
        # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
        unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
        # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
        test_loss, test_acc, test_f1 = testins(model, device, test1_loader)

        str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
            unlearn_acc) + ', D_test acc : ' + str(test_acc)
        print(str_list)
        print('before D_test - D_forget f1:', other_f1, 'D_forget f1:', unlearn_f1, 'D_test f1:', test_f1)
        adv_dataset = JointDataset(adv_images, target_labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)

        j = 0
        unlearn_loader_cycle = cycle(unlearn_loader)

        # while unlearn_acc != 0:
        while unlearn_acc > 70:
            for i, data in enumerate(zip(adv_loader, unlearn_loader_cycle)):

                model.train()

                (adv_data, adv_target), (data, target) = data

                optimizer.zero_grad()

                output_adv = model(adv_data.to(device))
                output = model(data.to(device))
                # target = target.squeeze().long()  # 是path的
                # adv_target = adv_target.squeeze().long()  # 是path的
                tensor11 = torch.randn(1, 1)
                # print("target.size():", target.size())
                if target.size() == tensor11.size():
                    # 如果是标量，增加一个维度并转换为 long 类型
                    target = target.squeeze(0).long()
                else:
                    target = target.squeeze().long()  # 注意，这里的 squeeze 实际上不会改变形状

                if adv_target.size() == tensor11.size():
                    # 如果是标量，增加一个维度并转换为 long 类型
                    adv_target = adv_target.squeeze(0).long()

                    # print(clabels.dim())
                else:
                    adv_target = adv_target.squeeze().long()  # 注意，这里的 squeeze 实际上不会改变形状
                loss_unlearn = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                loss_adv = CE(output_adv, adv_target.to(device)) * (
                            adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))

                loss_reg = 0

                for n, p in model.named_parameters():
                    if n in importance.keys():
                        loss_reg += torch.sum(importance[n] * (p - origin_params[n]).pow(2)) / 2

                loss = loss_unlearn + loss_adv + loss_reg * args.reg_lamb

                loss.backward()
                optimizer.step()

                model.eval()
                unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)

                # if unlearn_acc == 0:
                if unlearn_acc <= 70:
                    #     print ('unlearn_acc <=50 , Break at j = ', j, ' i = ', i)
                    print('unlearn_acc == 0, Break at j = ', j, ' i = ', i)
                    break

            j += 1

            if max_iter < j:
                break

        model.eval()
        unlearn_loss, unlearn_acc, unlearn_f1 = testins(model, device, unlearn_loader)
        # unlearn_loss, unlearn_acc,unlearn_f1 = test(model, device, qf_100_loader)
        other_loss, other_acc, other_f1 = testins(model, device, other_loader)
        # other_loss, other_acc,other_f1 = test(model, device, base2_loader)
        test_loss, test_acc, test_f1 = testins(model, device, test1_loader)
        # test_loss, test_acc,test_f1 = test(model, device, test1_loader)

        str_list = '\n After | D_test - D_forget acc : ' + str(other_acc) + ', D_forget acc : ' + str(
            unlearn_acc) + ', D_test acc : ' + str(test_acc)
        logger.info(str_list)
        # logger.info(' after D_test - D_forget f1:', other_f1, 'D_forget_f1:', unlearn_f1, 'D_test_f1:', test_f1)
        print(' after D_test - D_forget f1:', other_f1, 'D_forget_f1:', unlearn_f1, 'D_test_f1:', test_f1)

        logger.info( f'  after D_test - D_forget f1: {other_f1}, D_forget_f1: {unlearn_f1}, D_test_f1: {test_f1}')
        # member_gt = [1 for i in range(1000)]
        member_gt = [1 for i in range(100)]
        qf1_model = deepcopy(model)
        qf1_model.load_state_dict(best_model_stateqf1)
        u1=model.state_dict()
        average_euclidean0, average_manhattan0, average_cosine_similarity0 = analyze_sample_similarity(qf1_model,best_model_stateqf1,
                                                                                                       device,
                                                                                                       train_dataset,
                                                                                                       CONFIG)
        average_euclidean1, average_manhattan1, average_cosine_similarity1 = analyze_sample_similarity(model,u1, device,
                                                                                                       train_dataset,
                                                                                                       CONFIG)
        average_kl_div = calculate_kl_divergence(model,best_model_state_retrained ,qf1_model, unlearn_loader, device)
        f_u,u=model.feature_extractor.state_dict(),model.state_dict()
        model1=get_student_model().to(device)
        _t, pv, EMA_res, risk_score = api(device,model1,u, f_u ,unlearn_loader, member_gt, cal_1000_loader, caltest1_loader)
        print(
            f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
        print(
            f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
        print(
            f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
        print(f'average_kl_div: {average_kl_div}')
        logger.info(
            f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
        logger.info(
            f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
        logger.info(
            f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')

        num = 1
        tsnes(unlearn_loader, other_loader, kd0_5_loader_no, model, model, model, u, f_u, num)

        logger.info(f'average_kl_div: {average_kl_div}')
        case3_D_test.append(test_acc)
        case3_D_r.append(other_acc)
        case3_D_f.append(unlearn_acc)

def train_and_forget(network,base2_loader,kd0_5_loader_no,test1_loader ,best_model_state_strained,best_model_state_retrained, dataset, classes,qf_1_dataset, qf_1_loader,base2_dataset , method,gpu,train_dataset,CONFIG,cal_1000_loader,caltest1_loader):
    # batch_size = args.b

    # get network
    # net = getattr(models, args.net)(num_classes=args.classes)
    net = get_student_model()
    net.load_state_dict(best_model_state_strained)

    unlearning_teacher = get_student_model()
    # unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)

    if gpu:
        net = net.cuda()
        unlearning_teacher = unlearning_teacher.cuda()

    # 读取数据的root
    # root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

    # 照片的尺寸

    img_size = 224 if network == "ViT" else 32

    # 数据集 以及数据集加载
    # trainset = getattr(datasets, args.dataset)(
    #     root=root, download=True, train=True, unlearning=True, img_size=img_size
    # )
    # validset = getattr(datasets, args.dataset)(
    #     root=root, download=True, train=False, unlearning=True, img_size=img_size
    # )

    trainloader = base2_loader
    # trainloader = DataLoader(trainset, num_workers=4, batch_size=args.b, shuffle=True)
    validloader = test1_loader
    # validloader = DataLoader(validset, num_workers=4, batch_size=args.b, shuffle=False)
    # 数据集划分
    # print("Dataset length:", len(trainset))
    # dataset_length=len(trainset)
    # split1 = int(dataset_length * 0.002)  # 80% 用于训练
    # split2 = dataset_length - split1    # 剩余的 20% 用于验证
    # forget_train, retain_train = torch.utils.data.random_split(trainset, [split1, split2])
    # forget_train, retain_train = torch.utils.data.random_split(
    #     trainset, [args.forget_perc, 1 - args.forget_perc]
    # )

    # 数据集的加载
    # forget_train_dl = DataLoader(list(forget_train), batch_size=128)

    # 修改1000的时候这里需要修改
    forget_train_dl = qf_1_loader
    # forget_train_dl = qf_100_loader
    # 修改1000的时候这里需要修改
    retain_train_dl = base2_loader
    # retain_train_dl = base2_loader
    # retain_train_dl = DataLoader(list(retain_train), batch_size=128, shuffle=True)
    forget_valid_dl = forget_train_dl
    retain_valid_dl = base2_loader
    # retain_valid_dl = validloader

    # Change alpha here as described in the paper
    model_size_scaler = 1
    if net == "ViT":
        model_size_scaler = 1
    else:
        model_size_scaler = 1

    # full_train_dl = DataLoader(
    #     ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    #     batch_size=batch_size,
    # )
    full_train_dl = base2_loader
    # 修改1000的时候这里需要修改
    kwargs = {
        "model": net,
        "unlearning_teacher": unlearning_teacher,
        "retain_train_dl": retain_train_dl,
        "retain_valid_dl": retain_valid_dl,
        "forget_train_dl": forget_train_dl,
        "forget_valid_dl": forget_valid_dl,
        "full_train_dl": full_train_dl,
        "valid_dl": validloader,
        "dampening_constant": 1,
        "selection_weighting": 10 * model_size_scaler,
        "num_classes": classes,
        "dataset_name": dataset,
        "device": "cuda" if gpu else "cpu",
        "model_name": network,
        "cal_1000_loader": cal_1000_loader,
        "caltest1_loader": caltest1_loader,
        # "base3_loader": base_1000_loader,
        "qf_100_dataset": qf_1_dataset,
        "base2_dataset": base2_dataset,
        # "base2_dataset":base2_dataset,
        "method": method,
        "best_model_state_retrained": best_model_state_retrained,
        "train_dataset": train_dataset,
        "CONFIG": CONFIG,
        'kd0_5_loader_no':kd0_5_loader_no
    }



    # wandb.init(project=f"{args.dataset}_forget_random_{args.forget_perc}", name=f'{args.method}')

    import time

    start = time.time()

    testacc, retainacc, zrf, mia, d_f,curent = getattr(forget_random_strategies, method)(
        **kwargs
    )
    # testacc, retainacc, zrf, mia, d_f,curent = getattr(forget_random_strategies_dermaa, method)(
    #     **kwargs
    # )
    end = time.time()
    time_elapsed = end - start

    print(testacc, retainacc, zrf, mia,curent)
    logger.info(f"TestAcc: {testacc}, RetainTestAcc: {retainacc}, ZRF: {zrf}, MIA: {mia}, Df: {d_f},current:{curent}")
    # wandb.log(
    #     {
    #         "TestAcc": testacc,
    #         "RetainTestAcc": retainacc,
    #         "ZRF": zrf,
    #         "MIA": mia,
    #         "Df": d_f,
    #         "model_scaler": model_size_scaler,
    #         "MethodTime": time_elapsed,
    #         # do not forget to deduct baseline time from it to remove results calc (acc, MIA, ...)
    #     }
    # )
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
def aggregate_predictions(models, X_test, weights):
    with torch.no_grad():
        # 使用加权平均合并模型的 logits
        weighted_logits = sum(weight * model(X_test) for model, weight in zip(models, weights))
        # 计算最终的概率分布
        probabilities = softmax(weighted_logits, dim=1)
        final_prediction = probabilities.max(1)[1]
    return final_prediction
def sisa(train_dataset,CONFIG,qf_50_loader,test1_loader,base2_loader,kd0_5_loader_no,cal_100_loader, caltest1_loader,device,random_seed):
    # 首先生成随机打乱的索引
    indices = np.random.permutation(len(Subset(train_dataset, CONFIG['BASE1']['BASE'])))

    # 分片
    shards = np.array_split(indices, 10)

    # 调整分片以确保每个分片的大小不满足 len(shard) % 32 == 1
    shards = adjust_shards(shards)

    # 划分数据为三个片段
    # shards = np.array_split(np.random.permutation(len(Subset(train_dataset, CONFIG['BASE1']['BASE']))), 10)
    print(len(Subset(train_dataset, CONFIG['BASE1']['BASE'])))
    models = []
    accuracies = []
    # for i, shard in enumerate(shards):
    #     print(f"Length of shard {i + 1}: {len(shard)}")
    # 训练模型并计算准确率
    for shard in shards:
        # X_shard = X[shard]
        # y_shard = y[shard]
        subset = Subset(Subset(train_dataset, CONFIG['BASE1']['BASE']), shard)
        # dataset = TensorDataset(X_shard, y_shard)
        # loader = DataLoader(dataset, batch_size=5, shuffle=True)
        dataloader = DataLoader(subset, batch_size=32, shuffle=True,
                                generator=torch.Generator().manual_seed(random_seed))

        model = get_teacher_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        model.train()
        for epoch in range(20):
            for data, target in dataloader:
                data = data.to(device)
                target = target.to(device)
                target = target.squeeze().long()  # 是path的
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
                inputs, labels = data[0].to(device), data[1].to(device)
                # inputs, labels = data[0].to(device), data[1].to(device)
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
        # f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        # print(f'Precision: {precision}')
        # print(f'Recall: {recall}')
        # print(f'F1 Score: {f1}')

        # print(f'Accuracy on retain data: {100 * correctt / totalt}%')
        # print(f'Accuracy on test data: {100 * correcttt / totaltt}%')
        accuracies.append(correcttt / totaltt)

        models.append(model)

    # 根据准确率计算权重
    weights = torch.tensor(accuracies) / sum(accuracies)
    # print(weights)

    # 对测试数据进行预测
    # X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.3)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # predicted = aggregate_predictions(models, X_test, weights)
    # print("Initial Accuracy:", accuracy_score(y_test, predicted.numpy()))

    # 假设我们需要遗忘第一个片段的数据
    # indices_to_forget = shards[0][:10]
    # print(shards[0])
    # member_gt = [1 for i in range(1000)]
    member_gt = [1 for i in range(100)]
    _t, pv, EMA_res, risk_score = api_sisa(device, models, weights, qf_50_loader, member_gt, cal_100_loader, caltest1_loader)
    print("EMA_res:", EMA_res, "risk_score:", risk_score, "pvalue:", pv)
    # 更新shards[0]，排除需要遗忘的数据
    # shards[0] = np.setdiff1d(shards[0], indices_to_forget)
    forget_set = set(CONFIG['QF_100']['QUERY'])
    # forget_set = set(CONFIG['QF_100']['QUERY'])
    updated_shards = []
    for shard in shards:
        shard_set = set(shard)
        updated_shard = list(shard_set - forget_set)
        updated_shards.append(updated_shard)
    updated_shards = adjust_shards(updated_shards)
    # 使用更新后的shards[0]重新训练模型
    # for i, shard in enumerate(updated_shards):
    #     print(f"Length of shard {i + 1}: {len(shard)}")
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
        dataloader = DataLoader(subset, batch_size=32, shuffle=True,
                                generator=torch.Generator().manual_seed(random_seed))

        model = get_teacher_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        model.train()
        for epoch in range(20):
            for data, target in dataloader:
                target = target.squeeze().long()  # 是path的
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
                inputs, labels = data[0], data[1]
                # inputs, labels = data[0].to(device), data[1].to(device)
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
        # f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'F1 m: {f1}_macro')

        # print(f'Accuracy on retain data: {100 * correctt / totalt}%')
        print(f'Accuracy on test data: {100 * correcttt / totaltt}%')
        accuracies.append(correcttt / totaltt)

        models[i - 1] = model
    # 根据准确率计算权重
    weights = torch.tensor(accuracies) / sum(accuracies)
    weights = weights.to(device)
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
            models = [model.to(device) for model in models]
            out_Y = aggregate_predictions(models, X, weights)
            predicted.append(out_Y)
            Y = Y.squeeze().long()  # pathmnist
            zY.append(Y)
    predicted = torch.cat(predicted).cpu().numpy()
    zY = torch.cat(zY).cpu().numpy()
    # predicted = aggregate_predictions(models, X_test, weights)
    print("Updated Accuracy:", accuracy_score(zY, predicted))
    logger.info(f"Updated Accuracy: {accuracy_score(zY, predicted)}")
    print("F1 Score (Macro):", calculate_f12_score(zY, predicted, average='macro'))
    logger.info(f"F1 Score (Macro): {calculate_f12_score(zY, predicted, average='macro')}")
    f1_score = calculate_f1_score(zY, predicted)
    print("F1 Score:", f1_score)
    logger.info(f"F1 Score: {f1_score}")
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_test = torch.tensor(y_test, dtype=torch.int64)
    # dataset_test = TensorDataset(X_test,y_test)
    # loader_test = DataLoader(dataset_test, batch_size=5, shuffle=True)
    # member_gt = [1 for i in range(50)]
    member_gt = [1 for i in range(100)]
    # print(models)
    # print(weights)
    _t, pv, EMA_res, risk_score = api_sisa(device, models, weights, qf_50_loader, member_gt, cal_100_loader, caltest1_loader)
    print("EMA_res:", EMA_res, "risk_score:", risk_score, "pvalue:", pv)
    logger.info(f"EMA_res: {EMA_res}, risk_score: {risk_score}, pvalue: {pv}")
    average_euclidean_retrained1, average_manhattan_retrained1, average_cosine_similarity_retrained1 = analyze_sample_similarity_sisa(models, weights,device,train_dataset, CONFIG)
    print(f'average_euclidean_retrained1: {average_euclidean_retrained1}, average_manhattan_retrained1: {average_manhattan_retrained1}, average_cosine_similarity_retrained1: {average_cosine_similarity_retrained1}')
    logger.info(f'average_euclidean_retrained1: {average_euclidean_retrained1}, average_manhattan_retrained1: {average_manhattan_retrained1}, average_cosine_similarity_retrained1: {average_cosine_similarity_retrained1}')

    num = 1
    tsnessisa(qf_50_loader, base2_loader, kd0_5_loader_no,models, models, models, weights, weights,num)
    # tsnes(qf_50_loader, base2_loader, kd0_5_loader_no,model, model, model, u, f_u,num)

    torch.save(model.state_dict(), "model_weights.pth")


def testk(model, test_loader, device):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, Y in test_loader:
                X, Y = X.to(device), Y.to(device)
                Y = Y.squeeze()  # 在path中使用

                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += Y.size(0)
                correct += (predicted == Y).sum().item()

                # 将预测值和真实标签保存起来，用于计算 F1 score
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(Y.cpu().numpy())

        accuracy = 100 * correct / total
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_labels_np, all_preds_np)
        # TP = conf_matrix[1, 1]  # 真正例
        # FP = conf_matrix[0, 1]  # 假正例
        # FN = conf_matrix[1, 0]  # 假负例
        TP = conf_matrix[1, 1]  # 真正例
        TN = conf_matrix[0, 0]  # 真负例
        FP = conf_matrix[0, 1]  # 假正例
        FN = conf_matrix[1, 0]  # 假负例
        # f1 = 2 * TP / (2 * TP + FP + FN)
        # accuracy = (TP + TN) / (TP + TN + FP + FN)
        # 计算 F1 score（宏平均）
        f1 = f1_score(all_labels, all_preds, average='macro')

        return accuracy, f1
def initialize_last_n_layers(model, n=10):
        layers = list(model.modules())
        for layer in layers[-n:]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)  # 使用 Xavier 初始化
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # 偏置初始化为0
            elif isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

def euk(train_dataset, CONFIG,best_model_state_trained,base2_loader,test1_loader,qf_100_loader, cal_1000_loader, caltest1_loader,device,kd0_5_loader_no):
    modelmlp = get_student_model().to(device)
    modelmlp.load_state_dict(best_model_state_trained)

    # def initialize_weights(model):
    #     for m in model.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_uniform_(m.weight)  # 使用 He 初始化方法
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
    #         elif isinstance(m, nn.LogSoftmax):
    #             pass  # LogSoftmax 不需要初始化
    # def initialize_weights(model):
    #     for m in model.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)  # 使用 Xavier 初始化方法（正态分布）
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
    #         elif isinstance(m, nn.LogSoftmax):
    #             pass  # LogSoftmax 不需要初始化
    import torch.nn as nn

    def freeze_first_n_layers(model, n=8):
        layers = list(model.modules())
        for layer in layers[:n]:
            for param in layer.parameters():
                param.requires_grad = False


    # 冻结前8层
    # freeze_first_n_layers(modelmlp, n=8)

    # 初始化最后10层
    initialize_last_n_layers(modelmlp, n=10)

    # 初始化分类器的所有层
    # initialize_weights(modelmlp.classifier)

    # u = modelmlp.state_dict()
    # 重新初始化最后一层分类器
    # modelmlp.classifier.network[0] = nn.Linear(256, 2).to(device)
    # modelmlp.classifier.network[1] = nn.LogSoftmax(dim=-1).to(device)

    # 冻结特征提取器的所有层
    # for param in modelmlp.feature_extractor.parameters():
    #     param.requires_grad = False

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    import torch.optim as optim
    import torch.nn as nn

    # 冻结模型的所有参数
    for param in modelmlp.parameters():
        param.requires_grad = False

    # 获取模型的所有层
    layers = list(modelmlp.children())  # 或 list(modelmlp.modules())，具体视模型结构而定

    # 解冻最后十层的参数
    last_layers = layers[-10:]  # 获取最后十层
    params_to_optimize = []
    for layer in last_layers:
        for param in layer.parameters():
            param.requires_grad = True
            params_to_optimize.append(param)

    # 使用 Adam 优化器，仅优化最后十层的参数
    optimizer = optim.Adam(params_to_optimize, lr=0.001)

    # criterion = nn.NLLLoss()
    # optimizer = optim.Adam(modelmlp.classifier.parameters(), lr=0.01)

    # 训练最后一层
    modelmlp.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(base2_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = modelmlp(inputs)
            labels = labels.squeeze().long()  # 是path的
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(base2_loader)}')

    # 测试重新训练最后一层的模型
    # def test(model, test_loader, device):
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for X, Y in test_loader:
    #             X, Y = X.to(device), Y.to(device)
    #             outputs = model(X)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += Y.size(0)
    #             correct += (predicted == Y).sum().item()
    #     accuracy = 100 * correct / total
    #     return accuracy



    accuracy_retrained = testk(modelmlp, test1_loader, device)
    print(f'Retrained Classifier Test Accuracy: {accuracy_retrained}%')
    logger.info(f'Retrained Classifier Test Accuracy: {accuracy_retrained}%')
    f_u1, u1 = modelmlp.feature_extractor.state_dict(), modelmlp.state_dict()
    # logger.info(f'>> evaluate membership attack on snet after training')
    # member_gt = [1 for i in range(1000)]
    member_gt = [1 for i in range(100)]
    # member_gt = [1 for i in range(50)]

    _t, pv, EMA_res, risk_score = api(device, modelmlp, u1, f_u1, qf_100_loader, member_gt, cal_1000_loader, caltest1_loader)
    print("EMA_res:", EMA_res, "risk_score:", risk_score, "pvalue:", pv)
    logger.info(f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
    average_euclidean_retrained, average_manhattan_retrained, average_cosine_similarity_retrained = analyze_sample_similarity(modelmlp, u1,device,train_dataset, CONFIG)

    print(f'average_euclidean_retrained: {average_euclidean_retrained}, average_manhattan_retrained: {average_manhattan_retrained}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained}')
    logger.info(f'average_euclidean_retrained: {average_euclidean_retrained}, average_manhattan_retrained: {average_manhattan_retrained}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained}')

    num = 1
    tsnes(qf_100_loader, base2_loader, kd0_5_loader_no, modelmlp, modelmlp, modelmlp, u1, f_u1, num)


def cfk(train_dataset, CONFIG,best_model_state_trained,base2_loader,test1_loader,qf_100_loader, cal_1000_loader, caltest1_loader,device,kd0_5_loader_no):

    # 微调最后一层
    # 这里我们仅微调最后一层，所以不解冻特征提取器层
    modelmlp2 =get_student_model().to(device)
    criterion2 = nn.CrossEntropyLoss()

    # criterion = nn.NLLLoss()
    modelmlp2.load_state_dict(best_model_state_trained)
    # for param in modelmlp2.feature_extractor.parameters():
    #     param.requires_grad = False

    # optimizer2 = optim.Adam(modelmlp2.classifier.parameters(), lr=0.001)

    for param in modelmlp2.parameters():
        param.requires_grad = False

        # 获取模型的所有层
    layers = list(modelmlp2.children())  # 或 list(modelmlp.modules())，具体视模型结构而定

    # 解冻最后十层的参数
    last_layers = layers[-10:]  # 获取最后十层
    params_to_optimize = []
    for layer in last_layers:
        for param in layer.parameters():
            param.requires_grad = True
            params_to_optimize.append(param)

    # 使用 Adam 优化器，仅优化最后十层的参数
    optimizer2 = optim.Adam(params_to_optimize, lr=0.001)

    modelmlp2.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(base2_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer2.zero_grad()
            outputs = modelmlp2(inputs)
            labels = labels.squeeze().long()
            loss = criterion2(outputs, labels)
            loss.backward()
            optimizer2.step()

            running_loss += loss.item()
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(base2_loader)}')
    # 测试微调后的模型
    accuracy_finetuned = testk(modelmlp2, test1_loader, device)
    print(f'Finetuned Classifier Test Accuracy: {accuracy_finetuned}%')
    logger.info(f'Finetuned Classifier Test Accuracy: {accuracy_finetuned}%')
    f_u2, u2 = modelmlp2.feature_extractor.state_dict(), modelmlp2.state_dict()
    # logger.info(f'>> evaluate membership attack on snet after training')
    # member_gt = [1 for i in range(1000)]
    member_gt = [1 for i in range(100)]

    _t, pv, EMA_res, risk_score = api(device, modelmlp2, u2, f_u2, qf_100_loader, member_gt, cal_1000_loader, caltest1_loader)
    print("EMA_res:", EMA_res, "risk_score:", risk_score, "pvalue:", pv)
    logger.info(f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
    average_euclidean_retrained1, average_manhattan_retrained1, average_cosine_similarity_retrained1 = analyze_sample_similarity(modelmlp2, u2,device,train_dataset, CONFIG)
    print(f'average_euclidean_retrained: {average_euclidean_retrained1}, average_manhattan_retrained: {average_manhattan_retrained1}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained1}')
    logger.info(f'average_euclidean_retrained: {average_euclidean_retrained1}, average_manhattan_retrained: {average_manhattan_retrained1}, average_cosine_similarity_retrained: {average_cosine_similarity_retrained1}')
    num = 2
    tsnes(qf_100_loader, base2_loader, kd0_5_loader_no, modelmlp2, modelmlp2, modelmlp2, u2, f_u2, num)
