from __future__ import print_function
import sys
sys.path.append('/root/autodl-fs/AdaptForget-main/machine_unlearninng/')
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
# from Domainadaptation_csv_three_wu import domainadaptation
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
from models.net_train_three import get_student_model, get_teacher_model
# from mlp_three_csv_wu import FeatureExtractor, Classifier, CombinedModel,get_student_model, get_teacher_model
# from tsne_mnist_guding2_tsne import tsnes

from torch.utils.data import Dataset, DataLoader, Subset
import os
from utilsinstance import JointDataset, NormalizeLayer, naive_train, train, adv_attack, testins, estimate_parameter_importance

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# from plot_with_new_data import update_plot_with_new_data
# from torch.utils.data import DataLoader, Subset, Dataset
# from torchvision import datasets, transforms
# # from tsne_mnist_samedata import tsne
# # from tsne_mnist_tuo import tsne
# from tsne_mnist_guding1 import tsnet
# # from tsne_mnist_guding2 import tsnes
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
# import datasets
# # import models
# import conf
from training_utils import *
logging.basicConfig(filename='./tc/training_log_qf1circulate_asdv7.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
def adaptforget(lambda_domain,lambda_risk,lambda_kd,train_dataset,num_epochsall, device, qf_100_loader, kd, test1_loader, cal_1000_loader,
                               caltest1_loader, best_model_state_retrained, best_model_state_trained,CONFIG,kd0_5_loader_no,base2_loader):
    smodelmlp = get_student_model().to(device)
    smodelmlp1 = get_student_model().to(device)
    smodelmlp_base2 = get_student_model().to(device)
    tmodelmlp = get_student_model().to(device)
    tmodelmlp.apply(init_weights)
    smodelmlp.apply(init_weights)
    u = smodelmlp.state_dict()
    f_u = smodelmlp.feature_extractor.state_dict()
    modelmlp = get_teacher_model().to(device)

    smodelmlp_base2.load_state_dict(best_model_state_retrained)
    modelmlp.load_state_dict(best_model_state_trained)

    member_gt = [1 for _ in range(1)]
    num = 0
    # best_accuracy = 0.0

    for epoch in range(num_epochsall):
        start_time = time.time()
        print(f"Epoch {epoch} starting...")
        logger.info(f"Epoch {epoch} starting...")

        print(f"Epoch {epoch} starting...")
        logger.info(f"Epoch {epoch} starting...")

        #  machine unlearning
        f_u, u, c_u = train_student_model_random(lambda_risk,lambda_kd,qf_100_loader, kd,test1_loader, tmodelmlp, modelmlp, smodelmlp, u, f_u)
        current_accuracy, accuracy1,f1 = test_model(test1_loader, qf_100_loader, kd, device, modelmlp,
                                                 smodelmlp, tmodelmlp, u, f_u)
        logger.info(f'>>current_accuracy2: {current_accuracy}, accuracy2: {accuracy1}, f12: {f1}')
        average_euclidean_adapt, average_manhattan_adapt, average_cosine_similarity_adapt = analyze_sample_similarity(smodelmlp, u,
                                                                                                                device,
                                                                                                                train_dataset,
                                                                                                                CONFIG)

        filename = f"smodel_weights_epoch_{epoch}.pth"
        torch.save(u, filename)
        print(f"Saved model weights for epoch {epoch} to {filename}")
        # logger
        print(f'>>average_euclidean_adapt: {average_euclidean_adapt}, average_manhattan_adapt: {average_manhattan_adapt}, average_cosine_similarity_adapt: {average_cosine_similarity_adapt}')
        logger.info(f'>>average_euclidean_adapt2: {average_euclidean_adapt}, average_manhattan_adapt2: {average_manhattan_adapt}, average_cosine_similarity_adapt2: {average_cosine_similarity_adapt}')


        average_kl_div_adapt=calculate_kl_divergence(smodelmlp,best_model_state_retrained,smodelmlp_base2, qf_100_loader, device)
        print(f'>>average_kl_div_adapt2: {average_kl_div_adapt}')
        logger.info(f'>>average_kl_div_adapt: {average_kl_div_adapt}')
        _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_100_loader, member_gt, cal_1000_loader,
                                          caltest1_loader)
        logger.info(f'Test value: {_t}, p-value: {pv}, EMA: {EMA_res}, Risk score: {risk_score}')


        f_u = domainadaptation(f_u, c_u, qf_100_loader, kd0_5_loader_no,lambda_domain)
        # f_u = domainadaptation(f_u, c_u, qf_100_loader, kd0_5_loader_no)
        smodelmlp1.load_state_dict(u)
        smodelmlp1.feature_extractor.load_state_dict(f_u)
        u = smodelmlp1.state_dict()
        average_euclidean_adapt, average_manhattan_adapt, average_cosine_similarity_adapt= analyze_sample_similarity(smodelmlp,u,device,train_dataset,CONFIG)
        print(
            f'>>average_euclidean_adapt1: {average_euclidean_adapt}, average_manhattan_adapt1: {average_manhattan_adapt}, average_cosine_similarity_adapt1: {average_cosine_similarity_adapt}')
        logger.info(
            f'>>average_euclidean_adap1t: {average_euclidean_adapt}, average_manhattan_adapt1: {average_manhattan_adapt}, average_cosine_similarity_adapt1: {average_cosine_similarity_adapt}')

        average_kl_div_adapt1=calculate_kl_divergence(smodelmlp, best_model_state_retrained,smodelmlp_base2, qf_100_loader, device)
        print(f'>>average_kl_div_adapt1: {average_kl_div_adapt1}')
        logger.info(f'>>average_kl_div_adapt1: {average_kl_div_adapt1}')
        _t, pv, EMA_res, risk_score = api(device, smodelmlp, u, f_u, qf_100_loader, member_gt, cal_1000_loader, caltest1_loader)
        current_accuracy,c2,f1 = test_model(test1_loader, qf_100_loader, kd, device, modelmlp, smodelmlp, tmodelmlp, u,f_u)
        num += 1
        logger.info(f'>>current_accuracy: {current_accuracy}, accuracy1: {c2},f1: {f1}')
        # tsnes(qf_100_loader, base2_loader, kd0_5_loader_no, smodelmlp, smodelmlp, smodelmlp, u, f_u, num)
        num += 1
        print(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
        logger.info(f'ad test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}')
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")
        logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")
