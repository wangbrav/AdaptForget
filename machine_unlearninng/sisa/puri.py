#coding:utf-8
import argparse
import sys
import torch
from utils.MIA import mia
import os
import pickle
from scipy.stats import ks_2samp, ttest_ind
import numpy as np
from utils.Log import log_creater
from utils.Risk import riskScore, riskScore2
import time
import torch.nn.functional as F
from torch.nn.functional import softmax
metrics = ['correctness', 'confidence', 'entropy']
def aggregate_predictions(models, X_test, weights):
    with torch.no_grad():
        # 使用加权平均合并模型的 logits
        weighted_logits = sum(weight * model(X_test) for model, weight in zip(models, weights))
        # 计算最终的概率分布
        # probabilities = softmax(weighted_logits, dim=1)
        # print(probabilities)
        # final_prediction = probabilities.max(1)[1]
        # print(weighted_logits)
    return weighted_logits
def get_member_ratio(datadf, thre_cnt=1, skip=[], mode='t'):
    sample2mem = []
    for idx, row in datadf.iterrows():
        sample2mem.append(np.sum([row[m] for m in (set(metrics) - set(skip))]))

    members_bool = np.asarray(sample2mem) >= thre_cnt

    if mode == 'ks':
        _, pvalue = ks_2samp(members_bool, [1 for i in range(10000)], mode='asymp')
    elif mode == 't':
        if members_bool.mean() == 1:
            return 1, 1
        else:
            _, pvalue = ttest_ind(members_bool, [1 for i in range(len(members_bool))], equal_var=True,
                                  nan_policy='raise')

    return np.sum(members_bool) / len(datadf), pvalue

def api(device, models,weights,qf_100_loader, member_gt, cal_1000_loader, caltest1_loader):
    queryTestDataLoader = qf_100_loader
    calTrainDataLoader = cal_1000_loader
    calTestnDataLoader = caltest1_loader

    # modelCal.eval()

    calTrainPred = []
    calTrainY = []
    calTestPred = []
    calTestY = []
    queryPred = []
    queryY = []

    # first, get query_output
    with torch.no_grad():
        for X, Y in queryTestDataLoader:

            X = X.to(device)
            Y = Y.to(device)
            out_Y = aggregate_predictions(models, X, weights)
            # print(out_Y)
            # out_Y = torch.exp(smodel18(X))  #mnist autism
            Y = Y.squeeze().long()  # pathmnist

            queryPred.append(out_Y)
            queryY.append(Y)
            # queryY = [item.unsqueeze(0) for item in queryY]# zhishiyongyu dange yangben de yiwagn

    # print(queryPred)
    queryY = torch.cat(queryY).detach().cpu().numpy()  # 这里进行了更改 加上了 unsqueeze(0)
    queryPred = torch.cat(queryPred).detach().cpu().numpy()

    # then， get calibration set output
    with torch.no_grad():
        for X, Y in calTrainDataLoader:
            X = X.to(device)
            Y = Y.to(device)
            Y = Y.squeeze().long()  # pathmnist

            out_Y = aggregate_predictions(models, X, weights)
            # out_Y = torch.exp(smodel18(X)) # mnist
            calTrainPred.append(out_Y)
            calTrainY.append(Y)
    calTrainY = torch.cat(calTrainY).detach().cpu().numpy()
    calTrainPred = torch.cat(calTrainPred).detach().cpu().numpy()

    with torch.no_grad():
        for X, Y in calTestnDataLoader:
            X = X.to(device)
            Y = Y.to(device)
            out_Y = aggregate_predictions(models, X, weights)
            Y = Y.squeeze().long()  # pathmnist

            # print(out_Y)

            # out_Y = torch.exp(smodel18(X))  # mnist

            calTestPred.append(out_Y)
            calTestY.append(Y)
    calTestY = torch.cat(calTestY).detach().cpu().numpy()
    calTestPred = torch.cat(calTestPred).detach().cpu().numpy()

    # run MIA
    MIA = mia(calTrainPred,
              calTrainY,
              calTestPred,
              calTestY,
              queryPred,
              queryY,
              num_classes=10)

    MIA_results = MIA._run_mia()

    t, pv = get_member_ratio(MIA_results['query_values_bin'], skip=['modified entropy'], mode='t')
    EMA_res = np.around(pv, decimals=2)
    risk_score = riskScore2(MIA_results['query_values_bin'], member_gt)
    return t, pv, EMA_res, risk_score