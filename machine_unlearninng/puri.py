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

metrics = ['correctness', 'confidence', 'entropy']
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

def api(device, smodel18,u,f_u,qf_100_loader, member_gt, cal_1000_loader,caltest1_loader):
    queryTestDataLoader = qf_100_loader
    calTrainDataLoader = cal_1000_loader
    calTestnDataLoader = caltest1_loader
    smodel18.load_state_dict(u)
    smodel18.feature_extractor.load_state_dict(f_u)
    smodel18.eval()
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
            Y = Y.squeeze().long()  # pathmnist
            out_Y = torch.exp(F.log_softmax(smodel18(X), dim=-1)) 
            # out_Y = torch.exp(smodel18(X))  #mnist autism

            queryPred.append(out_Y)
            queryY.append(Y)
            # queryY = [item.unsqueeze(0) for item in queryY]


    queryY = torch.cat(queryY).detach().cpu().numpy()
    queryPred = torch.cat(queryPred).detach().cpu().numpy()

    # then， get calibration set output
    with torch.no_grad():
        for X, Y in calTrainDataLoader:
            X = X.to(device)
            Y = Y.to(device)
            Y = Y.squeeze().long()  
            out_Y = torch.exp(F.log_softmax(smodel18(X), dim=-1)) 
            # out_Y = torch.exp(smodel18(X))
            calTrainPred.append(out_Y)
            calTrainY.append(Y)
    calTrainY = torch.cat(calTrainY).detach().cpu().numpy()
    calTrainPred = torch.cat(calTrainPred).detach().cpu().numpy()

    with torch.no_grad():
        for X, Y in calTestnDataLoader:
            X = X.to(device)
            Y = Y.to(device)
            Y = Y.squeeze().long()  
            out_Y = torch.exp(F.log_softmax(smodel18(X), dim=-1))  # path mnist

            # out_Y = torch.exp(smodel18(X)) 

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