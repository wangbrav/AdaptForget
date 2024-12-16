"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate random forgetting.
Seperate file to allow for easy reuse.
"""

import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy
from puri import api
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset
from tqdm import tqdm
from test_model_path import test_model

from sklearn import linear_model, model_selection, logger

from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils4 import *
# import ssd as ssd
from ssd import ParameterPerturber  # 从 ssd 模块中导入 ParameterPerturber 类
from qf1kosiam import analyze_sample_similarity

from calculate_kl_divergence import calculate_kl_divergence
from tsne_mnist_guding2_tsne import tsnes



def filter_dataset_by_label(dataset, unlearninglabels):
    indices = []
    for idx, (_, label) in enumerate(dataset):
        rnd = random.choice(unlearninglabels)
        while rnd == label:
            rnd = random.choice(unlearninglabels)
        indices.append(idx)  # 只记录与随机标签不同的索引
    return indices

def get_metric_scores(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    cal_1000_loader,
    caltest1_loader,
    method,
    best_model_state_retrained,
    train_dataset,
    CONFIG,
    kd0_5_loader_no
):
    qf1_model = deepcopy(model)
    qf1_model.load_state_dict(best_model_state_retrained)
    u =model.state_dict()
    # qf1_model.load_state_dict(torch.load('/root/autodl-tmp/wangbin/selective-synaptic-dampening-main/src/weight/trained_model_pathmnistqf1.pth'))
    average_euclidean0, average_manhattan0, average_cosine_similarity0 = analyze_sample_similarity(qf1_model,best_model_state_retrained ,device, train_dataset, CONFIG)
    average_euclidean1, average_manhattan1, average_cosine_similarity1 = analyze_sample_similarity(model, u,device, train_dataset, CONFIG)
    average_kl_div = calculate_kl_divergence(model,best_model_state_retrained,qf1_model, forget_train_dl,device)
    current_accuracy, accuracy1 ,f1= test_model(valid_dl, valid_dl, valid_dl, device, model,
                                             model, model, u, u)
    retain_acc_dict = evaluate(model, retain_valid_dl, device)
    loss_acc_dict = evaluate(model, valid_dl, device)
    zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 128, device)
    d_f = evaluate(model, forget_valid_dl, device)
    mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)
    #修改1000的时候这里也需要修改
    # member_gt = [1 for i in range(1000)]
    member_gt = [1 for i in range(100)]
    u,f_u =model.state_dict(),model.feature_extractor.state_dict()
    num = 2
    _t, pv, EMA_res, risk_score = api(device,model,u,f_u , forget_train_dl, member_gt, cal_1000_loader,caltest1_loader)
    print(f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}, loss_acc_dict["Acc"]: {loss_acc_dict["Acc"]},loss_f1_dict["F1"]:{loss_acc_dict["F1"]},method: {method}')
    logger.info(f' test value: {_t}, p_value: {pv}, ema: {EMA_res}, risk_score: {risk_score}, loss_acc_dict["Acc"]: {loss_acc_dict["Acc"]},loss_f1_dict["F1"]:{loss_acc_dict["F1"]},method: {method}')
    logger.info(f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
    print(f'average_euclidean0: {average_euclidean0}, average_manhattan0: {average_manhattan0}, average_cosine_similarity0: {average_cosine_similarity0}')
    logger.info(f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
    # print(f'average_euclidean1: {average_euclidean1}, average_manhattan1: {average_manhattan1}, average_cosine_similarity1: {average_cosine_similarity1}')
    print(f'average_kl_div: {average_kl_div}')
    logger.info(f'average_kl_div: {average_kl_div}')
    tsnes(forget_train_dl, retain_train_dl, kd0_5_loader_no,model, model, model, u, f_u,num)
    # num += 1


    return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], zrf, mia, d_f["Acc"],current_accuracy)


def baseline(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    cal_1000_loader,
    caltest1_loader,
    method,
    best_model_state_retrained,
    train_dataset,
    **kwargs,
):
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        cal_1000_loader,
        caltest1_loader,
        method,
        best_model_state_retrained,
    )


def retrain(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dataset_name,
    model_name,
    device,
    cal_1000_loader,
    caltest1_loader,
    method,
    best_model_state_retrained,
    train_dataset,
    **kwargs,
):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    if model_name == "ViT":
        epochs = getattr(conf, f"{dataset_name}_{model_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_{model_name}_MILESTONES")
    else:
        epochs = getattr(conf, f"{dataset_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_MILESTONES")
    _ = fit_one_cycle(
        epochs,
        model,
        retain_train_dl,
        retain_valid_dl,
        milestones=milestones,
        device=device,
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        cal_1000_loader,
        caltest1_loader,
        method,
        train_dataset,
    )


def finetune(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    cal_1000_loader,
    caltest1_loader,
    method,
    **kwargs,
):
    _ = fit_one_cycle(
        5, model, retain_train_dl, retain_valid_dl, lr=0.02, device=device
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        cal_1000_loader,
        caltest1_loader,
        method,
    )


def blindspot(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    cal_1000_loader,
    caltest1_loader,
    # base3_loader,
    base2_dataset,
    qf_100_dataset,
    method,
    best_model_state_retrained,
    train_dataset,
    CONFIG,
    kd0_5_loader_no,
    **kwargs,
):
    student_model = deepcopy(model)
    # base2_indices = CONFIG['BASE2']['BASE']
    base2_indices = CONFIG['BASE2']['BASE']
    # base2_dataset = Subset(train_dataset, base2_indices)

    # 直接从索引中随机采样 30%
    retain_indices = random.sample(base2_indices, int(0.3 * len(base2_indices)))
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = Subset(base2_dataset, retain_indices)

    # retain_train_subset = random.sample(
    #     base2_list, int(0.3 * len(base2_dataset))
    # )
    # retain_train_subset = random.sample(
    #     retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset))
    # )

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        # retain_data=base3_loader,
        retain_data=retain_train_subset,
        forget_data=qf_100_dataset,
        # forget_data=forget_train_dl.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=0.001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
    )

    return get_metric_scores(
        student_model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        cal_1000_loader,
        caltest1_loader,
        method,
        best_model_state_retrained,
        train_dataset,
        CONFIG,
        kd0_5_loader_no,

    )


def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    device,
    cal_1000_loader,
    caltest1_loader,
    qf_100_dataset,
    base2_dataset,
    method,
    best_model_state_retrained,
    train_dataset,
    CONFIG,
    kd0_5_loader_no,
    **kwargs,
):
    best_model_state_retrained =best_model_state_retrained
    model1 = model
    # unlearninglabels = list(range(9))
    unlearninglabels = list(range(num_classes))
    # unlearning_trainset = []
    # unlearninglabels = list(range(num_classes))
    # unlearning_trainset = []


    # for x, clabel in qf_100_dataset:
    # # for x, _, clabel in forget_train_dl.dataset:
    #     rnd = random.choice(unlearninglabels)
    #     while rnd == clabel:
    #         rnd = random.choice(unlearninglabels)
    #     unlearning_trainset.append((x,  rnd))
    #     # unlearning_trainset.append((x, _, rnd))
    #     print(unlearning_trainset)
    #
    # for x,  y in base2_dataset:
    # # for x, _, y in retain_train_dl.dataset:
    #     unlearning_trainset.append((x,  y))
    #     # unlearning_trainset.append((x, _, y))
    # 处理 qf_100_dataset


    qf_100_indices = filter_dataset_by_label(qf_100_dataset, unlearninglabels)
    qf_100_subset = Subset(qf_100_dataset, qf_100_indices)

    base2_indices = range(len(base2_dataset))  #
    base2_subset = Subset(base2_dataset, base2_indices)

    #
    unlearning_trainset = ConcatDataset([qf_100_subset, base2_subset])


    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )
    # unlearning_train_set_dl = DataLoader(
    #     unlearning_trainset, 32, pin_memory=True, shuffle=True
    # )

    history, u = fit_one_unlearning_cycle(
        3, model1, unlearning_train_set_dl, retain_valid_dl, device=device, lr=0.01
    )
    #
    # _ = fit_one_unlearning_cycle(
    #     2, model, unlearning_train_set_dl, retain_valid_dl, device=device, lr=0.0001
    # )
    model1.load_state_dict(u)
    return get_metric_scores(
        model1,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        cal_1000_loader,
        caltest1_loader,
        method,
        best_model_state_retrained,
        train_dataset,
        CONFIG,
        kd0_5_loader_no,



    )


def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    device,
    cal_1000_loader,
    caltest1_loader,
    method,
    best_model_state_retrained,
    train_dataset,
    CONFIG,
    kd0_5_loader_no,
    **kwargs,
):
    model1 = model
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, orig_target in tqdm(train_loader):
        # for data, _, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            orig_target = orig_target.squeeze().long()

            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                # target = target.long()
                target = target.unsqueeze(0)  # 增加一个维度

                # print(target)
                # print("Output shape:", output.shape)
                # print("Output type:", output.dtype)
                # print("Target shape:", target.shape)
                # print("Target type:", target.dtype)



                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model1.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_train_dl.dataset, model1)

    fisher_dir = []
    alpha = 1e-6
    # alpha = 1e-6
    for i, p in enumerate(model1.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    return get_metric_scores(
        model1,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        cal_1000_loader,
        caltest1_loader,
        method,
        best_model_state_retrained,
        train_dataset,
        CONFIG,
        kd0_5_loader_no,
    )


def ssd_tuning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    cal_1000_loader,
    caltest1_loader,
    method,
    best_model_state_retrained,
    train_dataset,
    CONFIG,
    kd0_5_loader_no,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }
    model1 = model
    # load the trained model
    optimizer = torch.optim.SGD(model1.parameters(),  lr=0.001)

    # ssd = ssd.ParameterPerturber(model, optimizer, device, parameters)
    # 
    ssd_instance = ParameterPerturber(model1, optimizer, device, parameters)
    model1= model1.eval()

    sample_importances = ssd_instance.calc_importance(forget_train_dl)

    original_importances = ssd_instance.calc_importance(full_train_dl)
    ssd_instance.modify_weight(original_importances, sample_importances)
    return get_metric_scores(
        model1,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
        cal_1000_loader,
        caltest1_loader,
        method,
        best_model_state_retrained,
        train_dataset,
        CONFIG,
        kd0_5_loader_no,
    )
