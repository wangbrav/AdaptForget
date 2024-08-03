
from resnet18_domain import ResNet18Modified
# from mlp_domain import get_student_model
from mu.net_three import get_student_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from KDloss import SoftTarget
from tqdm import tqdm
import pandas as pd
import sys
def domainadaptation(weights, forget_loader,test_loader):
    print("domainadaptation adversarial")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model_root = 'models'
    cuda = True

    lr = 0.0001
    batch_size = 128
    image_size = 28
    n_epoch = 8  # kd 0.25 qf1000
    # n_epoch = 4 #原来
    # 加载模型并设置到正确的设备
    smodel18 = get_student_model()
    # smodel18 = ResNet18Modified().to(device)
    smodel18 =smodel18.to(device)
    # model.load_state_dict(torch.load(weights))
    smodel18.feature_extractor.load_state_dict(weights)

    optimizer = optim.Adam(smodel18.parameters(), lr=lr)

    # loss_class = torch.nn.NLLLoss() mnist

    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.NLLLoss()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()
    dataloader_source = test_loader
    # dataloader_source = forget_loader
    dataloader_target = forget_loader
    # dataloader_target = test_loader
    for p in smodel18.parameters():
        p.requires_grad = True

    features_list = []
    num_epochs = 1
    # 遍历数据加载器

    best_accu_t = 0.0
    for epoch in range(n_epoch):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source

            smodel18.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                s_label = s_label.squeeze().long()
                domain_label = domain_label.cuda()

            class_output, domain_output = smodel18(s_img, alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            if cuda:
                t_img = t_img.cuda()
                domain_label = domain_label.cuda()

            _, domain_output = smodel18(t_img, alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            # err =  err_t_domain + err_s_domain +  1.3*err_s_label   # path kd 0.25 qf 1000
            err =  err_t_domain + err_s_domain +  err_s_label
            err.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                             % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                                err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            sys.stdout.flush()
            # torch.save(my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))
    return smodel18.feature_extractor.state_dict()


