
# from res.resnet18_domain import ResNet18Modified
from mlp_domain import get_student_model
# from mu.net import get_student_model
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
from grad_reverse import grad_reverse
from mlp_three_csv_wu import FeatureExtractor,Classifier,CombinedModel
from mlp_domain_three_csv_wu import DomainDiscriminator


# 分类损失那块有点问题，看看是否需要改，并且域判别这里那个分类还是否需要？
def domainadaptation(f_u,weights, forget_loader,test_loader):
    print("domainadaptation adversarial")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model_root = 'models'
    cuda = True

    # n_epoch = 1
    # n_epoch = 5  # 表格数据这里需要修改
    n_epoch = 3

    # 表格数据这里需要修改
    feature_extractor = FeatureExtractor(dim_in=20, dim_hidden=64)
    # feature_extractor = FeatureExtractor(dim_in=28, dim_hidden=64)
    feature_extractor.to(device)
    classifier = Classifier(dim_hidden=64, dim_out=2)
    # classifier = Classifier(dim_hidden=64, dim_out=10)
    classifier.to(device)
    domain_discriminator = DomainDiscriminator(dim_hidden=64)
    domain_discriminator.to(device)

    # modelmlp = CombinedModel(feature_extractor, classifier).to(device)
    # modelmlp = modelmlp.to(device)
    # modelmlp.load_state_dict(weights)
    #
    # # 加载模型并设置到正确的设备
    # # model.load_state_dict(torch.load(weights))
    # feature_extractor.load_state_dict(f_u) # 单独训练domain对抗训练的时候
    # # feature_extractor.load_state_dict(modelmlp.feature_extractor.state_dict())
    # classifier.load_state_dict(modelmlp.classifier.state_dict())
    modelmlp = CombinedModel(feature_extractor, classifier).to(device)
    modelmlp = modelmlp.to(device)
    modelmlp.classifier.load_state_dict(weights)
    # 加载模型并设置到正确的设备
    # model.load_state_dict(torch.load(weights))
    modelmlp.feature_extractor.load_state_dict(f_u)  # 单独训练domain 对抗训练的时候
    # feature_extractor.load_state_dict(modelmlp.feature_extractor.state_dict())
    modelmlp.classifier.load_state_dict(modelmlp.classifier.state_dict())

    # 为不同部分设置不同的学习率
    learning_rate_feature_classifier = 0.001
    # learning_rate_feature_classifier = 0.00008
    learning_rate_domain_discriminator = 0.0001
    # learning_rate_domain_discriminator = 0.00008

    # 初始化优化器并为各个部分指定不同的学习率
    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()),
                           lr=learning_rate_feature_classifier)
    domain_optimizer = optim.Adam(list(feature_extractor.parameters())+list(domain_discriminator.parameters()), lr=learning_rate_domain_discriminator)
    #  损失函数
    # loss_class = torch.nn.NLLLoss() mnist
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.NLLLoss()
    classification_loss = nn.CrossEntropyLoss()
    domain_loss = torch.nn.CrossEntropyLoss()
    # domain_loss = nn.BCELoss()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

    dataloader_source = test_loader
    # dataloader_source = forget_loader
    dataloader_target = forget_loader
    # dataloader_target = test_loader
    for p in feature_extractor.parameters():
        p.requires_grad = True
    for p in classifier.parameters():
        p.requires_grad = True
    for p in domain_discriminator.parameters():
        p.requires_grad = True

    best_accu_t = 0.0
    for epoch in range(n_epoch):
        # print(len(dataloader_source))
        feature_extractor.train()
        classifier.train()
        domain_discriminator.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        i = 0
        for (source_data, source_labels), (forget_data, _) in zip(dataloader_source, dataloader_target):


            source_data, source_labels = source_data.to(device), source_labels.to(device)
            forget_data = forget_data.to(device)

            # 合并源域和遗忘域数据
            mixed_data = torch.cat([source_data, forget_data], dim=0).to(device)
            # domain_labels = torch.cat([torch.zeros(source_data.size(0), 1), torch.ones(forget_data.size(0), 1)], dim=0)
            domain_labels = torch.cat([torch.zeros(source_data.size(0), 1), torch.ones(forget_data.size(0), 1)], dim=0)
            domain_labels = torch.cat([domain_labels, 1 - domain_labels], dim=1).float().to(device)

            # 特征提取
            features = feature_extractor(mixed_data)

            # 分类损失
            class_preds = classifier(features[:source_data.size(0)])  # 只有源域数据有标签
            c_loss = classification_loss(class_preds, source_labels)

            # 域判别损失前添加梯度反转层
            # features = feature_extractor(mixed_data)
            reversed_features = grad_reverse(features, alpha=0.99)  # 使用梯度反转层
            # reversed_features = grad_reverse(features, alpha=0.5)  # 使用梯度反转层
            domain_preds = domain_discriminator(reversed_features)
            d_loss = domain_loss(domain_preds, domain_labels)
            d_loss = d_loss

            # 更新分类器和特征提取器
            optimizer.zero_grad()
            # c_loss.backward(retain_graph=True)  # 保留计算图用于域判别器更新(表格数据是不是可以注释)
            # c_loss.backward(retain_graph=True)  # 保留计算图用于域判别器更新

            # 更新域判别器
            domain_optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()
            domain_optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                             % (epoch, i + 1, len_dataloader, c_loss.data.cpu().numpy(),
                                d_loss.data.cpu().numpy(), d_loss.data.cpu().item()))
            sys.stdout.flush()
            i = i+1
            # torch.save(my_net, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))
    return feature_extractor.state_dict()

# # 分类损失那块有点问题，看看是否需要改，并且域判别这里那个分类还是否需要？
# def domainadaptation(f_u, weights, forget_loader, test_loader):
#     print("domainadaptation adversarial")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     n_epoch = 30
#
#     # 调整特征提取器和分类器以适配表格数据
#     feature_extractor = FeatureExtractor(dim_in=8, dim_hidden=64)
#     feature_extractor.to(device)
#     classifier = Classifier(dim_hidden=64, dim_out=2)  # 适配2分类任务
#     classifier.to(device)
#     domain_discriminator = DomainDiscriminator(dim_hidden=64)  # 假设域判别器结构适用
#     domain_discriminator.to(device)
#
#     modelmlp = CombinedModel(feature_extractor, classifier).to(device)
#     modelmlp.load_state_dict(weights)
#     feature_extractor.load_state_dict(f_u)
#     classifier.load_state_dict(modelmlp.classifier.state_dict())
#
#     # 为不同部分设置不同的学习率
#     learning_rate_feature_classifier = 0.001
#     learning_rate_domain_discriminator = 0.001
#
#     # 初始化优化器
#     optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()),
#                            lr=learning_rate_feature_classifier)
#     domain_optimizer = optim.Adam(list(domain_discriminator.parameters()), lr=learning_rate_domain_discriminator)
#
#     # 损失函数
#     classification_loss = nn.CrossEntropyLoss().to(device)
#     domain_loss = nn.BCELoss().to(device)
#
#     dataloader_source = test_loader
#     dataloader_target = forget_loader
#
#     for epoch in range(n_epoch):
#         feature_extractor.train()
#         classifier.train()
#         domain_discriminator.train()
#
#         len_dataloader = min(len(dataloader_source), len(dataloader_target))
#         i = 0
#         for (source_data, source_labels), (forget_data, _) in zip(dataloader_source, dataloader_target):
#             source_data, source_labels = source_data.to(device), source_labels.to(device)
#             forget_data = forget_data.to(device)
#
#             # 合并源域和遗忘域数据
#             mixed_data = torch.cat([source_data, forget_data], dim=0).to(device)
#             domain_labels = torch.cat([torch.zeros(source_data.size(0), 1), torch.ones(forget_data.size(0), 1)], dim=0)
#             domain_labels = torch.cat([domain_labels, 1 - domain_labels], dim=1).float().to(device)
#
#             # 特征提取和分类损失
#             features = feature_extractor(mixed_data)
#             class_preds = classifier(features[:source_data.size(0)])  # 只有源域数据有标签
#             c_loss = classification_loss(class_preds, source_labels)
#
#             # 域判别损失
#             reversed_features = grad_reverse(features, alpha=0.5)  # 使用梯度反转层
#             domain_preds = domain_discriminator(reversed_features)
#             d_loss = domain_loss(domain_preds, domain_labels)
#
#             # 更新分类器和特征提取器
#             optimizer.zero_grad()
#             c_loss.backward(retain_graph=True)
#
#             # 更新域判别器
#             domain_optimizer.zero_grad()
#             d_loss.backward()
#             optimizer.step()
#             domain_optimizer.step()
#
#             i += 1
#             sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f' \
#                              % (epoch, i + 1, len_dataloader, c_loss.item(), d_loss.item()))
#             sys.stdout.flush()
#
#     return feature_extractor.state_dict()

