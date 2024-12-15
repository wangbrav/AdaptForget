
# from res.resnet18_domain import ResNet18Modified

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
from mu.net_three import get_student_model,get_teacher_model
# from mlp_three_csv_wu import FeatureExtractor, Classifier, CombinedModel,get_student_model, get_teacher_model



# def domainadaptation(f_u,c_u, forget_loader,test_loader):
def domainadaptation(f_u,c_u, forget_loader,test_loader,lambda_domain):
    print("domainadaptation adversarial")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model_root = 'models'
    cuda = True


    # n_epoch = 4
    # n_epoch = 10
    # n_epoch = 7
    # n_epoch = 3

    n_epoch = 30
    # n_epoch = 1
    # n_epoch = 10
    # n_epoch = 30


    # modelmlp = get_teacher_model()
    modelmlp = get_student_model()
    modelmlp = modelmlp.to(device)
    modelmlp.classifier.load_state_dict(c_u)
    # 加载模型并设置到正确的设备
    # model.load_state_dict(torch.load(weights))
    modelmlp.feature_extractor.load_state_dict(f_u)#单独训练domain 对抗训练的时候
    # feature_extractor.load_state_dict(modelmlp.feature_extractor.state_dict())
    modelmlp.classifier.load_state_dict(modelmlp.classifier.state_dict())

    # 为不同部分设置不同的学习率
    # learning_rate_feature_classifier = 0.00008  #qf100 kd0.5
    # learning_rate_feature_classifier = 0.00008  #qf100 kd0.5  调整看看效果
    # learning_rate_feature_classifier = 0.00008  #qf100 kd0.5  调整看看效果
    # learning_rate_feature_classifier = 0.001  #qf100 kd0.5  调整看看效果
    learning_rate_feature_classifier = 0.00000008   #qf1的时候选的
    # learning_rate_domain_discriminator = 0.00000008
    # learning_rate_domain_discriminator = 0.000008#qf100 kd0.5  调整看看效果
    # learning_rate_domain_discriminator = 0.001#qf100 kd0.5  调整看看效果
    learning_rate_domain_discriminator = 0.00008#qf100 kd0.5
    print("learning_rate_feature_classifier",learning_rate_feature_classifier)
    print("learning_rate_domain_discriminator",learning_rate_domain_discriminator)
    print( "n_epoch",n_epoch)

    # 初始化优化器并为各个部分指定不同的学习率
    optimizer = optim.Adam(list(modelmlp.feature_extractor.parameters()) + list(modelmlp.classifier.parameters()),
                           lr=learning_rate_feature_classifier)
    domain_optimizer = optim.Adam(list(modelmlp.feature_extractor.parameters())+list(modelmlp.domain_classifier.parameters()), lr=learning_rate_domain_discriminator)
    #  损失函数
    # loss_class = torch.nn.NLLLoss() mnist
    # loss_class = torch.nn.CrossEntropyLoss()
    # loss_domain = torch.nn.NLLLoss()
    classification_loss = nn.CrossEntropyLoss()
    # domain_loss = nn.BCELoss()
    domain_loss = torch.nn.CrossEntropyLoss()

    # loss_class = loss_class.cuda()
    # loss_domain = loss_domain.cuda()

    dataloader_source = test_loader
    # dataloader_source = forget_loader
    dataloader_target = forget_loader
    # dataloader_target = test_loader
    for p in modelmlp.feature_extractor.parameters():
        p.requires_grad = True
    for p in modelmlp.classifier.parameters():
        p.requires_grad = True
    for p in modelmlp.domain_classifier.parameters():
        p.requires_grad = True

    best_accu_t = 0.0
    for epoch in range(n_epoch):
        # print(1111111111111111111111111)
        # print(len(dataloader_source))
        modelmlp.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        i = 0
        for (source_data, source_labels), (forget_data, _) in zip(dataloader_source, dataloader_target):

            source_data = source_data.to(device)  # 将数据移到GPU
            forget_data = forget_data.to(device)  # 将数据移到GPU
            # 假设 source_data 和 forget_data 已经是 [批大小, 784] 的形状
            # 这里我们将其变回 [批大小, 通道数, 高度, 宽度] 的形状
            source_data = source_data.view(-1, 3, 28, 28)
            forget_data = forget_data.view(-1, 3, 28, 28)

            # source_data = source_data.view(source_data.size(0), -1)  # 假设是图像数据，展平
            # forget_data = forget_data.view(forget_data.size(0), -1)
            source_labels= source_labels.to(device)
            if source_labels.dim() > 1:
                source_labels = source_labels.squeeze(1).long()
            else:
                source_labels = source_labels.long()
            # source_labels = source_labels.squeeze(1).long()  #（这是因为要做ood数据而主食的）
            # 合并源域和遗忘域数据
            mixed_data = torch.cat([source_data, forget_data], dim=0).to(device)
            # domain_labels = torch.cat([torch.zeros(source_data.size(0), 1), torch.ones(forget_data.size(0), 1)], dim=0)
            domain_labels = torch.cat([torch.zeros(source_data.size(0), 1), torch.ones(forget_data.size(0), 1)], dim=0)
            domain_labels = torch.cat([domain_labels, 1 - domain_labels], dim=1).float().to(device)
            # print(domain_labels)

            # 特征提取
            features = modelmlp.feature_extractor(mixed_data)
            print(f"Labels shape: {source_labels.shape}")

            print(f"source_labels range: {source_labels.min().item()} to {source_labels.max().item()}")
            # valid_indices = (source_labels >= 0) & (source_labels <= 8)
            # source_labels = source_labels[valid_indices]
            # class_preds = class_preds[valid_indices]
            # 分类损失
            # print(source_labels)
            class_preds = modelmlp.classifier(features[:source_data.size(0)])  # 只有源域数据有标签
            # print(class_preds)
            print(f"Predictions shape: {class_preds.shape}")
            print(f"Labels shape: {source_labels.shape}")
            c_loss = classification_loss(class_preds, source_labels)

            # 假设 `domain_labels` 是你提供的张量
            domain_labels_simplified = domain_labels[:, 1]
            domain_labels_simplified = domain_labels_simplified.long()

            # 域判别损失前添加梯度反转层
            # features = feature_extractor(mixed_data)
            reversed_features = grad_reverse(features, alpha=0.5)  # 使用梯度反转层
            domain_preds = modelmlp.domain_classifier(reversed_features)
            d_loss = domain_loss(domain_preds, domain_labels_simplified)
            d_loss = lambda_domain * d_loss


            # 更新分类器和特征提取器
            optimizer.zero_grad()
            # print(c_loss)
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
    return modelmlp.feature_extractor.state_dict()


