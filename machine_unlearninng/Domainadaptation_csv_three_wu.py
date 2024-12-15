
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


def domainadaptation(f_u,weights, forget_loader,test_loader):
    print("domainadaptation adversarial")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model_root = 'models'
    cuda = True

    n_epoch = 3

    # 表格数据这里需要修改
    feature_extractor = FeatureExtractor(dim_in=20, dim_hidden=64)
    feature_extractor.to(device)
    classifier = Classifier(dim_hidden=64, dim_out=2)
    classifier.to(device)
    domain_discriminator = DomainDiscriminator(dim_hidden=64)
    domain_discriminator.to(device)

    modelmlp = CombinedModel(feature_extractor, classifier).to(device)
    modelmlp = modelmlp.to(device)
    modelmlp.classifier.load_state_dict(weights)
    modelmlp.feature_extractor.load_state_dict(f_u) 
    modelmlp.classifier.load_state_dict(modelmlp.classifier.state_dict())

    learning_rate_feature_classifier = 0.001
    learning_rate_domain_discriminator = 0.0001

    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()),
                           lr=learning_rate_feature_classifier)
    domain_optimizer = optim.Adam(list(feature_extractor.parameters())+list(domain_discriminator.parameters()), lr=learning_rate_domain_discriminator)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.NLLLoss()
    classification_loss = nn.CrossEntropyLoss()
    domain_loss = torch.nn.CrossEntropyLoss()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

    dataloader_source = test_loader
    dataloader_target = forget_loader
    for p in feature_extractor.parameters():
        p.requires_grad = True
    for p in classifier.parameters():
        p.requires_grad = True
    for p in domain_discriminator.parameters():
        p.requires_grad = True

    best_accu_t = 0.0
    for epoch in range(n_epoch):
        feature_extractor.train()
        classifier.train()
        domain_discriminator.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        i = 0
        for (source_data, source_labels), (forget_data, _) in zip(dataloader_source, dataloader_target):


            source_data, source_labels = source_data.to(device), source_labels.to(device)
            forget_data = forget_data.to(device)

            mixed_data = torch.cat([source_data, forget_data], dim=0).to(device)
            domain_labels = torch.cat([torch.zeros(source_data.size(0), 1), torch.ones(forget_data.size(0), 1)], dim=0)
            domain_labels = torch.cat([domain_labels, 1 - domain_labels], dim=1).float().to(device)

            features = feature_extractor(mixed_data)

            class_preds = classifier(features[:source_data.size(0)])  
            c_loss = classification_loss(class_preds, source_labels)

            reversed_features = grad_reverse(features, alpha=0.99)  
            domain_preds = domain_discriminator(reversed_features)
            d_loss = domain_loss(domain_preds, domain_labels)
            d_loss = d_loss

            optimizer.zero_grad()

            domain_optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()
            domain_optimizer.step()
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                             % (epoch, i + 1, len_dataloader, c_loss.data.cpu().numpy(),
                                d_loss.data.cpu().numpy(), d_loss.data.cpu().item()))
            sys.stdout.flush()
            i = i+1
    return feature_extractor.state_dict()
