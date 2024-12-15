
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
    cuda = Tru

    n_epoch = 30



    modelmlp = get_student_model()
    modelmlp = modelmlp.to(device)
    modelmlp.classifier.load_state_dict(c_u)
    # model.load_state_dict(torch.load(weights))
    modelmlp.feature_extractor.load_state_dict(f_u)
    # feature_extractor.load_state_dict(modelmlp.feature_extractor.state_dict())
    modelmlp.classifier.load_state_dict(modelmlp.classifier.state_dict())

    learning_rate_feature_classifier = 0.00000008

    learning_rate_domain_discriminator = 0.00008
    print("learning_rate_feature_classifier",learning_rate_feature_classifier)
    print("learning_rate_domain_discriminator",learning_rate_domain_discriminator)
    print( "n_epoch",n_epoch)

    optimizer = optim.Adam(list(modelmlp.feature_extractor.parameters()) + list(modelmlp.classifier.parameters()),
                           lr=learning_rate_feature_classifier)
    domain_optimizer = optim.Adam(list(modelmlp.feature_extractor.parameters())+list(modelmlp.domain_classifier.parameters()), lr=learning_rate_domain_discriminator)

    classification_loss = nn.CrossEntropyLoss()
    domain_loss = torch.nn.CrossEntropyLoss()

    dataloader_source = test_loader
    dataloader_target = forget_loader
    for p in modelmlp.feature_extractor.parameters():
        p.requires_grad = True
    for p in modelmlp.classifier.parameters():
        p.requires_grad = True
    for p in modelmlp.domain_classifier.parameters():
        p.requires_grad = True

    best_accu_t = 0.0
    for epoch in range(n_epoch):
        modelmlp.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        i = 0
        for (source_data, source_labels), (forget_data, _) in zip(dataloader_source, dataloader_target):

            source_data = source_data.to(device)  
            forget_data = forget_data.to(device)  
            source_data = source_data.view(-1, 3, 28, 28)
            forget_data = forget_data.view(-1, 3, 28, 28)
            source_labels= source_labels.to(device)
            if source_labels.dim() > 1:
                source_labels = source_labels.squeeze(1).long()
            else:
                source_labels = source_labels.long()

            mixed_data = torch.cat([source_data, forget_data], dim=0).to(device)
            domain_labels = torch.cat([torch.zeros(source_data.size(0), 1), torch.ones(forget_data.size(0), 1)], dim=0)
            domain_labels = torch.cat([domain_labels, 1 - domain_labels], dim=1).float().to(device)

            features = modelmlp.feature_extractor(mixed_data)
            print(f"Labels shape: {source_labels.shape}")

            print(f"source_labels range: {source_labels.min().item()} to {source_labels.max().item()}")

            class_preds = modelmlp.classifier(features[:source_data.size(0)]) 
            print(f"Predictions shape: {class_preds.shape}")
            print(f"Labels shape: {source_labels.shape}")
            c_loss = classification_loss(class_preds, source_labels)

            domain_labels_simplified = domain_labels[:, 1]
            domain_labels_simplified = domain_labels_simplified.long()

            reversed_features = grad_reverse(features, alpha=0.5) 
            domain_preds = modelmlp.domain_classifier(reversed_features)
            d_loss = domain_loss(domain_preds, domain_labels_simplified)
            d_loss = lambda_domain * d_loss


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
    return modelmlp.feature_extractor.state_dict()


