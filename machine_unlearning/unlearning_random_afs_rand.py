import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys
sys.path.append('/root/autodl-tmp/wangbin/yiwang/')
from KDloss import SoftTarget
from tqdm import tqdm
import torch.nn.functional as F
import logging

from utils.Metric import AverageMeter, accuracy, Performance
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for X, Y in tqdm(test_loader):
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)
                # _, predicted = torch.max(outputs.data, 1)
                pred = outputs.data.max(1)[1]
                total += Y.size(0)
                # Y = Y.squeeze(1).long()  
                correct += pred.eq(Y.view(-1)).sum().item()
    print(f"Correct: {correct}, Total: {total}")
    accuracy = 100 * correct / total
    return accuracy

def train_student_model_random(lambda_risk,lambda_kd,forget_loader, retain_loader,test1loader, tmodel18, model34, smodel18, u,f_u):
# def train_student_model_random(forget_loader, retain_loader,test1loader, tmodel18, model34, smodel18, u,f_u):
# def train_student_model_random(forget_loader, retain_loader, tmodel18, model34, smodel18, u,f_u,lambda_risk,lambda_kd):
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    risk_losses = AverageMeter()
    total_losses = AverageMeter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    smodel18.load_state_dict(u)
    smodel18.feature_extractor.load_state_dict(f_u)
    T = 6
    lambda_kd = lambda_kd
    lambda_risk =lambda_risk 
    best_accuracy = 0

    num_epochs = 1 #
    print(f'T: {T}')
    print(f'num_epochs: {num_epochs}')
    # criterionCls = nn.CrossEntropyLoss().to(device)
    criterionKD = SoftTarget(T)
    criterionKD2 = SoftTarget(9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterionCls = nn.CrossEntropyLoss().to(device)
    feature_loss_fn = nn.MSELoss()

    # criterionCls = nn.CrossEntropyLoss().to(device)
    optimizer18s = optim.Adam(smodel18.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # model34.train()
        model34.eval()
        tmodel18.eval()
        smodel18.train()
        correct = 0
        train_loss = 0
        
        with tqdm(total=len(retain_loader)) as t:
            for X, Y in tqdm(retain_loader):
                X = X.to(device)
                Y = Y.to(device)
                optimizer18s.zero_grad()
                snet_pred = smodel18(X)
                tnet_pred = model34(X)


                Y = Y.squeeze().long() 
                # Y = Y.squeeze(1).long() 
 
                cls_loss = criterionCls(snet_pred, Y)
                kd_loss = criterionKD(snet_pred, tnet_pred.detach())
                loss = cls_loss + kd_loss * lambda_kd

                for _X, _Y in forget_loader:
                    _X = _X.to(device)
                    _Y = _Y.to(device)
                    num_classes = tmodel18.classifier.num_classes 

                    random_probs = torch.rand(_X.size(0), num_classes).to(device)

                    # teacher_probs = F.softmax(random_probs, dim=-1)  
                    student_probs = smodel18(_X)
                    # student_probs = F.softmax(smodel18(_X), dim=-1)
                    teacher_probs = tmodel18(_X)
                    # teacher_probs = F.softmax(tmodel18(_X), dim=-1)
                    risk_loss = criterionKD2(student_probs, teacher_probs.detach())
                    loss = loss + risk_loss * torch.tensor(lambda_risk).to(device)
                loss.backward()
                train_loss += loss.item()
                optimizer18s.step()
                pred = snet_pred.data.max(1)[1]
                correct += pred.eq(Y.view(-1)).sum().item()
        train_loss /= len(retain_loader.dataset)
        train_accuracy = 100 * correct / len(retain_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        test_accuracy = test(smodel18, test1loader, device)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"Saving best model with accuracy {best_accuracy}")
        
    return smodel18.feature_extractor.state_dict(),smodel18.state_dict(),smodel18.classifier.state_dict()



