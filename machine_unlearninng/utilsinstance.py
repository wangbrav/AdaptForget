import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

from typing import *

import itertools
from itertools import cycle

import numpy as np
import random

class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self._len = len(inputs)

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds

def naive_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    CE = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # target = target.squeeze().long()  # 是path的
        tensor11 = torch.randn(1, 1)
        # print("target.size():", target.size())
        if target.size() == tensor11.size():
            # 如果是标量，增加一个维度并转换为 long 类型
            target = target.squeeze(0).long()

            # print(clabels.dim())
        else:
            # 如果不是标量，仅转换类型为 long
            target = target.squeeze().long()  # 注意，这里的 squeeze 实际上不会改变形状
        loss = -CE(output, target)
        loss.backward()
        optimizer.step()
        
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    CE = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CE(output, target)
        loss.backward()
        optimizer.step()
        
def adv_attack(args, model, device, train_loader, adversary, unlearn_k, num_adv_images = None):
    model.eval()
    
    attacked_image_arr = []
    target_label_arr = []
    
    if num_adv_images == None:

        if 1024 % unlearn_k == 0:
            num_iters = 1024 // unlearn_k
        else:
            num_iters = 1024 // unlearn_k + 1
    else:
        num_iters = num_adv_images

    for i in range (num_iters):

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            attack_label = torch.randint(0, 4, (data.shape[0],)).cuda()  
            attack_label = attack_label.to(torch.long)
            tensor11 = torch.randn(1, 1)
            if target.size() == tensor11.size():
                target = target.squeeze(0).long()
            else:
                target = target.squeeze().long() 
            attack_label = torch.where(attack_label == target,(attack_label + torch.randint(1, 3, (data.shape[0],)).cuda()) % 3, attack_label)

            adv_example = adversary.perturb(data, attack_label)

            inputs_numpy = adv_example.detach().cpu().numpy()
            labels_numpy = attack_label.cpu().numpy()

            for j in range(inputs_numpy.shape[0]):

                attacked_image_arr.append(inputs_numpy[j])
                target_label_arr.append(labels_numpy[j])
            
            
    return attacked_image_arr, target_label_arr
        


def testins(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_pred = []
    total_target = []
    CE = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # target = target.squeeze().long() # 是path的
            tensor11 = torch.randn(1, 1)
            # print("target.size():", target.size())
            if target.size() == tensor11.size():
            #     如果是标量，增加一个维度并转换为 long 类型
                target = target.squeeze(0).long()

            else:
                target = target.squeeze().long()  
            # target = target.long()
            # test_loss += CE(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True).squeeze(1)  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
            total_pred.extend(pred.cpu().numpy())
            if target.ndim == 0:
                total_target.append(target.item())  
        
                total_target.append(target) 
            else:
                total_target.extend(target.cpu().numpy())           

    f1 = f1_score(total_target, total_pred, average='macro')  # Calculate macro F1 score

    return test_loss, 100. * correct / len(test_loader.dataset), f1


def estimate_parameter_importance(trn_loader, model, device, num_samples, optimizer):
    importance = {n: torch.zeros(p.shape).to(device) for n, p in model.named_parameters()
                  if p.requires_grad}
    
    n_samples_batches = (num_samples // trn_loader.batch_size + 1) if num_samples > 0 \
        else (len(trn_loader.dataset) // trn_loader.batch_size)
    
    model.train()
    for images, targets in itertools.islice(trn_loader, n_samples_batches):
        outputs = model.forward(images.to(device))
        loss = torch.norm(outputs, p=2, dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                importance[n] += p.grad.abs() * len(targets)
    n_samples = n_samples_batches * trn_loader.batch_size
    importance = {n: (p / n_samples) for n, p in importance.items()}
    return importance
