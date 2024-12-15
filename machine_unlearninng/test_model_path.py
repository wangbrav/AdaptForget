import torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet18 import ResNet18
# from resnet34 import ResNet34
from mu.net_train import get_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from KDloss import SoftTarget
from tqdm import tqdm

def test_model(test_loader,retain_loader, forget_loader, device, model34, smodel18, tmodel18,u,f_u):
    smodel18.load_state_dict(u)
    smodel18.eval()  
    tmodel18.eval()  
    model34.eval()   
    correctt = 0
    totalt = 0
    correctu = 0
    totalu = 0
    correcttt = 0
    totaltt = 0
    all_labels = []
    all_preds = []
    with torch.no_grad(): 
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs18s = smodel18(inputs)
            # outputs18t = tmodel18(inputs)
            _, predicted = torch.max(outputs18s.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            totaltt += labels.size(0)
            labels = labels.squeeze()  
            correcttt += (predicted == labels).sum().item()  
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    conf_matrix = confusion_matrix(all_labels_np, all_preds_np)
    TP = conf_matrix[1, 1]  
    TN = conf_matrix[0, 0]  
    FP = conf_matrix[0, 1]  
    FN = conf_matrix[1, 0]  
    f1 = 2 * TP / (2 * TP + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')
    print(f'Accuracy: {accuracy}')
    print(f'Accuracy on test data: {100 * correcttt / totaltt}%')
    return 100 * correcttt / totaltt,correcttt/totaltt,f1_score
