import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
import os
import medmnist
from mu.net_train_three import get_student_model,get_teacher_model

from medmnist import INFO, Evaluator
from tqdm import tqdm
def aggregate_predictions(models, X_test, weights):
    with torch.no_grad():
        weighted_logits = sum(1 * model.feature_extractor(X_test) for model, weight in zip(models, weights))
    return weighted_logits

def set_seed(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(62)
random_seed = 62
def set_seed(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(62)
from torch.utils.data import Dataset, DataLoader, Subset
def analyze_sample_similarity(model,u, device, train_dataset, config):
    set_seed(32)
    sample_index = config['QF_100']['QUERY'][0]
    print(sample_index)

    model = model.to(device)
    model.load_state_dict(u)
    model.eval()

    sample, label = train_dataset[sample_index]

    sample = sample.unsqueeze(0).to(device)

    feature = model.feature_extractor(sample)
    feature = feature.view(feature.size(0), -1)

 
    base_indices = config['BASE2']['BASE']
    base_samples = Subset(train_dataset, base_indices)
    base_loader = DataLoader(base_samples, batch_size=len(base_samples), shuffle=False)
    base_data, base_labels = next(iter(base_loader))

    label_value = label.item()  
    same_class_mask = base_labels == label_value
    same_class_indices = np.where(same_class_mask.numpy())[0]
    if len(same_class_indices) == 0:
        print("No same class samples found in BASE")
        return None
    else:

        if len(same_class_indices) > 9000:
            print("Same class samples found in BASE", len(same_class_indices))
            same_class_indices = np.random.choice(same_class_indices,9000, replace=False)
            print("Same class samples found in BASE", same_class_indices)

        same_class_samples = DataLoader(Subset(train_dataset, same_class_indices), batch_size=len(same_class_indices), shuffle=False)
        same_class_data, _ = next(iter(same_class_samples))

        same_class_data = same_class_data.to(device)


        same_class_features = model.feature_extractor(same_class_data)
        same_class_features = same_class_features.view(same_class_features.size(0), -1)
        mean_same_class_features = torch.mean(same_class_features, dim=0)
        mean_same_class_features = mean_same_class_features.unsqueeze(0) 

        euclidean_distance = torch.norm(feature - mean_same_class_features, dim=1).item()

        manhattan_distance = torch.sum(torch.abs(feature - mean_same_class_features), dim=1).item()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(feature, mean_same_class_features).item()

        return euclidean_distance, manhattan_distance, cosine_similarity
def analyze_sample_similarity_sisa (models,weights, device, train_dataset, config):
    set_seed(62)
    sample_index = config['QF_100']['QUERY']
    print(sample_index)

    models = [model.to(device) for model in models]

    models = [model.eval() for model in models]
    forget_samples = Subset(train_dataset, sample_index)
    forget_loader = DataLoader(forget_samples, batch_size=len(forget_samples), shuffle=False)
    forget_data,forget_labels = next(iter(forget_loader))
    forget_class_data = forget_data.to(device)


    forget__class_features = aggregate_predictions(models, forget_class_data, weights)

    forget_class_features = forget__class_features.view(forget__class_features.size(0), -1)
    mean_forget_class_features = torch.mean(forget_class_features, dim=0)
    mean_forget_class_features = mean_forget_class_features.unsqueeze(0)  

    base_indices = config['BASE2']['BASE']
    base_samples = Subset(train_dataset, base_indices)
    base_loader = DataLoader(base_samples, batch_size=len(base_samples), shuffle=False)
    base_data, base_labels = next(iter(base_loader))

    if len(base_indices) == 0:
        print("No same class samples found in BASE")
        return None
    else:

        if len(base_indices) >=399:

            same_class_indices = np.random.choice(base_indices,399, replace=False)

        same_class_samples = DataLoader(Subset(train_dataset, same_class_indices), batch_size=len(same_class_indices), shuffle=False)
        same_class_data, _ = next(iter(same_class_samples))
        same_class_data = same_class_data.to(device)
        same_class_features = aggregate_predictions(models, same_class_data, weights)
        same_class_features = same_class_features.view(same_class_features.size(0), -1)
        mean_same_class_features = torch.mean(same_class_features, dim=0)
        mean_same_class_features = mean_same_class_features.unsqueeze(0)  

        euclidean_distance = torch.norm(mean_forget_class_features - mean_same_class_features, dim=1).item()
        manhattan_distance = torch.sum(torch.abs(mean_forget_class_features - mean_same_class_features), dim=1).item()
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(mean_forget_class_features, mean_same_class_features).item()
        return euclidean_distance, manhattan_distance, cosine_similarity
