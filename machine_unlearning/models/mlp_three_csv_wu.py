
import torch.nn as nn
import torch.nn.functional as F
import torch
import time




class FeatureExtractor(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),  
            # nn.BatchNorm1d(dim_hidden), 
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_hidden),
            # nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.network(x)

import torch.nn as nn



class Classifier(nn.Module):
    def __init__(self, dim_hidden, dim_out,):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.LogSoftmax(dim=-1)         
        )

    def forward(self, x):
        return self.network(x)


class CombinedModel(nn.Module):
    def __init__(self, feature_extractor, classifier,num_classes=2):
        super(CombinedModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.classifier.num_classes = num_classes

    def forward(self, x):
        features = self.feature_extractor(x)  
        output = self.classifier(features)
        return output


def get_teacher_model():
    feature_extractor_teacher = FeatureExtractor(dim_in=20, dim_hidden=256)
    classifier_teacher = Classifier(dim_hidden=256, dim_out=2)
    tmodelmlp = CombinedModel(feature_extractor_teacher, classifier_teacher)
    return tmodelmlp


def get_student_model():
    feature_extractor_student_v1 = FeatureExtractor(dim_in=20, dim_hidden=64)
    classifier_student_v1 = Classifier(dim_hidden=64, dim_out=2)
    smodelmlp = CombinedModel(feature_extractor_student_v1, classifier_student_v1)
    return smodelmlp