
from functions import ReverseLayerF
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, hidden, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),  # 后续添加的
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DomainClassifier(nn.Module):
    def __init__(self,hidden):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Net(nn.Module):
    def __init__(self, in_channels, hidden, num_classes):
        super(Net, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels)
        self.classifier = Classifier(hidden, num_classes)
        self.domain_classifier = DomainClassifier(hidden)

    def forward(self, x, alpha=None):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)

        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(features, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output

        return class_output


def get_model():
    return Net(3,128,4)

def get_teacher_model():
    return Net(3,128,4)

def get_student_model():
    return Net(3,128,5)