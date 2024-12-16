
# from functions import ReverseLayerF
# import torch.nn as nn
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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
    def __init__(self, hidden, num_classes=9):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),  
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)





class Net(nn.Module):
    def __init__(self, in_channels, hidden, num_classes=9):
        super(Net, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels)
        self.classifier = Classifier(hidden, num_classes)


    def forward(self, x, alpha=None):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)

        return class_output

def get_model():
    # return Net(3,256,9)
    return Net(3,256,9)
    # return Net(3,128,9)

def get_teacher_model():
    return Net(3,256,9)
    # return Net(3,128,9)
def get_student_model():
    # return Net(3,128,5)
    return Net(3,128,9)
