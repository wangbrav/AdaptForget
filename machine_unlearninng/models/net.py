
from functions import ReverseLayerF
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_channels, hidden, num_classes):
        super(Net, self).__init__()

        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),  # 第1层卷积
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3),  # 第2层卷积
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3),  # 第3层卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),  # 第4层卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 第5层卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # 域分类部分
        self.domain_classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 调整形状以匹配全连接层的期望输入

        # 如果提供了alpha（即正在进行域适应），则计算域输出
        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(x, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            class_output = self.classifier(x)
            return class_output, domain_output

        class_output = self.classifier(x)  # 通过分类部分
        return class_output

def get_model():
    return Net(3,128,9)

def get_teacher_model():
    return Net(3,128,9)

def get_student_model():
    return Net(3,32,9)