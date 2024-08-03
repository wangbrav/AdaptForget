import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()

        # 特征提取部分
        self.features = nn.Sequential(
            nn.Linear(dim_in * dim_in*3 , dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=True),
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28*3)  # 调整输入数据的形状以匹配网络的期望输入
        x = self.features(x)  # 通过特征提取部分
        x = self.classifier(x)  # 通过分类部分
        return x


def get_model():
    return MLP(28, 256, 10)

def get_teacher_model():
    return MLP(28, 256, 10)

def get_student_model():
    return MLP(28, 64, 10)
