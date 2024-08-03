


import torch.nn as nn
from functions import ReverseLayerF


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.dim_hidden = dim_hidden

        # 特征提取部分
        self.features = nn.Sequential(
            nn.Linear(dim_in * dim_in*3, dim_hidden),
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

        # 域分类阶段
        self.domain_classifier = nn.Sequential(
            nn.Linear(dim_hidden, 100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha=None):
        x = x.view(-1, 28 * 28*3)  # 调整输入数据的形状以匹配网络的期望输入
        features = self.features(x)
        class_output = self.classifier(features)  # 通过分类部分

        # 如果提供了alpha（即正在进行域适应），则计算域输出
        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(features, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output

        return class_output

# 获取模型的函数
def get_model():
    return MLP(28, 256, 10)

def get_teacher_model():
    return MLP(28, 256, 10)

def get_student_model():
    return MLP(28, 64, 10)