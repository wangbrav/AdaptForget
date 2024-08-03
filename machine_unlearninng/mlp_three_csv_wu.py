
import torch.nn as nn
import torch.nn.functional as F
import torch
import time


# class FeatureExtractor(nn.Module):
#     def __init__(self, dim_in, dim_hidden):
#         super(FeatureExtractor, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(dim_in * dim_in * 3, dim_hidden),
#             nn.BatchNorm1d(dim_hidden),
#             nn.ReLU(inplace=False),
#             nn.Linear(dim_hidden, dim_hidden),
#             nn.BatchNorm1d(dim_hidden),
#             nn.ReLU(inplace=False),
#         )
#
#     def forward(self, x):
#         return self.network(x)
#
# class Classifier(nn.Module):
#     def __init__(self, dim_hidden, dim_out):
#         super(Classifier, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(dim_hidden, dim_out),
#             nn.LogSoftmax(dim=-1)
#         )
#
#     def forward(self, x):
#         return self.network(x)
#
#
# class CombinedModel(nn.Module):
#     def __init__(self, feature_extractor, classifier):
#         super(CombinedModel, self).__init__()
#         self.feature_extractor = feature_extractor
#         self.classifier = classifier
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # 调整x的形状以匹配FeatureExtractor的输入期望
#         features = self.feature_extractor(x)
#         output = self.classifier(features)
#         return output

class FeatureExtractor(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),  # 直接使用8个特征作为输入
            # nn.BatchNorm1d(dim_hidden),  # 预测单样本的时候
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_hidden),
            # nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.network(x)

import torch.nn as nn

# class FeatureExtractor(nn.Module):
#     def __init__(self, dim_in, dim_hidden):
#         super(FeatureExtractor, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(dim_in, dim_hidden),  # 直接使用8个特征作为输入
#             nn.LayerNorm(dim_hidden),  # 使用层归一化,  # 使用实例归一化替换批归一化
#             nn.ReLU(inplace=False),
#             nn.Linear(dim_hidden, dim_hidden),
#             nn.LayerNorm(dim_hidden),  # 使用层归一化,  # 使用实例归一化替换批归一化
#             nn.ReLU(inplace=False),
#         )
#
#     def forward(self, x):
#         return self.network(x)
#

class Classifier(nn.Module):
    def __init__(self, dim_hidden, dim_out,):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.LogSoftmax(dim=-1)         # 使用LogSoftmax进行分类
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
        features = self.feature_extractor(x)  # 直接提取特征
        output = self.classifier(features)
        return output

feature_extractortt = FeatureExtractor(dim_in=56, dim_hidden=64)
classifierstt = Classifier(dim_hidden=64, dim_out=2)

# 创建组合模型实例
modelmlp = CombinedModel(feature_extractortt, classifierstt)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
params_256 = count_parameters(modelmlp)

print(f'参数量 (Net(24, 64, 9)): {params_256}')
params = count_parameters(modelmlp)
print(f'参数量 (CombinedModel): {params}')

# 生成随机输入数据
input_data = torch.randn(100,56)  # 100个样本，每个样本3个通道，32x32的图像

# 将模型移动到GPU上（如果有GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelmlp.to(device)
input_data = input_data.to(device)

# 计算推理时间
modelmlp.eval()
with torch.no_grad():

    start_time = time.time()
    torch.cuda.reset_max_memory_allocated(device)  # 重置最大内存分配

    output = modelmlp(input_data)
    end_time = time.time()

inference_time = end_time - start_time
print(f'推理100个样本的时间: {inference_time:.6f} 秒')
inference_time_microseconds = inference_time * 1_000_000
print(f'推理100个样本的时间: {inference_time_microseconds:.2f} 微秒')
inference_time_milliseconds = inference_time * 1000
print(f'推理100个样本的时间: {inference_time_milliseconds:.2f} 毫秒')

# 计算GPU内存占用
memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # 转换为MB
print(f'推理100个样本占用的GPU内存: {memory_allocated:.2f} MB')