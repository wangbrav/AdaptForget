
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
            nn.Linear(hidden, hidden),  #后续添加的
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
    return Net(3,32,9)
    # return Net(3,128,9)

def get_teacher_model():
    return Net(3,256,9)
    # return Net(3,128,9)
def get_student_model():
    # return Net(3,128,5)
    return Net(3,128,9)
# def get_model1(hidden):
#     return Net(3, hidden, 9)
# # 定义模型
# model_256 = get_model1(256)
# model_32 = get_model1(32)
# model_128 = get_model1(128)
#
#
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# # 计算每个模型的参数量
# params_256 = count_parameters(model_256)
# params_32 = count_parameters(model_32)
# params_128 = count_parameters(model_128)
#
# print(f'参数量 (Net(3, 256, 9)): {params_256}')
# print(f'参数量 (Net(3, 32, 9)): {params_32}')
# print(f'参数量 (Net(3, 128, 9)): {params_128}')
# torch.manual_seed(int(time.time()))
#
# input_data = torch.randn(100, 3, 28, 28)  # 100个样本，每个样本3个通道，28x28的图像
#
# # 将模型移动到GPU上（如果有GPU）
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_32.to(device)
# input_data = input_data.to(device)
#
# # 计算推理时间和GPU内存占用
# model_32.eval()
# with torch.no_grad():
#     start_time = time.time()
#     torch.cuda.reset_max_memory_allocated(device)  # 重置最大内存分配
#     output = model_32(input_data)
#     end_time = time.time()
#
# inference_time = end_time - start_time
# print(f'推理100个样本的时间: {inference_time:.6f} 秒')
# inference_time_microseconds = inference_time * 1_000_000
# print(f'推理100个样本的时间: {inference_time_microseconds:.2f} 微秒')
# inference_time_milliseconds = inference_time * 1000
# print(f'推理100个样本的时间: {inference_time_milliseconds:.2f} 毫秒')
#
# # 计算GPU内存占用
# memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # 转换为MB
# print(f'推理100个样本占用的GPU内存: {memory_allocated:.2f} MB')