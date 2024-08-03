import torch.nn as nn
from functions import ReverseLayerF


class FeatureExtractor(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in * dim_in * 3, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.network(x)

class Classifier(nn.Module):
    def __init__(self, dim_hidden, dim_out):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, dim_hidden):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_hidden, 100),
            nn.ReLU(inplace=False),
            nn.Linear(100, 2),
            # nn.LogSoftmax(dim=1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)