# pyright: reportMissingImports=true, reportUntypedBaseClass=false, reportGeneralTypeIssues=false
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

class NTK_Linear(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(NTK_Linear, self).__init__() 
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim,bias=False)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out

class LinearNeuralTangentKernel(nn.Linear): 
    
    def __init__(self, in_features, out_features, bias=True, beta=np.sqrt(0.1), w_sig = np.sqrt(2.0)):
        self.beta = beta
        super(LinearNeuralTangentKernel, self).__init__(in_features, out_features)
        self.reset_parameters()
        self.w_sig = w_sig
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input):
        return F.linear(input, self.w_sig * self.weight/np.sqrt(self.in_features), self.beta * self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta)

class NTK_MLP(nn.Module):
    def __init__(self, num_classes=10, filters_percentage=1.0, beta=np.sqrt(0.1)):
        super(NTK_MLP, self).__init__()
        self.n_wid = int(32*filters_percentage)
        self.fc1 = LinearNeuralTangentKernel(1024, self.n_wid, beta=beta)
        self.fc2 = LinearNeuralTangentKernel(self.n_wid, num_classes, beta=beta)
#         self.fc3 = LinearNeuralTangentKernel(self.n_wid, self.n_wid, beta=beta)
#         self.fc4 = LinearNeuralTangentKernel(self.n_wid, self.n_wid, beta=beta)
#         self.fc5 = LinearNeuralTangentKernel(self.n_wid, num_classes, beta=beta)

    def forward(self, x):
        x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
        x = self.fc2(x)
        return x

class Affine(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return x * self.weight + self.bias
    
class StandardLinearLayer(nn.Linear): 
    
    def __init__(self, in_features, out_features, bias=True, beta=np.sqrt(0.1), w_sig = np.sqrt(2.0)):
        self.beta = beta
        self.w_sig = w_sig
        super(StandardLinearLayer, self).__init__(in_features, out_features)
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/np.sqrt(self.in_features))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=self.beta)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, beta={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.beta)
    
class MLP(nn.Module):

    def __init__(self, num_layer=1, num_classes=10, filters_percentage=1., hidden_size=32, input_size=1024):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.layers = self._make_layers()

    def _make_layers(self):
        layer = []
        layer += [
            StandardLinearLayer(self.input_size,self.hidden_size),#nn.Linear(self.input_size, self.hidden_size),
            # Affine(self.hidden_size),
            nn.ReLU()]
        for i in range(self.num_layer - 2):
            layer += [StandardLinearLayer(self.hidden_size,self.hidden_size)]#[nn.Linear(self.hidden_size, self.hidden_size)]
            # layer += [Affine(self.hidden_size)]
            layer += [nn.ReLU()]
        layer += [StandardLinearLayer(self.hidden_size,self.num_classes)]#[nn.Linear(self.hidden_size, self.num_classes)]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = x.reshape(x.size(0), self.input_size)
        return self.layers(x)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)
    
class ConvStandard(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 np.sqrt(1.0)):
        super(ConvStandard, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/(self.in_channels*np.prod(self.kernel_size)))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding)
            
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
    
class ConvNeuralTangentKernel(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 np.sqrt(1.0)):
        super(ConvNeuralTangentKernel, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input, self.w_sig*self.weight/np.sqrt(self.in_channels*np.prod(self.kernel_size)),\
                        self.bias,self.stride,self.padding)
     
class ntk_Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
#         if not transpose:
        model += [ConvNeuralTangentKernel(in_channels,out_channels,kernel_size,stride=stride,padding=padding,
                                         output_padding=output_padding)] 
#         else:
#             model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                                          output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(ntk_Conv, self).__init__(*model)

class ntk_AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(ntk_AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            ntk_Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            ntk_Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            ntk_Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            ntk_Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ntk_Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ntk_Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            ntk_Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            ntk_Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
#             LinearNeuralTangentKernel(n_filter2, num_classes, beta=np.sqrt(0.1)),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _ResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    
class ResNet18(nn.Module):
    def __init__(self, filters_percentage=1.0, n_channels = 3, num_classes=10, block=_ResBlock, num_blocks=[2,2,2,2], n_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, int(64*filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*filters_percentage), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*filters_percentage), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*filters_percentage)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ResNet18_small(nn.Module):
    def __init__(self, filters_percentage=0.5, n_channels = 3, num_classes=10, block=_ResBlock, num_blocks=[2,2,2], n_classes=10):
        super(ResNet18_small, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, int(64*filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*filters_percentage), num_blocks[2], stride=2)
        self.linear = nn.Linear(int(256*filters_percentage)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
#def conv3x3(in_planes, out_planes, stride=1):
#    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
class Wide_ResNet(nn.Module):
    def __init__(self, depth=4, filters_percentage=1, widen_factor=5, dropout_rate=0.0, num_classes=10):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class ConvImprovedStandard(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=(0,0), output_padding=0, w_sig =\
                 np.sqrt(2.0),s=10000):
        super(ConvImprovedStandard, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.s = s
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1/np.sqrt(self.in_channels*np.prod(self.kernel_size)))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input, self.weight/np.sqrt(self.s),self.bias,self.stride,self.padding)

class wide_basicIS(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basicIS, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = ConvImprovedStandard(in_planes, planes, kernel_size=3, padding=(1,1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = ConvImprovedStandard(planes, planes, kernel_size=3, stride=stride, padding=(1,1))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                ConvImprovedStandard(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
class Wide_ResNetIS(nn.Module):
    def __init__(self, depth=4, filters_percentage=1.0, widen_factor=1, dropout_rate=0.0, num_classes=10):
        super(Wide_ResNetIS, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = ConvImprovedStandard(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basicIS, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basicIS, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basicIS, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    
class ConvNTK(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=(0,0), output_padding=0, w_sig =\
                 np.sqrt(2.0)):
        super(ConvNTK, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input, self.w_sig*self.weight/np.sqrt(self.in_channels*np.prod(self.kernel_size))\
                        ,self.bias,self.stride,self.padding)

class wide_basicNTK(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basicNTK, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = ConvNTK(in_planes, planes, kernel_size=3, padding=(1,1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = ConvNTK(planes, planes, kernel_size=3, stride=stride, padding=(1,1))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                ConvNTK(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
class Wide_ResNetNTK(nn.Module):
    def __init__(self, depth=4, filters_percentage=1.0, widen_factor=1, dropout_rate=0.0, num_classes=10):
        super(Wide_ResNetNTK, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = ConvNTK(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basicNTK, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basicNTK, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basicNTK, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def _reinit(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

'''
Added by Shash: CIFAR Specific ResNets from https://github.com/akamaster/pytorch_resnet_cifar10
Adapted to the style of calls of Golatkar though.
'''
def _weights_init(m):
    init_mult = 0.01
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        m.weight.requires_grad = False
        m.weight*=init_mult
        m.weight.requires_grad = True

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FeatureExtractor_mnist(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(FeatureExtractor_mnist, self).__init__()
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

class Classifier_mnist(nn.Module):
    def __init__(self, dim_hidden, dim_out):
        super(Classifier_mnist, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)
class CombinedModel_mnist(nn.Module):
    def __init__(self, feature_extractor, classifier,dim_out):
        super(CombinedModel_mnist, self).__init__()
        self.dim_out = dim_out
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 调整x的形状以匹配FeatureExtractor的输入期望
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class FeatureExtractor_Pima(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(FeatureExtractor_Pima, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),  # 直接使用8个特征作为输入
            nn.BatchNorm1d(dim_hidden),
            # nn.GroupNorm(1, dim_hidden),  # 使用组归一化，1表示每个组包含所有的通道
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            # nn.GroupNorm(1, dim_hidden),  # 同上
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.network(x)

class Classifier_Pima(nn.Module):
    def __init__(self, dim_hidden, dim_out):
        super(Classifier_Pima, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_hidden, dim_out),
            nn.LogSoftmax(dim=-1)         # 使用LogSoftmax进行分类
        )

    def forward(self, x):
        return self.network(x)


class CombinedModel_Pima(nn.Module):
    def __init__(self, feature_extractor, classifier,dim_out):
        super(CombinedModel_Pima, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x)  # 直接提取特征
        output = self.classifier(features)
        return output


class ResNet(nn.Module):
    def __init__(self, filters_percentage=1.0, n_channels=3, num_classes=10, block=BasicBlock, num_blocks=[3, 3, 3], n_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, int(16*filters_percentage), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(32*filters_percentage), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(64*filters_percentage), num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class FeatureExtractor_pathmnist(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor_pathmnist, self).__init__()
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


class Classifier_pathmnist(nn.Module):
    def __init__(self, hidden, num_classes):
        super(Classifier_pathmnist, self).__init__()
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


class Net_pathmnist(nn.Module):
    def __init__(self, in_channels, hidden, num_classes):
        super(Net_pathmnist, self).__init__()
        self.feature_extractor = FeatureExtractor_pathmnist(in_channels)
        self.classifier = Classifier_pathmnist(hidden, num_classes)


    def forward(self, x, alpha=None):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)

        return class_output
def ResNet20(**kwargs):
    return ResNet(num_blocks=[3, 3, 3], **kwargs)


def ResNet32(**kwargs):
    return ResNet(num_blocks=[5, 5, 5], **kwargs)


def ResNet44(**kwargs):
    return ResNet(num_blocks=[7, 7, 7], **kwargs)


def ResNet56(**kwargs):
    return ResNet(num_blocks=[9, 9, 9], **kwargs)


def ResNet110(**kwargs):
    return ResNet(num_blocks=[18, 18, 18], **kwargs)


def ResNet1202(**kwargs):
    return ResNet(num_blocks=[200, 200, 200], **kwargs)

_MODELS = {}

def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn

@_add_model
def resnet20(**kwargs):
    return ResNet20(**kwargs)

@_add_model
def resnet32(**kwargs):
    return ResNet32(**kwargs)

@_add_model
def resnet44(**kwargs):
    return ResNet44(**kwargs)

@_add_model
def resnet56(**kwargs):
    return ResNet56(**kwargs)

@_add_model
def resnet110(**kwargs):
    return ResNet110(**kwargs)

@_add_model
def resnet1202(**kwargs):
    return ResNet1202(**kwargs)

@_add_model
def mlp(**kwargs):
    return MLP(**kwargs)

@_add_model
def ntk_linear(**kwargs):
    return NTK_Linear(**kwargs)

@_add_model
def ntk_mlp(**kwargs):
    return NTK_MLP(**kwargs)

@_add_model
def allcnn(**kwargs):
    return AllCNN(**kwargs)

@_add_model
def ntk_allcnn(**kwargs):
    return ntk_AllCNN(**kwargs)

@_add_model
def allcnn_no_bn(**kwargs):
    return AllCNN(batch_norm=False, **kwargs)

@_add_model
def resnet(**kwargs):
    return ResNet18(**kwargs)

@_add_model
def resnet_small(**kwargs):
    return ResNet18_small(**kwargs)

@_add_model
def wide_resnet(**kwargs):
    return Wide_ResNet(**kwargs)

@_add_model
def is_wide_resnet(**kwargs):
    return Wide_ResNetIS(**kwargs)

@_add_model
def ntk_wide_resnet(**kwargs):
    return Wide_ResNetNTK(**kwargs)

def get_model(name, **kwargs):
    return _MODELS[name](**kwargs)


def mlpmnist(num_classes=10,**kwargs):

    feature_extractors = FeatureExtractor_mnist(dim_in=28, dim_hidden=64)  # 假设输入尺寸为28*28
    classifiers = Classifier_mnist(dim_hidden=64, dim_out=10)
    return CombinedModel_mnist(feature_extractors, classifiers, dim_out=num_classes, **kwargs)

def mlp_pima(num_classes=2,**kwargs):

    feature_extractors = FeatureExtractor_Pima(dim_in=24, dim_hidden=64)  # 假设输入尺寸为8
    classifiers = Classifier_Pima(dim_hidden=64, dim_out=2)
    return CombinedModel_Pima(feature_extractors, classifiers,dim_out=num_classes ,**kwargs)
def mlp_diabetes(num_classes=2,**kwargs):

    feature_extractors = FeatureExtractor_Pima(dim_in=20, dim_hidden=64)  # 假设输入尺寸为8
    classifiers = Classifier_Pima(dim_hidden=64, dim_out=3)
    return CombinedModel_Pima(feature_extractors, classifiers,dim_out=num_classes ,**kwargs)



def mlp_death(num_classes=2,**kwargs):

    feature_extractors = FeatureExtractor_Pima(dim_in=56, dim_hidden=64)  # 假设输入尺寸为8
    classifiers = Classifier_Pima(dim_hidden=64, dim_out=2)
    return CombinedModel_Pima(feature_extractors, classifiers,dim_out=num_classes ,**kwargs)


def net_pathmnist(in_channels=3,**kwargs):

    return Net_pathmnist(in_channels=in_channels,hidden=256,num_classes=9)

def net_dermamnist(in_channels=3,**kwargs):

    return Net_pathmnist(in_channels=in_channels,hidden=256,num_classes=7)
def net_retinamnist(in_channels=3,**kwargs):

    return Net_pathmnist(in_channels=in_channels,hidden=256,num_classes=5)
def net_octmnist(in_channels=3,**kwargs):

    return Net_pathmnist(in_channels=in_channels,hidden=128,num_classes=4)
# 使用_add_model装饰器来注册mlpmnist函数
@_add_model
def build_mlpmnist_model(**kwargs):
    return mlpmnist(**kwargs)
@_add_model
def build_mlp_pima_model(**kwargs):
    return mlp_pima(**kwargs)
@_add_model
def build_mlp_diabetes_model(**kwargs):
    return mlp_diabetes(**kwargs)
@_add_model
def build_mlp_death_model(**kwargs):
    return mlp_death(**kwargs)
@_add_model
def build_net_pathmnist_model(**kwargs):
    return net_pathmnist(**kwargs)

@_add_model
def build_net_dermamnist_model(**kwargs):
    return net_dermamnist(**kwargs)
@_add_model
def build_net_retinamnist_model(**kwargs):
    return net_retinamnist(**kwargs)
@_add_model
def build_net_octmnist_model(**kwargs):
    return net_octmnist(**kwargs)