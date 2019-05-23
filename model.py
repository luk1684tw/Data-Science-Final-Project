import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

class VGGNet(nn.Module):
    def __init__(self, net_arch, num_classes=10):
        # net_arch: a list of vggXX layer info
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        layers = []
        in_channels = 3 
        for arch in net_arch:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif arch == 'M5':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            elif arch == "FC1":
                layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC2":
                layers.append(nn.Conv2d(1024,1024, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC":
                layers.append(nn.Conv2d(1024,self.num_classes, kernel_size=1))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=arch, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels=arch
        self.vgg = nn.ModuleList(layers)
    def forward(self, input_data):
        x = input_data
        for layer in self.vgg:
            x = layer(x)
        out = x
        return out