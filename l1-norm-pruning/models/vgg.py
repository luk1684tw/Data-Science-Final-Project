import math

import torch
import torch.nn as nn
from torch.autograd import Variable


__all__ = ['vgg']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, transfer=False):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        if transfer:
            self.classifier = nn.Sequential(
                nn.Linear(16384, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
              nn.Linear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_classes)
            )
            
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        # print ('[INFO] Shape of input is', x.size())
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        # print ('[INFO] Shape of X is', x.size())
        x = x.view(x.size(0), -1)
        # print ('[INFO] Shape of X is', x.size())
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    net = vgg()
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y = net(x)
    print(y.data.shape)