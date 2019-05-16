import torch
import torchvision
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from datasets import GenerateCifar10Dataset as get
from model import VGGNet

training, testing = get(root, batch_size)
# vgg16
net_arch16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5', "FC1", "FC2", "FC"]

vgg16 = VGGNet(net_arch=net_arch16)
print(vgg16)