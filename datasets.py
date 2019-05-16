import torch
import torchvision
import os
from torchvision import transforms

def GenerateCifar10Dataset(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training = torchvision.datasets.CIFAR10(os.path.join(root, 'datasets/cifar10'), download=True, train=True, transform=transform)
    testing = torchvision.datasets.CIFAR10(os.path.join(root, 'datasets/cifar10'), download=True, train=False, transform=transform)

    trainLoader = torch.utils.data.DataLoader(training, batch_size, shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(testing, batch_size, shuffle=True, num_workers=2)
    
    return trainLoader, testLoader