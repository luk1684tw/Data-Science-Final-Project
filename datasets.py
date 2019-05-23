import torch
import torchvision
import os
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

def GenerateCifar10Dataset(root, batch_size):
    print ('start create datasets')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training = torchvision.datasets.CIFAR10(os.path.join(root, 'datasets/cifar10'), download=True, train=True, transform=transform)
    testing = torchvision.datasets.CIFAR10(os.path.join(root, 'datasets/cifar10'), download=True, train=False, transform=transform)

    # print ('create sampler')
    sampler = WeightedRandomSampler([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 50000)
    print (list(sampler))

    print ('create dataloader')
    trainLoader = torch.utils.data.DataLoader(training, batch_size, shuffle=False, num_workers=2, sampler=sampler)
    testLoader = torch.utils.data.DataLoader(testing, batch_size, shuffle=False, num_workers=2, sampler=sampler)

    print ('return loaders')
    return trainLoader, testLoader