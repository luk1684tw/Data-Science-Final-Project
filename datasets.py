import torch
import torchvision
import os
from torchvision import transforms
import torch.utils.data.sampler.WeightedRandomSampler as WeightedRandomSampler

def GenerateCifar10Dataset(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    training = torchvision.datasets.CIFAR10(os.path.join(root, 'datasets/cifar10'), download=True, train=True, transform=transform)
    testing = torchvision.datasets.CIFAR10(os.path.join(root, 'datasets/cifar10'), download=True, train=False, transform=transform)

    sampler = WeightedRandomSampler([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 10)

    trainLoader = torch.utils.data.DataLoader(training, batch_size, shuffle=True, num_workers=2, sampler=sampler)
    testLoader = torch.utils.data.DataLoader(testing, batch_size, shuffle=True, num_workers=2, sampler=sampler)
    
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    count = dict()
    for img, label in trainLoader:
        if label in count:
            count[label] += 1
        else:
            count[label] = 1
    
    trainLen = len(trainLoader)
    for label in labels:
        count[label] /= trainLen
    print (count)

    return trainLoader, testLoader