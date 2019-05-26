import torch
import torchvision
import os
import random
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler


def GenerateCifar10Dataset(root, trainBatchSize, testBatchSize, dist_index):
    dist = [
        [3, 3, 3, 3, 1, 1, 1, 1, 1, 1],
        [5, 5, 5, 5, 1, 1, 1, 1, 1, 1],
        [3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [1, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [150, 150, 150, 150, 1, 1, 1, 1, 1, 1],
        [200, 200, 200, 200, 1, 1, 1, 1, 1, 1],
        [150, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [200, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
    ]
    print('start create datasets')
    trainTransform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    testTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    training = torchvision.datasets.CIFAR10(os.path.join(
        root, 'datasets/cifar10'), download=True, train=True, transform=trainTransform)
    testing = torchvision.datasets.CIFAR10(os.path.join(
        root, 'datasets/cifar10'), download=True, train=False, transform=testTransform)

    print('create sampler')
    weights = [0]*len(training.data)
    classRation = dist[dist_index].copy()
    random.shuffle(classRation)
    print ('Label distribution ratio:',classRation)
    
    for idx, val in enumerate(training):
        weights[idx] = classRation[val[1]]
    sampler = WeightedRandomSampler(weights, 50000)
    # print (list(sampler))

    print('create dataloader')
    trainLoader = torch.utils.data.DataLoader(
        training, trainBatchSize, shuffle=False, num_workers=2, sampler=sampler)
    testLoader = torch.utils.data.DataLoader(
        testing, testBatchSize, shuffle=True, num_workers=2)

    print('return loaders')
    return trainLoader, testLoader
