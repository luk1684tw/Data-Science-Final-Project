import torch
import torchvision
import os
import random
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

'''
define x > y
distribution A:[x, y, y, y, y, y, y, y, y, y] 
distribution B:[x, x, x, x, y, y, y, y, y, y] 
distribution C:[x, x, x, x, x, x, x, x, x, y] 

'''


def GenerateCifar10Dataset(root, trainBatchSize, testBatchSize, dist, test):
    distTest = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 75],
        [75, 75, 1, 1, 75, 1, 1, 1, 75, 1],
        [75, 75, 75, 75, 1, 75, 75, 75, 75, 75],
        [100, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [100, 1, 1, 100, 1, 1, 100, 1, 1, 100],
        [100, 100, 100, 100, 100, 100, 100, 1, 100, 100]        
    ]
    testSet = {'A75': 0, 'B75': 1, 'C75': 2, 'A100': 3, 'B100': 4, 'C100': 5}

    distType = dist[0]
    distNum = int(dist[1:])

    print('[INFO] Start creating datasets')
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

    classRation = list()
    if not test:
        if distType == 'A':
            classRation = [1, 1, 1, 1, 1, 1, 1, 1, 1] + [distNum]
            print ('[INFO] Distribution type is', distType, ': One class is more than the others.')
        elif distType == 'B':
            classRation = [1, 1, 1, 1, 1, 1] + [distNum]*4
            print ('[INFO] Distribution type is', distType, ': Some classes are more than the majority.')
        elif distType == 'C':
            classRation = [distNum]*9 + [1]
            print ('[INFO] Distribution type is', distType, ': One class is less than the others.')
        else:
            print ('[INFO] No matched distribution')
        random.shuffle(classRation)
    else:
        index = testSet[dist]
        classRation = distTest[index]

    print ('[INFO] Distribution number is', distNum)
    print ('[INFO] Label distribution ratio:',classRation)

    training = torchvision.datasets.CIFAR10(os.path.join(
        root, 'datasets/cifar10'), download=True, train=True, transform=trainTransform)
    testing = torchvision.datasets.CIFAR10(os.path.join(
        root, 'datasets/cifar10'), download=True, train=False, transform=testTransform)

    print('[INFO] Creating sampler')
    weights = [0]*len(training.data)


    for idx, val in enumerate(training):
        weights[idx] = classRation[val[1]]
    sampler = WeightedRandomSampler(weights, 50000)
    # print (list(sampler))

    print('[INFO] Creating dataloader')
    trainLoader = torch.utils.data.DataLoader(
        training, trainBatchSize, shuffle=False, num_workers=2, sampler=sampler)
    testLoader = torch.utils.data.DataLoader(
        testing, testBatchSize, shuffle=True, num_workers=2)

    print('[INFO] Return dataloaders')
    return trainLoader, testLoader
