import torch
import torchvision
import os
import random
from torchvision import transforms

'''
define x > y
distribution A:[x, y, y, y, y, y, y, y, y, y] 
distribution B:[x, x, x, x, y, y, y, y, y, y] 
distribution C:[x, x, x, x, x, x, x, x, x, y] 

'''


def GenerateOCTDatasets(root, trainBatchSize, testBatchSize):
    print('[INFO] Start creating datasets')
    trainTransform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    testTransform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    training = torchvision.datasets.ImageFolder(os.path.join(
        root, 'datasets/STFDogs/train'), transform=trainTransform)
    testing = torchvision.datasets.ImageFolder(os.path.join(
        root, 'datasets/STFDogs/test'), transform=testTransform)

    print('[INFO] Creating dataloader')
    trainLoader = torch.utils.data.DataLoader(
        training, trainBatchSize, shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(
        testing, testBatchSize, shuffle=True, num_workers=2)

    print('[INFO] Return dataloaders')
    return trainLoader, testLoader
