from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import f1_score

import models
import sys
sys.path.insert(0, '..')
from datasets import GenerateCifar10Dataset as get
root = 'content/Drive/My Drive/Colab Notebooks'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='../saves', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--dist', default=0, type=int, nargs='+',
                    help='distribution of dataset')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print (args.dist)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# if args.dataset == 'cifar10':
#     train_loader, test_loader = get(root, args.batch_size, args.test_batch_size, args.dist)
#     # train_loader = torch.utils.data.DataLoader(
#     #     datasets.CIFAR10('./data.cifar10', train=True, download=True,
#     #                    transform=transforms.Compose([
#     #                        transforms.Pad(4),
#     #                        transforms.RandomCrop(32),
#     #                        transforms.RandomHorizontalFlip(),
#     #                        transforms.ToTensor(),
#     #                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     #                    ])),
#     #     batch_size=args.batch_size, shuffle=True, **kwargs)
#     # test_loader = torch.utils.data.DataLoader(
#     #     datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
#     #                        transforms.ToTensor(),
#     #                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     #                    ])),
#     #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
# else:
#     train_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR100('./data.cifar100', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.Pad(4),
#                            transforms.RandomCrop(32),
#                            transforms.RandomHorizontalFlip(),
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                        ])),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                        ])),
#         batch_size=args.test_batch_size, shuffle=True, **kwargs)


def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    #predict = []
    #true_value = []
    classnum = 10
    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))
    
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            #print (pred)
            
            print(pred.eq(target.data.view_as(pred)).cpu())
            # pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
            # predict_num += pre_mask.sum(0)
            # tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
            # target_num += tar_mask.sum(0)
            # acc_mask = pre_mask*tar_mask
            # acc_num += acc_mask.sum(0)

        # recall = acc_num/target_num
        # precision = acc_num/predict_num
        # F1 = 2*recall*precision/(recall+precision)
        # test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath, dist):
    torch.save(state, os.path.join(filepath, f'checkpointDist{dist}.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

for dist in args.dist:
    train_loader, test_loader = get(root, args.batch_size, args.test_batch_size, dist)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        model.cuda()
    best_prec1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(epoch)
        prec1 = test()
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'cfg': model.cfg
        }, is_best, filepath=args.save, dist=dist)
