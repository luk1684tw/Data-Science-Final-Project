from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import matplotlib.pyplot as plt
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
root = '/content/Drive/My Drive/Colab Notebooks'

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
parser.add_argument('--save', default='/content/Drive/My Drive/Colab Notebooks/models/baseline', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--dist', default=0, type=str, nargs='+',
                    help='distribution of dataset')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print (args.dist)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

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
    predict = []
    true_value = []

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # Target Tensor: target.data.view_as(pred)
        # Predict Tensor: pred
        predict += pred.tolist()[0]
        true_value += target.data.view_as(pred).tolist()[0]


    # test_loss /= len(test_loader.dataset)
    F1 = f1_score(true_value, predict, average='macro')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1: {:.2f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), F1))
    
    return correct.item() / float(len(test_loader.dataset))

def plot_kernels(tensor, num_cols=6):
    print(tensor.shape)
    tn = tensor.view(1, tensor.shape[0]*tensor.shape[1]*tensor.shape[2]*tensor.shape[3]).numpy()
    # print (tn)

    return tn[0]


def save_checkpoint(state, is_best, filepath, dist):
    torch.save(state, os.path.join(filepath, f'checkpointDist{dist}.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, f'checkpointDist{dist}.pth.tar'), os.path.join(filepath, f'model{dist}_best.pth.tar'))

for dist in args.dist:
    train_loader, test_loader = get(root, args.batch_size, args.test_batch_size, dist, False)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model._initialize_weights()

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
        print ('[INFO] Start printing weight distribution of Conv2D layers...')
        plt.figure()
        fig, ax = plt.subplots(3, 5, tight_layout=True, sharey=True,)
        
        weightInfo = list()
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                weightInfo.append(plot_kernels(m.weight.data.cpu()))
        ax = ax.flatten()
        for idx, weight in enumerate(weightInfo):
            print(f'[INFO] Setting up NO.{idx} layer...')
            ax[idx].hist(weight)
        plt.show()
        print ('[INFO] End of printing weight distribution of Conv2D layers')

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
