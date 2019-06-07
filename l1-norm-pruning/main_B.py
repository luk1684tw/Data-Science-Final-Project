from __future__ import print_function
import argparse
import numpy as np
import os
import shutil
%matplot inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# New Import
import sys
sys.path.insert(0, '..')
from datasets import GenerateCifar10Dataset as get
from sklearn.metrics import f1_score
import models
from compute_flops import print_model_param_flops
from matplotlib import pyplot as plt

modelRoot = '/content/Drive/My Drive/Colab Notebooks/models/pruned'
datasetRoot = '/content/Drive/My Drive/Colab Notebooks'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--scratch', default='', type=str, metavar='PATH',
                    help='path to the pruned model')
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
parser.add_argument('--save', default='/content/Drive/My Drive/Colab Notebooks/models/scratchB', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--dist', default=0, type=str,
                    help='distribution of dataset')                     

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)


train_loader, test_loader = get(datasetRoot, args.batch_size, args.test_batch_size, args.dist, True)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.scratch:
    modelPath = os.path.join(modelRoot, args.scratch)
    print ('[INFO] Loading model from', modelPath)
    checkpoint = torch.load(modelPath)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])

model_ref = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

flops_std = print_model_param_flops(model_ref, 32)
flops_small = print_model_param_flops(model, 32)
args.epochs = int(120 * (flops_std / flops_small))

if args.cuda:
    model.cuda()

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
    # F1-score 
    predict = []
    true_value = []
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #F1-score
        predict += pred.tolist()[0]
        true_value += target.data.view_as(pred).tolist()[0]
    test_loss /= len(test_loader.dataset)
    F1 = f1_score(true_value, predict, average='macro')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1: {:.2f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), F1))
    return correct.data.item() / float(len(test_loader.dataset)), F1

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, f'scratchB{args.dist}.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, f'scratchB{args.dist}.pth.tar'), os.path.join(filepath, f'model{args.dist}_best.pth.tar'))

def plot_kernels(tensor, num_cols=6):
    # if not tensor.ndim==4:
    #     raise Exception("assumes a 4D tensor")
    # if not tensor.shape[-1]==3:
    #     raise Exception("last dim needs to be 3 to plot")
    print(tensor.shape)
    tn = tensor.view(1, tensor.shape[0]*tensor.shape[1]*tensor.shape[2]*tensor.shape[3]).numpy()
    plt.hist(tn)
    plt.show()
    # num_kernels = tensor.shape[0]
    # num_rows = 1+ num_kernels // num_cols
    # fig = plt.figure(figsize=(num_cols,num_rows))
    # for i in range(tensor.shape[0]):
    #     ax1 = fig.add_subplot(num_rows,num_cols,i+1)
    #     ax1.imshow(tensor[i])
    #     ax1.axis('off')
    #     ax1.set_xticklabels([])
    #     ax1.set_yticklabels([])

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()

best_prec1 = 0.
F1 = 0.
for epoch in range(0, 1):#range(args.start_epoch, args.epochs):
    if epoch in [int(args.epochs*0.5), int(args.epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)

    # Plot 
    i = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            plot_kernels(m.weight.data.clone().cpu())
            
            #    plt.savefig('result.png')
            

    prec1, f1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
        F1 = f1
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': model.cfg
    }, is_best, filepath=args.save)
with open(os.path.join(args.save, f'bestAccu{args.dist}.txt'), 'w') as file:
    file.write(f'Accu: {best_prec1}, F1: {F1}\n')