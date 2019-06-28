# Overview
Originally from this paper's code implementation - [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning).

We use the training and pruning settings of this repo in pursuit of exactly the same experimental settings as the author of the paper did.

But modify the dataloader to our imbalanced dataset generator to test whether the conclusion of the paper still holds under different imbalanced situations.

# Datasets
We use CIFAR-10 as a base dataset to create imbalanced datasets.To test the models in different "Imbalanced Degree", we oversamples the dataset in two different dimensions: "Class Distribution" and "Class Ratio"

## Class Distribution
We define 3 different distributions as follows 
1. Dist A: one label is much more than all the others e.g. [100, 1, 1, 1, 1, 1, 1, 1, 1, 1]

2. Dist B: 4 labels are much more than the other 6 labels e.g. [100, 100, 100, 100, 1, 1, 1, 1, 1, 1]

3. Dist C: one label is much less than all the others e.g. [100, 100, 100, 100, 100, 100, 100, 100, 100, 1]

## Class Ratio
To simulate a variety of imbalanced degrees, we use 4 class ratios [75, 100, 200, 500]. Examples as follows
* 75: [75, 1, 1, 1, 1, 1, 1, 1, 1, 1]
* 100: [100, 1, 1, 1, 1, 1, 1, 1, 1, 1]
* 200: [200, 1, 1, 1, 1, 1, 1, 1, 1, 1]
* 500: [500, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Experiments
This directory contains all the CIFAR experiments in the paper, where there are four pruning methods in total:  

1. [L1-norm based channel pruning](https://arxiv.org/abs/1608.08710)
2. [Network Slimming](https://arxiv.org/abs/1708.06519)
3. [Soft filter pruning](https://www.ijcai.org/proceedings/2018/0309.pdf)
4. [Non-structured weight-level pruning](https://arxiv.org/abs/1506.02626)

For each method, check out example commands for baseline training, finetuning, scratch-E training and scratch-B training  in the corresponding directorys.  

We only use [L1-norm based channel pruning](https://arxiv.org/abs/1608.08710) as the pruning method. Then we compare the 4 methods' accuracy to find out the result.

