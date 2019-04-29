#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division
import os
import torch
from skimage import io, transform 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms, utils
import random
import math
import argparse
import json
import pickle

# Return only indices of certain classes
def get_indices_label(dataset, labels_list):
    label_indices = []

    for i in range(dataset.__len__()):
        _, target = dataset.__getitem__(i)
        if target in labels_list:
            label_indices.append(i)

    return label_indices

def load_mnist(labels_list, batch_size=1):
    """ Load MNIST dataset"""

    #transform input/output to tensor
    transform = transforms.Compose([
        transforms.Pad(2), # Makes 32x32, Log2(32) = 5  
        transforms.ToTensor(),  
    ])


    #train set
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Get train indices corresponding to the labels_list
    train_indices = get_indices_label(train_set, labels_list)
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     sampler=SubsetRandomSampler(train_indices),
                     shuffle=False)

    #test set
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Get test indices corresponding to the labels_list
    test_indices = get_indices_label(test_set, labels_list)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    sampler=SubsetRandomSampler(test_indices),
                    shuffle=False)

    return [train_loader, test_loader] 