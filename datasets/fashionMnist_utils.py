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

def load_fashionMnist(labels_list=[], batch_size=1):
    """ Load fashion mnist dataset"""

    #transform input/output to tensor
    transform = transforms.Compose([
        transforms.Pad(2), # Makes 32x32
        transforms.ToTensor(),  
    ])


    #train set
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    # Get train indices corresponding to the labels_list
    if labels_list: #check if it's not empty
        train_indices = get_indices_label(train_set, labels_list)
        train_loader = torch.utils.data.DataLoader(
                         dataset=train_set,
                         batch_size=batch_size,
                         sampler=SubsetRandomSampler(train_indices),
                         shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(
                         dataset=train_set,
                         batch_size=batch_size,
                         shuffle=False)

    #test set
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Get test indices corresponding to the labels_list
    if labels_list: #check if it's not empty
        test_indices = get_indices_label(test_set, labels_list)
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=batch_size,
                        sampler=SubsetRandomSampler(test_indices),
                        shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(
                        dataset=test_set,
                        batch_size=batch_size,
                        shuffle=False)

    return [train_loader, test_loader] 




