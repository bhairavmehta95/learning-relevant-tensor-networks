#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division
import os
import torch
from skimage import io, transform 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import random
import math
import argparse
import json
import pickle

from datasets.mnist_utils import load_mnist
from feature_vectors import local_feature_vectors, custom_feature
from ucg import reduced_covariance, generate_new_phi

from _constants import FEATURE_MAP_D, HEIGHT, WIDTH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning Relevant Features (Stoudemire 2018)')
    parser.add_argument('--eps', type=float, default=1e-3, help='Truncation epsilon')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for MNIST')
    parser.add_argument('--seed', type=int, default=123, help='Seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, _ = load_mnist()
    Phi = custom_feature(train_loader, args.batch_size, fake_img=False)
    
    print('Size of train_loader: {}'.format(len(train_loader)))
    print('Size of initial Phi: {}'.format(Phi.shape))

    tree_depth = int(math.log2(HEIGHT * WIDTH)) 
    iterates = HEIGHT * WIDTH

    tree_tensor = dict()

    for layer in range(tree_depth):
        print('Layer: {}, Iterates: {}'.format(layer, iterates))

        for i in range(iterates):
            if i % 2: continue 
            
            #compute rho
            ind1 = i
            ind2 = i + 1
            rho = reduced_covariance(Phi, ind1, ind2)

            # Calculate Eigenvalues
            e_val, U = np.linalg.eigh(rho) # eigenvalues arranged in ascending order
            e_val, U = np.flip(e_val), np.flip(U, axis=1) # eigenvalues arranged in descending order
            trace = np.sum(e_val)

            truncation_sum = 0
            # Gross notation, but makes indexing nicer
            first_truncated_eigenvalue = 0

            for eig_idx, e in enumerate(e_val):
                truncation_sum += e
                first_truncated_eigenvalue += 1

                if (truncation_sum / trace) > (1 - args.eps):
                    break
            
            # truncation
            truncated_U = U[:, :first_truncated_eigenvalue] # keep first r cols of U

            # store U
            tree_tensor[layer, ind1, ind2] = truncated_U
            
        #compute new feature map
        Phi = generate_new_phi(Phi, tree_tensor, layer)
        #update number of local feature vectors for each image
        iterates = iterates // 2 

    print(tree_tensor[8,0,1].shape)
    with open("treeU_max", "wb") as file:
        pickle.dump(tree_tensor, file)

    #read
    with open("treeU_max", "rb") as file:
        tree=pickle.load(file)
    print(type(tree))
    print(tree[8,0,1].shape)