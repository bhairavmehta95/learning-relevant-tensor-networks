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
from functools import partial
import multiprocessing

from datasets.mnist_utils import load_mnist
from feature_vectors import local_feature_vectors, custom_feature
from ucg import reduced_covariance, generate_new_phi, precalculate_traces, rho_ij

from _constants import FEATURE_MAP_D, HEIGHT, WIDTH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning Relevant Features (Stoudemire 2018)')
    parser.add_argument('--logdir', type=str, default='saved-models/', help='default log directory')
    parser.add_argument('--eps', type=float, default=1e-3, help='Truncation epsilon')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for MNIST')
    parser.add_argument('--seed', type=int, default=123, help='Seed')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of multiprocessing workers')

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, _ = load_mnist()
    Phi = custom_feature(train_loader, args.batch_size, fake_img=False)

    print('Size of train_loader: {}'.format(len(train_loader)))
    print('Size of initial Phi: {}'.format(Phi.shape))

    tree_depth = int(math.log2(HEIGHT * WIDTH)) 
    iterates = HEIGHT * WIDTH

    manager = multiprocessing.Manager()
    tree_tensor = manager.dict()

    with multiprocessing.Pool(processes=args.nworkers) as pool:
        for layer in range(tree_depth):
            print('Layer: {}, Iterates: {}'.format(layer, iterates))
            traces = precalculate_traces(Phi)
            print('Traces shape: {}'.format(traces.shape))

            pairs = np.array_split(range(iterates), iterates // 2)
            pool.map(partial(rho_ij, Phi, traces, tree_tensor, layer, args.eps), pairs)

            print(tree_tensor[layer, 0, 1])

            #compute new feature map
            Phi = generate_new_phi(Phi, tree_tensor, layer)
            #update number of local feature vectors for each image
            iterates = iterates // 2 

    # Write to file
    print(tree_tensor[8,0,1].shape)
    with open(os.path.join(args.logdir, "treeU_max"), "wb") as file:
        pickle.dump(tree_tensor, file)

    # Read for testing
    with open(os.path.join(args.logdir, "treeU_max"), "rb") as file:
        tree=pickle.load(file)

    print(type(tree))
    print(tree[8,0,1].shape)