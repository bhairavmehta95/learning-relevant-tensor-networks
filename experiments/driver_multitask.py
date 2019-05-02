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
import json
import pickle
from functools import partial
import multiprocessing
import time

from experiments.args import get_args, get_mtl_args
from datasets.mnist_utils import load_mnist
from datasets.fashionMnist_utils import load_fashionMnist
from feature_vectors import local_feature_vectors, custom_feature
from ucg import reduced_covariance, generate_new_phi, precalculate_traces, rho_ij_mtl

from _constants import FEATURE_MAP_D, HEIGHT, WIDTH

if __name__ == '__main__':
    args = get_args()
    mtl_args = get_mtl_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not mtl_args.diff_datasets:
        train_loader1, _ = load_mnist([0, 1, 2])
        train_loader2, _ = load_mnist([3, 4, 5])
    else:
        train_loader1, _ = load_mnist()
        train_loader2, _ = load_fashionMnist()

    # two Phis
    Phi1 = custom_feature(train_loader1, args.batch_size, args.parser_type, args.feature_type, fake_img=False)
    Phi2 = custom_feature(train_loader2, args.batch_size, args.parser_type, args.feature_type, fake_img=False)

    print('Size of train_loader: {}'.format(len(train_loader1)))
    print('Size of initial Phi: {}'.format(Phi1.shape))

    tree_depth = int(math.log2(HEIGHT * WIDTH)) 
    iterates = HEIGHT * WIDTH

    manager = multiprocessing.Manager()
    tree_tensor = manager.dict()

    start_time = time.time()

    with multiprocessing.Pool(processes=args.nworkers) as pool:
        for layer in range(tree_depth):
            print('Layer: {}, Iterates: {}'.format(layer, iterates))

            # change here
            traces1 = precalculate_traces(Phi1)
            traces2 = precalculate_traces(Phi2)
            print('Traces shape: {}'.format(traces1.shape))

            pairs = np.array_split(range(iterates), iterates // 2)
            pool.map(partial(rho_ij_mtl, Phi1, Phi2, traces1, traces2, 
                tree_tensor, layer, args.eps, mtl_args.mixing_mu), pairs)

            print(tree_tensor[layer, 0, 1])

            #compute new feature map
            Phi1 = generate_new_phi(Phi1, tree_tensor, layer)
            Phi2 = generate_new_phi(Phi2, tree_tensor, layer)
            #update number of local feature vectors for each image
            iterates = iterates // 2 

    print('Time for {} Images: {}'.format(args.batch_size, time.time() - start_time))

    # Write to file
    print(tree_tensor[8,0,1].shape)
    with open(os.path.join(args.logdir, 'MTL-{}-BSz{}'.format(args.filename, args.batch_size)), "wb") as file:
        pickle.dump(tree_tensor, file)

    # Read for testing
    with open(os.path.join(args.logdir, 'MTL-{}-BSz{}'.format(args.filename, args.batch_size)), "rb") as file:
        tree=pickle.load(file)

    print(type(tree))
    print(tree[8,0,1].shape)