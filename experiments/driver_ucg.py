#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division
import os

os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6

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
import sys

from experiments.args import get_args
from datasets.mnist_utils import load_mnist
from feature_vectors import local_feature_vectors, custom_feature
from ucg import reduced_covariance, generate_new_phi, precalculate_traces, rho_ij

from _constants import FEATURE_MAP_D, HEIGHT, WIDTH

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, _ = load_mnist()
    Phi = custom_feature(train_loader, args.batch_size, args.parser_type, fake_img=False)

    print('Size of train_loader: {}'.format(len(train_loader)))
    print('Size of initial Phi: {}'.format(Phi.shape))

    tree_depth = int(math.log2(HEIGHT * WIDTH)) 
    iterates = HEIGHT * WIDTH
    start_time = time.time()

    for layer in range(tree_depth):
        tree_tensor = []
        print('Layer: {}, Iterates: {}'.format(layer, iterates))
        traces = precalculate_traces(Phi)

        pairs = np.array_split(range(iterates), iterates // 2)
        for i, indices in enumerate(pairs):
            tree_tensor = rho_ij(Phi, traces, tree_tensor, layer, args.eps, indices)

        #compute new feature map
        Phi = generate_new_phi(Phi, tree_tensor)
        #update number of local feature vectors for each image
        iterates = iterates // 2 

        print('Saving Model')
        with open(os.path.join(args.logdir, '{}{}-BSz{}-Layer{}'.format(
                args.prefix, args.filename, args.batch_size, layer)), "wb") as file:
            pickle.dump(tree_tensor, file)

        print(tree_tensor[0].shape)

    # Write to file
    print('Time for {} Images: {}'.format(args.batch_size, time.time() - start_time))
    

"""
    manager = multiprocessing.Manager()
    with multiprocessing.Pool(processes=args.nworkers) as pool:
        for layer in range(tree_depth):
            tree_tensor = manager.dict()

            print('Layer: {}, Iterates: {}'.format(layer, iterates))
            traces = precalculate_traces(Phi)
            print('Traces shape: {}'.format(traces.shape))

            pairs = np.array_split(range(iterates), iterates // 2)
            pool.map(partial(rho_ij, Phi, traces, tree_tensor, layer, args.eps), pairs)

            print(tree_tensor[layer, 0, 1])

            with open(os.path.join(args.logdir, '{}{}-BSz{}-Layer{}'.format(
                args.prefix, args.filename, args.batch_size, layer)), "wb") as file:

                pickle.dump(tree_tensor, file)

            #compute new feature map
            Phi = generate_new_phi(Phi, tree_tensor, layer)
            #update number of local feature vectors for each image
            iterates = iterates // 2 
"""