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
    parser.add_argument('--parser_type', type=str, default='default', help='image parser')
    parser.add_argument('--filename', type=str, default='TODO', help='default file to load')
    parser.add_argument('--eps', type=float, default=1e-3, help='Truncation epsilon')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for MNIST')
    parser.add_argument('--seed', type=int, default=123, help='Seed')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of multiprocessing workers')

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, test_loader = load_mnist()
    Phi = custom_feature(train_loader, args.batch_size, args.parser_type, fake_img=False)

    with open(os.path.join(args.logdir, args.filename), "rb") as file:
        U = pickle.load(file)

    for layer in range(tree_depth):
        #compute new feature map
        Phi = generate_new_phi(Phi, U, layer)
        #update number of local feature vectors for each image
        iterates = iterates // 2 

    tree_depth = int(math.log2(HEIGHT * WIDTH)) 
    iterates = HEIGHT * WIDTH

    #construct new database
    t1 = Phi[0].shape[0]
    t2 = Phi[0].shape[1]
    X = np.zeros((args.batch_size, t1*t2))
    for i in range(args.batch_size):
        #get reduced Phi
        X[i,:] = Phi[i].flatten()
    print(X.shape)

    #fit logistic regression
    y = np.zeros(args.batch_size)
    for batch_idx, (x, target) in enumerate(train_loader):
        if batch_idx == args.batch_size:
            break
        y[batch_idx] = target[0]  
    logreg = LogisticRegression(C=1e5, solver='newton-cg', 
        multi_class='multinomial',max_iter=100000, warm_start=True)
    logreg.fit(X, y)
    print(logreg.score(X, y))

    # TODO: @TAYSSIR: PLEASE ADD YOUR TEST CODE