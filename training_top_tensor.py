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
from sklearn.linear_model import LogisticRegression

from datasets.mnist_utils import load_mnist
from feature_vectors import local_feature_vectors, custom_feature

from _constants import FEATURE_MAP_D, HEIGHT, WIDTH


def generate_new_phi(Phi, tree_tensor, layer):
    
    print('   ----> generate new phi: ')
    Phi_new = [] #layer 0: list because the dimension of each d may differt. so we can't have a tensor of (100,32,d)

    iterates = len(Phi[0])  
    for imageidx in range(len(Phi)):
        coarse_grained = []
        i = 0
        while True: #go from o to 32 reduced vectors of Phi(image)
            if i == iterates: break 
            ind1 = i 
            ind2 = i + 1
            x = Phi[imageidx][ind1] #local feature vector1 
            y = Phi[imageidx][ind2] #local feature vector2
            truncated_U = tree_tensor[layer, ind1, ind2] #get layer of isometries
            
            #-------Apply U to image--------
            phi_xy = np.outer(x,y).flatten()
            phi_reduced = np.dot(truncated_U.T,phi_xy)
            #append result to the new phi
            coarse_grained.append(phi_reduced)

            i +=2
        Phi_new.append(np.array(coarse_grained))
        
    print('     ---> new Phi: N=' + str(len(Phi_new[0])))
    print('   ----> END generate new phi: ')
    return Phi_new

################################################ Test #####################################
parser = argparse.ArgumentParser(description='Learning Relevant Features (Stoudemire 2018)')
parser.add_argument('--eps', type=float, default=1e-3, help='Truncation epsilon')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size for MNIST')
parser.add_argument('--seed', type=int, default=123, help='Seed')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

iterates = HEIGHT * WIDTH
tree_depth = int(math.log2(HEIGHT * WIDTH)) -1

#load mnist
train_loader, test_loader = load_mnist()

#compute feature map for all images
Phi = custom_feature(train_loader, fake_img=False)
print(Phi.shape)
#read file contraining isometry U
file_name = "treeU_epsilon_less_2_images_200"
with open(file_name, "rb") as file:
    U=pickle.load(file)

for layer in range(tree_depth):
    print('\n-------> layer: '+str(layer))
    print('iterates = '+str(iterates))
    #compute new feature map
    Phi = generate_new_phi(Phi, U, layer)
    #update number of local feature vectors for each image
    iterates = iterates // 2 
    
#construct new database
t1 = Phi[0].shape[0]
t2 = Phi[0].shape[1]
X = np.zeros((N_MAX, t1*t2))
for i in range(N_MAX):
    #get reduced Phi
    X[i,:] = Phi[i].flatten()
print(X.shape)

#fit logistic regression
y = np.zeros(N_MAX)
for batch_idx, (x, target) in enumerate(train_loader):
    if batch_idx ==N_MAX:
        break
    y[batch_idx] = target[0]  
logreg = LogisticRegression(C=1e5, solver='newton-cg', multi_class='multinomial',max_iter=100000, warm_start=True)
logreg.fit(X, y)
print(logreg.score(X, y))
