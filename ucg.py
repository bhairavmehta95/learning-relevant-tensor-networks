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

from _constants import FEATURE_MAP_D, HEIGHT, WIDTH


def generate_new_phi(Phi, tree_tensor, layer):
    print('Generating new phi.')
    Phi_new = [] 
    iterates = len(Phi[0])

    for imageidx in range(len(Phi)):
        coarse_grained = []
        i = 0
        while i < iterates: #go from o to 32 reduced vectors of Phi(image)
            ind1 = i 
            ind2 = i + 1
            x = Phi[imageidx][ind1] #local feature vector1 type: numpy.ndarray
            y = Phi[imageidx][ind2] #local feature vector2
            truncated_U = tree_tensor[layer, ind1, ind2] #get layer of isometries

            phi_xy = np.outer(x,y).flatten()
            phi_reduced = np.dot(truncated_U.T,phi_xy)
            coarse_grained.append(phi_reduced)

            i += 2

        Phi_new.append(np.array(coarse_grained))
        
    print('New Phi with N {}'.format(len(Phi_new[0])))
    return Phi_new


def reduced_covariance(Phi, s1, s2):
    """ Compute the reduced covariance matrix given the two position (s1,s2) in feature matrix of an image.
        Example: to compute the reduced covariance matrix ro34, s1=2 and s2=3"""
    
    #Phi is a tensor as in the case for the first layer
    #Phi is a list starting from the second layer
    
    Nt = len(Phi)     #number of images
    N = len(Phi[0])       #number of local features vectors in Phi
       
    ro = None
    trace_tracker = 1

    for img_idx in range(Nt):
        #get the two local feature vectors 
        phi1 = Phi[img_idx][s1]
        phi2 = Phi[img_idx][s2]
        
        #trace over all the indices except s1 and s2
        # TODO: Parallelize
        trace_tracker = 1
        for s in range(N):
            if s != s1 and s != s2:
                x = Phi[img_idx][s]   
                trace_tracker *= np.inner(x, x)
                
        #compute the order 4 tensor
        phi12 = np.outer(phi1, phi2).flatten()
        ro_j = np.outer(phi12, phi12)
        
        #add result to ro
        if ro is None:
            ro = np.zeros_like(ro_j)

        ro += trace_tracker*ro_j
        
    return ro / Nt


################################### Test ###############################################################
parser = argparse.ArgumentParser(description='Learning Relevant Features (Stoudemire 2018)')
parser.add_argument('--eps', type=float, default=1e-3, help='Truncation epsilon')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size for MNIST')
parser.add_argument('--seed', type=int, default=123, help='Seed')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_loader, _ = load_mnist()

print('Size of train_loader: {}'.format(len(train_loader)))

Phi = custom_feature(train_loader, fake_img=False)
print('Size of initial Phi: {}'.format(Phi.shape))

tree_depth = int(math.log2(HEIGHT * WIDTH)) 
iterates = HEIGHT * WIDTH

tree_tensor = dict()

for layer in range(tree_depth):
    print('\n-------> layer: '+str(layer))
    print('iterates = '+str(iterates))
    for i in range(iterates):
        if i % 2 != 0: continue 
        
        #compute ro
        ind1 = i
        ind2 = i + 1
        ro = reduced_covariance(Phi, ind1, ind2)

        # Calculate Eigenvalues
        e_val, U = np.linalg.eigh(ro) # eigenvalues arranged in ascending order
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
