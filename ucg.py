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
import sys

from datasets.mnist_utils import load_mnist
from feature_vectors import local_feature_vectors, custom_feature

from _constants import FEATURE_MAP_D, HEIGHT, WIDTH, TOL_RHO


def rho_ij(Phi, traces, tree_tensor, layer, eps, indices):
    ind1, ind2 = indices
    rho = reduced_covariance(Phi, ind1, ind2, traces)

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

        if (truncation_sum / trace) > (1 - eps):
            break
    
    # Truncate U
    truncated_U = U[:, :first_truncated_eigenvalue] # keep first r cols of U

    # Store U
    tree_tensor.append(truncated_U)


def rho_ij_mtl(Phi1, Phi2, traces1, traces2, tree_tensor, layer, eps, mixing_mu, indices):
    ind1, ind2 = indices
    rho1 = reduced_covariance(Phi1, ind1, ind2, traces1)
    rho2 = reduced_covariance(Phi2, ind1, ind2, traces2)

    rho = mixing_mu * rho1 + (1 - mixing_mu) * rho2

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

        if (truncation_sum / trace) > (1 - eps):
            break
    
    # Truncate U
    truncated_U = U[:, :first_truncated_eigenvalue] # keep first r cols of U

    # Store U
    tree_tensor.append(truncated_U)


def generate_new_phi(Phi, tree_tensor):
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
            truncated_U = tree_tensor[i // 2] #get layer of isometries

            phi_xy = np.outer(x,y).flatten()
            phi_reduced = np.dot(truncated_U.T,phi_xy)
            coarse_grained.append(phi_reduced)

            i += 2

        Phi_new.append(np.array(coarse_grained))
        
    print('New Phi with N: {}'.format(len(Phi_new[0])))
    return Phi_new


def precalculate_traces(Phi):
    Nt = len(Phi)     #number of images
    N = len(Phi[0])       #number of local features vectors in Phi

    traces = np.zeros((Nt, N))

    for img_idx in range(Nt):
        for s in range(N):
            x = Phi[img_idx][s]
            traces[img_idx][s] = np.inner(x, x)

    return traces


def reduced_covariance(Phi, s1, s2, traces):
    """ Compute the reduced covariance matrix given the two position (s1,s2) in feature matrix of an image.
        Example: to compute the reduced covariance matrix ro34, s1=2 and s2=3"""
    
    #Phi is a tensor as in the case for the first layer
    #Phi is a list starting from the second layer
    
    Nt = len(Phi)     #number of images
    N = len(Phi[0])       #number of lo
    rho = None
       
    trace_tracker = 1
    norm_old = 0
    norm_diff_old = 0

    print('Inside Reduced Covariance [{}, {}]: Nt={}, N={}'.format(s1, s2, Nt, N))
    for image_num in range(Nt):
        #get the two local feature vectors
        img_idx = random.choice(range(Nt))

        phi1 = Phi[img_idx][s1]
        phi2 = Phi[img_idx][s2]
        
        #trace over all the indices except s1 and s2
        trace_tracker = 1
        for s in range(N):
            if s != s1 and s != s2:   
                trace_tracker *= traces[img_idx][s]

        phi12 = np.outer(phi1, phi2).flatten()
        rho_j = np.outer(phi12, phi12)

        #add result to rho
        if rho is None:
            rho = np.zeros_like(rho_j)

        rho += trace_tracker*rho_j

        norm_new = np.linalg.norm(rho, 'fro')
        norm_diff_new = norm_new - norm_old
        if image_num > 25 and (abs(norm_diff_new - norm_diff_old)) <= TOL_RHO:
            print('BREAKING @ Img {} / {}, PhiShapes: {}, \nSizePhi12 {}, SizeRhoJ {}, SizeRho {}'.format(image_num, Nt, phi1.shape, phi12.shape, rho_j.shape, rho.shape))
            break


        norm_old = norm_new
        norm_diff_old = norm_diff_new
            
    return rho / Nt

