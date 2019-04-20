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



FEATURE_MAP_D = 2
TRUNCATION_EPS = 1e-3


def loadMnist(batch_size=1):
    """ Load MNIST dataset"""

    #transform input/output to tensor
    transform = transforms.Compose([
        # transforms.Pad(2), # Makes 32x32, Log2(32) = 5  
        transforms.ToTensor(),  
    ])

    #train set
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=False)

    #test set
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    return [train_loader, test_loader] 


def generate_new_phi(Phi, tree_tensor, layer):
    
    print('   ----> generate new phi: ')
    Phi_new = [] #layer 0: list because the dimension of each d may differt. so we can't have a tensor of (100,32,d)
    
    #Phi is a tensor as in the case for the first layer
    if type(Phi) is np.ndarray:      
        iterates = Phi.shape[1]  #layer0: Phi(100,8*8,2) 
        
        for imageidx, image in enumerate(Phi):
            coarse_grained = []
            i = 0
            while True: #go from o to 32 reduced vectors of Phi(image)
                if i == iterates: break 
                ind1 = i 
                ind2 = i + 1
                x = Phi[imageidx, ind1, :] #local feature vector1 type: numpy.ndarray
                y = Phi[imageidx, ind2, :] #local feature vector2
                truncated_U = tree_tensor[layer, ind1, ind2] #get layer of isometries

                #-------Apply U to image--------
                #1- reshape u as a third order tensor
                #s = truncated_U.shape[0] //2
                #t = truncated_U.shape[1]
                #truncated_U = np.reshape(truncated_U, (s, s, t))
                #2- mode_1 (vector) product to get a matrix 
                #phi_reduced = np.dot(truncated_U.transpose(),x)
                #3- inner product to get a vector
                phi_xy = np.outer(x,y).flatten()
                phi_reduced = np.dot(truncated_U.T,phi_xy)

                #append result to the new phi
                coarse_grained.append(phi_reduced)

                i +=2
            Phi_new.append(np.array(coarse_grained))
            
    #Phi is a tensor starting from the second layer     
    elif type(Phi) is list:
        iterates = len(Phi[0])  #layer1 len(Phi[0][0])=32
        for imageidx in range(len(Phi)):
            coarse_grained = []
            i = 0
            while True: #go from o to 32 reduced vectors of Phi(image)
                if i == iterates: break 
                ind1 = i 
                ind2 = i + 1
                x = Phi[imageidx][ind1] #local feature vector1 type: numpy.ndarray
                y = Phi[imageidx][ind2] #local feature vector2
                truncated_U = tree_tensor[layer, ind1, ind2] #get layer of isometries
                
                #-------Apply U to image--------
                #1- reshape u as a third order tensor
                #s1 = len(x)
                #s2 = len(y)
                #t = truncated_U.shape[1]
                #truncated_U = np.reshape(truncated_U, (s1, s2, t))
                #2- mode_1 (vector) product to get a matrix 
                #phi_reduced = np.dot(truncated_U.transpose(),x)
                #3- inner product to get a vector
                phi_xy = np.outer(x,y).flatten()
                phi_reduced = np.dot(truncated_U.T,phi_xy)
                #append result to the new phi
                coarse_grained.append(phi_reduced)

                i +=2
            Phi_new.append(np.array(coarse_grained))
        
    print('     ---> new Phi: N=' + str(len(Phi_new[0])))
    print('   ----> END generate new phi: ')
    return Phi_new


def local_feature_vectors(vector):
    """ Transform a vector representing an image to a matrix where the first row=[1,1,...,1] 
        and the elements of the second row are the elements of the vector  """

    N = HEIGHT * WIDTH
    phi = np.ones((2, N))
    phi[1, :] = np.squeeze(vector)
    norm = np.linalg.norm(phi, axis=0)
    norm = norm.reshape((1,len(norm)))
    phi /=norm

    return phi.T

def custom_feature(data_loader, fake_img=True):
    """ For each image: 
            Transform each pixel of each image to a vector of dimension 2 """
    
    #dimensions of feature tensor Phi
    dim1 = len(data_loader) #number of images

    dim2 = HEIGHT * WIDTH
    dim3 = 2 
    
    Phi = np.zeros((dim1, dim2, dim3))
   
    for batch_idx, (x, target) in enumerate(data_loader):
        if fake_img:
            # Expand
            x = x[None, :]

        image = x[0, 0, :, :]
        image = image.flatten() #vectorize the image
        image = local_feature_vectors(image)
        Phi[batch_idx, :, :] = image

    return Phi

def reduced_covariance(Phi, s1, s2):
    """ Compute the reduced covariance matrix given the two position (s1,s2) in feature matrix of an image.
        Example: to compute the reduced covariance matrix ro34, s1=2 and s2=3"""
    
    #Phi is a tensor as in the case for the first layer
    if type(Phi) is np.ndarray: 
        Nt = Phi.shape[0]      #number of images
        N = Phi.shape[1]       #number of local features vectors in Phi
        d = FEATURE_MAP_D

        ro = np.zeros((d**2,d**2))

        n_images = 0
        for j in range(Nt):
            if j == 1000: #compute the reduced covariance matrix using 1000 images
                break

            n_images += 1
            
            #get the two local feature vectors 
            phi1 = Phi[j, s1, :]
            phi2 = Phi[j, s2, :]

            #trace over all the indices except s1 and s2 
            trace_tracker = 1
            for s in range(N):
                if s != s1 and s != s2:
                    x = Phi[j, s, :]   
                    # outer_product = np.outer(x, x) 
                    # trace_tracker *= np.trace(outer_product)
                    trace_tracker *= np.inner(x, x)
                    #trace_tracker += np.inner(x, x)


            #compute the order 4 tensor
            phi12 = np.outer(phi1, phi2).flatten() 
            ro_j = np.outer(phi12,phi12)
            
            #add result to ro
            ro += trace_tracker*ro_j
        return ro / n_images
            
    #Phi is a list starting from the second layer
    elif type(Phi) is list:
        Nt = len(Phi)     #number of images
        N = len(Phi[0])       #number of local features vectors in Phi
        
        #---------get d1 and d2
        #get the two local feature vectors 
        phi1 = Phi[0][s1]
        phi2 = Phi[0][s2]
        #trace over all the indices except s1 and s2
        trace_tracker = 1
        for s in range(N):
            if s != s1 and s != s2:
                x = Phi[0][s]   
                # outer_product = np.outer(x, x) 
                trace_tracker *= np.inner(x,x)
                # trace_tracker += np.inner(x, x)
                #trace_tracker *= np.trace(outer_product)
                #trace_tracker += np.inner(x, x)


        #compute the order 4 tensor
        phi12 = np.outer(phi1, phi2).flatten() 
        ro_j = np.outer(phi12, phi12)
        
        #d1 = ro_j.shape[0]
        #d2 = ro_j.shape[1]
        n_images = 1
        
        #--------compute the rest
        #ro = np.zeros((d1,d2))
        ro = trace_tracker*ro_j;
        j = 1
        while True:
            if j == Nt: break
                
            n_images += 1
            
            #get the two local feature vectors 
            phi1 = Phi[j][s1]
            phi2 = Phi[j][s2]
            
            #trace over all the indices except s1 and s2
            trace_tracker = 1
            for s in range(N):
                if s != s1 and s != s2:
                    x = Phi[j][s]   
                    # outer_product = np.outer(x, x) 
                    # trace_tracker *= np.trace(outer_product)
                    trace_tracker *= np.inner(x, x)
                    #trace_tracker += np.inner(x, x)

                    
            #compute the order 4 tensor
            phi12 = np.outer(phi1, phi2).flatten()
            ro_j = np.outer(phi12, phi12)
            
            #add result to ro
            ro += trace_tracker*ro_j
            
            j += 1
            
        return ro / n_images


################################### Test ###############################################################

parser = argparse.ArgumentParser()
parser.add_argument("--fake", action="store_true")

args = parser.parse_args()

train_loader = None
Phi = None

if not args.fake:
    # #test load mnist
    train_loader, _ = loadMnist()
        
    HEIGHT = 28
    WIDTH = 28

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    # print('==>>> total testing batch number: {}'.format(len(test_loader)))

    Phi = custom_feature(train_loader, fake_img=False)
    print(Phi.shape)


else: 
# # Fake images for faster testing
    N_FAKE_IMGS = 100
    HEIGHT = 8
    WIDTH = 8

    np.random.seed(30)
    train_loader = np.random.random((N_FAKE_IMGS, 1, HEIGHT, WIDTH))
    #test feature map
    Phi = custom_feature(zip(train_loader, np.random.random(N_FAKE_IMGS)))

tree_depth = int(math.log2(HEIGHT * WIDTH)) -1
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
        #print('reduced ro shape' +str(ro.shape))
        #svd
        e_val, U = np.linalg.eigh(ro) # eigenvalues arranged in ascending order
        e_val, U = np.flip(e_val), np.flip(U, axis=1) # eigenvalues arranged in descending order
        #print("indices: ({}, {})\nU\n{}\nS{}\nV{}\n".format(ind1, ind2, u, s, v))
        #---------OLD eigenvalues = s**2
        #trace = np.sum(e_val)
        eigenvalues = s**2
        trace = np.sum(eigenvalues)

        truncation_sum = 0
        # Gross notation, but makes indexing nicer
        first_truncated_eigenvalue = 0

        for eig_idx, e in enumerate(e_val):
            truncation_sum += e
            first_truncated_eigenvalue += 1

            if (truncation_sum / trace) > (1 - TRUNCATION_EPS):
                break

        print(len(e_val), first_truncated_eigenvalue)
        print(len(eigenvalues), first_truncated_eigenvalue)
        print(trace)
        
        #truncation
        truncated_U = U[:, :first_truncated_eigenvalue] # keep first r cols of U

        #store U
        tree_tensor[layer, ind1, ind2] = truncated_U
        
    #compute new feature map
    Phi = generate_new_phi(Phi, tree_tensor, layer)
    #update number of local feature vectors for each image
    iterates = iterates // 2 
    
    
#print(tree_tensor[4,0,1].shape)
#print(type(tree_tensor[4,0,1]))

"""
For each pair of indices,
calculate ro
svd
truncation 
store U
"""




