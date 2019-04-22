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

FEATURE_MAP_D = 2
TRUNCATION_EPS = 1e-2


def loadMnist(batch_size=1):
    """ Load MNIST dataset"""

    #transform input/output to tensor
    transform = transforms.Compose([
        transforms.Pad(2), # Makes 32x32, Log2(32) = 5  
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
    dim1 = N_MAX #number of images

    dim2 = HEIGHT * WIDTH
    dim3 = 2 
    
    Phi = np.zeros((dim1, dim2, dim3))
   
    for batch_idx, (x, target) in enumerate(data_loader):
        if batch_idx ==N_MAX:
            break
        image = x[0, 0, :, :]
        image = image.flatten() #vectorize the image
        image = local_feature_vectors(image)
        Phi[batch_idx, :, :] = image
    
    return Phi


def generate_new_phi(Phi, tree_tensor, layer):
    
    print('   ----> generate new phi: ')
    Phi_new = [] #layer 0: list because the dimension of each d may differt. so we can't have a tensor of (100,32,d)
    
    #Phi is a tensor as in the case for the first layer
    if type(Phi) is np.ndarray:      
        iterates = Phi.shape[1]  
        
        for imageidx, image in enumerate(Phi):
            coarse_grained = []
            i = 0
            while True: #go from o to 32 reduced vectors of Phi(image)
                if i == iterates: break 
                ind1 = i 
                ind2 = i + 1
                x = Phi[imageidx, ind1, :] #local feature vector1 
                #print('x shape: ' +str(x.shape))
                y = Phi[imageidx, ind2, :] #local feature vector2
                #print('y shape: ' +str(y.shape))
                truncated_U = tree_tensor[layer, ind1, ind2] #get layer of isometries

                #-------Apply U to image--------
                phi_xy = np.outer(x,y).flatten()
                phi_reduced = np.dot(truncated_U.T,phi_xy)

                #append result to the new phi
                coarse_grained.append(phi_reduced)

                i +=2
            Phi_new.append(np.array(coarse_grained))
            
    #Phi is a tensor starting from the second layer     
    elif type(Phi) is list:
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
HEIGHT = 32
WIDTH = 32
N_MAX = 200
iterates = HEIGHT * WIDTH
tree_depth = int(math.log2(HEIGHT * WIDTH)) -1

#load mnist
train_loader, test_loader = loadMnist()

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
