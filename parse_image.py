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


class ParseImage:
    """Class: Provide different functions to parse an image represented as a 2D tensor."""
    def __init__(self, image):
        self.image = image
        
    def default(self):
        """ 
        Parse an image row by row.
        OUTPUT:
            vector representing the image 
        """
        image = self.image
          
        image_vector = image.flatten()
        
        return image_vector
    
    def columnParser(self):
        """ 
        Parse an image column by column.
        OUTPUT:
            vector representing the image 
        """
        image = self.image
        
        image = torch.transpose(image,0,1)
        image_vector = image.flatten()
        
        return image_vector
        
    def spiralParser(self):
        """ 
        Parse a square image in spiral.
        OUTPUT:
            vector representing the image 
        """
        image = self.image
        
        #get dimensions
        m, n = image.shape
        
        #initialize
        image_vector = torch.zeros(n*m)
        
        #begin spiral
        k = 0
        l = 0
        index = 0
        while (k < m and l < n) : 
            # add the first row from 
            # the remaining rows  
            for i in range(l, n) : 
                image_vector[index] = image[k,i]
                index = index + 1
            k += 1
            
            # add the last column from 
            # the remaining columns  
            for i in range(k, m) : 
                image_vector[index] = image[i,(n - 1)]
                index = index + 1
            n -= 1
            
            # add the last row from 
            # the remaining rows  
            if ( k < m) : 
                for i in range(n - 1, (l - 1), -1) :
                    image_vector[index] = image[(m - 1),i]
                    index = index + 1
                m -= 1
         
            # add the first column from 
            # the remaining columns  
            if (l < n) : 
                for i in range(m - 1, k - 1, -1) :
                    image_vector[index] = image[i,l] 
                    index = index + 1
                l += 1
        
        return image_vector
        
    def blockParser(self, window_dimension = 2):
        """ 
        Parse a square image in spiral.
        OUTPUT:
            vector representing the image 
        """
        image = self.image
        
        #get dimensions
        m, n = image.shape
        
        #initialize
        image_vector = torch.zeros(n*m)
        
        #begin block
        i = 0
        index = 0
        for row in range(int(m/window_dimension)):
            j = 0
            for column in range(int(n/window_dimension)):
                image_vector[index:(index+window_dimension**2)] = image[i:i+window_dimension,j:j+window_dimension].flatten()
                j = j+window_dimension
                index = index + window_dimension**2
            i = i+window_dimension
        
        return image_vector