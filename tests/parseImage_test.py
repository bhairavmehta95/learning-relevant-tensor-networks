#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
import torch
import parseImage 


# In[2]:


#create random tensor
torch.manual_seed(30)
n = 4;
image = torch.rand(n,n)
print(image)
print(image[1,:])

#call ParseImage class
parser = parseImage.ParseImage(image)

#call default function
print('default')
vector = parser.default()
print(vector)

#call columnParser function
print('columnParser')
vector = parser.columnParser()
print(vector)

#call spiralParser function
print('spiralParser')
vector = parser.spiralParser()
print(vector)

#call blockParser function
print('blockParser')
vector = parser.blockParser()
print(vector)


# In[ ]:




