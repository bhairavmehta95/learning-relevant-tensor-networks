import numpy as np
from _constants import FEATURE_MAP_D, HEIGHT, WIDTH
from datasets.parse_image import ImageParser
from math import cos, sin, pi

def local_feature_vectors(vector, feature_type):
    """ 
    Apply a local feature map to each pixel.
    INPUT: 
        vector: contains all the pixel of the images.
        feature_type: 
            It can take the values 'default' and 'cossin'. 
            If 'default': Transform a vector representing an image to a matrix 
                          where the first row = [1,1,...,1] and the elements of 
                          the second row are the elements of the vector
            If 'cossin': Transform each pixel x contained in vector to a two dimensional 
                         vector = [cos(0.5*pi*x), sin(0.5*pi*x)] 
    OUTPUT:
        phi: the new representation of the image after applying the local features maps
    """

    N = HEIGHT * WIDTH
    phi = np.ones((2, N))

    if feature_type=='default': #transform a 
        phi[1, :] = np.squeeze(vector)
        norm = np.linalg.norm(phi, axis=0)
        norm = norm.reshape((1,len(norm)))
        phi /=  norm

    if feature_type=='cossin':
        for i in range(N):
            phi[0,i] = cos(0.5*pi*vector[i])
            phi[1,i] = sin(0.5*pi*vector[i])
    

    return phi.T

def custom_feature(data_loader, batch_size, parser_type='default', feature_type='default', fake_img=True):
    """ For each image: 
            Transform each pixel of each image to a vector of dimension 2 """
    
    #dimensions of feature tensor Phi
    dim1 = batch_size #number of images
    dim2 = HEIGHT * WIDTH
    dim3 = FEATURE_MAP_D 
    
    Phi = np.zeros((dim1, dim2, dim3))
   
    for batch_idx, (x, target) in enumerate(data_loader):
        if batch_idx == batch_size:
            break
        image = x[0, 0, :, :]

        parser = ImageParser(image, parser_type)
        image = parser.parse()
        image = image.flatten() #vectorize the image
        
        image = local_feature_vectors(image, feature_type)
        Phi[batch_idx, :, :] = image

    return Phi