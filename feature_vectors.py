import numpy as np
from _constants import FEATURE_MAP_D, HEIGHT, WIDTH

def local_feature_vectors(vector):
    """ Transform a vector representing an image to a matrix where the first row=[1,1,...,1] 
        and the elements of the second row are the elements of the vector  """

    N = HEIGHT * WIDTH
    phi = np.ones((2, N))
    phi[1, :] = np.squeeze(vector)
    norm = np.linalg.norm(phi, axis=0)
    norm = norm.reshape((1,len(norm)))
    phi /=  norm

    return phi.T

def custom_feature(data_loader, batch_size, fake_img=True):
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
        image = image.flatten() #vectorize the image
        image = local_feature_vectors(image)
        Phi[batch_idx, :, :] = image

    return Phi