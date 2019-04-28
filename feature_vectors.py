import numpy as np

HEIGHT = 32
WIDTH = 32


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