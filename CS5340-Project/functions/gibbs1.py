# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:03:53 2018

@author: prana
"""

import numpy as np
import PIL
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
#%matplotlib inline
from skimage import io


def image_denoise(M, beta, sigma, url, burn=100):

    """ Utilizes Gibbs Sampling to denoise an image.
   
    Parameters
    ----------
    M: run length of interest
    beta: inverse temperature
    url: input image
    sigma: standard deviation
    burn: initial runs, ignored
 
    Output
    ------
	Y: original image
    (count/M): denoised pixel matrix
    """
    
	# prep original image
    image = io.imread(url)
    gray_image = (np.dot(image[...,:3], [0.299, 0.587, 0.114]))/255. 
    Y = (misc.imresize(gray_image, (100,100)))/255.
    
    L = Y.shape[1]
    count = np.matrix(np.zeros((L, L))) # keeps track of new pixels
    X = np.matrix(np.zeros((L+2, L+2))) # initial state

    # Ising model
    pixels = np.reshape(np.random.choice([-1,1], L*L), (L,L))
    X[1:L+1, 1:L+1] = pixels
    
    for sweep in range(burn + M):
        for ix in range(1,L+1):
            for iy in range(1,L+1):
                d = X[iy-1, ix] + X[iy+1, ix] + X[iy, ix+-1] + X[iy, ix+1] #assess neighboring pixels
                prob = 1. / (1 + np.exp(-2 * (beta * d + Y[iy-1, ix-1]/sigma**2))) #generate probabilities
                X[iy, ix] = np.random.choice((-1,1),p =[1-prob, prob]) #assign pixel value
    
        if sweep > burn:
            count = count + (X[1:L+1, 1:L+1]==1) #update new pixel matrix
            
    return Y, (count/M) # returns original and new matrix

img1, img2 = image_denoise(10, 0.1, 1, 'http://www.plasticbag.org/images/extra/grainy_one.jpg', burn=10)
plt.imshow(img1, cmap = plt.get_cmap('gray'))
plt.imshow(img2, cmap = plt.get_cmap('gray'))