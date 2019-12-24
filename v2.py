# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:29:19 2019

@author: prana
"""

%matplotlib inline
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy.special import expit as sigmoid
from scipy.stats import multivariate_normal

np.random.seed(0)
sns.set_style('whitegrid')

#load data
data = Image.open('check2.gif')
image = np.asarray(data).astype(np.float32)

for i in range(len(image)):
    for j in range(len(image[0])):
        if image[i,j] == 255:
            image[i,j] = 1
        else:
            image[i,j] = -1

#Create array to perform opearations on
ising = np.zeros((len(image)+2,len(image[0])+2))

for i in range(len(image)):
    for j in range(len(image[0])):
        ising[i+1,j+1] = image[i,j]

#Coupling strength
J=4

#Gibbs sampling 
for n in range(3):
    for i in range(1,len(ising[0])-1):
        for j in range(1,len(ising)-1):
            pot = []
            for x in [-1, 1]:
                edge_pot = np.exp(J*ising[j-1,i]*x) * np.exp(J*ising[j,i-1]*x) * np.exp(J*ising[j+1,i]*x) * np.exp(J*ising[j,i+1]*x)
                pot.append(edge_pot)
            prob1 = multivariate_normal.pdf(image[j-1,i-1], mean = 1, cov = 1)*pot[1]/(multivariate_normal.pdf(image[j-1,i-1], mean = 1, cov = 1)*pot[1] + multivariate_normal.pdf(image[j-1,i-1], mean = -1, cov = 1)*pot[0]) 
            if np.random.uniform() <= prob1:
                ising[j,i] = 1
            else:
                ising[j,i] = -1

#Retrieving the final array
final = np.zeros((len(image),len(image[0])))
final = ising[1:len(ising)-1,1:len(ising[0])-1]

#Converting it back to image
for i in range(len(final[0])):
    for j in range(len(final)):
        if final[j,i] == 1:
            final[j,i] = 255
        else:
            final[j,i] = 0

#Converting to 3-D array            
image_denoise = np.reshape(final, [len(image), len(image[0]), 1]).transpose((0, 1, 2))

cv2.imshow("", image_denoise)           
cv2.waitKey(0)
cv2.destroyAllWindows()
