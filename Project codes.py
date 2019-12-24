import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python
import random 
import scipy
from scipy.spatial import distance
from scipy.stats import multivariate_normal
import pandas as pd

def read_data(filename, is_RGB, visualize=False, save=False, save_name=None):
# read the text data file
#   data, image = read_data(filename, is_RGB) read the data file named 
#   filename. Return the data matrix with same shape as data in the file. 
#   If is_RGB is False, the data will be regarded as Lab and convert to  
#   RGB format to visualise and save.
#
#   data, image = read_data(filename, is_RGB, visualize)  
#   If visualize is True, the data will be shown. Default value is False.
#
#   data, image = read_data(filename, is_RGB, visualize, save)  
#   If save is True, the image will be saved in an jpg image with same name
#   as the text filename. Default value is False.
#
#   data, image = read_data(filename, is_RGB, visualize, save, save_name)  
#   The image filename.
#
#   Example: data, image = read_data("1_noise.txt", True)
#   Example: data, image = read_data("cow.txt", False, True, True, "segmented_cow.jpg")

	with open(filename, "r") as f:
		lines = f.readlines()

	data = []

	for line in lines:
		data.append(list(map(float, line.split(" "))))

	data = np.asarray(data).astype(np.float32)

	N, D = data.shape

	cols = int(data[-1, 0] + 1)
	rows = int(data[-1, 1] + 1)
	channels = D - 2
	img_data = data[:, 2:]

	# In numpy, transforming 1d array to 2d is in row-major order, which is different from the way image data is organized.
	image = np.reshape(img_data, [cols, rows, channels]).transpose((1, 0, 2))

	if visualize:
		if channels == 1:
			# for visualizing grayscale image
			cv2.imshow("", image)
		else:
			# for visualizing RGB image
			cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_Lab2BGR))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if save:
		if save_name is None:
			save_name = filename[:-4] + ".jpg"
		assert save_name.endswith(".jpg") or save_name.endswith(".png"), "Please specify the file type in suffix in 'save_name'!"

		if channels == 1:
			# for saving grayscale image
			cv2.imwrite(save_name, image)
		else:
			# for saving RGB image
			cv2.imwrite(save_name, (cv2.cvtColor(image, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))

	if not is_RGB:
		image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

	return data, image

################################################################################################################
    
#Gibbs Sampling

#Read the image
output = read_data("4_noise.txt",is_RGB = True)

data = output[0]
image = output[1]

#Convert image values to ising model
for i in range(len(image)):
    for j in range(len(image[0])):
        if image[i,j,:] == 255:
            image[i,j,:] = 1
        else:
            image[i,j,:] = -1

#Create array to perform opearations on
ising = np.zeros((len(image)+2,len(image[0])+2))

for i in range(len(image)):
    for j in range(len(image[0])):
        ising[i+1,j+1] = image[i,j,:]

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
            prob1 = multivariate_normal.pdf(image[j-1,i-1,:], mean = 1, cov = 1)*pot[1]/(multivariate_normal.pdf(image[j-1,i-1,:], mean = 1, cov = 1)*pot[1] + multivariate_normal.pdf(image[j-1,i-1,:], mean = -1, cov = 1)*pot[0]) 
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

#Visualize denoised image      
cv2.imshow("", image_denoise)           
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("4_denoise.png", image_denoise)


#############################################################################################################            
  
#Variational Inference

#Read in the image
output = read_data("4_noise.txt",is_RGB = True)

data = output[0]
image = output[1]

#Convert image values to ising model
for i in range(len(image)):
    for j in range(len(image[0])):
        if image[i,j,:] == 255:
            image[i,j,:] = 1
        else:
            image[i,j,:] = -1

#Create arrays to perform opearations on
ising = np.zeros((len(image)+2,len(image[0])+2))

for i in range(len(image)):
    for j in range(len(image[0])):
        ising[i+1,j+1] = image[i,j,:]
       
var = np.zeros((len(image)+2,len(image[0])+2))

for i in range(len(image)):
    for j in range(len(image[0])):
        var[i+1,j+1] = image[i,j,:]

#Coupling sterngth        
J = 1

#Variational Inference        
for n in range(15):
    mean_field = np.zeros((len(image),len(image[0])))
    ising = var.copy()
    for i in range(1,len(ising)-1):
        for j in range(1,len(ising[0])-1):
            mean_field[i-1,j-1] = J*ising[i-1,j]+ J*ising[i,j-1] + J*ising[i+1,j] + J*ising[i,j+1]
            log_pdf = multivariate_normal.logpdf(ising[i,j], mean=-1, cov=4) - multivariate_normal.logpdf(ising[i,j], mean=1, cov=4)
            approx_1 = 1/(1 + np.exp(-2*mean_field[i-1,j-1] + log_pdf))
            approx_2 = 1/(1 + np.exp(2*mean_field[i-1,j-1] - log_pdf))
            var[i,j] = approx_1 - approx_2

image2 = np.zeros((len(image),len(image[0])))     

for i in range(len(image2)):
    for j in range(len(image2[0])):
        image2[i,j] = var[i+1,j+1]

#Converting it back to image
for i in range(len(image2)):
    for j in range(len(image2[0])):
        if np.round(image2[i,j]) == 1:
            image2[i,j] = 255
        else:
            image2[i,j] = 0

#Converting to 3-D array            
image_denoise = np.reshape(image2, [len(image2), len(image2[0]), 1]).transpose((0, 1, 2))

#Visualize denoised image      
cv2.imshow("", image_denoise)           
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("4_denoise_vi.png", image_denoise)
     

###############################################################################################################

#Expectation Maximisation Algorithm

#Read in the image
output = read_data("owl.txt",is_RGB = False)

data = output[0]
image = output[1]

#Initializing with random parameters
centroid_old = []
center1 = data[random.randint(0,len(data)),2:5]
center2 = data[random.randint(0,len(data)),2:5]
centroid_old.append(center1)
centroid_old.append(center2)
centroid_old = np.asarray(centroid_old).astype(np.float32)

covar_old = []
covar1 = [[np.var(data[:len(data)//2,2]),0,0],[0,np.var(data[:len(data)//2,3]),0],[0,0,np.var(data[:len(data)//2,4])]]
covar2 = [[np.var(data[len(data)//2:,2]),0,0],[0,np.var(data[len(data)//2:,3]),0],[0,0,np.var(data[len(data)//2:,4])]]
covar_old.append(covar1)
covar_old.append(covar2)
covar_old = np.asarray(covar_old).astype(np.float32)

mix_old = []
mix1 = np.random.uniform()
mix2 = 1 - mix1
mix_old.append(mix1)
mix_old.append(mix2)
mix_old = np.asarray(mix_old).astype(np.float32)

log_likelihood_old = 0
convergence = 10

while abs(convergence) > 0.1:
    #Expectation step
    
    #Calculating Responsibility
    resp = np.zeros((len(data),2))
    resp = np.asarray(resp).astype(np.float64)
    
    for n in range(len(data)):
        for k in range(2):
            resp[n,k] = mix_old[k]*multivariate_normal.pdf(data[n,2:5], mean = centroid_old[k,:], cov = covar_old[k,:]) 
        resp[n,:] /= resp[n,:].sum()    
    
    #Maximization step
    
    #Calculating new mean
    centroid_new = np.zeros((2,3))
    centroid_new = np.asarray(centroid_new).astype(np.float32)
    
    for k in range(2):            
        for n in range(len(resp)):
            centroid_new[k,:] += resp[n,k]*data[n,2:5]
        centroid_new[k,:] /= resp[:,k].sum()     
    
    #Calculating new mixing factor 
    mix_new = np.zeros((2))
    mix_new = np.asarray(mix_new).astype(np.float32)
    
    for k in range(2):    
        for n in range(len(resp)):
            mix_new[k] += resp[n,k]
        mix_new[k] /= len(resp)             
    
    #Calculating new covariance
    covar_new = np.zeros((2,3,3))
    covar_new = np.asarray(covar_new).astype(np.float32)
    
    for k in range(2):    
        for n in range(len(resp)):
            diff = np.reshape(data[n,2:5]-centroid_new[k,:], (3,1))
            covar_new[k] += resp[n,k]*diff*diff.T
        covar_new[k] /= resp[:,k].sum()
    
    #Calculating log likelihood
    log_likelihood_new = 0
      
    for n in range(len(data)):
        likelihood = 0
        for k in range(2):
            likelihood += mix_new[k]*multivariate_normal.pdf(data[n,2:5], mean = centroid_new[k,:], cov = covar_new[k,:]) 
        log_likelihood_new += np.log(likelihood)
    
    convergence = log_likelihood_new - log_likelihood_old
    
    log_likelihood_old = log_likelihood_new.copy()
    
    centroid_old = centroid_new.copy()
    mix_old = mix_new.copy()
    covar_old = covar_new.copy()

#Segmenting images
assign = np.zeros((len(data)))

for n in range(len(resp)):
    if resp[n,0] > resp[n,1]:
        assign[n] = 1
    else:
        assign[n] = 2


#Image segment 1
seg_image1 = data[:,2:5].copy() 

for n in range(len(seg_image1)):
    if assign[n] == 1:
        seg_image1[n,:] = 0

image_seg1 = np.reshape(seg_image1, [len(image[0]), len(image), 3]).transpose((1, 0, 2))

cv2.imshow("", cv2.cvtColor(image_seg1, cv2.COLOR_Lab2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("owl_seg1.png", (cv2.cvtColor(image_seg1, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))

#Image segment 2
seg_image2 = data[:,2:5].copy() 

for n in range(len(seg_image2)):
    if assign[n] == 2:
        seg_image2[n,:] = 0

image_seg2 = np.reshape(seg_image2, [len(image[0]), len(image), 3]).transpose((1, 0, 2))

cv2.imshow("", cv2.cvtColor(image_seg2, cv2.COLOR_Lab2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("owl_seg2.png", (cv2.cvtColor(image_seg2, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))

#Both segments masked
seg_image3 = data[:,2:5].copy() 

seg_image3[:,:] = 0

for n in range(len(seg_image3)):
    if assign[n] == 1:
        seg_image3[n,0] = 100


image_seg3 = np.reshape(seg_image3, [len(image[0]), len(image), 3]).transpose((1, 0, 2))

cv2.imshow("", cv2.cvtColor(image_seg3, cv2.COLOR_Lab2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("owl_seg3.png", (cv2.cvtColor(image_seg3, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))


################################################################################################################
            