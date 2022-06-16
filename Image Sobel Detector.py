
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import math 

from PIL import Image

import imageio

import time

#supplemental functions 


#starting by implementing convolution function focusing on kernel of the original image

def ConvolutionFunction(OriginalImage, Kernel):
	
	#the image.shape[0] of the original image represents its height 
	
	ImageHeight = OriginalImage.shape[0]
	
	#the image.shape[1] of the original image represents its width
	
	ImageWidth = OriginalImage.shape[1]
	
	#since kernel here represents a small 2d matrix to blur the original image 
	
	#We represents Kernel.shape[0] as its height, and Kernel.shape[1] as its width 
	
	KernelHeight = Kernel.shape[0]
	
	KernelWidth =  Kernel.shape[1]
	
	#pad numpy arrays within the image
	
	#we consider OriginalImage as an array 
	
	#if the grayscale image gives three element as the number of channels.
	
	
	if(len(OriginalImage.shape) == 3):
		
		PaddedImage = np.pad(OriginalImage, pad_width = ((KernelHeight // 2, KernelHeight // 2), 
		(KernelWidth//2, KernelWidth//2), (0,0)), mode='constant', constant_values=0).astype(np.float32)
		
		
		#if the grayscale image gives two element as the number of channels.
		
		
	elif (len(OriginalImage.shape) == 2):
		
		PaddedImage = np.pad(OriginalImage, pad_width = (( KernelHeight // 2,  KernelHeight // 2),
			(KernelWidth//2, KernelWidth//2)), mode='constant', constant_values=0).astype(np.float32)
		
		
	#floor division result quotient of Kernel height and width divides by 2 
		
	height = KernelHeight // 2
	
	width = KernelWidth // 2
	
	
	
	
	#initialize a new array of given shape and type, filled with zeros from padded image 
	
	ConvolvedImage = np.zeros(PaddedImage.shape)
	
	#sum = 0
	
	#iterate the image convolution as 2d array as well 
	
	for i in range(height, PaddedImage.shape[0] - height):
		
		for j in range(width, PaddedImage.shape[1] - width):
			
			
			#2D matrix indexes 
			
			x = PaddedImage[i - height:i-height + KernelHeight, j-width:j-width + KernelWidth]
			
			#use flaten() to return a copy of the array collapsed into one dimension.
			
			x = x.flatten() * Kernel.flatten()
			
			#pass the sum of the array elements into the convolved image matrix
			
			ConvolvedImage[i][j] = x.sum()
			
	#assign endpoints of height and width in the 2D matrix 
			
	HeightEndPoint = -height
	
	WidthEndPoint  = -width 
	
	#when there is no height, return [height, width = width end point] 
	
	if(height == 0):
		
		return ConvolvedImage[height:, width : WidthEndPoint]
	
	#when there is no width, return [height = height end point, width ] 
	
	if(width  == 0):
		
		return ConvolvedImage[height: HeightEndPoint, width:]
	
	#return the convolved image
	
	return ConvolvedImage[height: HeightEndPoint,  width: WidthEndPoint]




#SobelImage function implementation 




def SobelImage(image):
	
	image = Image.open(image).convert('L')
	
	image = np.asarray(image)
	
	FilteredImage = np.zeros_like(image, dtype=np.float32)
	
	X =  ConvolutionFunction(image, np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
	
	Y =  ConvolutionFunction(image, np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
	
	Magnitude = np.sqrt(X**2 + Y**2)
	
	
	Orientation = np.arctan(Y , X)
	
	
	return (Magnitude.astype(np.uint8), Orientation.astype(np.uint8))




#Driver/Testing code

#Required test:

#Compute Sobel edge filter on ”LadyBug.jpg” and save as ”5a.png” and ”5b.png”.


a = SobelImage('hw1_data/LadyBug.jpg')[0]

plt.imshow(a, cmap='gray')

plt.imsave('5a.png', a, cmap = 'gray')


b = SobelImage('hw1_data/LadyBug.jpg')[1]

plt.imshow(a, cmap= 'gray')

plt.imsave('5b.png',b, cmap= 'gray')

#out of range if greater than 1



#Written Assognment:


b = SobelImage('hw1_data/TightRope.png')[0]

plt.imshow(b, cmap='gray')

plt.imsave('S2Q3E0.png', b, cmap = 'gray')




c = SobelImage('hw1_data/TightRope.png')[1]

plt.imshow(c, cmap='gray')

plt.imsave('S2Q3E1.png', c, cmap = 'gray')



