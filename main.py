import numpy as np
import cv2
import os
import imutils
from tkinter import Tk
from tkinter.filedialog import askopenfilename


    
# defining the canny detector function
# here weak_th and strong_th are thresholds for
# double thresholding step
def Edge_detector(img, weak_th = None, strong_th = None):
	
	# conversion of image to grayscale
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Noise reduction step
	img = cv2.GaussianBlur(img, (5, 5), 1.4)
	# Calculating the gradients
	gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
	gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
    # Define Sobel kernels
    
    # Compute magnitude and orientation of gradients
   
	# Conversion of Cartesian coordinates to polar
	mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
	
	# setting the minimum and maximum thresholds
	# for double thresholding
	mag_max = np.max(mag)
	if not weak_th:weak_th = mag_max * 0.1
	if not strong_th:strong_th = mag_max * 0.5
	
	# getting the dimensions of the input image
	height, width = img.shape
	
	# Looping through every pixel of the grayscale
	# image
	for i_x in range(width):
		for i_y in range(height):
			
			grad_ang = ang[i_y, i_x]
			grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
			
			# selecting the neighbours of the target pixel
			# according to the gradient direction
			# In the x axis direction
			if grad_ang<= 22.5:
				neighb_1_x, neighb_1_y = i_x-1, i_y
				neighb_2_x, neighb_2_y = i_x + 1, i_y
			
			# top right (diagonal-1) direction
			elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
				neighb_1_x, neighb_1_y = i_x-1, i_y-1
				neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
			
			# In y-axis direction
			elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
				neighb_1_x, neighb_1_y = i_x, i_y-1
				neighb_2_x, neighb_2_y = i_x, i_y + 1
			
			# top left (diagonal-2) direction
			elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
				neighb_1_x, neighb_1_y = i_x-1, i_y + 1
				neighb_2_x, neighb_2_y = i_x + 1, i_y-1
			
			# Now it restarts the cycle
			elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
				neighb_1_x, neighb_1_y = i_x-1, i_y
				neighb_2_x, neighb_2_y = i_x + 1, i_y
			
			# Non-maximum suppression step
			if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
				if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
					mag[i_y, i_x]= 0
					continue

			if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
				if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
					mag[i_y, i_x]= 0

	weak_ids = np.zeros_like(img)
	strong_ids = np.zeros_like(img)			
	ids = np.zeros_like(img)
	
	# double thresholding step
	for i_x in range(width):
		for i_y in range(height):
			
			grad_mag = mag[i_y, i_x]
			
			if grad_mag<weak_th:
				mag[i_y, i_x]= 0
			elif strong_th>grad_mag>= weak_th:
				ids[i_y, i_x]= 1
			else:
				ids[i_y, i_x]= 2
	
	
	# finally returning the magnitude of
	# gradients of edges
	return mag



Tk().withdraw()
filename = askopenfilename() 

#image = cv2.imread("1.jpg")
image = cv2.imread(filename)
image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)
cv2.waitKey(0)
# Reading the image
edges = Edge_detector(image,70,100)
cv2.imshow("Edged", edges)
cv2.waitKey(0)

#img = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
img = np.uint8(edges * 255)
contours, heirarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img1 = image.copy()
cv2.drawContours(img1, contours, -1, (0,255,0), 1)
cv2.imshow("All of the contours", img1)
cv2.waitKey(0)

contours=sorted(contours, key = cv2.contourArea, reverse = True)[:50]
Number_Plate_Contour = 0

for current_contour in contours:        
    perimeter = cv2.arcLength(current_contour, True)
    approx = cv2.approxPolyDP(current_contour, 0.02*perimeter, True) 
    if len(approx) == 4:  
           Number_Plate_Contour = approx 
           break 
 
print(Number_Plate_Contour) 
mask = np.zeros(img.shape,np.uint8)
new_image1 = cv2.drawContours(mask,[Number_Plate_Contour],0,255,-1,)
new_image1 =cv2.bitwise_and(image,image,mask=mask)
cv2.imshow("Number Plate",new_image1)
cv2.waitKey(0)

gray_scaled1 = cv2.cvtColor(new_image1, cv2.COLOR_BGR2GRAY)
ret,processed_img = cv2.threshold(np.array(gray_scaled1), 125, 255, cv2.THRESH_BINARY)
cv2.imshow("Number Plate",processed_img)
cv2.waitKey(0)

#image proccessing part start       