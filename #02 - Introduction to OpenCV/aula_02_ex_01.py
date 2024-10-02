 # Aula_01_ex_01.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

#import
import numpy as np
import cv2
import sys

# Read the image
image = cv2.imread( "../images/Orchid.bmp", cv2.IMREAD_UNCHANGED )
image_copy = image.copy()

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

# Image characteristics
height, width, x = image.shape

print("Image Size: (%d,%d)" % (height, width))
print("Image Type: %s" % (image.dtype))

for i in range(height):
	for j in range(width):
		for k in range(x):
			if image_copy[i,j,k] < 128:
				image_copy[i,j,k] = 0


# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size

# Show the image
cv2.imshow( "Display window", image )
cv2.imshow( "Display Copy Image", image_copy )

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
