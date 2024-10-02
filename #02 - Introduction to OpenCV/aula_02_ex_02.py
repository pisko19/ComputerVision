 # Aula_01_ex_02.py
 #
 # Pedro Afonso

#import
import numpy as np
import cv2
import sys

# Read the image
image1 = cv2.imread( "../images/deti.bmp", cv2.IMREAD_UNCHANGED )
image2 = cv2.imread( "../images/deti.jpg", cv2.IMREAD_UNCHANGED )

if  np.shape(image1) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)
	
if  np.shape(image2) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

if image1.shape != image2.shape:
	print("Formato diferente")
	exit(-1)

subtracao = cv2.subtract(image1,image2)


# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size

# Show the image
cv2.imshow( "Display Image1", image1 )
cv2.imshow( "Display Image2", image2 )
cv2.imshow( "Display Subtract", subtracao )

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
