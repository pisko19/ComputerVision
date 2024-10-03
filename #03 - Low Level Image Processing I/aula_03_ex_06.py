# aula_03_exe_06.py
#
# RGB and Gray-Level Histograms Visualization
#
# Paulo Dias

#import necessary libraries
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the RGB image
image = cv2.imread("../images/Fruits-RGB.tif", cv2.IMREAD_COLOR)

if image is None:
    print("Image file could not be opened!")
    exit(-1)

# Split the image into its Red, Green, and Blue components
b, g, r = cv2.split(image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the original RGB image and grayscale image
cv2.imshow("Original RGB Image", image)
cv2.imshow("Gray-Level Image", gray_image)

# Compute histograms for each color channel (Blue, Green, Red)
histSize = 256  # From 0 to 255
histRange = [0, 256]

# Compute histograms for the B, G, R components
hist_b = cv2.calcHist([b], [0], None, [histSize], histRange)
hist_g = cv2.calcHist([g], [0], None, [histSize], histRange)
hist_r = cv2.calcHist([r], [0], None, [histSize], histRange)

# Compute histogram for the grayscale image
hist_gray = cv2.calcHist([gray_image], [0], None, [histSize], histRange)

# Normalize histograms
cv2.normalize(hist_b, hist_b, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(hist_g, hist_g, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(hist_r, hist_r, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(hist_gray, hist_gray, 0, 255, cv2.NORM_MINMAX)

# Plot histograms using matplotlib

# Create a figure for RGB histograms
plt.figure()

# Plot the histogram for the Blue component
plt.subplot(2, 2, 1)
plt.plot(hist_b, color='b')
plt.xlim(histRange)
plt.title("Histogram - Blue Channel")

# Plot the histogram for the Green component
plt.subplot(2, 2, 2)
plt.plot(hist_g, color='g')
plt.xlim(histRange)
plt.title("Histogram - Green Channel")

# Plot the histogram for the Red component
plt.subplot(2, 2, 3)
plt.plot(hist_r, color='r')
plt.xlim(histRange)
plt.title("Histogram - Red Channel")

# Create a figure for the grayscale histogram
plt.subplot(2, 2, 4)
plt.plot(hist_gray, color='k')
plt.xlim(histRange)
plt.title("Histogram - Gray-Level Image")

# Show all the histograms
plt.tight_layout()
plt.show()

# Wait for key press to close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
