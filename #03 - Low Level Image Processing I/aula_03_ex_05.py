# aula_03_exe_05.py
#
# Histogram-Equalization with Histogram visualization
#
# Paulo Dias

#import
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread("../images/TAC_PULMAO.bmp", cv2.IMREAD_GRAYSCALE)

if image is None:
    # Failed Reading
    print("Image file could not be opened!")
    exit(-1)

# Ensure it is a gray-level image
if len(image.shape) > 2:
    print("The loaded image is NOT a GRAY-LEVEL image!")
    exit(-1)

# Display the original image
cv2.imshow("Original Image", image)

# Apply Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Display the equalized image
cv2.imshow("Equalized Image", equalized_image)

# Compute histograms for both images
histSize = 256  # From 0 to 255
histRange = [0, 256]

# Original image histogram
hist_original = cv2.calcHist([image], [0], None, [histSize], histRange)

# Equalized image histogram
hist_equalized = cv2.calcHist([equalized_image], [0], None, [histSize], histRange)

# Display histograms using OpenCV (for original image)
histImageWidth = 512
histImageHeight = 512
binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))

# Drawing the original image histogram
histImage_original = np.zeros((histImageHeight, histImageWidth), np.uint8)
cv2.normalize(hist_original, hist_original, 0, histImageHeight, cv2.NORM_MINMAX)

for i in range(histSize):
    cv2.rectangle(histImage_original, (i * binWidth, histImageHeight),
                  ((i + 1) * binWidth, histImageHeight - int(hist_original[i])),
                  (125), -1)

cv2.imshow("Original Image Histogram", histImage_original)

# Drawing the equalized image histogram
histImage_equalized = np.zeros((histImageHeight, histImageWidth), np.uint8)
cv2.normalize(hist_equalized, hist_equalized, 0, histImageHeight, cv2.NORM_MINMAX)

for i in range(histSize):
    cv2.rectangle(histImage_equalized, (i * binWidth, histImageHeight),
                  ((i + 1) * binWidth, histImageHeight - int(hist_equalized[i])),
                  (125), -1)

cv2.imshow("Equalized Image Histogram", histImage_equalized)

# Alternatively, display histograms using matplotlib
plt.figure()
plt.title("Histogram - Original Image")
plt.plot(hist_original, color='r')
plt.xlim(histRange)
plt.show()

plt.figure()
plt.title("Histogram - Equalized Image")
plt.plot(hist_equalized, color='r')
plt.xlim(histRange)
plt.show()

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
