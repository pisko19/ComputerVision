# aula_03_ex_04.py
#
# Contrast-Stretching with Histogram visualization
#
# Paulo Dias

#import
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image from argv
image = cv2.imread("../images/input.png", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("../images/input.png", cv2.IMREAD_GRAYSCALE)  # Uncomment to use 'input.png'

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

# Find the min and max intensity values using minMaxLoc
min_val, max_val, _, _ = cv2.minMaxLoc(image)
print(f"Min intensity value: {min_val}")
print(f"Max intensity value: {max_val}")

# Apply Contrast Stretching
contrast_stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Display the contrast-stretched image
cv2.imshow("Contrast-Stretched Image", contrast_stretched)

# Compute histograms for both images
histSize = 256  # From 0 to 255
histRange = [0, 256]

# Original image histogram
hist_original = cv2.calcHist([image], [0], None, [histSize], histRange)

# Contrast-stretched image histogram
hist_stretched = cv2.calcHist([contrast_stretched], [0], None, [histSize], histRange)

# Display histograms using OpenCV (for original image)
histImageWidth = 512
histImageHeight = 512
binWidth = int(np.ceil(histImageWidth * 1.0 / histSize))
histImage_original = np.zeros((histImageHeight, histImageWidth), np.uint8)

# Normalize the histogram to fit within the height of the histogram image
cv2.normalize(hist_original, hist_original, 0, histImageHeight, cv2.NORM_MINMAX)

# Draw the original image histogram
for i in range(histSize):
    cv2.rectangle(histImage_original, (i * binWidth, histImageHeight),
                  ((i + 1) * binWidth, histImageHeight - int(hist_original[i])),
                  (125), -1)

cv2.imshow("Original Image Histogram", histImage_original)

# Display histograms using OpenCV (for contrast-stretched image)
histImage_stretched = np.zeros((histImageHeight, histImageWidth), np.uint8)
cv2.normalize(hist_stretched, hist_stretched, 0, histImageHeight, cv2.NORM_MINMAX)

# Draw the contrast-stretched image histogram
for i in range(histSize):
    cv2.rectangle(histImage_stretched, (i * binWidth, histImageHeight),
                  ((i + 1) * binWidth, histImageHeight - int(hist_stretched[i])),
                  (125), -1)

cv2.imshow("Contrast-Stretched Image Histogram", histImage_stretched)

# Alternatively, display histograms using matplotlib
plt.figure()
plt.title("Histogram - Original Image")
plt.plot(hist_original, color='r')
plt.xlim(histRange)
plt.show()

plt.figure()
plt.title("Histogram - Contrast-Stretched Image")
plt.plot(hist_stretched, color='r')
plt.xlim(histRange)
plt.show()

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
