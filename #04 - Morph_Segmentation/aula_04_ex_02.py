import cv2
import numpy as np

def circular_kernel(diameter):
    radius = diameter // 2
    kernel = np.zeros((diameter, diameter), dtype=np.uint8)
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                kernel[i, j] = 1
    return kernel

# Load the image
image = cv2.imread("../images/wdg2.bmp")

# Check if the image was successfully read
if image is None:
    print("Image file could not be opened")
    exit(-1)

_, binary_image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)

# Step 2: Invert the binary image
inverted_image = cv2.bitwise_not(binary_image)

kernel = circular_kernel(11)
square_kernel = np.ones((11,11),np.uint8)

erosion = cv2.erode(image,kernel,iterations=1)
erosion2 = cv2.erode(image,kernel,iterations=2)
erosion4 = cv2.erode(image,kernel,iterations=4)
erosion10 = cv2.erode(image,kernel,iterations=10)
square_erosion = cv2.erode(image,square_kernel,iterations = 1)
# Show the image with the grid
cv2.imshow("Original Image",image)
cv2.imshow("Image with erosion", erosion)
cv2.imshow("Image with erosion2", erosion2)
cv2.imshow("Image with erosion4", erosion4)
cv2.imshow("Image with erosion10", erosion10)
cv2.imshow("Image with square erosion", square_erosion)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
