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
image = cv2.imread("../images/mon1.bmp")

# Check if the image was successfully read
if image is None:
    print("Image file could not be opened")
    exit(-1)

_, binary_image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

# Step 2: Invert the binary image
inverted_image = cv2.bitwise_not(binary_image)

kernel = circular_kernel(11)
square_kernel = np.ones((9,9),np.uint8)
erosion = cv2.erode(inverted_image,kernel,iterations=2)
square_erosion = cv2.erode(inverted_image,square_kernel,iterations=2)
# Show the image with the grid
cv2.imshow("Original Image",image)
cv2.imshow("Image with erosion", erosion)
cv2.imshow("Image with square erosion", square_erosion)



# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
