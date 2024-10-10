import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to create a circular structuring element
def circular_kernel(diameter):
    radius = diameter // 2
    kernel = np.zeros((diameter, diameter), dtype=np.uint8)
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                kernel[i, j] = 1
    return kernel

# Load the binary image art3.bmp
image1 = cv2.imread("../images/art3.bmp", cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully read
if image1 is None:
    print("Image file art3.bmp could not be opened")
    exit(-1)

# Step 1: Define the circular structuring element with diameter 11
circular_structuring_element = circular_kernel(11)

# Step 2: Apply morphological opening (erosion followed by dilation)
opened_image1 = cv2.morphologyEx(image1, cv2.MORPH_OPEN, circular_structuring_element)

# Load the binary image art2.bmp
image2 = cv2.imread("../images/art2.bmp", cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully read
if image2 is None:
    print("Image file art2.bmp could not be opened")
    exit(-1)

# Step 1: Define the rectangular structuring elements
rectangular_kernel_1 = np.ones((3, 9), np.uint8)  # Horizontal structuring element
rectangular_kernel_2 = np.ones((9, 3), np.uint8)  # Vertical structuring element

# Step 2: Apply morphological opening with the horizontal structuring element
opened_image2_horizontal = cv2.morphologyEx(image2, cv2.MORPH_OPEN, rectangular_kernel_1)

# Step 3: Apply morphological opening with the vertical structuring element
opened_image2_vertical = cv2.morphologyEx(image2, cv2.MORPH_OPEN, rectangular_kernel_2)

# Display the results for both images
plt.figure(figsize=(12, 8))

# Display for art3.bmp
plt.subplot(2, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title("Original Image (art3.bmp)")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(opened_image1, cmap='gray')
plt.title("Opened Image (Circular Kernel)")
plt.axis('off')

# Display for art2.bmp
plt.subplot(2, 3, 3)
plt.imshow(image2, cmap='gray')
plt.title("Original Image (art2.bmp)")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opened_image2_horizontal, cmap='gray')
plt.title("Opened Image (Horizontal Kernel)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(opened_image2_vertical, cmap='gray')
plt.title("Opened Image (Vertical Kernel)")
plt.axis('off')

plt.tight_layout()
plt.show()
