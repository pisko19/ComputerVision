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

# Load the binary image art4.bmp
image = cv2.imread("../images/art4.bmp", cv2.IMREAD_GRAYSCALE)

# Check if the image was successfully read
if image is None:
    print("Image file art4.bmp could not be opened")
    exit(-1)

# Define different sizes for the circular structuring element
diameters = [11, 22, 33]  # Example diameters: 11, 22, and 33 pixels
closed_images = []

# Apply morphological closing with different structuring element sizes
for diameter in diameters:
    circular_structuring_element = circular_kernel(diameter)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, circular_structuring_element)
    closed_images.append(closed_image)

# Display the original and closed images
plt.figure(figsize=(12, 8))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image (art4.bmp)")
plt.axis('off')

# Display the closed images
for i, closed_image in enumerate(closed_images):
    plt.subplot(1, 4, i + 2)
    plt.imshow(closed_image, cmap='gray')
    plt.title(f"Closed Image (Diameter {diameters[i]})")
    plt.axis('off')

plt.tight_layout()
plt.show()
