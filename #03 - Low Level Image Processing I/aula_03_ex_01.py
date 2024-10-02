import cv2
import numpy as np

# Load the image
image = cv2.imread("../images/deti.jpg")

# Check if the image was successfully read
if image is None:
    print("Image file could not be opened")
    exit(-1)

# Check the number of channels to determine if the image is RGB or Grayscale
is_gray = (len(image.shape) == 2)  # Single channel indicates grayscale

# Get the image dimensions
height, width = image.shape[:2]

# Define the grid size (distance between grid lines)
grid_size = 50  # Distance between lines (in pixels)

# Define grid color based on image type
grid_color = (200, 200, 200) if not is_gray else (255, 255, 255)  # Gray for RGB, White for grayscale

# Draw horizontal lines
for y in range(0, height, grid_size):
    cv2.line(image, (0, y), (width, y), grid_color, 1)  # Draw line with grid_color

# Draw vertical lines
for x in range(0, width, grid_size):
    cv2.line(image, (x, 0), (x, height), grid_color, 1)  # Draw line with grid_color

# Show the image with the grid
cv2.imshow("Image with Grid", image)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
