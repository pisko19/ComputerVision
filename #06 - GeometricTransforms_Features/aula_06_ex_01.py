import cv2
import numpy as np

# Load the image
src = cv2.imread("../images/deti.jpg")  # Replace 'your_image.jpg' with your image path
if src is None:
    print("Error: Could not load image.")
    exit()

# Get the dimensions of the image
rows, cols, channels = src.shape

# Create a rotation matrix
# Here, we rotate 25 degrees and keep the scale factor as 1
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 25, 1)

# Modify the translation part of the matrix
M[0][2] = M[0][2] - 50  # Translate x by -50 pixels
M[1][2] = M[1][2] + 100  # Translate y by +100 pixels

# Print the transformation matrix
print("Transformation Matrix:\n", M)

# Apply the affine transformation
dst = cv2.warpAffine(src, M, (cols, rows))

# Save the transformed image
cv2.imwrite('deti_tf.jpg', dst)

# Optionally display the images
cv2.imshow('Original Image', src)
cv2.imshow('Transformed Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()