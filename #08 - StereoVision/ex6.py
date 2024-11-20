import numpy as np
import cv2

# Load rectified images (ensure these are grayscale)
imgL_rectified = cv2.imread('left_rectified.jpg', cv2.IMREAD_GRAYSCALE)
imgR_rectified = cv2.imread('right_rectified.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize StereoBM with specified parameters
numDisparities = 16 * 5  # Must be divisible by 16
blockSize = 21  # Size of the block window. Must be an odd number >= 5
stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

# Compute the disparity map
disparity = stereo.compute(imgL_rectified, imgR_rectified)

# Normalize the disparity map to display as an 8-bit image
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_display = np.uint8(disparity_normalized)

# Display the results
cv2.imshow("Left Image - Rectified", imgL_rectified)
cv2.imshow("Right Image - Rectified", imgR_rectified)
cv2.imshow("Disparity Map", disparity_display)

cv2.waitKey(0)
cv2.destroyAllWindows()
