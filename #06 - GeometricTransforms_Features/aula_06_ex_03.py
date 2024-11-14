import cv2
import numpy as np

# Load the original and transformed images
src = cv2.imread('../images/deti.jpg')  # Replace with your original image path
dst = cv2.imread('deti_tf.jpg')  # Replace with your transformed image path

# Check if images are loaded
if src is None or dst is None:
    print("Error loading images. Check the image paths.")
    exit()

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT in both images
kp1, des1 = sift.detectAndCompute(src, None)  # For the original image
kp2, des2 = sift.detectAndCompute(dst, None)  # For the transformed image

# Draw keypoints on the images
img_kp1 = cv2.drawKeypoints(src, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_kp2 = cv2.drawKeypoints(dst, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the original image with keypoints
cv2.imshow("Original Image with Keypoints", img_kp1)
cv2.waitKey(0)  # Press any key to close the window

# Display the transformed image with keypoints
cv2.imshow("Transformed Image with Keypoints", img_kp2)
cv2.waitKey(0)  # Press any key to close the window

# Close all windows
cv2.destroyAllWindows()
