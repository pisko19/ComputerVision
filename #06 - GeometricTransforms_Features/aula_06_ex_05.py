import cv2
import numpy as np
import math

# Load the original and transformed images
src = cv2.imread('../images/deti.jpg')  # Load the original image
dst = cv2.imread('deti_tf.jpg')  # Load the transformed image

# Check if images are loaded
if src is None or dst is None:
    print("Error loading images. Check the image paths.")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(src, None)
kp2, des2 = orb.detectAndCompute(dst, None)

# Use Brute Force matcher to find matches
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (optional)
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Ensure at least 3 matches are found
if len(matches) >= 3:
    # Estimate the affine transformation matrix
    affine_matrix = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])  # Using the first three matches

    # Print the estimated affine transformation matrix
    print("Affine Transformation Matrix:")
    print(affine_matrix)

    # Warp the original image using the affine transformation
    warp_dst = cv2.warpAffine(src, affine_matrix, (src.shape[1], src.shape[0]))

    # Display the warped image
    cv2.imshow("Warped Image", warp_dst)

    # Extract parameters from the affine matrix
    a, c, tx = affine_matrix[0]
    b, d, ty = affine_matrix[1]

    # Compute translation
    t_x = tx
    t_y = ty

    # Compute scaling factors
    s_x = np.sign(a) * math.sqrt(a**2 + b**2)
    s_y = np.sign(d) * math.sqrt(c**2 + d**2)

    # Compute rotation angle
    psi = math.atan2(b, a)

    # Convert rotation angle from radians to degrees
    rotation_angle = np.degrees(psi)

    # Display computed parameters
    print(f"Translation: t_x = {t_x}, t_y = {t_y}")
    print(f"Scale: s_x = {s_x}, s_y = {s_y}")
    print(f"Rotation angle (in degrees): {rotation_angle}")

    # Convert both images to grayscale for subtraction comparison
    warp_dst_gray = cv2.cvtColor(warp_dst, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two images
    difference = cv2.absdiff(warp_dst_gray, dst_gray)

    # Display the difference
    cv2.imshow("Difference between Warped and Transformed", difference)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches found. Please ensure that there are at least 3 matches.")
