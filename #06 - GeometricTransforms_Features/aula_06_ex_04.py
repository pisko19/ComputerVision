import cv2
import numpy as np

# Load the two images
src = cv2.imread('../images/deti.jpg')  # Change to your image path
dst = cv2.imread('deti_tf.jpg')  # Change to your image path

# Check if images are loaded
if src is None or dst is None:
    print("Error loading images. Check the image paths.")
    exit()

# Convert images to grayscale
gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gray_src, None)
kp2, des2 = orb.detectAndCompute(gray_dst, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Modify the number of matches to consider
# Change the percentage to see the impact
numGoodMatches = int(len(matches) * 0.1)  # Keep 10% of the best matches
# numGoodMatches = int(len(matches) * 0.2)  # Uncomment to keep 20% of the best matches

matches = matches[:numGoodMatches]

# Draw matches
im_matches = cv2.drawMatches(src, kp1, dst, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow("Matches", im_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Evaluate transform
if len(matches) >= 4:  # Ensure enough matches to compute a homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # You can proceed to compute the homography here if needed
    homography_matrix, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    print("Homography Matrix:")
    print(homography_matrix)
else:
    print("Not enough matches to compute homography.")
