import numpy as np
import cv2

# Load calibration parameters
params = np.load("stereoParams.npz")
intrinsics1 = params["intrinsics1"]
distortion1 = params["distortion1"]
intrinsics2 = params["intrinsics2"]
distortion2 = params["distortion2"]
R = params["R"]  # Rotation matrix from stereo calibration
T = params["T"]  # Translation vector from stereo calibration

# Load stereo images
imgL = cv2.imread('..//images//left01.jpg')
imgR = cv2.imread('..//images//right01.jpg')
height, width = imgL.shape[:2]

# Define output matrices for stereo rectification
R1 = np.zeros((3, 3))
R2 = np.zeros((3, 3))
P1 = np.zeros((3, 4))
P2 = np.zeros((3, 4))
Q = np.zeros((4, 4))

# Stereo Rectification
cv2.stereoRectify(intrinsics1, distortion1, intrinsics2, distortion2,
                  (width, height), R, T, R1, R2, P1, P2, Q,
                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0, 0))

# InitUndistortRectifyMap
map1x, map1y = cv2.initUndistortRectifyMap(intrinsics1, distortion1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(intrinsics2, distortion2, R2, P2, (width, height), cv2.CV_32FC1)

# Apply remapping to obtain rectified images
imgL_rectified = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
imgR_rectified = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# Draw horizontal lines on the rectified images for verification
def draw_horizontal_lines(img, line_spacing=25, color=(0, 255, 0), thickness=1):
    for y in range(0, img.shape[0], line_spacing):
        cv2.line(img, (0, y), (img.shape[1], y), color, thickness)

# Copy rectified images to draw lines
imgL_lines = imgL_rectified.copy()
imgR_lines = imgR_rectified.copy()
draw_horizontal_lines(imgL_lines)
draw_horizontal_lines(imgR_lines)

# Display rectified images with horizontal lines
cv2.imshow("Left Image - Rectified with Lines", imgL_lines)
cv2.imshow("Right Image - Rectified with Lines", imgR_lines)

# Optional: Interactive row highlight based on mouse clicks
def mouse_handler_left(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a red line at the corresponding row in both images
        imgL_temp = imgL_rectified.copy()
        imgR_temp = imgR_rectified.copy()
        cv2.line(imgL_temp, (0, y), (width, y), (0, 0, 255), 2)
        cv2.line(imgR_temp, (0, y), (width, y), (0, 0, 255), 2)
        cv2.imshow("Left Image - Rectified with Interactive Line", imgL_temp)
        cv2.imshow("Right Image - Rectified with Interactive Line", imgR_temp)

def mouse_handler_right(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a red line at the corresponding row in both images
        imgL_temp = imgL_rectified.copy()
        imgR_temp = imgR_rectified.copy()
        cv2.line(imgL_temp, (0, y), (width, y), (0, 0, 255), 2)
        cv2.line(imgR_temp, (0, y), (width, y), (0, 0, 255), 2)
        cv2.imshow("Left Image - Rectified with Interactive Line", imgL_temp)
        cv2.imshow("Right Image - Rectified with Interactive Line", imgR_temp)

# Set mouse callback for interactive line drawing
cv2.namedWindow("Left Image - Rectified with Interactive Line")
cv2.namedWindow("Right Image - Rectified with Interactive Line")
cv2.setMouseCallback("Left Image - Rectified with Interactive Line", mouse_handler_left)
cv2.setMouseCallback("Right Image - Rectified with Interactive Line", mouse_handler_right)

print("Click on a row in the rectified images to see the corresponding row in the other image.")
print("Press any key to exit.")

# Save the rectified images to disk
cv2.imwrite('left_rectified.jpg', imgL_rectified)
cv2.imwrite('right_rectified.jpg', imgR_rectified)

# Optionally, save the images with horizontal lines as well
cv2.imwrite('left_rectified_with_lines.jpg', imgL_lines)
cv2.imwrite('right_rectified_with_lines.jpg', imgR_lines)

print("Rectified images saved as 'left_rectified.jpg' and 'right_rectified.jpg'.")

# Display interactive windows
cv2.imshow("Left Image - Rectified with Interactive Line", imgL_rectified)
cv2.imshow("Right Image - Rectified with Interactive Line", imgR_rectified)



cv2.waitKey(0)
cv2.destroyAllWindows()
