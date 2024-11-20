# stereo_exe_1.py
#
# Stereo Chessboard Calibration
#
# Paulo Dias modified for stereo calibration

import numpy as np
import cv2
import glob

# Chessboard dimensions
board_h = 9
board_w = 6

# Criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ....,(6,5,0)
objp = np.zeros((board_h * board_w, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Lists to store 3D points and 2D points for left and right images
objPoints = []  # 3D point coordinates in real world space
left_corners = []  # 2D points in left image plane
right_corners = []  # 2D points in right image plane

# Load stereo pair images
left_images = sorted(glob.glob('..//images//left*.jpg'))
right_images = sorted(glob.glob('..//images//right*.jpg'))

def FindAndDisplayChessboard(img, board_w, board_h):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    # Refine and display corners if found
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
        return True, corners2  # Return refined corners
    return False, None

# Process each pair of images
for left_img, right_img in zip(left_images, right_images):
    # Load left and right images
    imgL = cv2.imread(left_img)
    imgR = cv2.imread(right_img)

    # Find corners in both images
    retL, cornersL = FindAndDisplayChessboard(imgL, board_w, board_h)
    retR, cornersR = FindAndDisplayChessboard(imgR, board_w, board_h)

    # If corners are found in both images, add points to lists
    if retL and retR:
        objPoints.append(objp)
        left_corners.append(cornersL)
        right_corners.append(cornersR)

# Release display windows
cv2.destroyAllWindows()

# Stereo calibration: calibrate the stereo camera system
# Obtain the intrinsic and distortion parameters for both cameras, as well as the rotation and translation between them

# Initialize calibration parameters
_, left_mtx, left_dist, _, _ = cv2.calibrateCamera(objPoints, left_corners, imgL.shape[1::-1], None, None)
_, right_mtx, right_dist, _, _ = cv2.calibrateCamera(objPoints, right_corners, imgR.shape[1::-1], None, None)

# Stereo calibration to find relative positions of the cameras
ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
    objPoints, left_corners, right_corners,
    left_mtx, left_dist, right_mtx, right_dist,
    imgL.shape[1::-1],
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

# Save calibration results to a file
np.savez('stereo_calibration_params.npz',
         left_mtx=left_mtx, left_dist=left_dist,
         right_mtx=right_mtx, right_dist=right_dist,
         R=R, T=T, E=E, F=F)

print("Stereo calibration completed and saved.")
