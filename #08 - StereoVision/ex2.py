import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# Prepare object points, e.g., (0,0,0), (1,0,0), ..., (6,5,0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Arrays to store object points and image points for left and right images
objPoints = []         # 3D points
left_imgPoints = []    # 2D points in left images
right_imgPoints = []   # 2D points in right images

# Load stereo pair images
left_images = sorted(glob.glob('..//images//left*.jpg'))
right_images = sorted(glob.glob('..//images//right*.jpg'))

def find_corners(image, board_w, board_h):
    """Detect chessboard corners in an image and refine them."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

# Process each image pair
for left_img_path, right_img_path in zip(left_images, right_images):
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)

    # Find and refine chessboard corners in each image
    retL, cornersL = find_corners(imgL, board_w, board_h)
    retR, cornersR = find_corners(imgR, board_w, board_h)

    # Only use pairs where both images found corners
    if retL and retR:
        objPoints.append(objp)
        left_imgPoints.append(cornersL)
        right_imgPoints.append(cornersR)

# Calibrate each camera individually
_, intrinsics1, distortion1, _, _ = cv2.calibrateCamera(objPoints, left_imgPoints, imgL.shape[1::-1], None, None)
_, intrinsics2, distortion2, _, _ = cv2.calibrateCamera(objPoints, right_imgPoints, imgR.shape[1::-1], None, None)

# Stereo calibration using the intrinsic guesses and CV_CALIB_SAME_FOCAL_LENGTH
flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_SAME_FOCAL_LENGTH
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
ret, intrinsics1, distortion1, intrinsics2, distortion2, R, T, E, F = cv2.stereoCalibrate(
    objPoints, left_imgPoints, right_imgPoints,
    intrinsics1, distortion1, intrinsics2, distortion2,
    imgL.shape[1::-1],
    criteria=criteria, flags=flags
)

# Save the calibration parameters
np.savez("stereoParams.npz",
         intrinsics1=intrinsics1,
         distortion1=distortion1,
         intrinsics2=intrinsics2,
         distortion2=distortion2,
         R=R, T=T, E=E, F=F)

print("Stereo calibration completed and parameters saved to stereoParams.npz.")
