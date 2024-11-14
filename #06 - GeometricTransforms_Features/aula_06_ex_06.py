import cv2
import numpy as np

# Initialize global variables to store the selected points
srcPts = []

# Load the original image
src = cv2.imread('../images/homography_1.jpg')  # Update this to your specific image path

# Check if the image is loaded
if src is None:
    print("Error loading image. Check the image path.")
    exit()

# Function to select corners from the original image
def select_corners(event, x, y, flags, params):
    global srcPts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(srcPts) < 4:
            srcPts.append((x, y))
            cv2.circle(src, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(src, str(len(srcPts)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.imshow("Original", src)
            print(f"Point {len(srcPts)} selected: ({x}, {y})")

# Display the original image and allow point selection
cv2.imshow("Original", src)
cv2.setMouseCallback("Original", select_corners)

# Wait until 4 points are selected
print("Please select 4 corners of the book by clicking on them.")
while True:
    key = cv2.waitKey(1)  # Wait for a key event
    if len(srcPts) == 4:
        print("4 points selected, processing...")
        break  # Exit the loop once 4 points are selected

cv2.destroyAllWindows()

# Ensure exactly 4 points are selected
if len(srcPts) == 4:
    # Convert selected points to numpy array
    np_srcPts = np.array(srcPts).astype(np.float32)

    # Define the destination points for the rectangular area to fit the book
    width, height = 175, 235  # Size of the book in pixels
    dstPts = np.array([[0, 0],
                       [width - 1, 0],
                       [width - 1, height - 1],
                       [0, height - 1]], dtype=np.float32)  

    # Compute the homography matrix
    homography_matrix, status = cv2.findHomography(np_srcPts, dstPts)

    # Print the estimated homography matrix
    print("Homography Matrix:")
    print(homography_matrix)

    # Warp the original image using the homography transformation
    warped_image = cv2.warpPerspective(src, homography_matrix, (width, height))

    # Display the warped image
    cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Please select exactly 4 points in the image.")
