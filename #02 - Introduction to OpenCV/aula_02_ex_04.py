# Aula_01_ex_04.py
# Pedro Afonso


# Import necessary libraries
import numpy as np
import cv2
import sys

# Mouse callback function to detect clicks and draw a circle on left-click
def mouse_handler(event, x, y, flags, params):
    # Check if the right mouse button was clicked
    if event == cv2.EVENT_RBUTTONDOWN:
        print("right click")
    # Check if the left mouse button was clicked
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Draw a filled circle at the clicked position
        cv2.circle(image1, (x, y), 4, (0, 255, 0), -1)  # Green circle, radius 4
        cv2.imshow("Display Image", image1)  # Refresh the window to show the updated image

# Read the image
image1 = cv2.imread("../images/deti.jpg", cv2.IMREAD_UNCHANGED)

# Check if the image was successfully read
if image1 is None:
    # Failed reading the image
    print("Image file could not be opened")
    sys.exit(-1)

# Convert to a gray-level image
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Create visualization windows
cv2.imshow("Display Image", image1)
cv2.imshow("Display Image Gray Color", image2)

# Set the mouse callback function to handle right-click and left-click events
cv2.setMouseCallback("Display Image", mouse_handler)

# Wait for a key press indefinitely
cv2.waitKey(0)

# Destroy the windows
cv2.destroyAllWindows()
