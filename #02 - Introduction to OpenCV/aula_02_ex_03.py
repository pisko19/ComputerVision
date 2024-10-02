# Aula_01_ex_03.py
#
# Pedro Afonso
# Import necessary libraries
import numpy as np
import cv2
import sys

# Mouse callback function to detect clicks and draw a circle on right-click
def mouse_handler(event, x, y, flags, params):
    # Check if the right mouse button was clicked
    if event == cv2.EVENT_RBUTTONDOWN:
        # Draw a filled circle at the clicked position
        print("right click")  # Example behavior for left-click
    elif event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image1, (x, y), 4, (0, 255, 0), -1)  # Green circle, radius 10
        cv2.imshow("Display Image1", image1)  # Refresh the window to show the updated image
        

# Read the image
image1 = cv2.imread("../images/deti.jpg", cv2.IMREAD_UNCHANGED)

# Check if the image was successfully read
if image1 is None:
    # Failed reading the image
    print("Image file could not be opened")
    sys.exit(-1)

# Create a visualization window
cv2.imshow("Display Image1", image1)

# Set the mouse callback function to handle right-click events
cv2.setMouseCallback("Display Image1", mouse_handler)

# Wait for a key press indefinitely
cv2.waitKey(0)

# Destroy the window
cv2.destroyWindow("Display Image1")
