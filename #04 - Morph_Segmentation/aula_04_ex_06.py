import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables to hold the seed point
seed_point = None

# Mouse callback function to get the seed point
def get_seed_point(event, x, y, flags, param):
    global seed_point
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        seed_point = (x, y)
        print(f"Seed point selected: {seed_point}")

# Function to apply flood fill and display the result
def flood_fill_region(image_path):
    global seed_point

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image file {image_path} could not be opened")
        return

    # Create a mask for the flood fill operation
    h, w, _ = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Create a mask larger than the image

    # Set up the window and mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", get_seed_point)

    while True:
        # Display the image
        cv2.imshow("Image", image)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key to exit
            print("Exiting...")
            break
        
        if seed_point is not None:
            # Set new color for the filled area (e.g., green)
            new_value = (0, 255, 0)

            # Allowable intensity variation
            lo_diff = (5, 5, 5)  # Lower bound
            up_diff = (5, 5, 5)  # Upper bound

            # Perform flood fill
            cv2.floodFill(image, mask, seed_point, new_value, lo_diff, up_diff)

            # Reset seed_point to None to prevent multiple fillings
            seed_point = None

    cv2.destroyAllWindows()

# Test the interactive region segmentation on different images
images_to_test = ["../images/wdg2.bmp", "../images/tools_2.png", "../images/lena.jpg"]
for image_path in images_to_test:
    flood_fill_region(image_path)
