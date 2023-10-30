import cv2
import numpy as np

# Load an image
image = cv2.imread('1.jpg')

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for the yellow/green color of tennis balls
lower_bound = np.array([20, 100, 100])
upper_bound = np.array([40, 255, 255])

# Create a mask to isolate the yellow/green regions
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the detected contours
for contour in contours:
    # Calculate the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Draw a green rectangle around the detected tennis ball
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Tennis Ball Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
