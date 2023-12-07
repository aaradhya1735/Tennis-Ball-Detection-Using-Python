import cv2
import numpy as np
# Create a VideoCapture object to access the camera using the DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use camera index 0 (built-in camera) or change to your desired camera index

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use the Hough Circle Transform to detect circles (tennis balls)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50,
    )

    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.uint16(np.around(circles))

        # Iterate through detected circles and draw them
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # Draw the circle and its center on the frame
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 2, (0, 0, 255), 3)

    # Display the frame with detected circles
    cv2.imshow('Tennis Ball Detection', frame)

    # Press 'q' to exit the live stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()