import depthai as dai
import cv2
import numpy as np  # For stacking frames

# Function to detect a red ball in the frame
def detect_red_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    red_lower1 = np.array([0, 50, 50], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170, 50, 50], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    # Create masks for both red ranges and combine them
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours to check for a red ball
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Red Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print("Red ball detected!")
            return frame, True

   # print("No red ball detected.")
    return frame, False

# Create a pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use CAM_A instead of RGB
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 2)  # Adjust this to crop the image vertically

# Linking
camRgb.video.link(xoutVideo.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue to get RGB frames
    qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        inVideo = qVideo.get()  # Blocking call, will wait until a new data has arrived
        frame = inVideo.getCvFrame()

        if frame is not None:
            # Process the frame to get "up" and "down" views
            height, width, _ = frame.shape
            up_view = frame[:height // 2, :]  # Top half of the frame
            down_view = frame[height // 2:, :]  # Bottom half of the frame

            # Stack the two views vertically
            combined_view = np.vstack((up_view, down_view))

            # Perform red ball detection on the combined view
            color_detected_view, red_ball_detected = detect_red_ball(combined_view)

            # Show the original and color-detected views
            cv2.imshow("Combined View (Up and Down)", combined_view)
            cv2.imshow("Red Ball Detection", color_detected_view)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()

