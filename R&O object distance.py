import depthai as dai
import cv2
import numpy as np

# Function to detect balls and measure depth
def detect_ball(frame, depth_frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red and orange
    red_lower1 = np.array([0, 120, 70], np.uint8)
    red_upper1 = np.array([10, 255, 255], np.uint8)
    red_lower2 = np.array([170, 120, 70], np.uint8)
    red_upper2 = np.array([180, 255, 255], np.uint8)
    orange_lower = np.array([10, 100, 100], np.uint8)
    orange_upper = np.array([25, 255, 255], np.uint8)

    # Create masks for red and orange
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

    # Combine red and orange masks
    combined_mask = cv2.bitwise_or(red_mask, orange_mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            ball_roi = frame[y:y+h, x:x+w]

            # Apply edge detection to detect lines (black and white)
            edges = cv2.Canny(ball_roi, 100, 200)

            # Count black and white pixels to distinguish between balls
            black_lines = cv2.inRange(ball_roi, np.array([0, 0, 0]), np.array([50, 50, 50]))
            white_lines = cv2.inRange(ball_roi, np.array([200, 200, 200]), np.array([255, 255, 255]))

            # Calculate the ratio of black and white lines
            black_count = cv2.countNonZero(black_lines)
            white_count = cv2.countNonZero(white_lines)
            total_pixels = ball_roi.shape[0] * ball_roi.shape[1]

            black_ratio = black_count / total_pixels
            white_ratio = white_count / total_pixels

            # If more black pixels are found, classify as red, else orange
            if black_ratio > 0.05:  # Adjust threshold based on testing
                label = "Red Ball"
                color = (0, 0, 255)  # Red
            else:
                label = "Orange Ball"
                color = (0, 165, 255)  # Orange

            # Draw the bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Get depth at the center of the bounding box
            cx, cy = x + w // 2, y + h // 2
            cy_rescaled = int(cy * depth_frame.shape[0] / frame.shape[0])
            cx_rescaled = int(cx * depth_frame.shape[1] / frame.shape[1])
            depth_value = depth_frame[cy_rescaled, cx_rescaled] / 1000.0  # Convert to meters

            if depth_value > 0:
                return frame, depth_value, label

    return frame, None, None

# Create a pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")

# Configure the RGB camera
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Configure the mono cameras for stereo depth
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Configure stereo depth
stereo.initialConfig.setConfidenceThreshold(200)
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
stereo.setSubpixel(True)  # Better accuracy for depth data

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
camRgb.video.link(xoutRgb.input)
stereo.depth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    ball_detected = False

    while True:
        inRgb = qRgb.get()
        inDepth = qDepth.get()

        frame = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()

        if frame is not None and depthFrame is not None:
            # Process the frame and depth data
            color_detected_view, depth_in_meters, ball_type = detect_ball(frame, depthFrame)

            # Print the depth only once when a ball is detected
            if depth_in_meters and not ball_detected:
                print(f"{ball_type} detected! Depth: {depth_in_meters:.2f} meters")
                ball_detected = True
            elif not depth_in_meters and ball_detected:
                ball_detected = False

            # Show the combined view
            cv2.imshow("Ball Detection", color_detected_view)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
