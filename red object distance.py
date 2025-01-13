import depthai as dai
import cv2
import numpy as np

# Function to detect a red ball and measure its depth
def detect_red_ball(frame, depth_frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([160, 100, 100], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    # Create masks for both red ranges and combine them
    mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Red Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Get the depth value at the center of the bounding box
            cx, cy = x + w // 2, y + h // 2

            # Rescale coordinates from RGB to depth frame size
            cy_rescaled = int(cy * depth_frame.shape[0] / frame.shape[0])
            cx_rescaled = int(cx * depth_frame.shape[1] / frame.shape[1])

            # Ensure the coordinates are within bounds
            cy_rescaled = min(cy_rescaled, depth_frame.shape[0] - 1)
            cx_rescaled = min(cx_rescaled, depth_frame.shape[1] - 1)

            # Access the depth value at the rescaled coordinates
            depth_value = depth_frame[cy_rescaled, cx_rescaled]

            # Convert depth to meters, depth is likely in millimeters so divide by 1000
            depth_in_meters = depth_value / 1000.0

            if depth_value > 0:
                # Adjust this scaling factor depending on your camera and environment
                # If you notice significant errors in depth, consider adjusting this factor (such as 0.5 or 1.5)
                scaled_depth = depth_in_meters * 0.5
                return frame, scaled_depth

    return frame, None

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
            color_detected_view, depth_in_meters = detect_red_ball(frame, depthFrame)

            # Print the depth only once when the red ball is detected
            if depth_in_meters and not ball_detected:
                print(f"Red ball detected! Depth: {depth_in_meters:.2f} meters")
                ball_detected = True
            elif not depth_in_meters and ball_detected:
                ball_detected = False

            # Show the combined view
            cv2.imshow("Red Ball Detection", color_detected_view)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
