import depthai as dai
import cv2
import numpy as np  # For stacking frames

# Function to detect specific colors and return objects (contours and bounding boxes)
def detect_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        "green": ((35, 50, 50), (85, 255, 255)),  # Light green to dark green
        "blue": ((90, 50, 50), (130, 255, 255)),  # Light blue to dark blue
        "red": ((0, 50, 50), (10, 255, 255)),  # Red lower range
        "red_alt": ((170, 50, 50), (180, 255, 255)),  # Red upper range
        "orange": ((15, 50, 50), (25, 255, 255)),  # Light orange
        "white": ((0, 0, 200), (180, 50, 255))  # Bright white
    }

    # Create a mask for each color
    masks = {}
    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        masks[color] = cv2.inRange(hsv, lower_bound, upper_bound)

    # Combine the two red masks (lower and upper ranges)
    if "red" in masks and "red_alt" in masks:
        masks["red"] = cv2.bitwise_or(masks["red"], masks["red_alt"])
        del masks["red_alt"]

    # Create an output image highlighting the detected colors
    output = frame.copy()

    # Draw bounding boxes for each color's contours
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours (noise)
                # Get bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)
                # Draw the bounding box and label
                cv2.rectangle(output, (x, y), (x + w, y + h), {
                    "green": (0, 255, 0),
                    "blue": (255, 0, 0),
                    "red": (0, 0, 255),
                    "orange": (0, 165, 255),
                    "white": (255, 255, 255)
                }[color], 2)
                cv2.putText(output, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return output

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

            # Perform color detection on the combined view
            color_detected_view = detect_colors(combined_view)

            # Show the original and color-detected views
            cv2.imshow("Combined View (Up and Down)", combined_view)
            cv2.imshow("Color Detection (Objects)", color_detected_view)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
