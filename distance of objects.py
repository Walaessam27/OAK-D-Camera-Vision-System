import depthai as dai
import cv2
import numpy as np

# Function to detect red balls and get their distance
def detect_red_and_distance(frame, depth_frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = min(int(x + w / 2), depth_frame.shape[1] - 1)
            center_y = min(int(y + h / 2), depth_frame.shape[0] - 1)

            if 0 <= center_x < depth_frame.shape[1] and 0 <= center_y < depth_frame.shape[0]:
                distance = depth_frame[center_y, center_x]
                cv2.putText(frame, f"Distance: {distance} mm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame

# DepthAI Pipeline
pipeline = dai.Pipeline()

# Create nodes
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# Create XLinkOut nodes
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

# Set stream names
xoutVideo.setStreamName("video")
xoutDepth.setStreamName("depth")

# Configure the nodes
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

# Linking
camRgb.video.link(xoutVideo.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)  # Link the depth output to XLinkOut

# Connect to device
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        video_frame = video_queue.get().getCvFrame()
        depth_frame = depth_queue.get().getFrame()

        if video_frame is not None and depth_frame is not None:
            output_frame = detect_red_and_distance(video_frame, depth_frame)
            cv2.imshow("Red Ball Detection", output_frame)

        if cv2.waitKey(1) == ord('q'):
            break


cv2.destroyAllWindows()
