import depthai as dai
import cv2
import numpy as np

# Create a pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)
camDepth = pipeline.create(dai.node.StereoDepth)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

# Set stream names
xoutVideo.setStreamName("video")
xoutDepth.setStreamName("depth")

# Set camera properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use CAM_A for RGB
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(30)

# Set Mono cameras for depth (stereo cameras)
camLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Left mono camera (stereo)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # Corrected resolution
camLeft.setFps(30)

camRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Right mono camera (stereo)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # Corrected resolution
camRight.setFps(30)

# Set StereoDepth node (with confidence threshold via initialConfig)
stereo_config = camDepth.initialConfig
stereo_config.setConfidenceThreshold(200)  # Corrected method for confidence threshold

# Link Mono cameras to StereoDepth node
camLeft.out.link(camDepth.left)
camRight.out.link(camDepth.right)

# Link nodes
camRgb.video.link(xoutVideo.input)
camDepth.depth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues for RGB and depth frames
    qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        inVideo = qVideo.get()  # Get the RGB frame
        inDepth = qDepth.get()  # Get the depth frame

        # Convert to OpenCV format
        frameRgb = inVideo.getCvFrame()
        frameDepth = inDepth.getCvFrame()

        # Normalize the depth frame to [0, 255] and convert to 8-bit for applyColorMap
        frameDepthNormalized = cv2.normalize(frameDepth, None, 0, 255, cv2.NORM_MINMAX)
        frameDepth8U = np.uint8(frameDepthNormalized)

        # Apply color map to depth image for better visualization
        frameDepthColor = cv2.applyColorMap(frameDepth8U, cv2.COLORMAP_JET)

        # Display both RGB and Depth
        cv2.imshow("RGB", frameRgb)
        cv2.imshow("Depth", frameDepthColor)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
