import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Define sources for RGB and Stereo depth
camRgb = pipeline.create(dai.node.ColorCamera)
camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)
camDepth = pipeline.create(dai.node.StereoDepth)

# Define outputs
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

# Set the stream names
xoutVideo.setStreamName("video")
xoutDepth.setStreamName("depth")

# Properties for RGB camera
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use RGB camera
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Properties for mono cameras (left and right)
camLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Left camera
camRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Right camera
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# StereoDepth node settings
camDepth.setLeftInput(camLeft.out)
camDepth.setRightInput(camRight.out)

# Output depth information
camDepth.setOutputDepth(True)  # Enable depth output

# Linking the cameras and outputs
camRgb.video.link(xoutVideo.input)
camDepth.depth.link(xoutDepth.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    # Output queues to get frames
    qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        # Get the RGB frame
        inVideo = qVideo.get()
        frame = inVideo.getCvFrame()

        # Get the depth frame
        inDepth = qDepth.get()
        frameDepth = inDepth.getCvFrame()

        if frame is not None:
            cv2.imshow("RGB Video", frame)

        if frameDepth is not None:
            # Apply a color map to visualize depth
            frameDepthColor = cv2.applyColorMap(frameDepth, cv2.COLORMAP_JET)
            cv2.imshow("Depth", frameDepthColor)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
