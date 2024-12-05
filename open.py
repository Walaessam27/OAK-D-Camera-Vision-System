import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use CAM_A instead of RGB
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Linking
camRgb.video.link(xoutVideo.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the rgb frames from the output defined above
    qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        inVideo = qVideo.get()  # Blocking call, will wait until a new data has arrived
        frame = inVideo.getCvFrame()
        if frame is not None:
            cv2.imshow("video", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
