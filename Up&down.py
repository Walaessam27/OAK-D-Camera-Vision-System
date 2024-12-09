import depthai as dai
import cv2
import numpy as np  # For stacking frames

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
    # Output queue will be used to get the RGB frames from the output defined above
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

            # Show the combined view
            cv2.imshow("Combined View (Up and Down)", combined_view)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()


############################################################################    in two windows
# import depthai as dai
# import cv2

# # Create a pipeline
# pipeline = dai.Pipeline()

# # Define sources and outputs
# camRgb = pipeline.create(dai.node.ColorCamera)
# xoutVideo = pipeline.create(dai.node.XLinkOut)

# xoutVideo.setStreamName("video")

# # Properties
# camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Use CAM_A instead of RGB
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setIspScale(1, 2)  # Adjust this to crop the image vertically

# # Linking
# camRgb.video.link(xoutVideo.input)

# # Connect to device and start pipeline
# with dai.Device(pipeline) as device:
#     # Output queue will be used to get the RGB frames from the output defined above
#     qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)

#     while True:
#         inVideo = qVideo.get()  # Blocking call, will wait until a new data has arrived
#         frame = inVideo.getCvFrame()

#         if frame is not None:
#             # Process the frame to simulate "up" or "down" view
#             height, width, _ = frame.shape
#             up_view = frame[:height // 2, :]  # Top half of the frame
#             down_view = frame[height // 2:, :]  # Bottom half of the frame

#             # Show both views
#             cv2.imshow("Up View", up_view)
#             cv2.imshow("Down View", down_view)

#         if cv2.waitKey(1) == ord('q'):
#             break

# cv2.destroyAllWindows()
