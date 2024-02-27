# Import the necessary libraries
import os               # Operating system module
import cv2              # OpenCV library for computer vision tasks

# Set a threshold value for object detection confidence
thres = 0.45  

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Set camera parameters - width, height, and brightness
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load class names for object detection from a file
classNames = []
classFile = os.path.join('I:\\PROJECTS\\SH WORKSHOP\\TINYML', 'coco.names')

# Read class names from the file and store them in the classNames list
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  

# Define paths for the configuration and weights files for the pre-trained model
configPath = r'I:\\PROJECTS\\SH WORKSHOP\\TINYML\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'I:\\PROJECTS\\SH WORKSHOP\\TINYML\\frozen_inference_graph.pb'

# Create an instance of the object detection model
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Set input size, scale, mean, and swap color channels for the model
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start an infinite loop for continuously capturing and processing frames
while True:
    # Read a frame from the camera
    success, img = cap.read()

    # Check for errors in reading the frame
    if not success or img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        print("Error reading frame from the video source.")
        break  # Exit the loop if there's an issue with reading frames

    # Detect objects in the frame using the pre-trained model
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    # Draw bounding boxes and labels on the frame for detected objects
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the output frame with annotations
    cv2.imshow("Output", img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
