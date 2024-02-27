import os
import cv2

thres = 0.45  


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = os.path.join('I:\\PROJECTS\\SH WORKSHOP\\TINYML', 'coco.names')

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  

configPath = r'I:\\PROJECTS\\SH WORKSHOP\\TINYML\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'I:\\PROJECTS\\SH WORKSHOP\\TINYML\\frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    if not success or img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        print("Error reading frame from the video source.")
        break  # Exit the loop if there's an issue with reading frames

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
