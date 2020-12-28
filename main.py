import cv2

thres = 0.5 #threshhold to detect objects


cap = cv2.VideoCapture(0) #setting up the camera
cap.set(3, 640)
cap.set(4, 480)

#importing names (classes) from the COCO dataset
classNames =[]
classFile = 'coco.names'

with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#importing our files
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

# creating our Model - OpenCV provides a model that does all the processing for us
# you need to provide configuration path and weight path

net = cv2.dnn_DetectionModel(weightPath, configPath) #creating model

# Parameters set by default - found on documentation, it is required to run the model.
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    # Send our image to the model which will return a prediction
    classIds, confs, bbox = net.detect(img, confThreshold= thres ) # if confidence>50, return that class
    print(classIds, bbox)

    # looping through 3 variables with one for loop using zip function
    if len(classIds) != 0:     # need to check for an object detected first
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color = (0,255,0),thickness = 3)
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)


    cv2.imshow("Output", img)
    cv2.waitKey(1)