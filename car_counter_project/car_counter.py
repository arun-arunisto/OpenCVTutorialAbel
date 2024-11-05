import cv2
from ultralytics import YOLO
import utilsModule
import math
from sort import *
import numpy as np




cap = cv2.VideoCapture("cars.mp4")
model = YOLO("yolo11l.pt")
#label names 
names = model.names
# print(names)

mask = cv2.imread("mask.png") #opening the mask image

cap.set(3, 1280)
cap.set(4, 720)

#drawing line coordinates
limits = [150, 400, 673, 400]

total_count = []

#tracking cars
tracker = Sort(max_age=20, min_hits=2)

while True:
    success, img = cap.read()
    maskregion = cv2.bitwise_and(img, mask)
    result = model(maskregion, stream=True)

    #numpy array
    detections = np.empty((0, 5))
    # print(result)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    for r in result:
        # print(r)
        boxes = r.boxes
        for box in boxes:
            # print(box)
            x1, y1, x2, y2 = box.xyxy[0]
            # print(x1, y1, x2, y2)
            #converting to integer
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            w, h = x2-x1, y2-y1
            bbox = x1, y1, w, h
            

            # class name
            cls = box.cls[0]
            # print(cls)
            classname = names[int(cls)]
            # print(classname)

            #finding the center
            cx, cy = x1+w//2, y1+h//2
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # confidence level
            conf = box.conf[0]
            # print(conf)
            conf = math.ceil(conf*100)/100 #rounding of the decimal values to two 
            # print(conf)
            if conf >= 0.7 and classname=="car":
                #for tracker
                current_array = np.array([x1, y1, x2, y2, conf])
                #adding the above array to the detections
                detections = np.vstack((detections, current_array))

                utilsModule.cornerRect(img, bbox)
                if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20: 
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    tracker_results = tracker.update(detections)
    print(tracker_results)
    for tr in tracker_results:
        _, _, _, _, Id = tr
        if Id not in total_count:
            total_count.append(Id)
    #displaying car count
    utilsModule.putTextRect(img, f"Total cars: {len(total_count)}", (0, 38))
    cv2.imshow("frame", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break