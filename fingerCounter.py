import cv2
import mediapipe as mp
from cv_modules.hand_tracking_module import handDetector



##############################
wCam, hCam = 640, 480
##############################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = handDetector(detectionCon=0.7)

#tip ids
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    _, img = detector.findHands(img, draw=False)
    lmList, _ = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        #print("Landmark: for 4",lmList[4][1],"Landmark for 3:", print(lmList[3][1]))
        #thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #for other 4 fingers
        for id in range(1, 5):
            #print("tipid:", id, lmList[tipIds[id]][2], "tipid:", lmList[tipIds[id]-2][2])
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total_fingers = fingers.count(1)
        #print(total_fingers)
        #overlaying the number on my camera
        cv2.rectangle(img, (0, 0), (200, 200), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 150), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 20)
    cv2.imshow('window', img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break