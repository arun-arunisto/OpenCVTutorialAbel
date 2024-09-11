import os
import cv2
import numpy as np
from cv_modules.hand_tracking_module import handDetector


#------ painter images ----------#
folder_path = "paint_brushes"
mylist = os.listdir(folder_path)

overlay_list = []
for img_path in mylist:
    image = cv2.imread(folder_path+"/"+img_path)
    overlay_list.append(image)

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = handDetector()


header = overlay_list[0]
drawing_color = (0, 0, 255) #red for default
xp, yp = 0, 0

#creating canvas
img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    _, img = detector.findHands(img, draw=False)
    lmlist, _ = detector.findPosition(img)
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # index finger
        x2, y2 = lmlist[12][1:]  # middle finger

        fingers = detector.fingersUp(flip=True)

        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0 and fingers[0] == 0:
            xp, yp = 0, 0
            print("selection mode")
            print(x1, y1, x2, y2)
            if y1 < 150:
                #red
                if 460 < x1 < 490:
                    header = overlay_list[4]
                    drawing_color = (0, 0, 255)

                #blue - 680 710
                if 680 < x1 < 710:
                    header = overlay_list[3]
                    drawing_color = (255, 0, 0)

                #green color - 930 - 960
                if 930 < x1 < 960:
                    header = overlay_list[2]
                    drawing_color = (0, 255, 0)

                #eraser - 1120 - 1150
                if 1120 < x1 < 1150:
                    header = overlay_list[1]
                    drawing_color = (0, 0, 0)


        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0 and fingers[0] == 0:
            print("Drawing mode")
            cv2.circle(img, (x1, y1), 10, drawing_color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawing_color == (0, 0, 0):
                cv2.circle(img, (x1, y1), 20, drawing_color, cv2.FILLED)
                cv2.line(img,  (xp, yp), (x1, y1),  drawing_color, 20)
                cv2.line(img_canvas, (xp, yp), (x1, y1),  drawing_color, 20)

            cv2.line(img, (xp, yp), (x1, y1), drawing_color, 10)
            cv2.line(img_canvas, (xp, yp), (x1, y1), drawing_color, 10)
            xp, yp = x1, y1

    #merging the canvas and image
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    #overlaying image to header
    img[0:200, 0:1280] = header
    cv2.imshow("canvas", img_canvas)
    cv2.imshow("painter", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break