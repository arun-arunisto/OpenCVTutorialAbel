import cv2
import numpy as np

from cv_modules.hand_tracking_module import handDetector
import pyautogui
from pynput.mouse import Controller



# two simple functions
def scroll_up():
    mouse = Controller()
    mouse.scroll(0, 1) #scroll up the mouse 1 step

def scroll_down():
    mouse = Controller()
    mouse.scroll(0, -1) #scroll down one step down


#--------------height and width -----------------#
wCam, hCam = 640, 480
#------------------ variables using for logic -----------#
frameR = 100
smoothening = 2
plocx, plocy = 0, 0 #previous location
clocx, clocy = 0, 0 #current location

#accessing camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = handDetector()

#getting monitor screen size
wScr, hScr = pyautogui.size()
print(wScr, hScr)

while True:
    _, img = cap.read()
    detector.findHands(img, draw=False)
    #landmarks
    lmList, _ = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #getting the tips of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #setting a range for the hand movement
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (0, 0, 255), 2)
        #checking which finger is up
        fingers = detector.fingersUp()
        #print(fingers)
        if fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            #getting coordinates inside the rectangle
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            #finding the current location
            clocx = plocx + (x3-plocx)/smoothening
            clocy = plocy + (y3-plocy)/smoothening
            pyautogui.moveTo(wScr-clocx, clocy)
            #drawing circle on the index finger
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            plocx, plocy = clocx, clocy
        #for scroll up
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and fingers[0] == 0:
            print("scroll up")
            scroll_up()

        #scroll down
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0 and fingers[0] == 0:
            print("scroll down")
            scroll_down()

        #mouse click
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0 and fingers[0] == 0:
            length, img, lineinfo = detector.findDistance(8, 12, img)
            if int(length) < 30:
                # print(length)
                cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
    cv2.imshow("virtual mouse", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break