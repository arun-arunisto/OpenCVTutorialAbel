import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands  # the model we used to detect hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon)  # object that we created
        self.mpDraw = mp.solutions.drawing_utils  # to draw the landmarks
        self.tipIds =  [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        #converting the img into RGB becuase the object only uses rgb images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        #print(results.multi_hand_landmarks) #to get the landmarks of a hand
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                myLmlist = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x*w), int(lm.y*h), int(lm.z*w)
                    myLmlist.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ##bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax-xmin, ymax-ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0]+(bbox[2]//2), bbox[1]+(bbox[3]//2)
                myHand['lmList'] = myLmlist
                self.lmList = myHand['lmList']
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    #mpDraw.draw_landmarks(img, handLms)
                    # #we are drawing in img property with handLms for landmarks
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    #the above code only display marks this line will
                    # display the connections between the marks
                    #bbox
                    cv2.rectangle(img, (bbox[0]-20, bbox[1]-20),
                                  (bbox[0]+bbox[2]+20, bbox[1]+bbox[3]+20),
                                  (0, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0]-30, bbox[1]-30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
        return allHands, img

    def findPosition(self, img, handNo=0, draw=True):
        # for bounding box
        xList = []
        yList = []
        bbox = []
        # getting the information - like id and landmark
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                # getting height, width and channel of our image
                h, w, c = img.shape
                # finding the position
                # center-axis
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(cx, cy)
                # the above print will print the cx, cy value of every landmark
                # to know the value of the specified one
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                # the above id=landmark, cx, cy = for getting the axis
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    #fingers up
    def fingersUp(self, flip=False):
        fingers = []
        # thumb
        if flip:
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # other 4 fingers
        for id in range(1, 5):
            # index finger
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    #find distance
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1+x2) //2, (y1+y2) //2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]



def main():
    pTime = 0  # previous time
    cTime = 0  # current time
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        lmList, img = detector.findHands(img)
        #lmList = detector.findPosition(img)
        if len(lmList) > 1:
            print(lmList[0]["bbox"])
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # displaying FPS on screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        # for exit if you press 'q' it will break the loop
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()