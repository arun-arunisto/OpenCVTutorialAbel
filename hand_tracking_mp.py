import cv2
import mediapipe as mp
import time


#0 -> to use system camera
cap = cv2.VideoCapture(0)

#to detect hand
mp_hands = mp.solutions.hands #this is the endpoint going to use detect hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils #to draw the landmarks

#fps
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        #print(results.multi_hand_landmarks)
        for hand_lms in results.multi_hand_landmarks:
            #print(hand_lms)
            for id, lm in enumerate(hand_lms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                #center axis
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(cx, cy)
                print(id, cx, cy)

                #thumb tip 4
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
    #calculating the fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    print(int(fps))
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    pTime = cTime
    cv2.imshow("video", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break