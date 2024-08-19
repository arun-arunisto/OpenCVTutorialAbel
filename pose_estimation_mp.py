import cv2
import mediapipe as mp
import time


#mediapipe model for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("../data/video/dancing_video.mp4")
#setting width and height of the frame
cap.set(3, 1280)
cap.set(4, 720)
while True:
    success, img = cap.read()
    if not success:
        break
    # for video resizing
    img = cv2.resize(img, (1280, 720))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if id == 12:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    cv2.imshow("window", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break