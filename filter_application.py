import cv2 
import numpy as np
import sys

#filters
PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3

feature_params = {"maxCorners":500, "qualityLevel":0.2, "minDistance":15, "blockSize":9}

image_filter = PREVIEW
alive = True
window_name = "Camera filter"
cv2.namedWindow(window_name, cv2.WINDOW_FULLSCREEN)
result = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #fliping the frame
    frame = cv2.flip(frame, 1)

    if image_filter == PREVIEW:
        result = frame
    if image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    if image_filter == CANNY:
        result = cv2.Canny(frame, 10, 100)
    if image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            print(corners)
            for x, y in np.float32(corners).reshape(-1, 2):
                print(x, y)
                cv2.circle(result, (int(x), int(y)), 10, (255, 0, 255), 1, cv2.LINE_AA)


    """if cv2.waitKey(1) == ord('q'):
        break"""
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        image_filter = PREVIEW
    if key == ord('b'):
        image_filter = BLUR
    if key == ord('c'):
        image_filter = CANNY
    if key == ord('f'):
        image_filter = FEATURES
    cv2.imshow(window_name, result)

cv2.destroyAllWindows()