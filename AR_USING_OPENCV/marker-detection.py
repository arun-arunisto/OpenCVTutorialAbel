import cv2
import numpy as np
from cv2 import aruco


marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

param_markers = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    #converting the image to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detecting marker corner
    marker_corners, marker_ids, reject = aruco.detectMarkers(gray_img, marker_dict, parameters=param_markers)
    if marker_corners:
        # print(marker_corners, marker_ids)
        for ids, corners in zip(marker_ids, marker_corners):
            cv2.polylines(img, [corners.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
            corners = corners.reshape(4, 2)
            #print(corners)
            corners = corners.astype(int)
            #print(corners)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            cv2.putText(img, "TR 1000", top_right, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("camera", img)
    #cv2.imshow("gray", gray_img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break