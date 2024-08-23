import cv2
import numpy as np
from cv2 import aruco


def image_augmentation(frame, src_img, dst_point):
    src_h, src_w = src_img.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]])
    H, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_point)
    warp_image = cv2.warpPerspective(src_img, H, (frame_w, frame_h))
    cv2.fillConvexPoly(mask, dst_point, 255)
    results = cv2.bitwise_and(warp_image, warp_image, frame, mask=mask)


marker = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
param_markers = aruco.DetectorParameters()

aug_img = cv2.imread("images/1.png")
cap = cv2.VideoCapture(0)
while True:
    succ, img = cap.read()
    if not succ:
        break
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, reject = aruco.detectMarkers(gray_img, marker, parameters=param_markers)
    if marker_corners:
        #print(marker_corners)
        for ids, corners in zip(marker_ids, marker_corners):
            cv2.polylines(img, [corners.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            print("marker ids:", ids[0])
            #print(f"images/{ids[0]}.png")
            aug_img = cv2.imread(f"images/{ids[0]}.png")
            if aug_img is not None:
                image_augmentation(img, cv2.imread(f"images/{ids[0]}.png"), corners)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break