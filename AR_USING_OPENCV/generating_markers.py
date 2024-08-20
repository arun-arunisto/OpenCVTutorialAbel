import cv2
from cv2 import aruco

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

MARKER_SIZE = 400

#generating marker
for i in range(5):
    marker_img = aruco.generateImageMarker(marker_dict, i, MARKER_SIZE)
    #cv2.imshow("marker", marker_img)
    cv2.imwrite(f"markers/marker-{i}.png", marker_img)
print("Success!!")
key = cv2.waitKey(0)
if key == ord("q"):
    cv2.destroyAllWindows()