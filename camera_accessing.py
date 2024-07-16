#camera accessing
import cv2

cap = cv2.VideoCapture("video1.webm") #system camera
#print(cap.read())
while True:
    ret, frame = cap.read()
    waitkey = cv2.waitKey(1)
    if waitkey == ord('q'):
        break
    cv2.imshow("frame1", frame)
cv2.destroyAllWindows()
