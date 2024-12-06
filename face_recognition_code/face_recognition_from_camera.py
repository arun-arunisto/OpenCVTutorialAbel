import cv2
import face_recognition

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    results = face_recognition.face_locations(frame)
    if results:
        for i in results:
            top, right, bottom, left = i
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imshow("windows", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break