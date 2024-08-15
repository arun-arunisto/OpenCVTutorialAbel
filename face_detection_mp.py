import cv2
import mediapipe as mp
import time



#accessing camera
cap = cv2.VideoCapture(0)

#detection
face_detection = mp.solutions.face_detection
#drwaing utils
mp_draw = mp.solutions.drawing_utils
#initializing face detection model
face_detection_model = face_detection.FaceDetection()


while True:
    success, img = cap.read()

    #converting to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection_model.process(img_rgb)
    if results.detections:
        #print(results.detections)
        for id, detection in enumerate(results.detections):
            #print(id, detection)
            print(detection)
            #creating bbox
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
            #drawing own bbox
            cv2.rectangle(img, bbox, (0, 255, 0), 1)
            cv2.putText(img, f"{int(detection.score[0]*100)}%",
                        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 0), 2)

            #fancy bbox
            x, y, w, h = bbox
            l, t = 40, 10
            x1, y1 = x+w, y+w
            #top left x, y
            cv2.line(img, (x, y), (x+l, y), (0, 0, 255), t)
            cv2.line(img, (x, y), (x, y+l), (0, 0, 255), t)
            #top-right x1, y
            cv2.line(img, (x1, y), (x1-l, y), (0, 0, 255), t)
            cv2.line(img, (x1, y), (x1, y+l), (0, 0, 255), t)
            #bottom-left
            cv2.line(img, (x, y1), (x+l, y1), (0, 0, 255), t)
            cv2.line(img, (x, y1), (x, y1-l), (0, 0, 255), t)
            #bottom-right
            cv2.line(img, (x1, y1), (x1-l, y1), (0, 0, 255), t)
            cv2.line(img, (x1, y1), (x1, y1-l), (0, 0, 255), t)

    cv2.imshow("video", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break