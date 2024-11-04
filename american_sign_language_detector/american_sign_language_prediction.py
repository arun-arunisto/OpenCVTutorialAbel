import cv2
from ultralytics import YOLO
import numpy as np
import handTrackingModule
import time

#open the camera
cap = cv2.VideoCapture(0)


#predicting the result
def predicting_the_result(cropped_img, model):
    result = model(cropped_img)
    names_dict = result[0].names
    probs = result[0].probs.data.tolist()
    return names_dict[np.argmax(probs)]



#opening the sign language image
image = cv2.imread("C:\\Users\\aruna\\AbelCVAdvanced\\american_sign_language_detector\\asl.png")
#resizing the image
image = cv2.resize(image, (300, 300))
#setting offset to place the asl image
offset = 20
#setting window size
cap.set(3, 1280)
cap.set(4, 720)

#implementing hand detector
detector = handTrackingModule.handDetector(maxHands=1)

#loading the model
model = YOLO("C:\\Users\\aruna\\AbelCVAdvanced\\american_sign_language_detector\\runs\\classify\\train\\weights\\last.pt")


while True:
    _, frame = cap.read()

    hands, _ = detector.findHands(frame)
    if hands:
        # print(hands)
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        frameCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        #error handling
        try:
            # print(predicting_the_result(frameCrop, model))
            #visualizing the words
            word = predicting_the_result(frameCrop, model)
            cv2.rectangle(frame, (0, 0), (150, 150), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, word, (35, 100), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 15)
        except Exception as e:
            continue

    #overlaying signlanguage image
    h, w, c = image.shape
    frame[10:h+10, 980:w+980] = image

    cv2.imshow("Window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()