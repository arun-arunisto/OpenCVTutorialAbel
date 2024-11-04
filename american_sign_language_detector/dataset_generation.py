import cv2
import handTrackingModule
import numpy as np
import math
import time



cap = cv2.VideoCapture(0)
detector = handTrackingModule.handDetector(maxHands=1)
offset = 20
img_size = 300

folder_train = "C:\\Users\\aruna\\AbelCVAdvanced\\american_sign_language_detector\\sign_language_dataset\\train\\B"
folder_val = "C:\\Users\\aruna\\AbelCVAdvanced\\american_sign_language_detector\\sign_language_dataset\\val\\B"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        # print(hand)
        x, y, w, h  = hand['bbox']

        #creating a white background
        img_white = np.ones((img_size, img_size, 3), np.uint8)*255
        img_crop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        #shape of the cropped image
        img_crop_shape = img_crop.shape
        
        #overlayying the image
        try:
            aspect_ratio = h/w
            #fixing the height
            if aspect_ratio > 1:
                k = img_size/h #constant
                w_cal = math.ceil(k*w)
                img_resize = cv2.resize(img_crop, (w_cal, img_size))
                img_resize_shape = img_resize.shape
                w_gap = math.ceil((img_size-w_cal)/2)
                img_white[:, w_gap:w_cal+w_gap] = img_resize
            #fixing the width
            else:
                k = img_size/w #constant
                h_cal = math.ceil(k*h)
                img_resize = cv2.resize(img_crop, (img_size, h_cal))
                img_resize_shape = img_resize.shape
                h_gap = math.ceil((img_size-h_cal)/2)
                img_white[h_gap:h_cal+h_gap, :] = img_resize
        except:
            continue

        cv2.imshow("img_resize", img_resize)
        # cv2.imshow("img_white", img_white)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    # if we click the "s" button it will save the image
    if key == ord("s"):
        counter += 1
        #saving the image in training folder
        cv2.imwrite(f"{folder_train}/Image_{counter}.jpg", img_white)
        #saving the image in validation folder
        cv2.imwrite(f"{folder_val}/Image_{counter}.jpg", img_white)
        print(counter)
    if key == ord("q"):
        break