import face_recognition
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import numpy as np

#opening image
img = cv2.imread("family_photo.jpg")
results = face_recognition.face_locations(img)
if results:
    for i in results:
        top, right, bottom, left = i
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1, cv2.LINE_AA)
cv2.imwrite("results.jpg", img)