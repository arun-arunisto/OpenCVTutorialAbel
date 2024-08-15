import cv2
import mediapipe as mp
import time

#camera accessing
cap = cv2.VideoCapture(0)

mp_facemesh = mp.solutions.face_mesh
facemesh = mp_facemesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles
drawing_specc = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

while True:
    _, img = cap.read()

    #converting image to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = facemesh.process(img_rgb)
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            """mp_draw.draw_landmarks(
                image=img,
                landmark_list=facelms,
                connections=mp_facemesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_specc
            )"""
            mp_draw.draw_landmarks(
                image=img,
                landmark_list=facelms,
                connections=mp_facemesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_specc
            )
    cv2.imshow("window", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break