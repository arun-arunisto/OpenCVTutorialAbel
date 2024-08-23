import cv2
import mediapipe as mp
import time
import math


class hand_detector_module:
    def __init__(self, mode=False, max_hands=2, complexity=1, min_det_conf=0.5, min_tra_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.min_det_conf = min_det_conf
        self.min_tra_conf = min_tra_conf
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.min_det_conf, self.min_tra_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True, land_marks=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        ht_module_list = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # print(hand_lms)

                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                if land_marks:
                    for id, lm in enumerate(hand_lms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        ht_module_list.append((id, cx, cy))
        return img, ht_module_list