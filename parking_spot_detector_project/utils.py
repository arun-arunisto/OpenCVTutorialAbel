import pickle
from skimage.transform import resize
import numpy as np
import cv2


model_path = "model.p"
MODEL = pickle.load(open(model_path, "rb"))

def empty_or_not(spot_bgr):
    flat_data = []
    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = MODEL.predict(flat_data)
    if y_output == 0:
        return True
    else:
        return False
    

# img_path = cv2.imread("non-empty.jpg")
# print(empty_or_not(img_path))

def get_parking_spots_bboxes(connected_components):
    (totallabels, label_ids, values, centroid) = connected_components
    slot = []
    coef = 1
 
    for i in range(1, totallabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT]*coef)
        y1 = int(values[i, cv2.CC_STAT_TOP]*coef)
        w = int(values[i, cv2.CC_STAT_WIDTH]*coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT]*coef)

        slot.append([x1, y1, w, h])
    return slot
