import cv2
from utils import empty_or_not, get_parking_spots_bboxes
import json
import numpy as np

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask = "mask_1920_1080.png"

mask = cv2.imread(mask, 0)

cap = cv2.VideoCapture("parking_1920_1080.mp4")

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for j in spots]
diffs = [None for j in spots]

# print(spots_status)
# print(diffs)

previous_frame = None
frame_nmr = 0
ret = True
step = 30


with open("instances_default.json") as f:
    data = json.load(f)

bounding_box = [bbox["bbox"] for bbox in data["annotations"]]

while True:
    #video running infinite
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w]

            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1+h, x1:x1+w])
        # print([diffs[j] for j in np.argsort(diffs)][::-1])
    
    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j]/np.amax(diffs) > 0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status
    if frame_nmr % step == 0:
        previous_frame = frame.copy()
    
    for spot_indx, _ in enumerate(spots):
        # print(spot_indx, bounding_box[spot_indx])
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (0, 255, 0), 2)
    # print(spots_status)
    cv2.rectangle(frame, (80, 20), (550, 80), (255, 255, 255), -1)
    cv2.putText(frame, "Available spots: {} / {}".format(str(sum(spots_status)), len(spots_status)), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
