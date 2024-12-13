import cv2
import face_recognition
import numpy as np


cap = cv2.VideoCapture(0)

#loading the picture of donald trump
donald_trump = face_recognition.load_image_file("C:\\Users\\aruna\\AbelCVAdvanced\\face_recognition_code\\donald_trump.jpg")
donald_trump_encodings = face_recognition.face_encodings(donald_trump)[0]

#loading elon musk photo
elon_musk = face_recognition.load_image_file("C:\\Users\\aruna\\AbelCVAdvanced\\face_recognition_code\\elon_musk.jpg")
elon_musk_encodings = face_recognition.face_encodings(elon_musk)[0]

#loading my photo
arun_arunisto = face_recognition.load_image_file("C:\\Users\\aruna\\AbelCVAdvanced\\face_recognition_code\\arun_arunisto.jpeg")
arun_arunisto_encodings = face_recognition.face_encodings(arun_arunisto)[0]


#creating array of known face encodings
known_face_encodings = [
    donald_trump_encodings,
    elon_musk_encodings,
    arun_arunisto_encodings
]

known_face_names = [
    "Donald Trump",
    "Elon Musk",
    "Arun Arunisto"
]

face_locations = []
face_encodings = []
face_names = []
#process this frame
process_this_frame = True

while True:
    _, frame = cap.read()
    if process_this_frame:
        #resizing the current frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #converting bgr to rgb
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        #finding faces in current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        #taking the face_encodings from live
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # print(face_encodings)
        face_names = []
        for face_encoding in face_encodings:
            #see if the face encodings in any known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            #if any match found
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            #face distances
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
    process_this_frame = not process_this_frame

    #displaying the bbox
    # print(face_locations)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # print(top, right, bottom, left)
        #scale back up face locations
        top *= 4
        right *=4
        bottom *= 4
        left *= 4

        #drawing bbox around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #adding name
        cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
    # cv2.imshow("rgbframe", rgb_small_frame)
    cv2.imshow("window", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
