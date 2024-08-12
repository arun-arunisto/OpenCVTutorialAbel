import cv2 

source = cv2.VideoCapture(0)

#load the model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

#model parameters
in_width = 300
in_height = 300

mean = [104, 117, 123]
conf_threshold = 0.7


while True:
    _, frame = source.read()

    #fliping the frame
    frame = cv2.flip(frame, 1)
    #shape of the frame
    #print(frame.shape)
    #height
    frame_height = frame.shape[0]
    #width
    frame_width = frame.shape[1]

    #blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    #print(blob)

    #run the model
    net.setInput(blob)
    detections = net.forward()
    
    #detction'
    #print(detections.shape)
    for i in range(detections.shape[2]):
        #print(i)
        confidence = detections[0, 0, i, 2]
        #print(confidence)
        if confidence > conf_threshold:
            #for drawing the rectangle
            x_left_bottom = int(detections[0, 0, i, 3]*frame_width)
            y_left_bottom = int(detections[0, 0, i, 4]*frame_height)
            x_right_top = int(detections[0, 0, i, 5]*frame_width)
            y_right_top = int(detections[0, 0, i, 6]*frame_height)
            #print(x_left_bottom, y_left_bottom, x_right_top, y_right_top)
            #drawing the rectanglw
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = f"Confidence: {str(confidence)[:4]}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom-label_size[1]), (x_left_bottom+label_size[0], y_left_bottom+base_line), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    t, _ = net.getPerfProfile()
    label = f"Inference lime: {t*1000.0/cv2.getTickFrequency()} ms"
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    #waitkey
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow('frame', frame)

cv2.destroyAllWindows()