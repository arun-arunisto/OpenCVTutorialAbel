import pickle
from skimage.transform import resize
import cv2
import numpy as np

model_path = "C:\\Users\\aruna\\PycharmProjects\\advancedComputerVisionAbel\\model_training\\weather_model.p"

MODEL = pickle.load(open(model_path, "rb"))

classes = {0:"CLOUDY", 1:"RAIN", 2:"SHINE", 3:"SUNRISE"}


#print(MODEL)
def predict_weather_func(image):
    flat_data = []
    image_resized = resize(image, (15, 15, 3))
    flat_data.append(image_resized.flatten())
    flat_data = np.array(flat_data)
    #print(flat_data)
    y_output = MODEL.predict(flat_data)
    # print(y_output)
    return classes[y_output[0]]

image = cv2.imread("C:\\Users\\aruna\\PycharmProjects\\advancedComputerVisionAbel\\data\\datasets\\sklearn_data\\cloudy\\cloudy1.jpg")
print(predict_weather_func(image))
# predict_weather(image)