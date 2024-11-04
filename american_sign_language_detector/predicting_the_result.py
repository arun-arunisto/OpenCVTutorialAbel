#import package
from ultralytics import YOLO
import numpy as np

#loading the model
model = YOLO("C:\\Users\\aruna\\AbelCVAdvanced\\american_sign_language_detector\\runs\\classify\\train\\weights\\last.pt")

#predicting the result
result = model("C:\\Users\\aruna\\AbelCVAdvanced\\american_sign_language_detector\\sign_language_dataset\\val\\A\\Image_10.jpg")

# print(result)
names_dict = result[0].names

#probability list
probs = result[0].probs.data.tolist()

# print(probs)
# print(names_dict)
print(names_dict[np.argmax(probs)])