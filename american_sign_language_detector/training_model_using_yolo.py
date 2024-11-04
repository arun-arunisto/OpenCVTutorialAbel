#import the yolo package from ultralytics
from ultralytics import YOLO

#define the model
model = YOLO('yolo11n-cls.pt')

#train the model
results = model.train(data="C:\\Users\\aruna\\AbelCVAdvanced\\american_sign_language_detector\\sign_language_dataset", epochs=30, imgsz=64)