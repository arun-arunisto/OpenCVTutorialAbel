import cv2
import numpy as np
import gradio as gr
from workoutfile import predict_weather_func

# def sepia(input_img):
#     sepia_filter = np.array([
#         [0.393, 0.769, 0.189],
#         [0.349, 0.686, 0.168],
#         [0.272, 0.534, 0.131]
#     ])
#     sepia_img = input_img.dot(sepia_filter.T)
#     sepia_img /= sepia_img.max()
#     return sepia_img

def predict_weather(input_img):
    result = predict_weather_func(input_img)
    cv2.putText(input_img, result, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return input_img

demo = gr.Interface(predict_weather, gr.Image(),title="Weather Predict", outputs="image")
demo.launch(share=True)