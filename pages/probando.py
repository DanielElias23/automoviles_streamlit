from ultralytics import YOLO
import cv2
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import streamlit as st

#img_file_buffer = st.camera_input("Take a picture", type=Video)

model= YOLO("yolov8s.pt")

#results = model.predict(source="img_file_buffer", show=True)



from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

#faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')





webrtc_streamer(key="example")

results = model.predict(source="0", show=True)

st.write(results)
