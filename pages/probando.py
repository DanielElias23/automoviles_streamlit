from ultralytics import YOLO
import cv2
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import streamlit as st

#img_file_buffer = st.camera_input("Take a picture", type=Video)

model= YOLO("yolov8s.pt")

#results = model.predict(source="img_file_buffer", show=True)

#results = model.predict(source="0", show=True)

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')

#st.write(results)
