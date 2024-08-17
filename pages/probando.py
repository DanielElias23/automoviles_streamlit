from ultralytics import YOLO
import cv2
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import streamlit as st

model= YOLO("yolov8s.pt")

results = model.predict(source="0", show=True)

#st.write(results)
