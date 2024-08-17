from ultralytics import YOLO
import cv2
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import streamlit as st

#img_file_buffer = st.camera_input("Take a picture", type=Video)

model= YOLO("yolov8s.pt")

import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Cámara en Vivo con Streamlit")

    # Usa la cámara 0 (la cámara predeterminada)
    cap = cv2.VideoCapture(0)

    # Configura el marco de Streamlit
    frame_window = st.image([])

    while True:
        # Captura un fotograma de la cámara
        ret, frame = cap.read()
        
        if not ret:
            st.error("No se pudo capturar el video.")
            break
        
        # Convierte el fotograma de BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Muestra el fotograma en Streamlit
        frame_window.image(frame_rgb, channels="RGB")
        
        # Si la ventana de Streamlit se cierra, rompe el bucle
        if st.button("Detener"):
            break

    # Libera la cámara
    cap.release()

if __name__ == "__main__":
    main()


results = model.predict(source="0", show=True)

st.write(results)
