from ultralytics import YOLO
import cv2
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import streamlit as st

#img_file_buffer = st.camera_input("Take a picture", type=Video)

model= YOLO("yolov8s.pt")

import numpy as np

# Crear un botón para activar la cámara en vivo
if st.button("Activar cámara en vivo"):
    # Encender la cámara
    cap = cv2.VideoCapture(0)

    # Crear un contenedor para mostrar el video en vivo
    video_container = st.empty()

    while True:
        # Leer un frame de la cámara
        ret, frame = cap.read()

        # Convertir el frame a un formato compatible con Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)

        # Mostrar el frame en el contenedor
        video_container.image(frame, channels="RGB")

        # Esperar un poco antes de leer el siguiente frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara
    cap.release()
    cv2.destroyAllWindows()

results = model.predict(source="0", show=True)

st.write(results)
