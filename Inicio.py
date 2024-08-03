import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_extras.mention import mention


st.set_page_config(page_title="Proyecto", page_icon="📈")

[theme]
base="dark"

st.subheader(":orange[Bienvenidos]")

st.title("Proyectos de machine learning")

st.subheader("Autor: Daniel C. S.")

#mention(label="DanielElias23", icon="github", url="https://github.com/DanielElias23",)

st.sidebar.subheader(":blue[Selecciona un proyecto]")

#st.subheader("Problema")

#st.subheader("Descripción del problema")

st.write("""
         Estos proyectos son con el fin de mostrar habilidades de programación enfocado al área de ciencia de datos, los datos utilizados tienen sus contextos propios por lo que los modelos de inteligencia artificial no se pueden ocupar para uso general. Cada proyecto pretende mostrar habilidades diferentes en el contexto de machine learning, usando modelos diferentes y de diferentes categorías. Estos proyecto ya han sido realizados en los ejemplos mostrados en mí página de GitHub, pero no fueron implementados para la visualización de página web. Los análisis de los datos y las decisiones como la elección de modelos de machine learning está en el código en GitHub. 
         """)

st.write("https://github.com/DanielElias23")

st.write("www.linkedin.com/in/danielchingasilva")
         
st.subheader("Objetivos")

st.write("-Dar a conocer el conocimientos al respecto con machine learning")

st.write("-Demostrar dominio de conocimiento para solucionar problemas a diferentes problemáticas")

st.write("-Mostrar habilidades de programación enfocado al contexto ciencia de datos en las empresas")



#st.write("-Predecir el valor de diferentes automoviles segun sus caracterisitcas para que la empresa pueda definir un rango de precios para ofrecer")

#st.write("""El valor del precio predicho podria informarnos del rango de precio de venta puede optar la empresa y asi definir
 #         las posibles ganacias de la empresa""")


#data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv')

#st.subheader("Descripción del dataset")

#st.write("""
#         El dataset muestra la descripción automoviles de diferentes marcas con las especificaciones tecnicas
#         de cada modelo con su respectivo precio.
#         """)

#st.write(":blue[205 x 26]")

#st.write(data)


