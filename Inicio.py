import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_extras.mention import mention
import streamlit.components.v1 as components

st.set_page_config(page_title="Proyecto", page_icon="📈")
import base64
import streamlit as st



@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}
img = get_img_as_base64("difu6.jpg")
#https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqZ2lmjdQMNC3cyQ2g0i_wvigb5elGGBIPBg&s
#https://img.freepik.com/fotos-premium/imagen-borrosa-centro-comercial-fondo-luz-bokeh_1157641-5174.jpg
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://www.colorhexa.com/191b20.png");
background-size: 150%;
background-position: top left; 
background-repeat: repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

import streamlit as st


st.markdown(
    """
    <style>
    .top-bar {
        background-color: #009999;
        color: white;
        padding: 0px;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 70px;
        z-index: 9999;
        text-align: center;
    }
    .top-bar a {
            color: white;
            margin-right: 0px;
            text-decoration: none;
            font-size: 25px; /* Tamaño de fuente de los enlaces */
        }

    .main-content {
        padding-top: 60px;  /* Espacio para que no tape el contenido */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="top-bar">
        <h1></h1>
        <div>
        <a href="https://proyectdaniel.streamlit.app/" style="color:white; margin-right:100px;"></a>
        <a href="oculto" style="color:white; margin-right:20px;"></a>
        <a href="?section=Lenguajes" style="color:white; margin-right:20px;"></a>
        <a href="?section=Sobre" style="color:white; margin-right:20px;"></a>
    </div>
    """,
    unsafe_allow_html=True
)


def set_custom_css():
    st.markdown(
        """
        <style>
        /* Estilos para la barra de desplazamiento en la página */
        ::-webkit-scrollbar {
            width: 16px; /* Ajusta el ancho de la barra de desplazamiento */
            height: 16px; /* Ajusta la altura de la barra de desplazamiento (horizontal) */
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1; /* Color del fondo de la pista de desplazamiento */
        }

        ::-webkit-scrollbar-thumb {
            background: #888; /* Color de la parte deslizable de la barra */
            border-radius: 10px; /* Radio de borde de la parte deslizable */
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #555; /* Color de la parte deslizable cuando se pasa el ratón */
        }
        </style>
        """,
        unsafe_allow_html=True)

def main1():
    set_custom_css()

    st.write(''*1000)

if __name__ == "__main__":
    main1()


#embed_component = {
#    'Linkedin': """
#    <script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
#    <div class="badge-base LI-profile-badge" data-locale="es_ES" data-size="large" data-theme="light" data-type="VERTICAL" data-vanity="danielchingasilva" data-version="v1">
#        <a class="badge-base__link LI-simple-link" href="https://cl.linkedin.com/in/danielchingasilva?trk=profile-badge">Daniel Elías Chinga Silva</a>
#    </div>
#    """
#}



embed_component = {
    'Linkedin': """
    <style>
        .badge-base {
            height: 1000px;  /* Ajusta la altura según sea necesario */
            width: 200%;
        }
    </style>
    <script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
    <div class="badge-base LI-profile-badge" data-locale="es_ES" data-size="large" data-theme="light" data-type="VERTICAL" data-vanity="danielchingasilva" data-version="v1">
        <a class="badge-base__link LI-simple-link" href="https://cl.linkedin.com/in/danielchingasilva?trk=profile-badge">Daniel Elías Chinga Silva</a>
    </div>
    """
}



#embed_component = {'Linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script> <div class="badge-base LI-profile-badge" data-locale="es_ES" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="danielchingasilva" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://cl.linkedin.com/in/danielchingasilva?trk=profile-badge">Daniel Elías Chinga Silva</a></div> """}

#with st.sidebar:


#st.subheader(":orange[Bienvenido]")


import streamlit as st

# Estilo CSS con una clase .box para envolver el título
title_style = """
<style>
.box {
    color: white;
    font-family: Helvetica, Arial, sans-serif;
    font-size: 47px;
    font-weight: bold;
    text-align: center;
    background-color: #191b20;
    padding: 0px;
    border: 4px solid #009999;
    border-radius: 10px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    margin: 5 auto;
    width: 676px;  /* Ancho de la caja */
    height: 124px;
}
.subtext {
    font-family: 'Inter', sans-serif;
    font-size: 19px; /* Tamaño del texto secundario */
    font-weight: normal;
}
</style>
"""

# Aplica el estilo
st.markdown(title_style, unsafe_allow_html=True)

# Muestra el título dentro de la caja con la clase .box
#Bienvenido a proyectos de machine learning
#st.markdown('<div class="box">Daniel Elías Chinga Silva</div>', unsafe_allow_html=True)

st.markdown("""
<div class="box">
    Daniel Elías Chinga Silva
    <p class="subtext">(Profesional con formación científica, orientado en gestión en datos)</p>
    <p class="subtext"></p>
</div>
""", unsafe_allow_html=True)

import streamlit as st




left, right = st.columns(2)
imagen1="linkedin2.png"


with left:
    #components.html(embed_component["Linkedin"], height=370)
    #components.html(embed_component["Linkedin"], height=344)
    #st.subheader("  Contacto")
    st.write("")
    #st.image(linkedin, width=300)  
    #st.write(linkedin)
    image_data = open(imagen1, "rb").read()
    encoded_image = base64.b64encode(image_data).decode()
    #<a href="www.linkedin.com/in/danielchingasilva" target="_blank">
    st.write(f'''
    <style>
        .resized-image {{
            display: block;
            margin-left: auto -1;
            margin-right: auto;
            border-radius: 10px;
            width: 306px;  /* Ajusta el tamaño aquí */
        }}
    </style>
     <a href="https://www.linkedin.com/in/danielchingasilva" target="_blank">
        <img src="data:image/jpeg;base64,{encoded_image}" class="resized-image"/>
    </a>
    ''', unsafe_allow_html=True)
    st.markdown("""
        <style>
        .box1 {
            border: 1px solid #009999; 
            padding: 23px;
            border-radius: 0px;
            background-color: #191b20;
            width: 310px;  /* Ancho de la caja */
            height: 168px;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="box1">  
            <h3>Contacto:</h3>          
            <ul>
                <li>danielchingasilva@gmail.com</li>
                <li>+569 3148 8069</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    
    #st.text("danielchingasilva@gmail.com")
    #st.write("egerg")
#with right:

#    st.subheader("Autor: Daniel C. S.")
#    st.write("https://github.com/DanielElias23")

#    st.write("www.linkedin.com/in/danielchingasilva")
     
#    st.subheader("Objetivos")

#    st.write("-Dar a conocer el conocimientos al respecto con machine learning")

#    st.write("-Demostrar dominio de conocimiento para solucionar problemas a diferentes problemáticas")

#    st.write("-Mostrar habilidades de programación enfocado al contexto ciencia de datos en las empresas")
#mention(label="DanielElias23", icon="github", url="https://github.com/DanielElias23",)

#left, right = st.columns(2)

#with left:
#    components.html("<iframe src='https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:12345678901234567890' height='370' width='400'></iframe>", height=370)
#4CAF50
with right:
    # Estilo CSS para aplicar un borde
    st.markdown("""
        <style>
        .box2 {
            border: 1px solid #009999; 
            padding: 23px;
            border-radius: 0px;
            background-color: #191b20;
            width: 315px;
            height: 530px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Envuelve el contenido en un div con la clase 'box'
    st.markdown("""
        <div class="box2">
            <h3>Autor: Daniel C. S.</h3>
            <p><a href="https://github.com/DanielElias23" target="_blank">https://github.com/DanielElias23</a></p>
            <p><a href="https://www.linkedin.com/in/danielchingasilva" target="_blank">www.linkedin.com/in/danielchingasilva</a></p>
            <h3>Objetivos</h3>
            <ul>
                <li>Dar a conocer el conocimiento al respecto con machine learning</li>
                <li>Demostrar dominio de diferentes tipos de modelos de machine learning en un aspecto más general posible</li>
                <li>Mostrar habilidades de programación enfocadas al contexto ciencia de datos en las empresas</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
#<li>Demostrar dominio de conocimiento para solucionar problemas a diferentes problemáticas</li>
st.sidebar.header("Barra de proyectos")
st.sidebar.write("Selecciona un proyecto")
#st.subhe
#ader("Problema")

st.logo("https://images.emojiterra.com/google/noto-emoji/unicode-15/color/512px/1f4c3.png")
#st.subheader("Descripción del problema")

#st.write("""
#         Estos proyectos son con el fin de mostrar habilidades de programación enfocado al área de ciencia de datos, los datos utilizados tienen sus contextos propios por lo que los modelos de inteligencia artificial no se pueden ocupar para uso general. Cada proyecto pretende mostrar habilidades diferentes en el contexto de machine learning, usando modelos diferentes y de diferentes categorías. Estos proyecto ya han sido realizados en los ejemplos mostrados en mí página de GitHub, pero no fueron implementados para la visualización de página web. Los análisis de los datos y las decisiones como la elección de modelos de machine learning está en el código en GitHub. 
#         """)


# Definir el estilo CSS
css = """
    <style>
    body {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    .badge-container {
        text-align: left;
        margin-left: -39px;
    }
    .badge-container img {
        transform: scale(1.6);  /* Escala las imágenes al 150% de su tamaño original */
        transform-origin: 0 0; /* Ajusta el origen de la transformación */
        margin: 35px;   /* Espacio entre los badges */
    }
    </style>
"""


st.markdown(css, unsafe_allow_html=True)

st.markdown("""
    <div class="badge-container">
        <img src="https://img.shields.io/badge/Python-Avanzado-orange.svg" alt="Python">
        <img src="https://img.shields.io/badge/SQL-Avanzado-orange.svg" alt="SQL">
        <img src="https://img.shields.io/badge/R-Intermedio-yellowgreen.svg" alt="R">
        <img src="https://img.shields.io/badge/CSS-Intermedio-yellowgreen.svg" alt="CSS">
        <img src="https://img.shields.io/badge/HTML-Intermedio-yellowgreen.svg" alt="HTML">
        <img src="https://img.shields.io/badge/NoSQL-Intermedio-yellowgreen.svg" alt="NoSQL">
    </div>
    """, unsafe_allow_html=True)         



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


