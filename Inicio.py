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
img = get_img_as_base64("image.jpg")
#https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqZ2lmjdQMNC3cyQ2g0i_wvigb5elGGBIPBg&s
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/fotos-premium/imagen-borrosa-centro-comercial-fondo-luz-bokeh_1157641-5174.jpg");
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
    font-family: 'Playfair Display', serif;
    font-size: 46px;
    font-weight: bold;
    text-align: center;
    background-color: #191b20;
    padding: 0px;
    border: 1px solid #4CAF50;
    border-radius: 10px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    margin: 5 auto;
    width: 676px;  /* Ancho de la caja */
    height: 150px;
}
</style>
"""

# Aplica el estilo
st.markdown(title_style, unsafe_allow_html=True)

# Muestra el título dentro de la caja con la clase .box
st.markdown('<div class="box">Bienvenido a proyectos de machine learning</div>', unsafe_allow_html=True)


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
            border: 1px solid #4CAF50;
            padding: 23px;
            border-radius: 40px;
            background-color: #191b20;
            width: 312px;  /* Ancho de la caja */
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

with right:
    # Estilo CSS para aplicar un borde
    st.markdown("""
        <style>
        .box2 {
            border: 1px solid #4CAF50; 
            padding: 23px;
            border-radius: 40px;
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
                <li>Demostrar dominio de conocimiento para solucionar problemas a diferentes problemáticas</li>
                <li>Mostrar habilidades de programación enfocadas al contexto ciencia de datos en las empresas</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.header("Barra de proyectos")
st.sidebar.write("Selecciona un proyecto")
#st.subhe
#ader("Problema")



#st.subheader("Descripción del problema")

#st.write("""
#         Estos proyectos son con el fin de mostrar habilidades de programación enfocado al área de ciencia de datos, los datos utilizados tienen sus contextos propios por lo que los modelos de inteligencia artificial no se pueden ocupar para uso general. Cada proyecto pretende mostrar habilidades diferentes en el contexto de machine learning, usando modelos diferentes y de diferentes categorías. Estos proyecto ya han sido realizados en los ejemplos mostrados en mí página de GitHub, pero no fueron implementados para la visualización de página web. Los análisis de los datos y las decisiones como la elección de modelos de machine learning está en el código en GitHub. 
#         """)


         



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


