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

#https://www.blogdelfotografo.com/wp-content/uploads/2021/12/Fondo_Negro_3.webp
#https://w0.peakpx.com/wallpaper/596/336/HD-wallpaper-azul-marino-agua-clima-nubes-oscuro-profundo.jpg
#
#https://i.pinimg.com/564x/8b/5a/a4/8b5aa4783578968cd257b0b5418f3645.jpg
#https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkwNy1hdW0tNDQteC5qcGc.jpg
#https://i.pinimg.com/736x/f6/c1/0a/f6c10a01b8f7c3285fc660b0b0664e52.jpg
#https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkwNy1hdW0tNDQteC5qcGc.jpg
#https://c.wallhere.com/photos/ac/a0/color_pattern_texture_background_dark-672126.jpg!d

#https://e0.pxfuel.com/wallpapers/967/154/desktop-wallpaper-solid-black-1920%C3%971080-black-solid-color-background-top.jpg
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.colorhexa.com/191b20.png");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg");
background-size: 30%;
background-position: top left; 
background-repeat: no-repeat;
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


embed_component = {'Linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script> <div class="badge-base LI-profile-badge" data-locale="es_ES" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="danielchingasilva" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://cl.linkedin.com/in/danielchingasilva?trk=profile-badge">Daniel Elías Chinga Silva</a></div> """}

#with st.sidebar:


st.subheader(":orange[Bienvenidos] 👋")

st.title("Proyectos de machine learning")

left, right = st.columns(2)

with left:
    components.html(embed_component["Linkedin"], height=370)

with right:

    st.subheader("Autor: Daniel C. S.")
    st.write("https://github.com/DanielElias23")

    st.write("www.linkedin.com/in/danielchingasilva")
     
    st.subheader("Objetivos")

    st.write("-Dar a conocer el conocimientos al respecto con machine learning")

    st.write("-Demostrar dominio de conocimiento para solucionar problemas a diferentes problemáticas")

    st.write("-Mostrar habilidades de programación enfocado al contexto ciencia de datos en las empresas")
#mention(label="DanielElias23", icon="github", url="https://github.com/DanielElias23",)



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


