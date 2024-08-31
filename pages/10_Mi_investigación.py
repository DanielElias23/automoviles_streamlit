from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import requests
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import base64
import streamlit as st

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}
img = get_img_as_base64("de_chat.png")

#https://www.blogdelfotografo.com/wp-content/uploads/2021/12/Fondo_Negro_3.webp
#https://w0.peakpx.com/wallpaper/596/336/HD-wallpaper-azul-marino-agua-clima-nubes-oscuro-profundo.jpg
#
#https://i.pinimg.com/564x/8b/5a/a4/8b5aa4783578968cd257b0b5418f3645.jpg
#https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkwNy1hdW0tNDQteC5qcGc.jpg
#https://i.pinimg.com/736x/f6/c1/0a/f6c10a01b8f7c3285fc660b0b0664e52.jpg
#https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkwNy1hdW0tNDQteC5qcGc.jpg
#https://c.wallhere.com/photos/ac/a0/color_pattern_texture_background_dark-672126.jpg!d

#Colores
#https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg
img = get_img_as_base64("fondo1.jpg")
#https://e0.pxfuel.com/wallpapers/967/154/desktop-wallpaper-solid-black-1920%C3%971080-black-solid-color-background-top.jpg
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: center;
background-repeat: repeat;
background-attachment: local;
background-attachment: scroll;
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
st.markdown(
    """
    <style>
    .top-bar {
        background-color: #424949;
        color: white;
        padding: 0px;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 20px;
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
st.logo("https://images.emojiterra.com/google/noto-emoji/unicode-15/color/512px/1f4c3.png")

title_style = """
<style>
.box {
    color: white;
    font-family: 'Playfair Display', serif;
    font-size: 46px;
    font-weight: bold;
    text-align: center;
    background-color: #148f77;
    padding: 5px; 
    border: 1px solid #009999;
    border-radius: 0px;
    box-shadow: 10px 10px 20px rgba(0, 0, 0, 0.3);
    margin: -68px -1px 0px 0px;
    width: 676px;  
    height: 84px;  
}
.subtext {
    font-family: 'Inter', sans-serif;
    font-size: 13px; /* Tamaño del texto secundario */
    font-weight: normal;
}
</style>
"""

# Aplica el estilo
st.markdown(title_style, unsafe_allow_html=True)

# Muestra el título y el texto adicional dentro de la misma caja con la clase .box
st.markdown("""
<div class="box">
    Investigación astronómica
</div>
""", unsafe_allow_html=True)

st.write("")

custom_css = """
<style>
.badge {
    display: inline-block;
    padding: 0.5em 1em;
    font-size: 0.9em;
    font-weight: 700;
    text-align: center;
    white-space: nowrap;
    border-radius: 0.25em;
    color: white;
    margin: 1px;
    transform: translateY(0px) translateX(7px);
}

.badge-primary1 {
    background-color: #17a2b8;
}

.badge-secondary {
    background-color: #6c757d;
}

.badge-success {
    background-color: #2ecc71
}

.badge-danger {
    background-color: #6c757d;
}

.badge-warning {
    background-color: #6c757d;
}

.badge-info {
    background-color: #157fe2;
}

.badge-light {
    background-color: #e2b015;
    color: #212529;
}
</style>
"""

# Aplica el estilo CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Muestra los badges usando HTML
st.markdown('''
<span class="badge badge-primary1">Ciencia astronomía</span>
<span class="badge badge-secondary">Area extragaláctica</span>
<span class="badge badge-info">Enfoque Conglomerados</span>
''', unsafe_allow_html=True)

#st.write("")
#st.write("")

st.subheader("Introducción a mi investigación")


st.write("La investigación se centra en galaxias distantes que experimentan sobredensidades o aglomeraciones. Estas son de gran relevancia, ya que según estudios, la evolución de dichas galaxias está estrechamente ligada a su conglomerado. A lo largo del tiempo, esos conglomerados lejanos se convierten en cúmulos galácticos, como el que habitamos.")
st.write("Un reto significativo de este campo radica en que, dada la distancia, es complicado identificar con precisión los conglomerados. Por ello, se ha seguido una línea investigativa para definirlos. Mi trabajo en particular se enfoca en el estudio del conglomerado z=2.80.")


conglo=Image.open("conglo1.png")

st.write(conglo)

st.subheader('Investigación: "Formación estelar y evolución de una sobredensidad de galaxias a redshift 2.8".')

st.write("En esta investigación se analizan y describen todas las características del conglomerado z=2.80, incluyendo parámetros como su masa, edad, tamaño y tasa de formación estelar, los cuales proporcionan información clave sobre su posible evolución. Además, se estudia la interacción del conglomerado con otros, así como los efectos que estas interacciones podrían tener.")

st.write('Un aspecto relevante es el significado de "redshift", representado por la letra "z". En términos generales, está relacionado con la distancia de las galaxias, pero también considera factores cruciales como la expansión del universo, el efecto Doppler, y los efectos gravitatorios y cuánticos relacionados con la interacción de los fotones.')


# Declarar variable de estado
if 'pdf_ref' not in ss:
    ss.pdf_ref = None

# Subir archivo PDF
#uploaded_file = st.file_uploader("Subir archivo PDF", type='pdf', key='pdf')

pdf_path = "inves.pdf"  # Reemplaza "archivo.pdf" con el nombre de tu archivo

try:
    with open(pdf_path, "rb") as pdf_file:
        binary_data = pdf_file.read()
        ss.pdf_ref = binary_data
except FileNotFoundError:
    st.error(f"El archivo {pdf_path} no se encontró en el directorio.")

# Si se ha cargado un archivo PDF, guardarlo
#if binary_data:
#    ss.pdf_ref = binary_data

# Visualizar el PDF si se ha cargado
if ss.pdf_ref:
    #binary_data = ss.pdf_ref.getvalue()
    # Obtener el tamaño de la ventana del navegador
    st.markdown(
        """
        <style>
        .pdf-viewer-container {{
            width: 100vw;
            height: 80vh; /* Ajusta la altura al porcentaje deseado */
            border: none;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    pdf_viewer(input=binary_data, width=705, height=900)

st.subheader("Contribución a la ciencia")

st.write("El conglomerado muestra una formación estelar activa, en contraste con otros que han revelado una formación más pasiva. Se estima que su masa podría estar entre 2.88 y 3.85 billones de masas solares, con un tamaño aproximado de 48.51 a 52.81 cMpc³. Las galaxias de este conglomerado exhiben comportamientos particulares, con una relación entre masa y tamaño que parece estar influenciada por el entorno. No se observaron efectos directos de las sobredensidades de Lyman-alfa mencionadas en el estudio de Zheng.")

st.write("Este conglomerado tiene el potencial de convertirse en un cúmulo galáctico similar al de Fornax, y su estudio podría ofrecer un gran aporte en la comprensión de la formación estelar en el universo.")

st.subheader("Referencias principales")

st.write("La investigación se basa principalmente en cinco investigaciones que te las describo a continuación:")

st.write("* Creación de base de datos y definición de parámetros, [en este sitio web.](%s)." % "https://arxiv.org/pdf/1510.02106")

st.write("* Descubrimiento de los conglomerados, [en este sitio web.](%s)." % "https://arxiv.org/pdf/2007.12314")

st.write("* Definición de evolución y sus parámetros, [en este sitio web.](%s)." % "https://arxiv.org/pdf/1310.2938")

st.write("* Definición de evolución morfológica, [en este sito web.](%s)." % "https://arxiv.org/pdf/1705.01634")

st.write("* Conglomerados cercanos, [en este sitio web.](%s)." % "https://arxiv.org/pdf/1606.07073")




