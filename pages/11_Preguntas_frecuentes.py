import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_extras.mention import mention
import base64
import streamlit as st

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}
img = get_img_as_base64("fondo1.jpg")
#https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqZ2lmjdQMNC3cyQ2g0i_wvigb5elGGBIPBg&s
#https://img.freepik.com/fotos-premium/imagen-borrosa-centro-comercial-fondo-luz-bokeh_1157641-5174.jpg
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
        height: 23px;
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
    height: 158px;  
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
    Preguntas frecuentes de Astronomía
</div>
""", unsafe_allow_html=True)

st.write("")
st.subheader("Introducción")

st.write("Esta sección está dedicada a profesionales de otras áreas que desean saber por qué un astrónomo puede ser científico de datos o analista de datos. La respuesta tiene muchas aristas, es por eso que en esta sección se responderán esas preguntas, además de entender todo lo que sabe un astrónomo. Estoy consciente que los astrónomos son pocos por lo tanto mucha gente desconoce lo que hacen.")




import streamlit as st

# Estilo CSS personalizado para un acordeón simulado
st.markdown("""
    <style>
    .accordion {
        background-color: #008CBA;
        color: white;
        cursor: pointer;
        padding: 18px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        font-size: 18px;
        border-radius: 10px;
        transition: 0.4s;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .accordion:hover {
        background-color: #005f73;
    }

    .panel {
        padding: 0 18px;
        background-color: #2c3e50;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.2s ease-out;
        border-radius: 0 0 10px 10px;
        margin-bottom: 10px;
        color: white; /* Color del texto */
        border: none; /* Sin borde inicialmente */
    }

    input[type="checkbox"] {
        display: none;
    }

    input[type="checkbox"]:checked ~ .panel {
        max-height: 600px;
        padding: 18px;
        border: 2px solid #008CBA; /* Borde cuando el panel está desplegado */
    }

    </style>
""", unsafe_allow_html=True)

# HTML con el truco de checkbox para controlar la visibilidad
st.markdown("""
<div>
    <label class="accordion" for="accordion-1">1. ¿Qué es lo que realmente sabe un Astrónomo?</label>
    <input type="checkbox" id="accordion-1">
    <div class="panel">
        <p>Un astrónomo es un profesional con un alto nivel en matemática, física y programación. Están 100% dedicados a la investigación, logrando un nivel alto en estas disciplinas.</p>
        <p>Un astrónomo debe tener manejo de ingles deseablemente hablado y uno de los sellos de la carrera es que tiene un nivel teórico de los más altos estándares, logrando obtener un gran desempeño en las disciplinas antes mencionadas. Los investigadores deben exhibir sus trabajos mundialmente.</p>
        <p>Una investigación con machine learning como la que muestro a continuación puede tardar años, incluso trabajando con grupos de investigación, en <a href="https://iopscience.iop.org/article/10.3847/1538-4365/ad5c66/pdf" target="_blank">este sitio web.</a></p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-2">2. ¿Cuál es la diferencia de astronomía con otras carreras?</label>
    <input type="checkbox" id="accordion-2">
    <div class="panel">
        <p>La diferencia es que las astronomía se enfoca en la investigación siendo su principal resultado la resolución de problemas o respuestas de carácter teórico. Como el nivel teórico es tan alto no necesita supervisión, ni limitaciones, se exigen resultados. Eso lo hace un profesional muy adaptable a grupos de trabajo con enfoque en resultados, como podría ser la toma de decisiones. Un profesional de otra área se limita a hacer exclusivamente su trabajo.</p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-3">3. ¿Cuál es la diferencia entre ciencias aplicadas y ciencias teóricas?</label>
    <input type="checkbox" id="accordion-3">
    <div class="panel">
        <p>La diferencia de una ciencia teórica es que no tiene la posibilidad de realizar intentos de prueba y error como lo hace una ciencia aplicada, es por ello que la analítica sea un papel muy importante en la astronomía, esto debe estar acompañado de muchas investigaciones que la apoyen, puesto que toda la investigación se basará en este análisis, eso logra que sea un profesional que tenga mucho mayor nivel teórico y de interpretación de datos, más exigente en estos aspectos.</p>
        <p>El beneficio de realizar estas ciencias teóricas es que no gastan recursos económicos como lo hace una ciencia aplicada. La inversión se hace cuando los científicos teóricos están seguros de que la inversión tendrá resultados.</p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-4">4. ¿Por qué un astrónomo es un buen analítico?</label>
    <input type="checkbox" id="accordion-4">
    <div class="panel">
        <p>Esto se da porque un astrónomo basara TODO su trabajo en el análisis, toda la investigación posterior dependerá de ello, por lo tanto debe lograr concordancia matemática, física y la relación con otras investigaciones. Además es por no tiene la posibilidad de experimentar de forma práctica, esto eleva demasiado los estándares teóricos.</p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-5">5. ¿Con qué trabajan los astrónomos?</label>
    <input type="checkbox" id="accordion-5">
    <div class="panel">
        <p>Actualmente un astrónomo trabaja con programación en lenguajes lógicos como python, ocupando librerías que sirvan para hacer cálculos como numpy, pandas, matplotlib, scipy, sckit-learn, tensorflow, AstroPy y software exclusivos de astronomía. Eso es solo el nivel básico, en la práctica un astrónomo con años en el rubro tiene un alto nivel de aprendizaje, trabajando con infinidades de lenguajes como C++ o Java y software exclusivos ya sea de astronomía o estadísticas.</p>
        <p>Algunas organizaciones astronómicas trabajan con lenguajes que son privados como S-Lang, IDL, Shell Scripting también algunas astrónomos trabajan con simulaciones matemáticas de alto nivel teórico mundial que requieren especializaciones como SAM (Semi-Analytical Models).</p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-6">6. ¿Los Astrónomos trabajan con machine learning?</label>
    <input type="checkbox" id="accordion-6">
    <div class="panel">
        <p>Sí, hay muchos astrónomos que trabajan con machine learning, en general en astronomía se prefiere los cálculos exactos y que entreguen certezas. Machine learning se ocupa cuando no es posible obtener esa respuesta de ninguna otra forma que no sea con una predicción.</p>
        <p>Por ejemplo en la investigación que muestro a continuación podría tomar mucho tiempo definir galaxias por galaxia como AGN (Es un tipo de galaxias), en la investigación clasifica a 14.245 galaxias como AGN gracias a un modelo CNN en <a href="https://iopscience.iop.org/article/10.3847/1538-4365/aaf9a2/pdf" target="_blank">este sitio web.</a></p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-7">7. ¿Los datos en astronomía son más complejos?</label>
    <input type="checkbox" id="accordion-7">
    <div class="panel">
        <p>Sí, para realizar una investigación de carácter científico se debe entender que los datos dependerán fuertemente de la herramienta tecnológica que se ocupen, estos entregaran la calidad de los datos y su posible interpretación. En el caso de Astronomía estos procesos pueden ser muy complejos y desafíos mundiales, llega a tratarse de apoyos internacionales. Un astrónomo debe conocer las especificaciones de estas tecnologías.</p>
        <p>Recordando que los datos tienen unidades de medida y rangos de medición, además deben ser trasformador porque son solo fotones y señales.</p>
        <p>Usualmente se describen estos aparatos en documentaciones o páginas web, aunque solo lo puede entender un astrónomo como por ejemplo el observatorio ALMA en <a href="https://www.eso.org/public/teles-instr/alma/" target="_blank">este sitio web.</a></p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-8">8. ¿Qué complejidades tiene ocupar un dataset de astronomía?</label>
    <input type="checkbox" id="accordion-8">
    <div class="panel">
        <p>Los datos de astronomía pueden ser imágenes, señales o tablas, pero en este caso me voy a referir a las tablas de datos. Las tablas de datos son muy diferentes, están generadas por otros astrónomos que se dedican a especificar cada elemento de esas tablas.</p>
        <p>Lo primero que se debe entender que los datos tienen rangos de medición, por lo que tienen un valor como tal, se divide en 3 valores, el valor estimado de la medición, el valor mínimo y el máximo del rango de medición, por lo que al hacer cualquier operación con una sola columna se deben ocupar estos 3 valores en conjunto, Los datos no pueden estar sin su error de medición no se considera válido.</p>
        <p>Además se debe considerar que los datos tienen unidades de medida por lo que no es posible combinar con otras columnas que tenga unidades de medida diferentes.</p>
        <p>Por otro lado cada columna en astronomía de una tabla son líneas investigativas, incluso cada fila puede tener su línea investigativa, esto hace muy complejo trabajarlas.</p>
        <p>Por ejemplo en mi investigación me referí a esta investigación <a href="https://iopscience.iop.org/article/10.1088/0067-0049/203/2/24/pdf" target="_blank">este sitio web</a>, porque en esta investigación se definió un radio para las galaxias que estaba ocupando, es muy importante porque en esa investigación se definió como se cálculo.</p>
        
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-9">9. ¿Qué profesionales pueden hacer investigaciones astronómicas?</label>
    <input type="checkbox" id="accordion-9">
    <div class="panel">
        <p>Solo profesionales que hayan obtenido algún grado académico de astronomía. La astronomía es una ciencia compleja que requiere especialización. Algunos profesionales del área de la biología, química que deseen trabajar en astronomía deben sacar algún grado académico en astronomía, ellos usualmente trabajan la astrobiología o generación de teorías de formación química. Algunos geólogos se pueden especializar en astronomía, puesto que existe la astronomía planetaria, aportan mucho conocimiento de la formación de planetas.</p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-10">10. ¿La Astronomía tiene muchas especialidades?</label>
    <input type="checkbox" id="accordion-10">
    <div class="panel">
        <p>Si, dado la complejidad de astronomía tiene muchas especialidades con muchas temáticas que son complemente diferentes entre objetos astronómicos, incluso gente que se limita a solo hacer investigaciones de creación de software, modelos, mediciones, simulaciones, teorías matemáticas exclusivas para astronomía.</p>
        <p>Acá te muestro las principales especialidades de astronomía con investigaciones de ejemplos:</p>
        <p>* Astronomía estelar, por ejemplo en <a href="https://arxiv.org/pdf/2408.12765" target="_blank">este sitio web.</a></p>
        <p>* Astronomía galáctica, por ejemplo en <a href="https://arxiv.org/pdf/2402.12443" target="_blank">este sitio web.</a></p>
        <p>* Astronomía extragalácitca cercana, por ejemplo en <a href="https://arxiv.org/pdf/2408.08026" target="_blank">este sitio web.</a></p>
        <p>* Astronomía extragaláctica lejana, por ejemplo en <a href="https://arxiv.org/pdf/1510.05647" target="_blank">este sitio web.</a></p>
        <p>* Astronomía planetaria, por ejemplo en <a href="https://arxiv.org/pdf/2403.12512" target="_blank">este sitio web.</a></p>
        <p>* Astronomía cosmológica, por ejemplo en <a href="https://arxiv.org/pdf/1412.4872" target="_blank">este sitio web.</a></p>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div>
    <label class="accordion" for="accordion-11">11. ¿Por qué un astrónomo puede querer trabajar con empresas?</label>
    <input type="checkbox" id="accordion-11">
    <div class="panel">
        <p>Es una pregunta común que se repite constantemente, acá te respondo algunas de las razones más importantes:</p>
        <p>* Algunos astrónomos les gusta trabajar con empresas, ya que se dan cuenta de que el trabajo es muy similar y tienen habilidades de sobra para demostrar en ambientes empresariales</p>
        <p>* Astronomía es muy exigente teóricamente requiere estudio continuo diario y el estudio no termina nunca, la ciencia de alto nivel teórico, es por ello que los ingenieros sufren tanto con los ramos de física, no se puede enseñar de forma facil. Un astronomo puede cansarse de tanta exigencia teorica y querer buscar otros horizontes.</p>
        <p>* Los astrónomo tienen pocas ofertas laborales y no pueden elegir en gran medida donde pueden trabajar, por lo que requieren emigrar a otras comunas, regiones o países. Trabajar con empresas les da esa garantía de poder elegir una localidad.</p>
</div>""", unsafe_allow_html=True)


st.subheader("Fin de las respuestas")


st.write("Agradezco la lectura de esta información, es especial para personas que desean entender por qué un profesional de la astronomía puede trabajar con datos. En general las empresas tienen desafíos diferentes, pero para lo que maneja un astrónomo es fácil adaptarse, ya que maneja un nivel teórico más exigente que lo que ocupan el común de las empresas. Puede trabajar en otras áreas como economía, programación, ingeniería, minería que tengan que ver con manejo de datos debido a su perfil investigativo de aprendizaje continuo.")













