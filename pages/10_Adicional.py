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

st.write("Esta sección está dedicada a profesionales de otras áreas que no logran entender porque un astrónomo puede ser científico de datos o analista de datos. La respuesta tiene muchas aristas, es por eso que en esta sección se responderán esas preguntas, además de entender bien todo lo que sabe un astrónomo considerando que los profesionales que se dedican a la astronomía son pocos en general.")

st.subheader("¿Qué es lo que realmente sabe un Astrónomo?")

st.write("La astronomía es una ciencia física tan compleja que requiere una especialización, los astrónomos son 100% dedicados a la investigación, la astronomía es una ciencia que requiere alto nivel de :red[matemática, física y programación].")

st.write('Estan enfocados en las :blue[investigaciones "papers"], un astronomo de pregrado puede entender en gran medida la lectura de investigaciones, pero para realizaras se requiere un doctorado, puesto que se necesita alto nivel tecnico y tener grupos de investigacion en las diferentes areas.')

st.write("Al hacer una investigación un astronomo no puede equivocarse en ningun paso tanto a nivel de calculos como a nivel teorico, se debe tener concordancia con otras investigaciones, cualquier error podria anular la investigación.")

st.subheader("¿Cual es la diferencia de astronomía con otras carreras?")

st.write("La astronomía es una ciencia exacta del :blue[más alto nivel teórico], es por ello que un astrónomo se puede desenvolver bien en areas que estan relacionadas con las disciplinas que ocupa como lo es fisica, matematica y programación.") 

st.write("Hablando como una persona que conoce tanto la matemática como la física, te puedo explicar porque muchas carreras tienen los primeros años materias de física sin necesitarlo realmente. Esto sucede porque la matemática es una ciencia que es difícil de entender por si sola, es difícil entender el contexto o los reales resultados de la aplicación de sus diferentes áreas. Esto se complementa muy bien con la física, una persona que sabe de física puede entender muy bien estos resultados, puesto que la estadística puede aplicarse a la física y lograr resultados notorios. La mecánica cuántica es completamente estadística y ha logrado resultados sorprendentes, solo con buena interpretación de la matemática.")

st.write("Esto resultados se pueden conseguir incluso si los objetos a analizar están a cientos de miles de millones de años luz de distancia. La especialidad de la ciencia física que puede obtener esto es la astronomía. Es por ello que un astrónomo es un buen conocedor de técnicas estadística.")

st.subheader("¿Cuál es la diferencia entre ciencias aplicadas y ciencias teóricas?")

st.write("La diferencia de una ciencia teórica es que no tiene la posibilidad de realizar intentos de prueba y error como lo hace una ciencia aplicada, es por ello que la analítica sea un papel muy importante en la astronomía, esto debe estar acompañado de muchas investigaciones que la apoyen, puesto que toda la investigación se basará en este análisis, eso logra que sea un profesional que tenga mucho mayor nivel teórico y de interpretación de datos, más exigente en estos aspectos.")

st.write("El beneficio de realizar estas ciencias teóricas es que no gastan recursos económicos como lo hace una ciencia aplicada. :blue[La inversión se hace cuando los científicos teóricos están seguros] de que la inversión tendrá resultados.")

st.subheader("¿Por que un astronomo es un buen analitico?")

st.write("La razon es porque un astronomo debe basara TODA su investigación en sus :red[analisis previo], logrando concordancia matematica, fisica y la relación con otras investigaciones. Ademas es por no tiene la posibilidad de experimentar de forma practica, esto eleva demaciado los estandares teoricos.")

st.subheader("¿Con que trabajan los astronomos?")

st.write("Actualmente los Astronomos estan trabajando con programación, para llevar a cabo esta ciencia se necesita mucho calculo y entendimiento de muchas investigaciones. Esto hace que la Astronomía sea una ciencia altamente efectiva en sus analisis ya que puede otorgar soluciones sin necesidad de gastar recursos.")

st.write("Un astronomo trabaja con lenguajes logicos como :blue[python], ocupando librerias que sirvan para hacer calculos como :blue[numpy, pandas, matplotlib, scipy, sckit-learn, tensorflow y software esclusivos de astronomía]. En la practica un astronomo profesional tiene un alto nivel de aprendizaje, trabajando con infinidades de lenguajes y software exclusivos ya sea de astronomia o estadisictas, en ocaciones algunos astronomos tienen conocimientos de programacion que conocen muy pocas personas en el mundo.")

st.write("Algunas organizaciones trabajan con lenguajes que son privados, tambien algunas astronomos trabajan con simulaciones matematicas de alto nivel teorico mundial que requieren especializaciones.")

st.subheader("¿Los Astronomos trabajan con machine learning?")

st.write("Si, hay muchos Astronomos que trabajan con machine learning, pero es mas importante hacer calculos exactos y explicitos. Por esto se ocupa machine learning cuando no es posible obtener esa respuesta de ninguna otra forma que no sea con una predicción. Esto tiene muchas mas complejidades para la interpretación de resultados.")

st.subheader("¿Que diferencia hay entre los datos de las empresas y astronomia?")

st.write("Para realizar una investigación se caracter cientifico se debe entender que los datos dependeran fuertemente de la herramienta tecnologica que se ocupen, estos entregaran la calidad de los datos y su posible interpretación. En el caso de Astronomía estos procesos pueden ser muy complejos y desafios mundiales, llega a tratarse de apoyos internacionales. Los datos de una mala herramienta pueden entregar resultados erroneos, el cientifico a cargo puede entender cuando los resultados pueden ser erroneos, puesto que no se condicen con otras investigaciones o con leyes fisicas fundamentales.")

st.write("Un astronomo debe conocer las partes de los observatorios, sus capacidad, limites y especificaciones, estos estan descritos en las paginas web oficiales de cada observatorio.")

st.write("Usualmente los datos mas importantes para los astronomos tiene mas elementos adicionales a considerar, los del aparato tecnologicos que obtuvieron los datos deben considerar el dia, el clima, calidad del aire, entre otros factores importantes, que debe conocer al realizar el analisis.")

st.subheader("¿Que complejidades tiene ocupar un dataset de astronomia?")

st.write("Los datos en astronomía son mas complejos de tratar que los de las empresas, por lo que no es comparable el nivel de complejidad en la 'cantidad de datos', algunas personas creen erronemanete que los datos de astronomía son comparables con los que se ocuparan en otros rubros. Esto es incorrecto, los datos en astronomía tienen comportamientos especiales.")

st.write("Los datos en Astronomía tanto cada fila y cada columna pueden ser investigaciones o lineas de investaciones historicas complejas que incluso van cambiando con el tiempo, esto vuelve muy complejo de tratar un dataset, puesto que tomar una simple decision de eliminar un dato puede ser crucial. Un dataset por ejemplo de 10 columnas puede implicar que debes saber que se descubrio en 10 investigaciones diferentes. En algunos casos puede conocerse el significado de los datos, pero como mencione anteriormente siempre se debe considerar la forma como se obtuvieron esos datos, esto usualmente tienen una investigación de respaldo que debes conocer.")

st.write("Otra complicación adicional, es que los datos en astronomía no son puntuales, esto dado que vienen desde mediciones, las mediciones tienen errores de medicion asociado como todo en astronomia, es decir el llamado dato en si tiene asociado un rango, por lo que al procesar ese dato se debe procesar con el error que es un rango. Un proceso simple como sumar una valor a la columna tiene una complejidad adicional que se debe sumar a los errores de medicion en algunos casos con funciones particulares. Usualmente los errores estan representados como columnas adicionales.")

st.write("Los datos tienen unidades de medidas por lo que al hacer cualquier tipo de operación con esas columnas se debe considerar la unidad de medida, es decir puede darse el caso que las columnas no correspondan las unidades de medidas y necesitan calculos adicionales.")

st.write("Se debe entender que en algunos casos los datos tienen valores incoherentes, esto solo puede entender un cientifico, ya que los datos muestras numeros totalmente anormales o que no consideren con las leyes fisicas.")

st.subheader("¿Se puede llegar y hacer una investigación astronomica?")

st.write("La astronomía es una ciencia que tiene alrededor de 100 años aproximadamente por lo tanto hay muchas investigaciones que ya se hicieron y que obtuvieron resultados, no es posible repetir una de investigacines se considera no valida o sera rechazada y tampoco mostrar algo que no esta cientificamente sustentado con investigaciones adicionales. Por ello tambien un cientifico debe estar al dia en las investigaciones para saber que investigacion es validad o cuales ya fueron hechas. Tambien conocer nuevos descubrimientos.")

st.subheader("¿Cualquier profesional puede hacer una investigación?")

st.write("Las teorias fisicas y astronomicas fueron demostradas matematicamente, es por ello que las investigaciones solo las puede hacer un astronomo, ya que cualquier inconsistencias con otras investigaciones anulara completamente la investigacion. Muchas de las investigaciones son de alto nivel teorico y tiene conceptos muy complejos.")

st.subheader("¿La Astronomía tiene muchas especialidades?")

st.write("Si, dado la complejidad de astronomia hace que la Astronomía tenga muchas especialidades, tiene muchas tematicas que son complemente diferentes entre objetos astronomicos, incluso gente que se limita a solo hacer investigaciones de creacion de software, modelos, mediciones, simulaciones, teorias matematicas exclusivas para astronomia.")

st.write("Aca te muestro las principales especialidad de astronomia:")

st.write("* Astronomía estelar")

st.write("* Astronmía galactica")

st.write("* Astronomía extragalacitca cercana")

st.write("* Astronomía extragalactica lejana")

st.write("* Astronomia planetaria")

st.write("* Astronomía cosmologica")

st.subheader("¿Por que un astronomo puede querer trabajar con empresas?")

st.write("Es una pregunta comun que se repite constantemente, aca te respondo algunas de las razones mas importantes:")

st.write("* Algunos astronomos les gusta trabajar con empresas, ya que se dan cuenta que el trabajo es muy similar y tienen habilidades de sobra para demostrar en ambientes empresariales")

st.write("* Astronomía es muy exigente teoricamente requiere estudio continuo diario y el estudio no termina nunca, la cienica es una sola y de alto nivel teorico es por ello que los ingenieros sufren tanto con los ramos de fisica, no se puede enseñar desde poco.")

st.write("* Los astronomo tienen pocas ofertas laborales y no pueden elegir en gran medida donde pueden trabajar, por lo que requieren emigrar a otras comunas, regiones o paises. Trabajar con empresas les da esa garantia de porder elegir una localidad.")

st.subheader("Fin de las respuestas")


st.write("Agradezco la lectura de esta información es espacial para personas que desean entender porque un profesional de la astronomía puede trabajar con datos. En general las empresas tienen desafios diferentes, pero para lo que maneja un astronomo es facil adaptarse. Ademas posee capacidad de entender investigaciones de otros ambitos ya sea de economicos como de programación, incluso fisicos de otras areas.")















