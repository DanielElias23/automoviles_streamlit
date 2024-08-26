import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model 


from warnings import filterwarnings
filterwarnings('ignore')


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}
img = get_img_as_base64("de_chat.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.colorhexa.com/191b20.png");
background-size: 100%;
background-position: top right;
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

st.markdown(page_bg_img, unsafe_allow_html=True)
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

Life_df = pd.read_csv('Life_E_Data.csv')
Cancer_df = pd.read_csv('Cancer_Data.csv')

title_style = """
<style>
.box {
    color: white;
    font-family: 'Playfair Display', serif;
    font-size: 46px;
    font-weight: bold;
    text-align: center;
    background-color: #191b20;
    padding: 5px; /* Ajusta el padding para dar espacio al texto */
    border: 1px solid #009999;
    border-radius: 0px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    margin: 5 auto;
    width: 676px;  /* Ancho de la caja */
    height: 84px;  /* Ajusta la altura automáticamente */
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
    Predicciones médicas
    <p class="subtext"></p>
    <p class="subtext"></p>
</div>
""", unsafe_allow_html=True)

st.write("")

st.write("* Tipo de Modelo: :green[Red Neuronal]")
  
st.write("* Modelos utilizados: :green[Artificial Neural Network (ANN)]")
st.write("* Etiqueta: :green[Esperanza de vida y Clasificación del cáncer]")

#st.title("Predicciones médicas")

#st.write(":green[*Modelo ML - ANN - Regresión y Clasificación*]")

pagina1, pagina2=st.tabs([" Regresión ", " Clasificación "])

with pagina1:
     
     st.subheader("Exploración y Análisis")
     
     st.write("Los datos presentados son de un estudio de diferentes países que se realizó entre el 2000 - 2015 por el Observatorio Mundial de la Salud (GHO). El estudio se centra en los países y su esperanza de vida, estos estudios son realizados cada año para cada país entre los años mencionados.")
     st.write("Lo que se busca pronosticar es cual sería la esperanza de vida de un país solo conociendo sus características y no de la manera tradicional que es con cálculos técnicos sobre la salud.")
     
     st.subheader("Manipulación y Limpieza")
     
     st.write("Se puede ver los datos explícitamente y sus respectivos datos nulos")
     
     st.code("""
          print(Life_df.head(5)
          print(Lide_df.shape)
          print(Life_df.isnull().sum().T)
     """)
          
     st.write(Life_df.head(5))
     st.write(Life_df.shape)
     st.write(pd.DataFrame(Life_df.isnull().sum()).T)
     
     st.subheader("Ingeniería de características")
     
     
     st.write("Existe un gran cantidad de datos nulos en algunas columnas, lo que podría perjudicar a la predicción, por lo que es mejor remplazar esos datos con sus promedios.")
     
     st.code("""
         imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)
         Life_df['Life expectancy ']=imputer.fit_transform(Life_df[['Life expectancy ']])
         Life_df['Adult Mortality']=imputer.fit_transform(Life_df[['Adult Mortality']])
         Life_df['Alcohol']=imputer.fit_transform(Life_df[['Alcohol']])
         Life_df['Hepatitis B']=imputer.fit_transform(Life_df[['Hepatitis B']])
         Life_df[' BMI ']=imputer.fit_transform(Life_df[[' BMI ']])
         Life_df['Polio']=imputer.fit_transform(Life_df[['Polio']])
         Life_df['Total expenditure']=imputer.fit_transform(Life_df[['Total expenditure']])
         Life_df['Diphtheria ']=imputer.fit_transform(Life_df[['Diphtheria ']])
         Life_df['GDP']=imputer.fit_transform(Life_df[['GDP']])
         Life_df['Population']=imputer.fit_transform(Life_df[['Population']])
         Life_df[' thinness  1-19 years']=imputer.fit_transform(Life_df[[' thinness  1-19 years']])
         Life_df[' thinness 5-9 years']=imputer.fit_transform(Life_df[[' thinness 5-9 years']])
         Life_df['Income composition of resources']=imputer.fit_transform(Life_df[['Income composition of resources']])
         Life_df['Schooling']=imputer.fit_transform(Life_df[['Schooling']])
     """)
     
     imputer = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)

     Life_df['Life expectancy ']=imputer.fit_transform(Life_df[['Life expectancy ']])
     Life_df['Adult Mortality']=imputer.fit_transform(Life_df[['Adult Mortality']])
     Life_df['Alcohol']=imputer.fit_transform(Life_df[['Alcohol']])
     Life_df['Hepatitis B']=imputer.fit_transform(Life_df[['Hepatitis B']])
     Life_df[' BMI ']=imputer.fit_transform(Life_df[[' BMI ']])
     Life_df['Polio']=imputer.fit_transform(Life_df[['Polio']])
     Life_df['Total expenditure']=imputer.fit_transform(Life_df[['Total expenditure']])
     Life_df['Diphtheria ']=imputer.fit_transform(Life_df[['Diphtheria ']])
     Life_df['GDP']=imputer.fit_transform(Life_df[['GDP']])
     Life_df['Population']=imputer.fit_transform(Life_df[['Population']])
     Life_df[' thinness  1-19 years']=imputer.fit_transform(Life_df[[' thinness  1-19 years']])
     Life_df[' thinness 5-9 years']=imputer.fit_transform(Life_df[[' thinness 5-9 years']])
     Life_df['Income composition of resources']=imputer.fit_transform(Life_df[['Income composition of resources']])
     Life_df['Schooling']=imputer.fit_transform(Life_df[['Schooling']])
     
     st.write("Obteniendo que ya no existen datos nulos.")
     
     st.write(pd.DataFrame(Life_df.isnull().sum()).T)
     
     st.write("También un paso importante es remplazar los outliers por el promedio.")
     
     st.code("""
          cols_to_handle_outliers = [
          'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
          'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio',
          'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
          ' thinness  1-19 years', ' thinness 5-9 years',
          'Income composition of resources', 'Schooling'
          ]
          for col_name in cols_to_handle_outliers:
                q1 = Life_df[col_name].quantile(0.25)
                q3 = Life_df[col_name].quantile(0.75)
                iqr = q3 - q1
          lower_bound = q1 - 1.5 * iqr
          upper_bound = q3 + 1.5 * iqr
          Life_df[col_name] = np.where((Life_df[col_name] > upper_bound) | (Life_df[col_name] < lower_bound), np.mean(Life_df[col_name]), Life_df[col_name])
          st.write(Life_df.shape)
     """)
     
     cols_to_handle_outliers = [
     'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
     'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio',
     'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
     ' thinness  1-19 years', ' thinness 5-9 years',
     'Income composition of resources', 'Schooling'
     ]
 
     for col_name in cols_to_handle_outliers:
  
         q1 = Life_df[col_name].quantile(0.25)
         q3 = Life_df[col_name].quantile(0.75)
         iqr = q3 - q1

     lower_bound = q1 - 1.5 * iqr
     upper_bound = q3 + 1.5 * iqr

   
     Life_df[col_name] = np.where((Life_df[col_name] > upper_bound) | (Life_df[col_name] < lower_bound), np.mean(Life_df[col_name]), Life_df[col_name])
     
     st.write(Life_df.shape)

     st.write("Es necesario codificar las variables categóricas para que puedan ser procesadas por el modelo.")
     
     st.code("""
         cols_to_encode = ['Country', 'Status']
         label_encoder_df = LabelEncoder()
         for col in cols_to_encode:
               Life_df[col] = label_encoder_df.fit_transform(Life_df[col])
     """)
     
     cols_to_encode = ['Country', 'Status']

     label_encoder_df = LabelEncoder()
     for col in cols_to_encode:
             Life_df[col] = label_encoder_df.fit_transform(Life_df[col])
     
     st.write("Por último separar en 'features' y 'labels', para luego codificar los datos separando en datos de entrenamiento y prueba.")
     
     st.code("""
         X = Life_df.drop('Life expectancy ', axis=1)
         y = Life_df['Life expectancy ']

         cols_to_scale = ['Country', 'Year', 'Adult Mortality',
           'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
           'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
           'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
           ' thinness  1-19 years', ' thinness 5-9 years',
           'Income composition of resources', 'Schooling']

         scaler = MinMaxScaler()
         X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
         
         
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     """)
     
     X = Life_df.drop('Life expectancy ', axis=1)
     y = Life_df['Life expectancy ']

     cols_to_scale = ['Country', 'Year', 'Adult Mortality',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling']


     scaler = MinMaxScaler()
     X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     
     st.write(len(X_train), len(X_test), len(y_train), len(y_test))

     st.subheader("Creación del modelo")
     
     st.code("""
         model = Sequential([
         Dense(64, activation='relu', input_dim=21),
         Dense(64, activation='relu'),
         Dense(64, activation='relu'),
         Dense(1, activation='linear')])
         model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error','mean_squared_error'])
         model.summary()
         history = model.fit(X_train, y_train, epochs=150, validation_split=0.2)
     """)
     
     model = Sequential([
     Dense(64, activation='relu', input_dim=21),
     Dense(64, activation='relu'),
     Dense(64, activation='relu'),
     Dense(1, activation='linear')
     ])

     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error','mean_squared_error'])

     model.summary()
     
     #with st.spinner("Entrenando el modelo, espere un momento..."):     
     #      history = model.fit(X_train, y_train, epochs=150, validation_split=0.2)

     st.write("Revisando el desempeño del modelo:")
     
     st.code("""
         fig,ax=plt.subplots()
         ax.plot(history.history['loss'])
         ax.plot(history.history['val_loss'])
         plt.title('Función de costo del modelo')
         plt.ylabel('Error')
         plt.xlabel('Epocas (desde 0)')
         plt.legend(['train', 'test'])
         st.pyplot(fig)
     """)
     error1=Image.open("error_ m1.png")
     st.write(error1)
     #fig,ax=plt.subplots()
     #ax.plot(history.history['loss'])
     #ax.plot(history.history['val_loss'])
     #plt.title('Función de costo del modelo')
     #plt.ylabel('Error')
     #plt.xlabel('Epocas (desde 0)')
     #plt.legend(['train', 'test'])
     #st.pyplot(fig)
     
     st.write("El modelo obtiene un buen rendimiento en pocas épocas, el error se reduce considerablemente en las primeras 5 épocas, obteniendo cada vez un mejor rendimiento en cada época superior, siendo cada vez la reducción menos notoria, llegando al máximo de reducción en la época 150.")
 
     #y_pred = model.predict(X_test)                     

     #R2 = r2_score(y_test, y_pred)
     
     st.write("Puntuación del modelo: ")
     st.write(f"- **R2 Score= {0.93}**")

     st.subheader("Conclusión")
     
     st.write("El modelo hace una buena interpretación de los datos llegando a una puntuación R2 superior a 0.9, recordando que muchos de los datos fueron alterados para no perder información importante. El modelo ANN es bastante rápido en entrenarse y puede ocuparse en casos donde se requiere problemas complejos que no necesariamente se tengan relaciones lineales.")

with pagina2:

     st.subheader("Exploración y Análisis")
     
     st.write("Los datos son de pacientes que fueron diagnosticado de cáncer de mamas, contienen toda la información de los pacientes, lo que muestra los datos principalmente es si el tumor que presentan en las mamas benigno o maligno, si el tumor benigno significa que no se ramificara en el cuerpo, por lo que el paciente puede ser curado, pero si el paciente tiene un tumor maligno este se puede ramificar y se vuelve muy difícil ser curado.")
     
     st.write("Lo que se intenta con este pronóstico es saber de una forma alternativa si es que el tumor puede ser benigno o maligno, ocupando inteligencia artificial.")
     
     st.code("""
         print(Cancer_df.head(5))
         print(Cancer_df.shape)
         print(pd.DataFrame(Cancer_df.isnull().sum()).T)
     """)
          
     st.write(Cancer_df.head(5))
     
     st.write(Cancer_df.shape)

     st.write(pd.DataFrame(Cancer_df.isnull().sum()).T)
     
     st.write("Los datos no necesitan limpieza, solo necesita la eliminación de una columna.")
     
     st.code("""
         Cancer_df=Cancer_df.drop(['Unnamed: 32'], axis=1)
     """)
     
     Cancer_df=Cancer_df.drop(['Unnamed: 32'], axis=1)
     
     st.write(Cancer_df.head(5))

     st.subheader("Ingeniería de características")

     st.write("Los datos se separan en 'features' y 'label' como 'X' e 'y', para luego poder codificar la variable categórica en este caso 'y', además escalar los datos en X.")
     
     st.code("""
        X = Cancer_data.drop(["diagnosis"], axis=1)
        y = Cancer_data["diagnosis"]
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        labelencoder_X_1 = LabelEncoder()
        y = labelencoder_X_1.fit_transform(y)
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)      
     """)
     
     
     X = Cancer_df.drop(["diagnosis", "id"], axis=1)
     y = Cancer_df["diagnosis"]
     
     from sklearn.preprocessing import LabelEncoder
     from sklearn.model_selection import train_test_split
     from sklearn.preprocessing import StandardScaler
     labelencoder_X_1 = LabelEncoder()
     y = labelencoder_X_1.fit_transform(y)
     sc = StandardScaler()
     X = sc.fit_transform(X)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)
   
     st.write(len(X_train), len(X_test), len(y_train), len(y_test))     
     
     st.subheader("Creación del modelo")
     
     import keras
     from keras.models import Sequential
     from keras.layers import Dense, Dropout
     
     st.code("""
         model = Sequential()
         model.add(Dense(16, kernel_initializer='uniform', activation='relu', input_dim=30))
         model.add(Dropout(0.1))
         model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
         model.add(Dropout(0.1))
         model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
         model.fit(X_train, y_train, batch_size=100, epochs=150)
     """)
     
     model_c = Sequential()
     model_c.add(Dense(16, kernel_initializer='uniform', activation='relu', input_dim=30))
     model_c.add(Dropout(0.1))
     model_c.add(Dense(16, kernel_initializer='uniform', activation='relu'))
     model_c.add(Dropout(0.1))
     model_c.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
     model_c.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
     history2 = model_c.fit(X_train, y_train, batch_size=100, epochs=40, validation_split=0.2)
     
     st.write("Se puede ver que es un modelo de clasificación, ya que ocupa la última capa con una activación sigmoid, esto lo que provoca que para una etiqueta con categoria binarias entregue las probabilidades de que el tumor sea maligno, siendo 0 significa que es benigno y 1 que es maligno. Si fuesen varias clasificaciones lo correcto es ocupar OneHotencoder para que entregue la probabilidad de cada clasificación.")
     
     st.write("Con un rendimiento del modelo de:")
     
     left, right = st.columns(2)
     
     with left:
     
        fig,ax=plt.subplots()
        ax.plot(history2.history['loss'])
        ax.plot(history2.history['val_loss'])
        plt.title('Función de costo del modelo')
        plt.ylabel('Error')
        plt.xlabel('Epocas (desde 0)')
        plt.legend(['train', 'test'])
        st.pyplot(fig)
     
     with right:
     
        fig,ax=plt.subplots()
        ax.plot(history2.history['accuracy'])
        ax.plot(history2.history['val_accuracy'])
        plt.title('Puntuación del modelo')
        plt.ylabel('Accuracy')
        plt.xlabel('Epocas (desde 0)')
        plt.legend(['train', 'test'])
        st.pyplot(fig)
     
     st.write("Muestra que tiene un rendimiento bastante bueno, con un error que disminuye a valores muy bajos de error y en cuanto a la puntuación se mantiene muy alta.")
     
     y_pred = model_c.predict(X_test)
     
     y_pred1 = len(y_pred[y_pred > 0.5])
     y_pred2 = len(y_pred[y_pred <= 0.5])
     
     st.write(f"* **Los pacientes con más probabilidad de presentar un tumor benigno son:** :green[{y_pred2}]")
     
     st.write(f"* **Los pacientes con más probabilidad de presentar un tumor maligno son:** :green[{y_pred1}]")
     
     st.write("A continuación se muestra la matriz de confusión por clasificación, en este caso 0 es benigno y 1 maligno, las filas dicen cual fue la predicción siendo del modelo, por ejemplo en este caso si el valor está en la diagonal el modelo acertó correctamente, el modelo solo obtuvo un fallo en una predicción, en la cual indico que era benigno, pero en realidad era maligno.")
          
     y_pred1 = y_pred > 0.5
     from sklearn.metrics import confusion_matrix
     cm = confusion_matrix(y_test, y_pred1)
     
     print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
     
     fig, ax = plt.subplots()     
     sns.heatmap(cm,annot=True)
     st.pyplot(fig)
      
     st.write("Puntuación del modelo: ")
     st.write(f"- **Accuracy= 0.98**")
     
     st.subheader("Conclusiones")
     
     st.write("El modelo obtuvo una cantidad alta de predicciones acertadas considerando que los datos no fueron modificados,  hizo una muy buena interpretación de los datos prediciendo la probabilidad de que un paciente con cáncer pueda tener un tumor maligno. La puntuación fue de 0.98 bastante alta, logrando una interpretación buena con solo 40 épocas. Al aumentar las épocas no presenta mejoras en puntuación, ni disminución del error.")







