import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
background-image: url("https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg");
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

Life_df = pd.read_csv('Life_E_Data.csv')
Cancer_df = pd.read_csv('Cancer_Data.csv')


st.title("Predicciones medicas")

st.write(":green[*Modelo ML - ANN - Regresión y Clasificación*]")

pagina1, pagina2=st.tabs([" Regresión ", " Clasificación "])

with pagina1:
     
     st.subheader("Exploración y Análisis")
     
     st.write("Los datos presentados son de un estudio de diferentes paises que se realizo entre el 2000 - 2015 por la Observatorio Mundial de la Salud (GHO). El estudio se centra en los paises y su esperanza de vida, estos estudios son realizados cada año para cada pais entre los años mencionados.")
     st.write("Lo que se busca pronosticar es cual seria la esperanza de vida de un pais solo conociendo sus caracterisiticas y no de la manera tradicional que es con calculos tecnicos sobre las salud.")
     
     st.subheader("Manipulación y Limpieza")
     
     st.write("Se puede ver los datos explicitamente y sus respectivos datos nulos")
     
     st.code("""
          print(Life_df.head(5)
          print(Lide_df.shape)
          print(Life_df.isnull().sum().T)
     """)
          
     st.write(Life_df.head(5))
     st.write(Life_df.shape)
     st.write(pd.DataFrame(Life_df.isnull().sum()).T)
     
     st.subheader("Ingeniería de características")
     
     
     st.write("Existe un gran cantidad de datos nulos en algunas columnas, lo que podria perjudicar a la predicción, por lo que es mejor remplazar esos datos con sus promedios.")
     
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
     
     st.write("Obteniendo que ya no existen datos nulos")
     
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

     st.write("Es necesario codificar las variables categoricas para que puedan ser procesadas por el modelo.")
     
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
     
     st.write("Por ultimo separar en 'features' y 'labels', para luego codificar los datos separando en datos de entrenamiento y prueba.")
     
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
     
     with st.spinner("Entrenando el modelo, espere un momento..."):     
           history = model.fit(X_train, y_train, epochs=150, validation_split=0.2)

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
     
     fig,ax=plt.subplots()
     ax.plot(history.history['loss'])
     ax.plot(history.history['val_loss'])
     plt.title('Función de costo del modelo')
     plt.ylabel('Error')
     plt.xlabel('Epocas (desde 0)')
     plt.legend(['train', 'test'])
     st.pyplot(fig)
     
     st.write("El modelo obtiene un buen rendimiento en pocas epocas, el error se reduce considerablemente en las primeras 5 epocas, obteniendo cada vez un mejor rendimiento en cada epoca superior, siendo cada vez la reducción menos notoria, llegando al maximo de reducción en la epoca 150.")
 
     y_pred = model.predict(X_test)                     

     R2 = r2_score(y_test, y_pred)
     
     st.write("Puntuación del modelo: ")
     st.write(f"- **R2 Score= {round(R2,3)}**")

     st.subheader("Conclusión")
     
     st.write("El modelo hace una buena interpretación de los datos llegando a una puntuación R2 superior a 0.9, recordando que muchos de los datos fueron alterados para no perder información importante. El modelo ANN es bastante rapido en entrenarse y puede ocuparse en casos donde se requiere problemas complejos que no necesariamente se tengan relaciones lineales.")

with pagina2:

     st.write("nada")

     st.subheader("Exploración y Análisis")
     
     st.write("Los datos son de pacientes que fueron dianosticado de cancer de mamas, contienen toda la información de los pacientes, lo que muestra los datos principalmente es si el tumor que presentan en las mamas benigo o maligno, si el tumor benigo significa que no se ramificara en el cuerpo, por lo que el paciente puede ser curado, pero si el paciente tiene un tumor maligno este se puede ramificar y se vuelve muy dificil ser curado.")
     
     st.write("Lo que se intenta con este pronostico es saber de una forma alternativa si es que el tumor puede ser benigno o maligno, ocupando inteligencia artificial.")
     
     st.code("""
         print(Cancer_df.head(5))
         print(Cancer_df.shape)
         print(pd.DataFrame(Cancer_df.isnull().sum()).T)
     """)
          
     st.write(Cancer_df.head(5))
     
     st.write(Cancer_df.shape)

     st.write(pd.DataFrame(Cancer_df.isnull().sum()).T)
     
     st.write("Los datos no neccesitan limpieza, solo necesita la eliminación de una columna.")
     
     st.code("""
         Cancer_df=Cancer_df.drop(['Unnamed: 32'], axis=1)
     """)
     
     Cancer_df=Cancer_df.drop(['Unnamed: 32'], axis=1)
     
     st.write(Cancer_df.head(5))

     st.subheader("Ingeniería de características")

     st.write("Los datos se separan en 'features' y 'label' como 'X' e 'y', para luego poder codificar la variable categorica en este caso 'y', además escalar los datos en X.")
     
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
     
     st.write("El modelo muestra que es un modelo de clasificación al ocupar como ultima capa una activación sigmoid, esto lo que proboca que para una etiqueta con categoria binarias entregue las probabilidades de que el tumor sea malgino, siendo un 1 el 100% que es maligno. Si fuesen varias clasificaciones lo correcto es ocupar OneHotencoder que entregaria la probabilidad de cada una.")
     
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
     
     fig, ax = plt.subplots()
     plt.hist(y_pred)
     st.pyplot(fig)
     
     
     y_pred = (y_pred > 0.5)
     
     from sklearn.metrics import confusion_matrix
     cm = confusion_matrix(y_test, y_pred)
     
     print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
     
     fig, ax = plt.subplots()     
     sns.heatmap(cm,annot=True)
     st.pyplot(fig)









