import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv')

st.subheader("Proyecto :orange[1] ")

st.title("Predictor de precios de automoviles 🚗")
#st.write("Seleccione las caracteriticas del modelo de automovil que quiere predecir el precio estimado en el mercado, luego presione enviar.")

st.write("Lo que hace el predictor, es que segun las caracterisitcas de cierto vehiculo puede predecir cual seria el precio en el mercado de ese vehiculo.")
st.write("El predictor de precios de vehiculos se utiliza para saber si una empresa automotriz puede entrar a un nuevo mercado, ya conocera el valor en el cual podra vender esos vehiculos, en el caso de ya contar con una empresa automotriz se puede utilizar para saber a cuanto se podria ofrecer un nuevo modelo de vehiculo segun sus caracterisitcas. Esto dado que los datos utilizados por el modelo de machine learning contiene los modelos de vehiculos que ofrece el mercado y sus respectivos precios.")
#st.write("""
#          Una empresa de automoviles pretender entrar al mercado de un pais, para ello necesita saber a cuanto podria vender sus automoviles
#          para saber si es viable el negocio, entonces hace un estudio para saber el precio de los automoviles que ofrece el mercado con sus
#          respectivas especificaciones de cada modelo automovilistico,  
           
#          """)

#st.subheader("Objetivos a lograr")

#st.write("-Analizar y limpiar la data, luego definir etiquetas precios para solucion del problema")

#st.write("-Crear un modelo de machine learning entrenado con la data que tiene especificaciones de autos del mercado")

#st.write("-Predecir diferentes modelos de autos segun sus caracterisitcas")

#st.subheader("Modelo predictivo de precios")

#st.write("La data es limpiada y las variables categoricas se codifican a numeros para no causar malas interpretaciones del modelo")


data = pd.DataFrame(data)

data2 = data.drop(["car_ID", "symboling"], axis=1)

data2[["Brand", "Car_Name1", "Car_Name2", "Car_Name3", "Car_Name4"]]=data2["CarName"].str.split(" ",expand=True)

data3 = data2.drop(["CarName","Car_Name1","Car_Name2","Car_Name3","Car_Name4"], axis=1)

data3["Brand"] = data3["Brand"].replace({ "Nissan": "nissan", "toyouta": "toyota", "vokswagen": "volkswagen", "vw": "volkswagen", "porcshce":"porsche", "maxda":"mazda"})

#st.write(data3.shape)

nombre_col=data3.columns.tolist()

#st.write(data3)

Marca=data3["Brand"].unique()

st.header("Selecciona un modelo de vehiculo:")
with st.form("auto", clear_on_submit=False, border=True):           
            marca_auto=st.selectbox("Marca del vehiculo", data3["Brand"].unique())
            left_column, right_column, three_column=st.columns(3)
            with left_column:
                          tipo_combustible=st.selectbox("Tipo de combustible", data3["fueltype"].unique())
                          sistema_de_combustible=st.selectbox("Sistema de combustible", data3["fuelsystem"].unique())
                          aspiracion=st.selectbox("Tipo de aspiración", data3["aspiration"].unique())
                          base_de_rueda=st.select_slider("Base de la rueda", np.sort(data3["wheelbase"].unique()))
                          largo_del_auto=st.select_slider("Largo del auto", np.sort(data3["carlength"].unique()))
                          stoke=st.select_slider("Ciclos del motor", np.sort(data3["stroke"].unique()))
                          caballos_de_fuerza=st.select_slider("Caballos de fuerza", np.sort(data3["horsepower"].unique()))
                          highwaympg=st.select_slider("Rendimiento mpg en carretera", np.sort(data3["highwaympg"].unique()))
            with right_column:
                          numero_puertas=st.selectbox("Numero de puertas", data3["doornumber"].unique())
                          cuerpo_del_auto=st.selectbox("Cuerpo del auto", data3["carbody"].unique())
                          cilindrado=st.selectbox("Cilindrado", data3["cylindernumber"].unique())
                          boreratio=st.select_slider("Relación diametro por carrera", np.sort(data3["boreratio"].unique()))
                          ancho_del_auto=st.select_slider("Ancho del auto", np.sort(data3["carwidth"].unique()))
                          Compressionratio=st.select_slider("Relación de compresión", np.sort(data3["compressionratio"].unique()))
                          peakrpm=st.select_slider("Peak rpm", np.sort(data3["peakrpm"].unique()))
            with three_column:
                          manubrio=st.selectbox("Manubrio", data3["drivewheel"].unique())
                          ubicacion_motor=st.selectbox("Ubicación del motor", data3["enginelocation"].unique())
                          tipo_de_motor=st.selectbox("Tipo del motor", data3["enginetype"].unique())
                          peso_del_auto=st.select_slider("Peso del auto", np.sort(data3["curbweight"].unique()))
                          altura_del_auto=st.select_slider("Altura del auto", np.sort(data3["carheight"].unique()))
                          tamaño_del_motor=st.select_slider("Tamaño del motor", np.sort(data3["enginesize"].unique()))
                          citympg=st.select_slider("Rendimiento mpg en ciudad", np.sort(data3["citympg"].unique()))
            variable_input_sumit = st.form_submit_button("Enviar")

data4=data3

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

ohe = OneHotEncoder()
le = LabelEncoder()

data_name_col = ["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "Brand"]
data_name_col2 = ["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem"]

#Para ocupar OneHotEncoder y ocupar ".categories_" se necesita que sean columnas tipo "category"


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

X = data4.drop("price", axis=1)

X[["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "Brand"]]= X[["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "Brand"]].astype("category")



y = data4["price"]



if variable_input_sumit:
        st.write(f":blue[El modelo elegido es {marca_auto}]")
        data_2=pd.DataFrame([marca_auto, tipo_combustible, aspiracion, numero_puertas, cuerpo_del_auto, manubrio, ubicacion_motor, base_de_rueda, largo_del_auto,  ancho_del_auto, altura_del_auto, peso_del_auto, tipo_de_motor, cilindrado, tamaño_del_motor, sistema_de_combustible, boreratio,stoke, Compressionratio, caballos_de_fuerza, peakrpm, citympg, highwaympg]).T
        data_2=data_2.rename(columns={0:"Brand", 1:"fueltype", 2:"aspiration", 3:"doornumber", 4:"carbody", 5:"drivewheel", 6:"enginelocation", 7:"wheelbase", 8:"carlength", 9:"carwidth", 10:"carheight", 11:"curbwight", 12:"enginetype", 13:"cylindernumber", 14:"enginesize", 15:"fuelsystem", 16:"boreratio", 17:"stroke", 18:"compressionratio", 19:"horsepower", 20:"peakrpm", 21:"citympg", 22:"highwaympg"})        
        st.write(data_2)
        
        
        data_2=pd.DataFrame([tipo_combustible, aspiracion, numero_puertas, cuerpo_del_auto, manubrio, ubicacion_motor, base_de_rueda, largo_del_auto,  ancho_del_auto, altura_del_auto, peso_del_auto, tipo_de_motor, cilindrado, tamaño_del_motor, sistema_de_combustible, boreratio,stoke, Compressionratio, caballos_de_fuerza, peakrpm, citympg, highwaympg, marca_auto]).T
        data_2=data_2.rename(columns={0:"fueltype", 1:"aspiration", 2:"doornumber", 3:"carbody", 4:"drivewheel", 5:"enginelocation", 6:"wheelbase", 7:"carlength", 8:"carwidth", 9:"carheight", 10:"curbweight", 11:"enginetype", 12:"cylindernumber", 13:"enginesize", 14:"fuelsystem", 15:"boreratio", 16:"stroke", 17:"compressionratio", 18:"horsepower", 19:"peakrpm", 20:"citympg", 21:"highwaympg", 22:"Brand"})   
        #st.write(data_2)
        X1=pd.concat([data_2, X], axis=0).reset_index()   
        for col in data_name_col:

              data_ohe = ohe.fit_transform(X1[[col]].values.reshape(-1, 1)).toarray()
              X1 = pd.concat([X1.drop(col, axis = 1), pd.DataFrame(data_ohe, columns = ohe.categories_[0])], axis = 1)  
        #st.write(X1)           
        #X.drop("fueltype", axis=1)
        #st.write(X)      
             
        data_2=X1.iloc[[0]]
        X1=X1.drop(0)
        X=pd.DataFrame(X1)
        X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=30)
        pipe_en = Pipeline([("polynomial", PolynomialFeatures(include_bias=False, degree=2)), ("ss", StandardScaler()), ("model", ElasticNet(tol=0.2, max_iter=5000, l1_ratio=0.75, alpha=10))]) 
        #st.write(X)       
        pipe_en.fit(X_train, y_train)
        y_pred4 = pipe_en.predict(X_test)

        
        
        y_pred_sol = pipe_en.predict(data_2)
        st.header("El precio estimado para este vehiculo es:")
        st.title(f"{round(float(y_pred_sol),2)} $")
        
        
        """Modelo ML: 
            :green[ElasticNet, alfa=10, L1=0.75]
        """
        st.write(f"Puntuación: :green[R^2: {round(r2_score(y_pred4,y_test),2)}, MSE: {round(mean_squared_error(y_pred4,y_test),2)}]")
#for col in data_name_col:
    
#    data_le = le.fit_transform(X[col])
#    data_le = pd.DataFrame(data_le)
#    data4[col] = data_le

#for col in data_name_col:

#        data4=le.fit_transform(data4[col])
#        data4=pd.concat([])

#st.write(data4)


       
#st.write(data4)
       
#print(data3.columns.tolist())

#Ahora esta lista para el modelo de prediccion







#pipe_en = Pipeline([("polynomial", PolynomialFeatures(include_bias=False, degree=2)), ("model", ElasticNet(tol=0.2, max_iter=5000, l1_ratio=0.75, alpha=10))]) 




#param_grid = {
    #"polynomial__degree": [ 1, 2,3],
    #"alpha":[0.001, 0.1,1,10,100],
    #"l1_ratio":[0.5,0.75, 1]
#}






 
        #for col in data_name_col:

        #        data_ohe = ohe.fit_transform(data_2[[col]].values.reshape(-1, 1)).toarray()
        #        data_2 = pd.concat([data_2.drop(col, axis = 1), pd.DataFrame(data_ohe, columns = ohe.categories_[0])], axis = 1)
        
            
        #unico=pd.DataFrame(X.head(1))
        #data_final=pd.merge(unico, data_2, how="outer")
        #
        
        #st.write(pd.DataFrame(pd.concat(pd.DataFrame([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,manurio, tipo_de_motor], axis=0)), nombre_col).T)
                        
#st.selectbox("nada", ("alfa-romero","audi","bmw","chevrolet","dodge","honda","isuzu","jaguar","mazda", "nada", "madadasd", "nada2"))

#data3.insert(24, 'Categoria', data3["price"])

#data3["Categoria"]=["Caro" if x>15000 else ("Medio"if (15000>=x>9000) else "Barato") for x in data3["Categoria"]]



#data_gr=pd.DataFrame(data3.groupby(["Brand", "Categoria"], as_index=False)["price"].agg("count"))
#data_gr2=pd.DataFrame(data3.groupby(["Brand", "Categoria"], as_index=False)["price"].agg("mean"))


#st.table(data_gr)




