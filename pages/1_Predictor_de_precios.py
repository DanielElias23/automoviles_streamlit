import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import requests
from PIL import Image
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

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv')
alfa_romeo=Image.open("alfa-romeo.jpg")
audi=Image.open("audi.jpg")
bmw=Image.open("bmw.jpg")
chevrolet=Image.open("chevrolet.jpg")
dodge=Image.open("dodge.jpg")
honda=Image.open("honda.jpg")
isuzu=Image.open("isuzu.jpg")
#st.subheader("Modelo ML -  :orange[Regresión] ")
st.subheader("Predicción de Precios de Automóviles en el Rubro Automotriz")
st.write(":green[Modelo ML - Regresión]")
#with open('googlelogo.png', 'wb') as f: 
#     f.write(alfa_romeo.content) 

pagina1, pagina2 =st.tabs(["Individual","Múltiple"])

with pagina1:

   st.write(":blue[Elige un modelo]")

   st.title("Predictor de precios")
#st.write("Seleccione las caracteriticas del modelo de automovil que quiere predecir el precio estimado en el mercado, luego presione enviar.")

   st.write("Lo que hace el predictor de precios, es que según las características de cierto producto puede predecir cual sería el precio en el mercado.")
   st.write("El predictor de precios de vehículos se utiliza para saber si una empresa automotriz puede entrar a un nuevo mercado, ya que conocerá el valor en el cual podrá vender esos vehículos. En el caso de ya contar con una empresa automotriz se puede utilizar para saber a cuanto se podría ofrecer un nuevo modelo de vehículo según sus características. Esto dado que los datos utilizados por el modelo de machine learning contiene los modelos de vehículos que ofrece el mercado y sus respectivos precios.")
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

   st.sidebar.write(":blue[Selecciona un vehículo con sus respectivas especificaciones]")

   data = pd.DataFrame(data)

   data2 = data.drop(["car_ID", "symboling"], axis=1)

   data2[["Brand", "Car_Name1", "Car_Name2", "Car_Name3", "Car_Name4"]]=data2["CarName"].str.split(" ",expand=True)

   data3 = data2.drop(["CarName","Car_Name1","Car_Name2","Car_Name3","Car_Name4"], axis=1)

   data3["Brand"] = data3["Brand"].replace({"alfa-romero":"alfa-romeo", "Nissan": "nissan", "toyouta": "toyota", "vokswagen": "volkswagen", "vw": "volkswagen", "porcshce":"porsche", "maxda":"mazda"})

#st.write(data3.shape)

   nombre_col=data3.columns.tolist()

#st.write(data3)

   Marca=data3["Brand"].unique()

   st.header("Selecciona un modelo de vehículo:")
   with st.form("auto", clear_on_submit=False, border=True):           
            marca_auto=st.selectbox("Marca del vehículo", data3["Brand"].unique())
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
                          numero_puertas=st.selectbox("Número de puertas", data3["doornumber"].unique())
                          cuerpo_del_auto=st.selectbox("Cuerpo del auto", data3["carbody"].unique())
                          cilindrado=st.selectbox("Cilindrado", data3["cylindernumber"].unique())
                          boreratio=st.select_slider("Relación diámetro por carrera", np.sort(data3["boreratio"].unique()))
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

   

   ohe = OneHotEncoder()
   le = LabelEncoder()

   data_name_col = ["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem", "Brand"]
   data_name_col2 = ["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber", "fuelsystem"]

#Para ocupar OneHotEncoder y ocupar ".categories_" se necesita que sean columnas tipo "category"




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
        st.header("El precio estimado para este vehículo es:")
        st.title(f"{round(float(y_pred_sol),2)} $")
        
        
        """Modelo ML: 
            :green[ElasticNet, alfa=10, L1=0.75]
        """
        st.write(f"Puntuación: :green[R^2: {round(r2_score(y_pred4,y_test),2)}, MSE: {round(mean_squared_error(y_pred4,y_test),2)}]")
        
        if marca_auto=="alfa-romeo":
                
                st.sidebar.subheader(f":blue[Selecionaste Alfa Romeo]")
                st.sidebar.write("Alfa Romeo Automobili S.p.A. es una marca italiana de automóviles de lujo, popularmente conocidos por elegante diseño y altas prestaciones. Fundada en 1910 en Milán, Italia; actualmente es propiedad del conglomerado de empresas Stellantis.")
                st.sidebar.write(alfa_romeo)
                
        if marca_auto=="audi":
         
                st.sidebar.subheader(f":blue[Selecionaste Audi]")
                st.sidebar.write('La compañía tiene su sede central en Ingolstadt, en el Estado Federado de Baviera en Alemania. En la actualidad, Audi sigue con su filosofía de "«a la vanguardia de la técnica»" e impulsa todo su conocimiento adquirido en sus coches.')
                st.sidebar.write(audi)
                
        if marca_auto=="bmw":
         
                st.sidebar.subheader(f":blue[Selecionaste Bmw]")
                st.sidebar.write("Es un fabricante alemán de automóviles y motocicletas de alta gama y lujo, cuya sede se encuentra en Múnich. Sus subsidiarias son Mini, Rolls-Royce, BMW i y BMW Bank. BMW es el líder mundial en ventas entre los fabricantes de gama alta; compite principalmente con Audi, Volvo, Lexus y Mercedes-Benz, entre otros vehículos de gama alta.")
                st.sidebar.write(bmw)

                
        if marca_auto=="chevrolet":
         
                st.sidebar.subheader(f":blue[Selecionaste Chevrolet]")
                st.sidebar.write("Chevrolet es un fabricante estadounidense de automóviles y camiones con sede en Detroit, perteneciente al grupo General Motors. Nació de la alianza de Louis Chevrolet y William Crapo Durant el 3 de noviembre de 1911.2​")
                st.sidebar.write(chevrolet)
                
        if marca_auto=="dodge":
         
                st.sidebar.subheader(f":blue[Selecionaste Dodge]")
                st.sidebar.write("Dodge es una marca de automóviles estadounidense, llamada originalmente Dodge Brothers Company (1900-1927),1​ actualmente propiedad de Stellantis. Chrysler adquirió la compañía Dodge en 1928 de la que seguía formando parte del FCA US LLC.")
                st.sidebar.write(dodge)

                
        if marca_auto=="honda":
         
                st.sidebar.subheader(f":blue[Selecionaste Honda]")
                st.sidebar.write("Honda Motor Co., Ltd., es una empresa de capital abierto de origen japonés fabricante de automóviles, motores para vehículos terrestres, acuáticos y aéreos, motocicletas, robots y demás refacciones para la industria automotriz.")
                st.sidebar.write(honda)

                
        if marca_auto=="isuzu":
         
                st.sidebar.subheader(f":blue[Selecionaste Isuzu]")
                st.sidebar.write("Isuzu Motors Ltd. es un fabricante de vehículos industriales y comerciales, así como de motores diésel, con sede mundial en Tokio, Japón. Su actividad se concentra en el diseño, producción, ensamblaje, venta y distribución de vehículos comerciales. ")
                st.sidebar.write(isuzu)
   

with pagina2:

     st.subheader("Instrucciones")
     
     st.write("-Sube un archivo con un dataset")
     
     st.write("-El dataset con los datos debe estar limpiado")
     
     st.write("-Debe ser un archivo '.csv'")
     
     st.write("-La tabla debe contener todas estas columnas en el siguiente orden:")
     
     if st.checkbox("Mostrar columnas"):
     
                ''':green["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation", "wheelbase", "carlength", "carwidth", "carheight", "curbweight", "enginetype", "cylindernumber", "enginesize", "fuelsystem", "boreratio", "stroke", "compressionratio", "horsepower", "peakrpm", "citympg", "highwaympg", "Brand"]'''
     
     st.write("-Puede contener algunas columnas adicional 'price' como los datos de prueba")
     
     st.write("-Puede descargar datos de prueba dando click [aquí](%s)" % "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv" )
     
     st.sidebar.write(":blue[Selecciona un vehículo con sus respectivas especificaciones]")



#st.write(data3.shape)

     #nombre_col=data3.columns.tolist()
     
     with st.form("carga archivo", clear_on_submit=True):
     
                data_up = st.file_uploader("Inserte un archivo", type=["csv"])
                
                recibido=st.form_submit_button("Enviar")
                
                if recibido and data_up is not None:
                      #ret = self.upload_file(data_up)
                      
                      #if ret is not False:
                      
                              data_up = pd.read_csv(data_up)
                      
                              data_up = pd.DataFrame(data_up)
                              
                              data = pd.DataFrame(data)

                              data_up = data.drop(["car_ID", "symboling"], axis=1)

                              data_up[["Brand", "Car_Name1", "Car_Name2", "Car_Name3", "Car_Name4"]]=data_up["CarName"].str.split(" ",expand=True)

                              data_up = data_up.drop(["CarName","Car_Name1","Car_Name2","Car_Name3","Car_Name4"], axis=1)

                              data_up["Brand"] = data_up["Brand"].replace({"alfa-romero":"alfa-romeo", "Nissan": "nissan", "toyouta": "toyota", "vokswagen": "volkswagen", "vw": "volkswagen", "porcshce":"porsche", "maxda":"mazda"})
                              
                              data_up = data_up.drop(["price"], axis=1, errors="ignore")
                              
                              data_up2 = data_up
                              
                              for col in data_name_col:

                                    data_ohe = ohe.fit_transform(data_up2[[col]].values.reshape(-1, 1)).toarray()
                                    data_up2 = pd.concat([data_up2.drop(col, axis = 1), pd.DataFrame(data_ohe, columns = ohe.categories_[0])], axis = 1) 
                                     
                                     #X1=pd.concat([data_2, X], axis=0).reset_index()   

                              #data = pd.DataFrame(data)

                              data5 = data.drop(["car_ID", "symboling"], axis=1)

                              data5[["Brand", "Car_Name1", "Car_Name2", "Car_Name3", "Car_Name4"]]=data5["CarName"].str.split(" ",expand=True)

                              data5 = data5.drop(["CarName","Car_Name1","Car_Name2","Car_Name3","Car_Name4"], axis=1)

                              data5["Brand"] = data5["Brand"].replace({"alfa-romero":"alfa-romeo", "Nissan": "nissan", "toyouta": "toyota", "vokswagen": "volkswagen", "vw": "volkswagen", "porcshce":"porsche", "maxda":"mazda"})    
                              
                              for col in data_name_col:

                                    data_ohe = ohe.fit_transform(data5[[col]].values.reshape(-1, 1)).toarray()
                                    data5 = pd.concat([data5.drop(col, axis = 1), pd.DataFrame(data_ohe, columns = ohe.categories_[0])], axis = 1) 
                              
                              X2 = data5.drop("price", axis=1)
                              y2 = data5["price"]
                              
                              X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=30)
                              pipe_en2 = Pipeline([("polynomial", PolynomialFeatures(include_bias=False, degree=2)), ("ss", StandardScaler()), ("model", ElasticNet(tol=0.2, max_iter=5000, l1_ratio=0.75, alpha=10))]) 
               
                              pipe_en2.fit(X_train, y_train) 
                              y_pred_up = pipe_en2.predict(data_up2)
                              
                              y_pred_up2 = pd.DataFrame(y_pred_up)
                              
                              mezcla = pd.DataFrame(pd.concat([data_up, y_pred_up2], axis=1))
                              mezcla = mezcla.rename(columns={0:"price"})
                              
                   
                              st.write(":blue[Los clientes que se pueden perder son: ]")
                              
                              st.write(mezcla)  
                  
                              st.write(":blue[(click arriba para descargar)]")
               
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




