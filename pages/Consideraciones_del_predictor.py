import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.galleries.tutorials.pyplot as plt #.pyplot as plt
import streamlit as st

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv')

st.title("Consideraciones del predictor")

#st.subheader("Resumen")

st.write("El predictor de precios de vehiculos se utiliza para saber si una empresa automotriz puede entrar a un nuevo mercado y si el negocio puede ser viable, al ya conocer a cuanto podria ofrecer los vehiculos, en el caso de ya contar con una empresa automotriz se puede utilizar para saber a cuanto se podria ofrecer un nuevo modelo de vehiculo.")

st.write("Lo que hace el predictor, es que segun las caracterisitcas de cierto vehiculo puede predecir cual seria el precio en el mercado de ese vehiculo.")

st.subheader("Datos utilizados")

st.write("Los datos utilizados son importantes, ya que solo hara buenas predicciones siempre y cuando todas las caracterisitcas del vehiculo seleccionado se encuentren dentro de los datos considerados para hacer el predictor. Tambien pueden haber sesgos o tendencias segun la distribución de los datos.")

st.write("Se puede ver explicitamente la data:")

data = pd.DataFrame(data)

data2 = data.drop(["car_ID", "symboling"], axis=1)

data2[["Brand", "Car_Name1", "Car_Name2", "Car_Name3", "Car_Name4"]]=data2["CarName"].str.split(" ",expand=True)

data3 = data2.drop(["CarName","Car_Name1","Car_Name2","Car_Name3","Car_Name4"], axis=1)

data3["Brand"] = data3["Brand"].replace({ "Nissan": "nissan", "toyouta": "toyota", "vokswagen": "volkswagen", "vw": "volkswagen", "porcshce":"porsche", "maxda":"mazda"})



nombre_col=data3.columns.tolist()

st.write(data3)
st.write(data3.shape)
data_gr=pd.DataFrame(data3.groupby("Brand")["price"].agg("mean"))

st.subheader("Precios")

minimo=np.min(data3["price"])
maximo=np.max(data3["price"])

st.write(f"El precio minimo de los vehiculos de los datos con :blue[{round(minimo)}$] y el maximo precio es :blue[{round(maximo)}$], el modelo puede mostrar menores o mayores precios que estos, ya que puede considerar que el modelo de vehiculo seleccionado tiene caracterisitcas peores o mejores que las presentadas en los datos.")

st.write("La distribución de los precios es la siguiente:")

fig,ax=plt.subplots()
ax.grid(axis="y", linewidth = 0.2)
ax.hist(data3["price"], bins=30)
plt.title("Distribución de precios de los datos")
plt.xlabel("Precios de vehiculos")
plt.ylabel("Cantidad de vehiculos")
#data_gr.plot(kind="bar")
st.pyplot(fig)
#st.bar_chart(data3["price"])

st.subheader("Precio promedio por marca de automovil")

st.write("El precio promedio por marca tiene que ver con los datos presentados por cada marca, no asi con las especificaciones, pero el modelo de predicion reconocera que hay ciertas marcas que tienen cierto valor adicional.")

st.bar_chart(data_gr)


