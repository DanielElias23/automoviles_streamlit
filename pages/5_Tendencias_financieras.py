import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.models import load_model
import keras
import h5py
import requests
import os

#print(os.listdir("../input"))

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
background-size: 80%;
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

data_price =  pd.read_csv("prices.csv")
data_fundament = pd.read_csv("fundamentals.csv")
data_security = pd.read_csv("securities.csv")
data_price_split = pd.read_csv("prices-split-adjusted.csv")

st.title("Tendencias Financieras")

st.subheader("Exploración y Análisis")

st.write("Los datos presentados provienen de la Bolsa de Nueva York, por lo que contienen información financiera, es un set de datos en los cuales vienen 'Fundamentos' que contiene información como ingresos, ganancias, activos, deudas, entre otros, tambien esta 'prices-split-adjusted' que contiene los precios ajustados por split o dividendo de las acciones, que contiene la información para que se refleje el rendimiento real de las acciones, 'precios' contiene lo precios de las acciones sin splits, 'seguros' contiene informaciones variadas como información de la empresa, simbolo bursatil, el sector la industria, etc")

st.subheader("Manipulación y Limpieza")

st.write("Se pueden mostrar las datas explicitamente y si tienen datos nulos:")

st.write("-Se muestra la tabla price.csv y sus datos nulos")

st.write(data_price.head(5))


st.write(pd.DataFrame(data_price.isna().sum()).T)
st.write(data_price.shape)

st.write("-Se muestra la tabla fundamentals.csv y sus datos nulos")

st.write(data_fundament.head(5))


st.write(pd.DataFrame(data_fundament.isna().sum()).T)
st.write(data_fundament.shape)

st.write("-Se muestra la tabla securities.csv y sus datos nulos")

st.write(data_security.head(5))


st.write(pd.DataFrame(data_security.isna().sum()).T)
st.write(data_security.shape)

st.write("-Se muestra la tabla prices-split-adjusted.csv y sus datos nulos")

st.write(data_price_split.head(5))


st.write(pd.DataFrame(data_price_split.isna().sum()).T)
st.write(data_price_split.shape)

st.write("Son pocos los datos nulos, en general esas columnas no se ocupan, asi que no hay necesidad de limpiar.")

st.subheader("Ingenieria de caracterisitcas")


st.write("En este proyecto no se ocuparan las cuatro tablas sino que se ocupara solo 'prices-split-adjusted.csv', pero esta data se puede explicar con las demas tablas. En especifico veremos cuales son los valores en la bolsa con los ajustes a los split de 'APPL'")

st.write(data_security[data_security["Ticker symbol"]=="AAPL"])

st.write("El simbolo bursatil 'AAPL' corresponde a Apple Inc. se analizaran solo los datos de esta empresa tecnologica  y sus valores ajustados con splits.")

st.code("""
   data_aapl = data_price_split[data_price_split.symbol == 'AAPL']
   data_aapl.drop(['symbol'],1,inplace=True)
   data_aapl["date"] = data_aapl.index  
   data_aapl['date'] = pd.to_datetime(data_aapl['date'])
   print(data_aapl.head())
   print(data_aapl.shape)
   """)

data_aapl = data_price_split[data_price_split.symbol == 'AAPL']
data_aapl.drop(['symbol'],1,inplace=True)
data_aapl["date"] = data_aapl.index  
data_aapl['date'] = pd.to_datetime(data_aapl['date'])
st.write(data_aapl.head())
st.write(data_aapl.shape)

st.write("Visualizamos como se ven las funciones de 'close'")

st.code("""
    fig, ax=plt.subplots()
    sns.set(style="darkgrid")
    sns.lineplot(x=data_aapl["date"], y=data_aapl["close"])
    plt.ylabel("Precio de cierre")
    plt.xlabel("Dias")
    plt.title("Valor del precio de cierre por día")
    plt.show()
   """)

fig, ax=plt.subplots()
sns.set(style="darkgrid")
sns.lineplot(x=data_aapl["date"], y=data_aapl["close"])
plt.ylabel("Precio de cierre")
plt.xlabel("Dias")
plt.title("Precio de cierre por día")
st.pyplot(fig)

st.write("Podemos ver como se comparan los cuatro valores en el tiempo")

st.code("""
   fig, ax=plt.subplots()
   sns.set(style="darkgrid")
   sns.lineplot(x=data_aapl["date"], y=data_aapl["close"], label="cierre")
   sns.lineplot(x=data_aapl["date"], y=data_aapl["open"], label="apertura")
   sns.lineplot(x=data_aapl["date"], y=data_aapl["low"], label="bajo")
   sns.lineplot(x=data_aapl["date"], y=data_aapl["high"], label="alto")
   plt.ylabel("Precio de cierre")
   plt.xlabel("Dias")
   plt.title("Precio de cierre por día")
   plt.show()
""")

fig, ax=plt.subplots()
sns.set(style="darkgrid")
sns.lineplot(x=data_aapl["date"], y=data_aapl["close"], label="cierre")
sns.lineplot(x=data_aapl["date"], y=data_aapl["open"], label="apertura")
sns.lineplot(x=data_aapl["date"], y=data_aapl["low"], label="bajo")
sns.lineplot(x=data_aapl["date"], y=data_aapl["high"], label="alto")
plt.ylabel("Precio de cierre")
plt.xlabel("Dias")
plt.title("Precio de cierre por día")
st.pyplot(fig)

st.write("Se ve que tanto los valores de cierre, apertura, bajo y alto tienen valores bastantes similares, esto es muy util ya que las prediciones pueden ser mejores.")

st.subheader("-Creación y ajuste de red neuronal")

st.write("Una red neuronal dedicada para pronotico de continuacion de funciones, es el caso de RNN con el modelo de memoria a largo plazo como LSTM.")

st.write("El modelo LSTM es muy sensible a la escalabilidad de los datos, por lo que deben este todos en los escalados")

st.code("""
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    dataset_c = min_max_scaler.fit_transform(data_aapl['close'].values.reshape(-1, 1))
    dataset_o = min_max_scaler.fit_transform(data_aapl['open'].values.reshape(-1, 1))
    dataset_l = min_max_scaler.fit_transform(data_aapl['low'].values.reshape(-1, 1))
    dataset_h = min_max_scaler.fit_transform(data_aapl['high'].values.reshape(-1, 1))
   """)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset_c = min_max_scaler.fit_transform(data_aapl['close'].values.reshape(-1, 1))
dataset_o = min_max_scaler.fit_transform(data_aapl['open'].values.reshape(-1, 1))
dataset_l = min_max_scaler.fit_transform(data_aapl['low'].values.reshape(-1, 1))
dataset_h = min_max_scaler.fit_transform(data_aapl['high'].values.reshape(-1, 1))

st.write(dataset_c[0:10])

st.write("Se separan los valores en datos de entrenamiento y prueba manualmente para que no pierdan el orden y para que los de prueba sean los ultimos de la función.")

st.code("""
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    """)
    
train_size = int(len(dataset_c) * 0.7)
test_size = len(dataset_c) - train_size
train, test = dataset_c[0:train_size,:], dataset_c[train_size:len(dataset_c),:]
st.write(len(train), len(test))

train_size2 = int(len(dataset_o) * 0.7)
test_size2 = len(dataset_o) - train_size2
train2, test2 = dataset_o[0:train_size2,:], dataset_o[train_size2:len(dataset_o),:]
st.write(len(train2), len(test2))

train_size3 = int(len(dataset_l) * 0.7)
test_size3 = len(dataset_l) - train_size3
train3, test3 = dataset_l[0:train_size3,:], dataset_l[train_size3:len(dataset_l),:]
st.write(len(train3), len(test3))

train_size4 = int(len(dataset_h) * 0.7)
test_size4 = len(dataset_h) - train_size4
train4, test4 = dataset_h[0:train_size4,:], dataset_h[train_size4:len(dataset_h),:]
st.write(len(train4), len(test4))


def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



x_train, y_train = create_dataset(train, look_back=15)
x_test, y_test = create_dataset(test, look_back=15)
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

x_train2, y_train2 = create_dataset(train2, look_back=15)
x_test2, y_test2 = create_dataset(test2, look_back=15)
x_train2 = np.reshape(x_train2, (x_train2.shape[0], 1, x_train2.shape[1]))
x_test2 = np.reshape(x_test2, (x_test2.shape[0], 1, x_test2.shape[1]))

x_train3, y_train3 = create_dataset(train3, look_back=15)
x_test3, y_test3 = create_dataset(test3, look_back=15)
x_train3 = np.reshape(x_train3, (x_train3.shape[0], 1, x_train3.shape[1]))
x_test3 = np.reshape(x_test3, (x_test3.shape[0], 1, x_test3.shape[1]))

x_train4, y_train4 = create_dataset(train4, look_back=15)
x_test4, y_test4 = create_dataset(test4, look_back=15)
x_train4 = np.reshape(x_train2, (x_train4.shape[0], 1, x_train4.shape[1]))
x_test4 = np.reshape(x_test2, (x_test4.shape[0], 1, x_test4.shape[1]))

#st.write(x_train.shape)
#st.write(y_train.shape)
#st.write(x_test.shape)
#st.write(y_test.shape)



#st.write(x_train.shape)
#st.write(y_train.shape)
#st.write(x_test.shape)
#st.write(y_test.shape)

look_back = 15
model = Sequential()
model.add(LSTM(20, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)
# invert predictions
trainPredict = min_max_scaler.inverse_transform(trainPredict)
trainY = min_max_scaler.inverse_transform([y_train])
testPredict = min_max_scaler.inverse_transform(testPredict)
testY = min_max_scaler.inverse_transform([y_test])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset_c)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset_c)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset_c)-1, :] = testPredict
# plot baseline and predictions
#plt.plot(min_max_scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()

fig, ax=plt.subplots()
sns.set(style="darkgrid")
#plt.plot(min_max_scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.ylabel("Valor de cierre")
plt.xlabel("Días")
plt.title("Precio de cierre pronosticado por día")
#plt.xticks(rotation=45)
st.pyplot(fig)













