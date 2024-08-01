import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import requests
from PIL import Image
import urllib.request


churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")

st.subheader("Proyecto :orange[2] ")

st.write(":blue[Rellena el formulario y predice la posible perdida]")

st.title("Perdida de clientes")

st.write(churn_df)

st.header("Formulario con los datos del cliente:")


churn_df["callcard"] = churn_df["callcard"].replace({1:"Si", 0:"No"})
churn_df[["equip"]] = churn_df[["equip"]].replace({1:"Si", 0:"No"})
churn_df["wireless"] = churn_df["wireless"].replace({1:"Si", 0:"No"})
churn_df["voice"] = churn_df["voice"].replace({1:"Si", 0:"No"})
churn_df["pager"] = churn_df["pager"].replace({1:"Si", 0:"No"})
churn_df["internet"] = churn_df["internet"].replace({1:"Si", 0:"No"})
churn_df["callwait"] = churn_df["callwait"].replace({1:"Si", 0:"No"})
churn_df["confer"] = churn_df["confer"].replace({1:"Si", 0:"No"})
churn_df["ebill"] = churn_df["ebill"].replace({1:"Si", 0:"No"})

churn_df["ed"] = churn_df["ed"].replace({1:"Educación Basica", 2:"Educación media", 3:"Educación superior", 4:"Magistrado/a", 5:"Doctorado/a"})

with st.form("cliente", clear_on_submit=False, border=True):
            st.subheader("Datos personales del cliente:")
            Nombre=st.text_input(label="Escriba el nombre del cliente")           
            #marca_auto=st.selectbox("Marca del vehículo", data3["Brand"].unique())
            
            left_column, right_column=st.columns(2)
            with left_column:
                          income=st.select_slider("Ingreso anual del cliente", np.sort(churn_df["income"].unique()))
                          años_trabajados=st.select_slider("Años trabajando", np.sort(churn_df["employ"].unique()))
                          numero_años_en_residencia=st.select_slider("Años en su vivienda actual", churn_df["address"].unique())                    
            with right_column:
                          nivel_de_estudio=st.selectbox("Nivel de estudio", churn_df["ed"].unique())
                          edad_del_cleinte=st.select_slider("Edad del cliente", np.sort(churn_df["age"].unique()))
                          tenure=st.select_slider("Meses que el cliente ha permanecido en la empresa", np.sort(churn_df["tenure"].unique()))
            st.subheader("Datos del servicio del cliente:")
            left_column, right_column, three_column=st.columns(3)
            with left_column:
                          servicio_de_internet=st.selectbox("¿Contrato servicios de internet?", churn_df["internet"].unique())
                          equipo_de_empresa=st.selectbox("¿Tiene equipos de la empresa?", churn_df["equip"].unique())
                          callwait=st.selectbox("¿Contrato servicio de llamada en espera?", churn_df["callwait"].unique())
                          longmon=st.select_slider("Gastos mensuales en llamadas larga distancia", np.sort(churn_df["longmon"].unique()))
                          cardmon=st.select_slider("Gastos mensuales en llamadas", np.sort(churn_df["cardmon"].unique()))
                          tollten=st.select_slider("Gastos totales de peajes (toll charger)", np.sort(churn_df["tollten"]))
                          
            with right_column:
                          servicio_inalambricos=st.selectbox("¿Contrato servicios inalambricos?", churn_df["wireless"].unique())
                          servicio_de_voz=st.selectbox("¿Contrato servicio de correo de voz?", churn_df["voice"].unique())
                          confer=st.selectbox("¿Contrato servicio de conferencias?", churn_df["confer"].unique())
                          tollmon=st.select_slider("Gastos mensuales en peajes (toll charger)", np.sort(churn_df["tollmon"].unique()))
                          wiremon=st.select_slider("Gastos mensuales en servicios inalambricos", np.sort(churn_df["wiremon"].unique()))
                          cardten=st.select_slider("Gastos totales en llamadas", np.sort(churn_df["cardten"].unique()))
            with three_column:
                          servicio_telefonico=st.selectbox("¿Contrato servicios telefonicos?", churn_df["callcard"].unique())
                          servicio_localizador=st.selectbox("¿Tiene un localizador?", churn_df["pager"].unique())
                          ebill=st.selectbox("¿El cliente utiliza facturación electronica?", churn_df["ebill"].unique())
                          equipmon=st.select_slider("Gastos mensuales en alquiler de equipos", np.sort(churn_df["equipmon"].unique()))
                          longten=st.select_slider("Gastos totales de llamadas larga distancia", np.sort(churn_df["longten"].unique()))
                          custcan=st.selectbox("Categoria del cliente", np.sort(churn_df["custcat"].unique()))
            variable_input_sumit = st.form_submit_button("Enviar")
                          

#X = churn_df.drop("churn", axis=1)


#X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', "callcard", "wireless", "longmon", "tollmon", "equipmon", "cardmon", "wiremon", "longten", "tollten", "cardten", "voice", "pager", "internet", "callwait", "confer", "ebill", "custcat"]]

#y = churn_df['churn'].astype('int')

#callcard, wireless, employ son otras clasificaciones por ende no sirven, los otros son solo numeros
#X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', "callcard", "wireless"]]


#y = churn_df['churn']
#print(y.head())

#from sklearn.preprocessing import MinMaxScaler

#mm = MinMaxScaler()

#X = mm.fit_transform(X)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, f1_score
#from sklearn.preprocessing import label_binarize
#from sklearn.tree import DecisionTreeClassifier


#from sklearn.ensemble import ExtraTreesClassifier


#ETC2 = ExtraTreesClassifier(oob_score=True, 
#                          random_state=5, 
#                          warm_start=True,
#                          bootstrap=True,
#                          n_jobs=-1,
#                          n_estimators=30)
#ETC2.fit(X_train, y_train)

#y_pred_ETC2 = ETC2.predict(X_test)

#precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_ETC2, beta=5, pos_label=1, average='weighted')
#auc = roc_auc_score(y_test, y_pred_ETC2, average='weighted')
#auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_ETC2,  classes=[1,2,3]), average='weighted')
#accuracy = accuracy_score(y_test, y_pred_ETC2)
#st.write("Extra Tree")
#st.write(f"Accuracy is: {accuracy:.2f}")
#st.write(f"Precision is: {precision:.2f}")
#st.write(f"Recall is: {recall:.2f}")
#st.write(f"Fscore is: {f_beta:.2f}")
#st.write(f"AUC is: {auc:.2f}")
#print(" ")



st.write(":blue[Proyecto en proceso, será terminado hoy 1/08/2024]")





