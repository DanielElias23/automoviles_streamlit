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

st.write(":blue[Rellena el formulario]")

st.title("Perdida de clientes")

st.write("El formulario predice que tan probable es que el cliente deje la empresa que le esta prestando algún servcio.")

st.write("Los datos presentados provienen de una empresa de telecomunicaciones, estos datos son de clientes y contienen informacion relevante para saber si el cliente permanecera o abandonara la empresa, los datos son antiguos por lo que el cliente ya tomo esta decision. Mediante estos datos se puede saber si otro cliente podria abandonar la empresa, ya que pueden cumplir perfiles parecidos. Esto es importante ya que permite a la empresa crear estrategias para que el cliente permanesca el mayor tiempo posible en la empresa.")

st.sidebar.write(":blue[Rellene el formularío y descubra si un cliente puede abandonar la empresa]")

st.header("Formulario con los datos del cliente:")

X = churn_df.drop(["churn", "loglong", "lninc", "logtoll"], axis=1)
y = churn_df['churn'].astype('int')

with st.form("cliente", clear_on_submit=False, border=True):
            churn_df["callcard"] = churn_df["callcard"].replace({1:"Si", 0:"No"})
            churn_df[["equip"]] = churn_df[["equip"]].replace({1:"Si", 0:"No"})
            churn_df["wireless"] = churn_df["wireless"].replace({1:"Si", 0:"No"})
            churn_df["voice"] = churn_df["voice"].replace({1:"Si", 0:"No"})
            churn_df["pager"] = churn_df["pager"].replace({1:"Si", 0:"No"})
            churn_df["internet"] = churn_df["internet"].replace({1:"Si", 0:"No"})
            churn_df["callwait"] = churn_df["callwait"].replace({1:"Si", 0:"No"})
            churn_df["confer"] = churn_df["confer"].replace({1:"Si", 0:"No"})
            churn_df["ebill"] = churn_df["ebill"].replace({1:"Si", 0:"No"})
            churn_df["ed"] = churn_df["ed"].replace({1:"Educación Basica", 2:"Educación Media", 3:"Educación Superior", 4:"Magistrado/a", 5:"Doctorado/a"})
            st.subheader("Datos personales del cliente:")
            Nombre=st.text_input(label="Escriba el nombre del cliente")           
            #marca_auto=st.selectbox("Marca del vehículo", data3["Brand"].unique())   
                     
            left_column, right_column=st.columns(2)
            with left_column:
                          income=st.select_slider("Ingreso anual del cliente", np.sort(churn_df["income"].unique()))
                          años_trabajados=st.select_slider("Años trabajando", np.sort(churn_df["employ"].unique()))
                          numero_años_en_residencia=st.select_slider("Años en su vivienda actual", churn_df["address"].unique())                    
            with right_column:
                          nivel_de_estudio=st.selectbox("Nivel de estudio", churn_df["ed"].unique(), index=True)                               
                          edad_del_cliente=st.select_slider("Edad del cliente", np.sort(churn_df["age"].unique()))
                          tenure=st.select_slider("Meses que el cliente ha permanecido en la empresa", np.sort(churn_df["tenure"].unique()))
            st.subheader("Datos del servicio del cliente:")
            left_column, right_column, three_column=st.columns(3)
            with left_column:
                          servicio_de_internet=st.selectbox("¿Contrato servicios de internet?", np.sort(churn_df["internet"].unique()))
                          
                          equipo_de_empresa=st.selectbox("¿Tiene equipos de la empresa?", np.sort(churn_df["equip"].unique()))
                          
                          callwait=st.selectbox("¿Contrato servicio de llamada en espera?", np.sort(churn_df["callwait"].unique()))
                          
                          longmon=st.select_slider("Gastos mensuales en llamadas larga distancia", np.sort(churn_df["longmon"].unique()))
                          cardmon=st.select_slider("Gastos mensuales en llamadas", np.sort(churn_df["cardmon"].unique()))
                          tollten=st.select_slider("Gastos totales de peajes (toll charger)", np.sort(churn_df["tollten"]))
                          
            with right_column:
                          servicio_inalambricos=st.selectbox("¿Contrato servicios inalambricos?", np.sort(churn_df["wireless"].unique()))
                          
                          servicio_de_voz=st.selectbox("¿Contrato servicio de correo de voz?", np.sort(churn_df["voice"].unique()))
                          
                          confer=st.selectbox("¿Contrato servicio de conferencias?", np.sort(churn_df["confer"].unique()))
                          
                          tollmon=st.select_slider("Gastos mensuales en peajes (toll charger)", np.sort(churn_df["tollmon"].unique()))
                          wiremon=st.select_slider("Gastos mensuales en servicios inalambricos", np.sort(churn_df["wiremon"].unique()))
                          cardten=st.select_slider("Gastos totales en llamadas", np.sort(churn_df["cardten"].unique()))
            with three_column:
                          servicio_telefonico=st.selectbox("¿Contrato servicios telefonicos?", np.sort(churn_df["callcard"].unique()))
                          
                          servicio_localizador=st.selectbox("¿Tiene un localizador?", np.sort(churn_df["pager"].unique()))
                          
                          ebill=st.selectbox("¿El cliente utiliza facturación electronica?", np.sort(churn_df["ebill"].unique()))
                          
                          equipmon=st.select_slider("Gastos mensuales en alquiler de equipos", np.sort(churn_df["equipmon"].unique()))
                          longten=st.select_slider("Gastos totales de llamadas larga distancia", np.sort(churn_df["longten"].unique()))
                          custcat=st.selectbox("Plan contratado", np.sort(churn_df["custcat"].unique()))
            variable_input_sumit = st.form_submit_button("Enviar")
                          




#X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', "callcard", "wireless", "longmon", "tollmon", "equipmon", "cardmon", "wiremon", "longten", "tollten", "cardten", "voice", "pager", "internet", "callwait", "confer", "ebill", "custcat"]]



#callcard, wireless, employ son otras clasificaciones por ende no sirven, los otros son solo numeros
#X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', "callcard", "wireless"]]


#y = churn_df['churn']
#print(y.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier


from sklearn.ensemble import ExtraTreesClassifier


ETC2 = ExtraTreesClassifier(oob_score=True, 
                          random_state=5, 
                          warm_start=True,
                          bootstrap=True,
                          n_jobs=-1,
                          n_estimators=30)
ETC2.fit(X_train, y_train)

y_pred_ETC2 = ETC2.predict(X_test)

precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_ETC2, beta=5, pos_label=1, average='weighted')
auc = roc_auc_score(y_test, y_pred_ETC2, average='weighted')
#auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_ETC2,  classes=[1,2,3]), average='weighted')
accuracy = accuracy_score(y_test, y_pred_ETC2)
#st.write("Extra Tree")
#st.write(f"Accuracy is: {accuracy:.2f}")
#st.write(f"Precision is: {precision:.2f}")
#st.write(f"Recall is: {recall:.2f}")
#st.write(f"Fscore is: {f_beta:.2f}")
#st.write(f"AUC is: {auc:.2f}")

if variable_input_sumit:
        data_3=pd.DataFrame([Nombre, tenure, edad_del_cliente, numero_años_en_residencia, income, nivel_de_estudio, años_trabajados , equipo_de_empresa, servicio_telefonico, servicio_inalambricos, longmon, tollmon, equipmon, cardmon, wiremon, longten, tollten, cardten, servicio_de_voz, servicio_localizador, servicio_de_internet, callwait, confer, ebill, custcat]).T
        data_3=data_3.rename(columns={0:"Nombre del cliente", 1:"tiempo en la empresa", 2:"Edad", 3:"años en residencia", 4:"ingreso anual", 5:"nivel educativo", 6:"Años de trabajo", 7:"Posee equipos de la empresa", 8:"Posee servicios de llamadas", 9:"Posee servicios inalambricos", 10:"Gastos mensuales en llamadas larga distancias", 11:"Gastos mensuales de peaje", 12:"Gastos mensuales en equipos", 13:"Gastos mensules en llamadas", 14:"Gastos mensuales en servicios inalambricos", 15:"Gastos totales en llamadas larga distancias", 16:"Gastos totales en peajes", 17:"Gastos totales en llamadas", 18:"Servicio de buzon de voz", 19:"Servicio de localizador", 20:"Servicio de internet", 21:"Serivcio de llamadas en espera", 22:"Servicio de conferencias", 23:"Utiliza facturación electronica", 24:"Plan del cliente"})
        
        st.write(f":blue[Los datos del cliente seleccionados son:]")                       
        st.write(data_3)
        
        if  nivel_de_estudio=="Educación Basica":
                     nivel_de_estudio=1
        if  nivel_de_estudio=="Educación Media":
                     nivel_de_estudio=2  
        if  nivel_de_estudio=="Educación Superior":
                     nivel_de_estudio=3
        if  nivel_de_estudio=="Magistrado/a":
                     nivel_de_estudio=4
        if  nivel_de_estudio=="Doctorado/a":
                     nivel_de_estudio=5 
                     
        if servicio_de_internet=="Si":
                     servicio_de_internet=1
        if servicio_de_internet=="No":
                     servicio_de_internet=0
        
        if equipo_de_empresa=="Si":
                     equipo_de_empresa=1
        if equipo_de_empresa=="No":
                     equipo_de_empresa=0
        
        if callwait=="Si":
                     callwait=1
        if callwait=="No":
                     callwait=0
        
        if servicio_inalambricos=="Si":
                     servicio_inalambricos=1
        if servicio_inalambricos=="No":
                     servicio_inalambricos=0
        
        if servicio_de_voz=="Si":
                     servicio_de_voz=1
        if servicio_de_voz=="No":
                     servicio_de_voz=0
        
        if confer=="Si":
                     confer=1
        if confer=="No":
                     confer=0
        
        if servicio_telefonico=="Si":
                     servicio_telefonico=1
        if servicio_telefonico=="No":
                     servicio_telefonico=0
        
        if servicio_localizador=="Si":
                     servicio_localizador=1
        if servicio_localizador=="No":
                     servicio_localizador=0
        
        if ebill=="Si":
                     ebill=1
        if ebill=="No":
                     ebill=0
                        
        data_2=pd.DataFrame([tenure, edad_del_cliente, numero_años_en_residencia, income, nivel_de_estudio, años_trabajados , equipo_de_empresa, servicio_telefonico, servicio_inalambricos, longmon, tollmon, equipmon, cardmon, wiremon, longten, tollten, cardten, servicio_de_voz, servicio_localizador, servicio_de_internet, callwait, confer, ebill, custcat]).T
        data_2=data_2.rename(columns={0:"tenure", 1:"age", 2:"address", 3:"income", 4:"ed", 5:"employ", 6:"equip", 7:"callcard", 8:"wireless", 9:"longmon", 10:"tollmon", 11:"equipmon", 12:"cardmon", 13:"wiremon", 14:"longten", 15:"tollten", 16:"cardten", 17:"voice", 18:"pager", 19:"internet", 20:"callwait", 21:"confer", 22:"ebill", 23:"custcat"}) 
        
        y_pred_ETC3 = ETC2.predict(data_2)
        
        if y_pred_ETC3==0:
              y_pred_ETC3=f":green[permanesca] en la empresa"
              mensaje=":blue[Felicitaciones estas entregando un buen servicio]"
        
        if y_pred_ETC3==1:
              y_pred_ETC3=":red[abandoné] la empresa"      
              mensaje=":blue[Recomendamos crear estrategias para mantener la permanencia del cliente]"    
        
        st.header(f"Es probable que el cliente :blue[{Nombre}] {y_pred_ETC3}")
        
        st.write(mensaje)
        
        st.write(f"Modelo ML: :green[ExtraTree]") 
        st.write(f"Accuracy: :green[{round(accuracy,2)}]     Recall: :green[{round(recall,2)}]")
        st.write(f"Presición: :green[{round(precision,2)}]     Fscore: :green[{round(f_beta,2)}]")
               
        
        









