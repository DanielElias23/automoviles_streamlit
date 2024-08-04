from groq import Groq
import streamlit as st

client = Groq(api_key="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")


st.sidebar.header("Opciones del chat")
       
numero = st.sidebar.select_slider("Ajuste de creatividad", ["Muy serio",0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "Normal", 1.1, 1.2, 1.3,1.4,1.5,1.6,1.7, 1.8, 1.9, "Muy creativo"])

if numero =="Serio":
    numero=0
if numero =="Normal":
    numero=1
if numero =="Creativo":
    numero=2

def get_ai_response(messages, numero):
      completion = client.chat.completions.create(
              model="llama-3.1-70b-versatile",
              messages=messages,
              temperature=numero,  #0.7,
              max_tokens=1024,
              stream=True,
      )
      
      response = "".join(chunk.choices[0].delta.content or "" for chunk in completion)
      return response
      

      
def chat():
      st.title("Chat con Llama 3.1")
      st.write("¡Bienvenidos al chat con IA! Escribe 'exist' para terminar la conversación. ")
      if "messages" not in st.session_state:
            st.session_state["messages"]=[]
      
      #if "numero" not in st.session_state:
      #      st.session_state["numero"]=0.7
      
      
      
      def submit():
            user_input = st.session_state.user_input
            if user_input.lower() == "exit":
                  st.write("!Gracias por chatear! ¡Adios!")
                  st.stop()
            #if i in numero:                  
            
            st.session_state["messages"].append({"role": "user", "content": user_input})
            
            with st.spinner("Obtieniendo respuesta..."):
                 ai_response = get_ai_response(st.session_state["messages"], numero)
                 st.session_state["messages"].append({"role": "assistant", "content": ai_response})  
                 
            st.session_state.user_input = ""
            
            
      for message in st.session_state["messages"]:
            role = "Tu" if message["role"] == "user" else "Bot"
            st.write(f"**{role}:** {message['content']}")
            
      with st.form(key="chat_form", clear_on_submit=True):
            st.text_input("Tu:", key="user_input")
            submit_button = st.form_submit_button(label="Enviar", on_click=submit)
            
      
if __name__ == "__main__":
      chat()
      
      
      
      
      
      
      
      
      
      
      
      
























