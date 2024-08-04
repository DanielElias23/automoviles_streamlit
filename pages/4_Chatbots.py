from groq import Groq
import streamlit as st

client = Groq(api_key="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")

pagina1, pagina2 =st.tabs(["Llama 3.1","Gemma 2"])
"""
with pagina1:
    st.sidebar.header("Opciones del chat")
       
    numero = st.sidebar.select_slider("Ajuste de creatividad", ["Muy serio",0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "Normal", 1.1, 1.2, 1.3,1.4,1.5,1.6,1.7, 1.8, 1.9, "Muy creativo"])

    if numero =="Muy serio":
      numero=0
    if numero =="Normal":
      numero=1
    if numero =="Muy creativo":
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
      st.write("¡Bienvenidos al chat con IA! Para refrescar la conversación actualiza la página.")
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
            role = "- 👨 **Tu**" if message["role"] == "user" else "- 🤖 **Bot**"
            st.write(f"{role}: {message['content']}")
            
      with st.form(key="chat_form", clear_on_submit=True):
            st.text_input("Tu:", key="user_input")
            submit_button = st.form_submit_button(label="Enviar", on_click=submit)
            
      
    if __name__ == "__main__":
      chat()
"""      
with pagina2:
    st.sidebar.header("Opciones del chat")
       
    #numero2 = st.sidebar.select_slider("Ajuste de creatividad", ["Muy serio",0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "Normal", 1.1, 1.2, 1.3,1.4,1.5,1.6,1.7, 1.8, 1.9, "Muy creativo"])

    #if numero2 =="Muy serio":
    #  numero2=0
    #if numero2 =="Normal":
    #  numero2=1
    #if numero2 =="Muy creativo":
    #  numero2=2
    cleint = Groq(api_key="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")
    def get_ai_response2(messages2):
      completion = client.chat.completions.create(
              model="gemma2-9b-it",
              messages=messages2,
              temperature=1,  #0.7,
              max_tokens=1024,
              stream=True,
      )
      
      response2 = "".join(chunk.choices[0].delta.content or "" for chunk in completion)
      return response2
      

      
    def chat2():
      st.title("Chat con Gemma 2")
      st.write("¡Bienvenidos al chat con IA! Para refrescar la conversación actualiza la página.")
      if "messages" not in st.session_state:
            st.session_state["messages"]=[]
      
      #if "numero" not in st.session_state:
      #      st.session_state["numero"]=0.7
      
      
      
      def submit2():
            user_input2 = st.session_state.user_input2
            if user_input2.lower() == "exit":
                  st.write("!Gracias por chatear! ¡Adios!")
                  st.stop()
            #if i in numero:                  
            
            st.session_state["messages2"].append({"role2": "user2", "content2": user_input2})
            
            with st.spinner("Obtieniendo respuesta..."):
                 ai_response2 = get_ai_response2(st.session_state["messages2"])
                 st.session_state["messages2"].append({"role2": "assistant2", "content2": ai_response2})  
                 
            st.session_state.user_input2 = ""
            
            
      for message2 in st.session_state["messages2"]:
            role2 = "- 👨 **Tu** " if message2["role2"] == "user2" else "- 🤖 **Bot**"
            st.write(f"{role2}: {message2['content2']}")
            
      with st.form(key="chat_form2", clear_on_submit=True):
            st.text_input("Tu:", key="user_input2")
            submit_button2 = st.form_submit_button(label="Enviar", on_click=submit2)
            
      
    if __name__ == "__main__":
      chat2()      
      
      
      
      
      
      
      
      
      
      
























