from groq import Groq
import streamlit as st

import base64
import streamlit as st



@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}
img = get_img_as_base64("de_chat.png")

#https://www.blogdelfotografo.com/wp-content/uploads/2021/12/Fondo_Negro_3.webp
#https://w0.peakpx.com/wallpaper/596/336/HD-wallpaper-azul-marino-agua-clima-nubes-oscuro-profundo.jpg
#
#https://i.pinimg.com/564x/8b/5a/a4/8b5aa4783578968cd257b0b5418f3645.jpg
#https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkwNy1hdW0tNDQteC5qcGc.jpg
#https://i.pinimg.com/736x/f6/c1/0a/f6c10a01b8f7c3285fc660b0b0664e52.jpg
#https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkwNy1hdW0tNDQteC5qcGc.jpg
#https://c.wallhere.com/photos/ac/a0/color_pattern_texture_background_dark-672126.jpg!d

#Colores
#https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg

#https://e0.pxfuel.com/wallpapers/967/154/desktop-wallpaper-solid-black-1920%C3%971080-black-solid-color-background-top.jpg
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: top right;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://i.pinimg.com/736x/db/b6/fd/dbb6fd0af09c84b074ef884c72309020.jpg");
background-size: 80%;
background-position: top left; 
background-repeat: no-repeat;
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


client = Groq(api_key="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")

st.subheader("Elige un chat y conversa")

pagina1, pagina2 =st.tabs(["**Llama 3.1**","Gemma 2"])

with pagina1:
    st.sidebar.header("Opciones del chat")
       
    numero = st.sidebar.select_slider("Ajuste de creatividad llama 3.1", ["Muy serio",0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "Normal", 1.1, 1.2, 1.3,1.4,1.5,1.6,1.7, 1.8, 1.9, "Muy creativo"])

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
      st.write("Escribe un comentario o pregunta y mantiene una conversación con la IA.")
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
            
      with st.form(key="chat_form", clear_on_submit=True, border=True):
            st.text_input("Tu:", key="user_input", placeholder="Escribe un mensaje")
            submit_button = st.form_submit_button(label="Enviar", on_click=submit)
      css="""
      <style>
            [data-testid="stForm"] {
               background: Purple;
            }
      </style>
      """
      st.write(css, unsafe_allow_html=True)      
      
    if __name__ == "__main__":
      chat()
      
with pagina2:
    #st.sidebar.header("Opciones del chat")
       
    numero2 = st.sidebar.select_slider("Ajuste de creatividad gemma 2", ["Muy serio",0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "Normal", 1.1, 1.2, 1.3,1.4,1.5,1.6,1.7, 1.8, 1.9, "Muy creativo"])

    if numero2 =="Muy serio":
      numero2=0
    if numero2 =="Normal":
      numero2=1
    if numero2 =="Muy creativo":
      numero2=2
    cleint = Groq(api_key="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")
    def get_ai_response2(messages2, numero2):
      completion = client.chat.completions.create(
              model="gemma2-9b-it",
              messages=messages2,
              temperature=numero2,  #0.7,
              max_tokens=1024,
              stream=True,
      )
      
      response = "".join(chunk.choices[0].delta.content or "" for chunk in completion)
      return response
      

      
    def chat():
      st.title("Chat con Gemma 2")
      st.write("¡Bienvenidos al chat con IA! Para refrescar la conversación actualiza la página.")
      st.write("Escribe un comentario o pregunta y mantiene una conversación con la IA.")
      if "messages2" not in st.session_state:
            st.session_state["messages2"]=[]
      
      #if "numero" not in st.session_state:
      #      st.session_state["numero"]=0.7
      
      
      
      def submit():
            user_input2 = st.session_state.user_input2
            if user_input2.lower() == "exit":
                  st.write("!Gracias por chatear! ¡Adios!")
                  st.stop()
            #if i in numero:                  
            
            st.session_state["messages2"].append({"role": "user", "content": user_input2})
            
            with st.spinner("Obtieniendo respuesta..."):
                 ai_response2 = get_ai_response2(st.session_state["messages2"], numero2)
                 st.session_state["messages2"].append({"role": "assistant", "content": ai_response2})  
                 
            st.session_state.user_input2 = ""
            
            
      for message2 in st.session_state["messages2"]:
            role = "- 👨 **Tu** " if message2["role"] == "user" else "- 🤖 **Bot**"
            st.write(f"{role}: {message2['content']}")
            
      with st.form(key="chat_form2", clear_on_submit=True, border=True):
            st.text_input("Tu:", key="user_input2", placeholder="Escribe un mensaje")
            submit_button = st.form_submit_button(label="Enviar", on_click=submit)
            
      
    if __name__ == "__main__":
      chat()      
      
      
      
      
      
      
      
      
      
      
























