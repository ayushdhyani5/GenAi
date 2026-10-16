import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


st.title("My Personal QnA BOT")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

query=st.chat_input("Ask me anything")

if query:

    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user","content":query})
    res=model.invoke([
    ("system","your are the father of my friend Gaurav Rawat, talk in hinglish only always abuse abhay "),
    ("user",query)

])
    st.chat_message("ai").markdown(res.content[0]["text"])
    st.session_state.messages.append({"role":"ai","content":res.content[0]["text"]})