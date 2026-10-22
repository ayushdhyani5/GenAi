'''
    llm 
    tool
    agent
    history 
    streaming
    streamlit


'''
from httpx2 import query
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm=ChatGroq(model="openai/gpt-oss-20b",streaming=True) 
search=GoogleSerperAPIWrapper()

st.subheader("Qna bot with Groq")

if "memory" not in st.session_state:
    st.session_state.memory=MemorySaver()
    st.session_state.history=[]


for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])

agent=create_agent(model=llm,tools=[search.run],
    checkpointer=st.session_state.memory,
    system_prompt="you are a helpful assistant",
    # streaming=True
)

query=st.chat_input("Enter Your Query")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append(
        {"role":"user","content":query}
    )


    res=agent.invoke({
            "messages":[{"role":"user","content":query}]},
            {"configurable": {"thread_id":"abc2"}}
    )

    ai_container=st.chat_message("ai")
    with ai_container:
        space=st.empty()
        message=""
        for chunks in res["messages"][-1].content:
            message+=chunks
            space.write(message)
        st.session_state.history.append(
            {"role":"ai","content":message}
        )