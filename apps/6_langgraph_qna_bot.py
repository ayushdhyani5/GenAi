from pydantic import BaseModel
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
load_dotenv()
class ChatState(BaseModel):
    messages:Annotated[list,add_messages]
llm=ChatGroq(model="openai/gpt-oss-20b")

def chatBotNode(state:ChatState)->ChatState:
    res=llm.invoke(state.messages)
    state.messages=[res]
    return state

graph=StateGraph(ChatState)
graph.add_node("chatbot",chatBotNode)
memory=InMemorySaver()

graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)

graph=graph.compile(checkpointer=memory)

config={"configurable":{"thread_id":"my-bot"}}

while True:
    query=input("User :")
    if query.lower() in["quit","exit"]:
        print("thanks for using me")
        break

    res=graph.invoke({"messages":[{"role":"user","content":query}]},config)
    print("Ai :",res["messages"][-1].content)
    