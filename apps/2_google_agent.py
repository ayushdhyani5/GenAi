from httpx2 import query
from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm=ChatGroq(model="openai/gpt-oss-20b")
search=GoogleSerperAPIWrapper()
agent=create_agent(model=llm,tools=[search.run],
    system_prompt="you are an agent who has access to google search",
    checkpointer=MemorySaver()
)
while True:
    query=input("User:  ")
    if(query=="exit"):
        break
    res=agent.invoke({
        "messages":[{"role":"user","content":query}]},
        {"configurable": {"thread_id":"abc2"}}
)
    print("Agent: ",res["messages"][-1].content)