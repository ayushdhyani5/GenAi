
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import streamlit as st
load_dotenv()

# Database connection
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

# Create table
db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        status TEXT CHECK(
            status IN ('pending', 'in progress', 'completed')
        ) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-20b")

# Initialize SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Memory


system_prompt = """
You are a task management assistant connected to a SQLite database.

Database schema:
tasks(id, title, status, created_at)

Rules:
1. Only perform operations on the tasks table.
2. Limit SELECT queries to 10 results.
3. Order task listings by created_at DESC.
4. After INSERT, UPDATE, or DELETE, verify the operation.
5. Valid statuses are pending, in progress, completed.
6. Never invent task IDs or database results.
7. Ask for clarification if a task is ambiguous.
"""

# Create agent
@st.cache_resource
def get_agent():
    agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt
    )
    return agent
agent=get_agent()
# Conversation loop
st.subheader("TaskBot: Manage your tasks")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt=st.chat_input("Ask me")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("ai"):
        with st.spinner("processing..."):
            response = agent.invoke(
            {
            "messages": [
                {"role": "user", "content": prompt}
            ]
            },
            {
            "configurable": {"thread_id": "1"}
            }
            )

            result = response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role":"ai", "content":result})


    print("\nAI:", result)
