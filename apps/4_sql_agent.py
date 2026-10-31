from sympy import content
from httpx2 import query
from langchain_classic.agents import agent
from langgraph.checkpoint.memory import InMemorySaver
from anthropic.types import model
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()
db=SQLDatabase.from_uri("sqlite:///my_tasks.db")


db.run('''
    create table if not exists tasks(
    id integer primary key AUTOINCREMENT ,
    title text,
    status text check(status in ('pending','in progress', 'completed')) default 'pending',
    created_at timestamp default current_timestamp
    )
''')
llm=ChatGroq(model="openai/gpt-oss-20b")

toolkit=SQLDatabaseToolkit(db=db,llm=llm)
tools=toolkit.get_tools()
memory=InMemorySaver()
system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table. 

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""

agent=create_agent(
    model=model,
    tool=tools,
    checkpointer=memory,
    system_prompt=system_prompt
    )

while True:
    query=input("User: ")
    response=agent.invoke(
        {'message':[{"role":"user","content":query}]},
        {"configurable":{"thread_id":1}}
    )
    result=response["message"][-1].content
    print("Ai:",result)


'''for tool in tools:
    print(tool.name)
'''


'print("DB created Successfully")'

