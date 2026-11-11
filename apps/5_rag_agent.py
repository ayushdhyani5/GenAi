
from streamlit import pdf
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
load_dotenv()

## data in st session
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False
if "agent" not in st.session_state:
    st.session_state.agent = None    
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None
if "messages" not in st.session_state:
    st.session_state.messages=[]
# 1. Load PDF

def process_document(path):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    llm = ChatGroq(
        model="openai/gpt-oss-20b"
    )
    @tool
    def retrieval_tool(question: str) -> str:
        """Retrieve relevant information from the uploaded medical PDF."""

        retrieved_docs = vector_store.similarity_search(
            question,
            k=3
        )

        if not retrieved_docs:
            return "No relevant information found."

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        return context


    # 7. System prompt
    system_prompt = """
    You are a helpful document question-answering assistant.

    Your knowledge base contains information from an uploaded PDF.

    For questions about the document:
    1. Always use the retrieval_tool.
    2. Answer using the retrieved context.
    3. Do not invent information that is not present in the context.
    4. If the information is unavailable, clearly say so.
    """

    # 8. Add conversation memory
    memory = InMemorySaver()

    # 9. Create agent
    agent = create_agent(
        model=llm,
        tools=[retrieval_tool],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent=agent
    st.session_state.document_uploaded=True

#### upload ui

if not st.session_state.document_uploaded:
    uploaded=st.file_uploader(label="upload your documents",type={"pdf"},accept_multiple_files=True)
    if uploaded:
        with st.spinner("processing"):
            path="./doc_files/"
            for file in uploaded:
                with open(path+file.name,"wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()


if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
       st.chat_message(message["role"]).markdown(message["content"])     
    query=st.chat_input("ask anything related to uploaded document")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        res=st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer=res["messages"][-1].content
        st.session_state.messages.append({"role":"ai","content":answer})

        st.chat_message("Ai").markdown(answer)
 