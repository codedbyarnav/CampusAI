import re
import os
import streamlit as st

from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CampusAI Chatbot",
    page_icon="🎓",
    layout="wide"
)

# ---------------- API KEY ----------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_API_KEY_HERE"

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;'>🎓 CampusAI – Student Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask anything about your university</p>", unsafe_allow_html=True)
st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("About")
    st.write("""
    AI chatbot for:
    - Admissions
    - Courses
    - Campus queries
    """)
    st.divider()
    st.write("Model: flan-t5-base")
    st.write("Mode: RAG")

# ---------------- CONSTANTS ----------------
VECTOR_STORE_PATH = "vectorstore/db_faiss"

PROMPT_TEMPLATE = """
Answer ONLY using the context.

If not found, say:
"I don't know based on the provided information."

Context:
{context}

Conversation:
{history}

Question:
{input}

Answer:
"""

# ---------------- LOADERS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ---------------- CHAIN ----------------
def get_chain():
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input"
    )

    vector_db = load_vectorstore()
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    prompt = PromptTemplate(
        input_variables=["context", "history", "input"],
        template=PROMPT_TEMPLATE
    )

    class RAGChain(LLMChain):
        def invoke(self, inputs, **kwargs):
            docs = retriever.get_relevant_documents(inputs["input"])
            context = "\n\n".join([d.page_content for d in docs])
            inputs["context"] = context
            return super().invoke(inputs, **kwargs)

    return RAGChain(llm=llm, prompt=prompt, memory=memory)

# ---------------- INIT ----------------
if "chain" not in st.session_state:
    st.session_state.chain = get_chain()

# ---------------- DISPLAY ----------------
for msg in st.session_state.chain.memory.chat_memory.messages:
    role = "🧑‍💻" if msg.type == "human" else "🤖"
    st.markdown(f"{role} {msg.content}")

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.markdown(f"🧑‍💻 {user_input}")

    result = st.session_state.chain.invoke({"input": user_input})

    raw = result["text"]
    cleaned = re.sub(r"<.*?>", "", raw)
    final = cleaned.strip()

    st.markdown(f"🤖 {final}")
