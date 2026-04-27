import re
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="CampusAI Chatbot",
    page_icon="🎓",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 2rem; }
h1 { text-align: center; font-weight: 700; }
.chat-message { padding: 12px; border-radius: 10px; margin-bottom: 10px; }
.user-msg { background-color: #1f77b4; color: white; }
.bot-msg { background-color: #262730; color: white; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>🎓 CampusAI – Intelligent Student Query Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Ask anything about your university</p>", unsafe_allow_html=True)
st.divider()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📌 About")
    st.write("""
    This AI helps students with:
    - Courses
    - Admissions
    - Campus details
    - General queries
    """)
    st.divider()
    st.write("⚙️ Model: Mistral-7B")
    st.write("📚 Mode: RAG (Context-based)")

# ---------- CONSTANTS ----------
VECTOR_STORE_PATH = "vectorstore/db_faiss"

PROMPT_TEMPLATE = """
You are a Student Query Assistant.

RULES:
- Answer ONLY from the provided context.
- If not found, say: "I don't know based on the provided information."
- Be clear and concise.

Context:
{context}

Conversation:
{history}

Question:
{input}

Answer:
"""

# ---------- LOADERS ----------
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

# ---------- CHAIN ----------
def get_chain():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
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

# ---------- INIT ----------
if "chain" not in st.session_state:
    st.session_state.chain = get_chain()

# ---------- CHAT DISPLAY ----------
for msg in st.session_state.chain.memory.chat_memory.messages:
    if msg.type == "human":
        st.markdown(f"<div class='chat-message user-msg'>🧑‍💻 {msg.content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message bot-msg'>🤖 {msg.content}</div>", unsafe_allow_html=True)

# ---------- INPUT ----------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.markdown(f"<div class='chat-message user-msg'>🧑‍💻 {user_input}</div>", unsafe_allow_html=True)

    result = st.session_state.chain.invoke({"input": user_input})

    raw = result["text"]
    cleaned = re.sub(r"<.*?>", "", raw)
    final = cleaned.strip()

    st.markdown(f"<div class='chat-message bot-msg'>🤖 {final}</div>", unsafe_allow_html=True)