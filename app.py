import re
import os
import streamlit as st

from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- API KEY ----------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_API_KEY_HERE"

# ---------------- PAGE ----------------
st.set_page_config(page_title="CampusAI Chatbot", page_icon="🎓")

st.title("🎓 CampusAI – Student Assistant")
st.write("Ask anything about your university")

# ---------------- CONSTANTS ----------------
VECTOR_STORE_PATH = "vectorstore/db_faiss"

PROMPT_TEMPLATE = """
Answer ONLY using the context below.
If not found, say:
"I don't know based on the provided information."

Context:
{context}

Chat History:
{history}

Question:
{question}

Answer:
"""

# ---------------- LOAD ----------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

vector_db = load_vectorstore()
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.2, "max_length": 512}
)

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

memory = ConversationBufferMemory(return_messages=True)

# ---------------- CHAT ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask your question...")

if user_input:
    # Retrieve context
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs])

    history_text = "\n".join(
        [f"User: {c['user']}\nBot: {c['bot']}" for c in st.session_state.chat_history]
    )

    # Create prompt
    final_prompt = prompt.format(
        context=context,
        history=history_text,
        question=user_input
    )

    # Get response
    response = llm(final_prompt)

    cleaned = re.sub(r"<.*?>", "", response).strip()

    # Save history
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": cleaned
    })

# ---------------- DISPLAY ----------------
for chat in st.session_state.chat_history:
    st.markdown(f"🧑‍💻 {chat['user']}")
    st.markdown(f"🤖 {chat['bot']}")
