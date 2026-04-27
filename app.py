import re
import streamlit as st
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- API ----------------
client = InferenceClient(
    token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
)

# ---------------- UI ----------------
st.set_page_config(page_title="CampusAI", page_icon="🎓")

st.title("🎓 CampusAI – Student Assistant")
st.write("Ask anything about your university")

# ---------------- VECTOR DB ----------------
VECTOR_STORE_PATH = "vectorstore/db_faiss"

@st.cache_resource
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error("❌ Vector store not found. Please check deployment.")
        st.stop()

vector_db = load_vectorstore()
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# ---------------- MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- PROMPT ----------------
def build_prompt(context, history, question):
    return f"""
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

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.markdown(f"🧑‍💻 {user_input}")

    # 🔥 SAFE RETRIEVAL
    try:
        docs = list(retriever.invoke(user_input))
    except Exception:
        docs = []

    context = "\n\n".join([d.page_content for d in docs]) if docs else "No context found."

    # History
    history_text = "\n".join(
        [f"User: {c['user']}\nBot: {c['bot']}" for c in st.session_state.chat_history]
    )

    prompt = build_prompt(context, history_text, user_input)

    # 🔥 SAFE LLM CALL
    try:
        response = client.text_generation(
            prompt,
            model="google/flan-t5-base",
            max_new_tokens=300
        )
        cleaned = re.sub(r"<.*?>", "", response).strip()
    except Exception:
        cleaned = "⚠️ Error generating response. Try again."

    st.markdown(f"🤖 {cleaned}")

    # Save history
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": cleaned
    })

# ---------------- DISPLAY ----------------
for chat in st.session_state.chat_history[:-1]:
    st.markdown(f"🧑‍💻 {chat['user']}")
    st.markdown(f"🤖 {chat['bot']}")
