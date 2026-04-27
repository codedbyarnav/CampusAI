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
    except Exception:
        st.error("❌ Vector store not found. Upload vectorstore folder to GitHub.")
        st.stop()

vector_db = load_vectorstore()
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# ---------------- MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- PROMPT ----------------
def build_prompt(context, history, question):
    return f"""
You are CampusAI, a student assistant.

RULES:
- Answer ONLY using the context.
- If not found, say: "I don't know based on the provided information."
- Keep answer short and clear.

Context:
{context}

Chat History:
{history}

Question:
{question}

Answer:
"""

# ---------------- DISPLAY OLD CHAT ----------------
for chat in st.session_state.chat_history:
    st.markdown(f"🧑‍💻 {chat['user']}")
    st.markdown(f"🤖 {chat['bot']}")

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.markdown(f"🧑‍💻 {user_input}")

    # -------- RETRIEVAL --------
    try:
        docs = retriever.invoke(user_input)
        context = "\n\n".join([d.page_content for d in docs])
    except Exception:
        context = "No context found."

    # -------- HISTORY --------
    history_text = "\n".join(
        [f"User: {c['user']}\nBot: {c['bot']}" for c in st.session_state.chat_history]
    )

    prompt = build_prompt(context, history_text, user_input)

    # -------- LLM (FIXED MODEL) --------
    try:
        response = client.chat_completion(
            model="HuggingFaceH4/zephyr-7b-beta",  # ✅ MUCH MORE STABLE
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        answer = response.choices[0].message["content"]

        cleaned = re.sub(r"<.*?>", "", answer).strip()

    except Exception as e:
        cleaned = f"⚠️ Error: {str(e)}"

    st.markdown(f"🤖 {cleaned}")

    # -------- SAVE --------
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": cleaned
    })
