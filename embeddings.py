from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdffiles(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(chunks, embedding_model, db_path):
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(db_path)

if __name__ == "__main__":
    print("Loading documents...")
    documents = load_pdffiles(DATA_PATH)

    print("Creating chunks...")
    chunks = create_chunks(documents)

    print("Generating embeddings...")
    embeddings = get_embedding_model()

    print("Creating FAISS index...")
    create_vector_store(chunks, embeddings, DB_FAISS_PATH)

    print("✅ Vector store created successfully!")