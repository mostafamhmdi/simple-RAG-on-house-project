import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

VECTOR_STORE_PATH = "vectore_store/faiss_index"

def create_vector_store(chunks):
    print("Creating embeddings...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("model loaded.")
    vector_store = FAISS.from_documents(chunks,embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("Vector store created and saved successfully.")
    
def load_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return FAISS.load_local(VECTOR_STORE_PATH,embeddings=embeddings,allow_dangerous_deserialization=True)
    return None