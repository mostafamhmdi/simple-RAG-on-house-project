from document_processor import load_and_splitter
from vector_store_manager import create_vector_store,load_vector_store
from llm_handler import load_llm


PDF_PATH = 'documents/house.pdf'

vector_store = load_vector_store()

if vector_store is None:
    print("No existing vector store found. Creating a new one...")
    chunks = load_and_splitter(PDF_PATH)
    create_vector_store(chunks)
    vector_store = load_vector_store()
    
print("\n✅ Vector store is ready.")

llm = load_llm()
print("\n✅ LLM is ready.")
