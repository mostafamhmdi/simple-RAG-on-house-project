from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_splitter(file_path):
    print("Loading document...")
    loader = PyPDFLoader(file_path=file_path)
    
    pdf = loader.load()
    
    
    print("splitting...")
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = textsplitter.split_documents(pdf)
    return chunks