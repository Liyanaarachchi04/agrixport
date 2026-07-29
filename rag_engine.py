import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_store():
    # Load 20+ documents
    loader = DirectoryLoader('data/', glob="**/*.*", show_progress=True)
    docs = loader.load()

    # Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Embedded Vector Database
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return vectorstore

if __name__ == "__main__":
    build_vector_store()
    print("RAG Pipeline Ingested Successfully.")