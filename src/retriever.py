import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_or_get_vector_store(chunks, persist_directory="./chroma_db"):
    """
    Initializes a local Chroma vector store. If chunks are provided, ingests them.
    """
    embeddings = get_embeddings()
    
    # If chunks are provided, ingest them
    if chunks:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        # Add documents in batches to avoid maximum batch size errors
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            vector_store.add_documents(batch)
    else:
        # Otherwise just connect to the existing index
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
    return vector_store, None  # None returned for client as Chroma manages it internally

def get_retriever(vector_store, k=4):
    """
    Returns a retriever configured to fetch the top k documents based on similarity.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
