import os
import glob
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents(data_dir: str = "data"):
    """
    Loads all .pdf and .docx files from the specified directory and splits them into chunks.
    """
    documents = []
    
    # Process PDF files
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        
    # Process DOCX files
    docx_files = glob.glob(os.path.join(data_dir, "*.docx"))
    for file_path in docx_files:
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
        
    print(f"Loaded {len(documents)} document pages.")
    
    if not documents:
        return []
        
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    return chunks

if __name__ == "__main__":
    # Test run
    chunks = load_and_chunk_documents()
    if chunks:
        print(f"First chunk sample: {chunks[0].page_content[:100]}...")
