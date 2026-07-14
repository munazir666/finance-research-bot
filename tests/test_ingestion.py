import os
import pytest
from src.ingestion import load_and_chunk_documents

def test_load_and_chunk_documents():
    """
    Test that the ingestion function successfully loads and chunks documents from the data directory.
    Assumes there is at least one valid document in the data directory.
    """
    data_dir = "data"
    
    # Ensure data directory exists
    assert os.path.exists(data_dir), f"Data directory '{data_dir}' does not exist."
    
    chunks = load_and_chunk_documents(data_dir)
    
    # Assert that chunks were created
    assert len(chunks) > 0, "No chunks were created. Ensure there are PDF or DOCX files in the data directory."
    
    # Assert properties of a chunk
    assert chunks[0].page_content is not None
    assert "source" in chunks[0].metadata
