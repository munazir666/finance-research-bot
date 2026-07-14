import streamlit as st
import os
import sys

# Force transformers to use legacy tf-keras to prevent Keras 3 conflict
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# Fix protobuf gencode/runtime mismatch
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv

# Add project root to sys.path so 'src' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set page config before any other Streamlit commands
st.set_page_config(page_title="AI Research Assistant", page_icon="🔍")

# Import custom modules
from src.ingestion import load_and_chunk_documents
from src.retriever import build_or_get_vector_store, get_retriever
from src.rag_chain import create_rag_chain

# Load environment variables
load_dotenv()

@st.cache_resource
def initialize_system():
    """
    Initializes the system: ingests documents (if needed), builds index, and returns the chain.
    Uses @st.cache_resource so it only runs once per app lifecycle.
    """
    try:
        # 1. Load and chunk documents
        with st.spinner("Loading index... (This will only take a few seconds)"):
            if not os.path.exists("chroma_db"):
                chunks = load_and_chunk_documents("data")
                vector_store, _ = build_or_get_vector_store(chunks)
            else:
                vector_store, _ = build_or_get_vector_store(None)
            
            # 3. Get retriever
            retriever = get_retriever(vector_store, k=4)
            
            # 4. Create chain
            chain = create_rag_chain(retriever)
            
        return chain
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return None

def main():
    st.title("🔍 AI Research Assistant")
    st.markdown("Ask questions about the uploaded PDF and DOCX research documents. The system will retrieve relevant chunks and provide a sourced answer.")
    
    # Check for required API keys
    if not os.getenv("GROQ_API_KEY"):
        st.warning("Please set your `GROQ_API_KEY` in the `.env` file.")
        return

    # Initialize the RAG chain
    chain = initialize_system()
    
    if chain is None:
        return

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
