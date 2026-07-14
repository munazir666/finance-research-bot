# AI Research Assistant (Phase 1)

This project is an AI Research Assistant that takes a user question, retrieves relevant chunks from a set of research documents (PDF/DOCX), and returns a sourced answer using a Retrieval-Augmented Generation (RAG) pipeline.

**Live Demo**: [Deploying to HuggingFace soon...]

## Architecture
The system follows a Basic RAG Pipeline data flow:
1. **Document Ingestion**: PDF and DOCX files from `data/` are loaded and chunked.
2. **Embedding & Storage**: Chunks are embedded using `sentence-transformers` and stored in a **Chroma** local vector database.
3. **Retrieval**: When a question is asked, the 4 most similar chunks are retrieved via vector search.
4. **Generation**: A Groq language model (`llama-3.1-70b-versatile`) reads the retrieved chunks and generates an answer with explicit source citations.

## How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd ai-research-assistant
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy the example environment file and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   You will need a Groq API Key.

4. **Add Data**:
   Ensure you have some PDF or DOCX files in the `data/` directory.

5. **Run the Streamlit Application**:
   ```bash
   streamlit run src/app.py
   ```
   The app will automatically ingest documents, build the Chroma index, and provide a chat interface.

## Tech Stack
- **Frontend**: Streamlit
- **LLM Orchestration**: LangChain
- **Language Model**: Groq (Llama 3 70B)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: Chroma
- **Document Processing**: PyPDF, docx2txt