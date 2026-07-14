import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """
    Formats retrieved documents into a string with numbered sources for citation.
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        content = doc.page_content
        formatted.append(f"[Source {i+1} - {os.path.basename(source)}, Page {page}]:\n{content}")
    return "\n\n".join(formatted)

def create_rag_chain(retriever):
    """
    Creates the RAG chain combining the retriever and the LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in .env")

    # Initialize Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
    )

    # Define the prompt template
    system_prompt = (
        "You are an AI Research Assistant. Use the following pieces of retrieved context to answer the user's question.\n"
        "If the context does not contain enough information to answer the question, clearly state 'I don't have enough information'.\n"
        "Do not make up facts or use outside knowledge.\n"
        "You MUST always cite which part of the context you used by referencing the Source numbers (e.g., [Source 1]).\n"
        "\n"
        "Context:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # Build the chain
    # We use RunnablePassthrough.assign to keep both context and question for the prompt
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
