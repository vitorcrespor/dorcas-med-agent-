from langchain_core.tools import tool
from rag.formatter import retrieve_context

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant information from the document search from the RAG."""
    return retrieve_context(query, k=5)

"""@tool
def fhir_qa

@tool
def fhir_extract_summary"""