from langchain_core.tools import tool
from rag.formatter import retrieve_context

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant information from the document search from the RAG from DB."""
    return retrieve_context(query, k=5)

@tool 
def pubmed(query: str) -> str:
    """Search PubMed for biomedical literature. Use for medical, clinical, biology, and scientific evidence questions."""
    
    
"""@tool
def fhir_qa

@tool
def fhir_extract_summary"""