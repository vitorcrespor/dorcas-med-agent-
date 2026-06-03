from langchain_core.tools import tool
from rag.formatter import retrieve_context
from pubmed.formatter import pubmed_context
import fhir.engine as fhir
from fhir.summarizer import retrieve_fhir_rag_context

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant information from the document search from the RAG from DB."""
    return retrieve_context(query)

@tool 
async def pubmed_tool(query: str) -> str:
    """Search PubMed for biomedical literature. Use for medical, clinical, biology, and scientific evidence questions."""
    return await pubmed_context(query)
    
@tool
def fhir_path_tool(expression: str) -> str:
    """Run a narrow read-only FHIRPath query over the local FHIR R4 log. Use only for patient-record questions.
    - If fhirpath_tool returns status="not_found", stop using fhirpath_tool for the current question.
    - Ask the user to verify the patient name or provide a patient ID.
    - Do not guess patient identities or retry with approximate names."""
    return fhir.execute_fhirpath(expression)



@tool
async def fhir_extract_summary(query: str) -> str:
    """Run a narrow read-only FHIRPath query to extract a summary of the patient record. Use it to improve readabilty
    for patient-related questions that may need medical literature help, 
    such as "Is it possible that Rafael may have a genetic predisposition to type 2 diabetes or hyperlipidemia?"
    - Do not guess patient identities or retry with approximate names."""
    return await retrieve_fhir_rag_context(query)
    