from langchain_core.tools import tool

@tool
def retriever(query: str) -> str:
    """Retrieve relevant information from the document."""
    docs= retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    
    results= []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: {doc.page_content}")
    return "\n\n".join(results)
