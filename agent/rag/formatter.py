from engine import get_retriever

def retrieve_context(query: str) -> str:
    retriever= get_retriever()
    nodes= retriever.retrieve(query)
    if not nodes:
        return "No relevant information found."

    results = []
    for i, node_with_score in enumerate(nodes, start=1):
        node= getattr(node_with_score, "node", node_with_score)
        score= getattr(node_with_score, "score", None)

        text= node.get_content()
        metadata= getattr(node, "metadata", {})
        source= (
            metadata.get("file_name")
            or metadata.get("source")
            or "unknown source")

        results.append(
            f"[Document {i} | score={score} | source={source}]\n{text}\n{metadata}")

    return "\n\n".join(results)