from dotenv import load_dotenv
import os
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pubmed.engine as pub


load_dotenv()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

#----------------->>>>>add full paper reader<<<<<---------------
def pubmed_context(query: str, max_results: int = 20, k: int = 5) -> str:
    """
    Search PubMed, fetch candidate articles, index their abstracts temporarily,
    and retrieve the most relevant chunks.

    PubMed supplies the PMIDs. The agent does not guess them.
    """
    articles= pub.search_pubmed_articles(query=query,
                                    max_results=max_results)
    if not articles:
        return "No relevant PubMed articles found."
    embed_model= HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model= embed_model
    documents= []

    for article in articles:
        text = f"""
        Title: {article.title}
        Journal: {article.journal}
        Year: {article.year}
        PMID: {article.pmid}
        Abstract:
        {article.abstract}
        """.strip()

        documents.append(Document(text=text,
                        metadata={"source": "pubmed",
                        "pmid": article.pmid,
                        "title": article.title,
                        "journal": article.journal,
                        "year": article.year}))

    index= VectorStoreIndex.from_documents(documents,
                                        embed_model=embed_model,)
    retriever= index.as_retriever(similarity_top_k=k)
    nodes= retriever.retrieve(query)

    if not nodes: return "No relevant PubMed chunks found."
    results= []

    for i, node_with_score in enumerate(nodes, start=1):
        node= getattr(node_with_score, "node", node_with_score)
        score= getattr(node_with_score, "score", None)
        text= node.get_content()
        metadata= node.metadata or {}
        pmid= metadata.get("pmid", "unknown PMID")
        title= metadata.get("title", "unknown title")
        journal= metadata.get("journal", "unknown journal")
        year= metadata.get("year", "unknown year")

        results.append(
            f"""[PubMed Document {i} | score={score} | PMID={pmid} | {journal} | {year}] 
            Title: {title}
            {text[:2000]}""")

    return "\n\n".join(results)