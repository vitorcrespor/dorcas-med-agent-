import asyncio
from dotenv import load_dotenv
from llama_index.core import Document

import pubmed.engine as pub
from rag.engine import retrieve_from_documents
import os

load_dotenv()
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

async def pubmed_context(query: str, max_results: int = 5, k: int = 5) -> str:
    """
    Search PubMed, fetch candidate articles, index their abstracts temporarily,
    and retrieve the most relevant chunks.

    PubMed supplies the PMIDs. The agent does not guess them.
    """
    articles= await pub.search_pubmed_articles(query=query,
                                    max_results=max_results)
    if not articles:
        return "No relevant PubMed articles found."
    documents= []

    for article in articles:
        article_text = article.full_text or article.abstract
        source= ("pmc_full_text"
                if article.full_text
                else "pubmed_abstract")

        documents.append(
            Document(
                    text=article_text,
                    metadata={
                        "source": source,
                        "pmid": article.pmid,
                        "title": article.title,
                        "journal": article.journal,
                        "year": article.year,
                    },
                )
            )


    nodes = await asyncio.to_thread(
    retrieve_from_documents,
    query=query,
    documents=documents,
    k=k,
)
    
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