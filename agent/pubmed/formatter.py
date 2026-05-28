# rag/pubmed_formatter.py

from rag.pubmed import search_pubmed_articles


def retrieve_pubmed_context(query: str, max_results: int = 5) -> str:
    articles = search_pubmed_articles(
        query=query,
        max_results=max_results,
    )

    if not articles:
        return "No relevant PubMed articles found."

    results = []

    for i, article in enumerate(articles, start=1):
        abstract = article.abstract[:2000]

        results.append(
            f"""[PubMed Article {i} | PMID: {article.pmid} | Journal: {article.journal} | Year: {article.year}]
Title: {article.title}

Abstract:
{abstract}"""
        )

    return "\n\n".join(results)