from dataclasses import dataclass
from typing import Any
import asyncio
import os
import xml.etree.ElementTree as ET

import httpx
from dotenv import load_dotenv


load_dotenv()

URL_BASE = os.getenv(
    "NCBI_URL",
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
).rstrip("/")

PMC_BIOC_URL = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/"
    "RESTful/pmcoa.cgi/BioC_json"
)


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    year: str | None = None
    full_text: str | None = None
    metadata: dict[str, Any] | None = None


def _base_params() -> dict[str, str]:
    email = os.getenv("NCBI_EMAIL")
    tool = os.getenv("NCBI_TOOL", "dorcas_agent")
    api_key = os.getenv("NCBI_API_KEY", "")

    if not email:
        raise ValueError("NCBI_EMAIL is not defined in .env")

    params = {
        "tool": tool,
        "email": email,
    }

    if api_key:
        params["api_key"] = api_key

    return params


async def search_pubmed_pmids(
    query: str,
    max_results: int = 5,
) -> list[str]:
    url = f"{URL_BASE}/esearch.fcgi"

    params = {
        **_base_params(),
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_results),
        "sort": "relevance",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()

    await asyncio.sleep(0.12)

    data = response.json()
    id_list = data["esearchresult"].get("idlist", [])

    if not id_list and len(query.split()) > 7:
        compact_query = " ".join(query.split()[:7])
        return await search_pubmed_pmids(
        query=compact_query,
        max_results=max_results)

    return id_list


def _safe_text(element) -> str:
    if element is None:
        return ""

    return "".join(element.itertext()).strip()


def _parse_pubmed_xml(xml_text: str) -> list[PubMedArticle]:
    root = ET.fromstring(xml_text)
    articles = []

    for item in root.findall(".//PubmedArticle"):
        pmid = item.findtext(".//PMID") or ""
        title = _safe_text(item.find(".//ArticleTitle"))

        abstract_parts = [
            _safe_text(element)
            for element in item.findall(".//AbstractText")
        ]

        abstract = "\n".join(
            part for part in abstract_parts if part
        )

        if not abstract:
            abstract = "No abstract available."

        journal = item.findtext(".//Journal/Title")

        year = (
            item.findtext(".//PubDate/Year")
            or item.findtext(".//PubDate/MedlineDate")
        )

        articles.append(
            PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                year=year,
            )
        )

    return articles


async def fetch_pubmed_articles(
    pmids: list[str],
) -> list[PubMedArticle]:
    if not pmids:
        return []

    url = f"{URL_BASE}/efetch.fcgi"

    params = {
        **_base_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()

    await asyncio.sleep(0.12)

    return _parse_pubmed_xml(response.text)


async def fetch_pmc_full_text(
    pmid: str,
    client: httpx.AsyncClient,
) -> str | None:
    url = f"{PMC_BIOC_URL}/{pmid}/unicode"

    try:
        response = await client.get(url)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        collections = response.json()

    except (httpx.HTTPError, ValueError):
        return None

    parts = []

    for collection in collections:
        for document in collection.get("documents", []):
            for passage in document.get("passages", []):
                text = passage.get("text", "").strip()

                if text:
                    parts.append(text)

    return "\n\n".join(parts) or None


async def add_pmc_full_text(
    articles: list[PubMedArticle],
) -> list[PubMedArticle]:
    async with httpx.AsyncClient(timeout=30) as client:
        for article in articles:
            article.full_text = await fetch_pmc_full_text(
                article.pmid,
                client,
            )

            await asyncio.sleep(0.35)

    return articles


async def search_pubmed_articles(
    query: str,
    max_results: int = 5,
) -> list[PubMedArticle]:
    pmids = await search_pubmed_pmids(
        query=query,
        max_results=max_results,
    )

    articles = await fetch_pubmed_articles(pmids)

    return await add_pmc_full_text(articles)