# rag/pubmed.py
from dataclasses import dataclass
from typing import Any
import os
import time
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import asyncio

load_dotenv()
EUTILS_BASE= os.getenv("NCBI_URL")


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str | None= None
    year: str | None= None
    metadata: dict[str, Any] | None= None


def _base_params() -> dict[str, str]:
    email= os.getenv("NCBI_EMAIL")
    tool= os.getenv("NCBI_TOOL", "dorcas_agent")
    api_key= os.getenv("NCBI_API_KEY", "")
    if not email:
        raise ValueError("NCBI_EMAIL is not defined in .env")

    params = {"tool": tool, 
              "email": email}
    if api_key: params["api_key"] = api_key
    return params

async def search_pubmed_pmids(query: str, max_results: int = 5) -> list[str]:
    url = f"{EUTILS_BASE}/esearch.fcgi"

    params = {**_base_params(),
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": str(max_results),
                "sort": "relevance"}

    response= requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    await asyncio.sleep(0.12)

    data= response.json()
    return data["esearchresult"].get("idlist", [])


def fetch_pubmed_articles(pmids: list[str]) -> list[PubMedArticle]:
    if not pmids:
        return []

    url = f"{EUTILS_BASE}/efetch.fcgi"

    params = {
        **_base_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    _rate_limit_sleep()

    return _parse_pubmed_xml(response.text)


def search_pubmed_articles(query: str, max_results: int = 5) -> list[PubMedArticle]:
    pmids= search_pubmed_pmids(query=query, max_results=max_results)
    return fetch_pubmed_articles(pmids)


def _safe_text(element) -> str:
    if element is None:
        return ""

    return "".join(element.itertext()).strip()


def _parse_pubmed_xml(xml_text: str) -> list[PubMedArticle]:
    root = ET.fromstring(xml_text)

    articles: list[PubMedArticle] = []

    for item in root.findall(".//PubmedArticle"):
        pmid = item.findtext(".//PMID") or ""

        title_element = item.find(".//ArticleTitle")
        title = _safe_text(title_element)

        abstract_parts = [
            _safe_text(abstract_text)
            for abstract_text in item.findall(".//AbstractText")
        ]

        abstract = "\n".join(part for part in abstract_parts if part)

        journal = item.findtext(".//Journal/Title")
        year = (
            item.findtext(".//PubDate/Year")
            or item.findtext(".//PubDate/MedlineDate")
        )

        if not abstract:
            continue

        metadata = {
            "source": "pubmed",
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "year": year,
        }

        articles.append(
            PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                year=year,
                metadata=metadata,
            )
        )

    return articles