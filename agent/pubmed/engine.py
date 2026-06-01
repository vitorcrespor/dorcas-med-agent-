# rag/pubmed.py
from dotenv import load_dotenv
import os

import chromadb
from llama_index.core import (
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


load_dotenv()

DB_PATH = os.getenv("DB_PATH")
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

_retriever = None
_embed_model = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Load the embedding model once and reuse it."""
    global _embed_model

    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME
        )

    return _embed_model


def retrieve_from_documents(
    query: str,
    documents: list[Document],
    k: int = 5,
):
    """
    Build a temporary in-memory index and retrieve relevant chunks.

    Useful for short-lived sources such as PubMed/PMC articles.
    It does not modify the persistent local-document Chroma database.
    """
    splitter = SentenceSplitter(
        separator=" ",
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = splitter.get_nodes_from_documents(documents)

    if not chunks:
        return []

    index = VectorStoreIndex(
        chunks,
        embed_model=get_embed_model(),
    )

    retriever = index.as_retriever(
        similarity_top_k=k
    )

    return retriever.retrieve(query)


def get_retriever():
    """
    Return the persistent retriever for local uploaded documents.
    Build it once per process, then reuse it.
    """
    global _retriever

    if _retriever is not None:
        return _retriever

    if DB_PATH is None:
        raise ValueError("DB_PATH is not defined in .env")

    llm = Ollama(
        model="llama3.2:3b"
    )

    embed_model = get_embed_model()

    documents = SimpleDirectoryReader(
        input_dir=DB_PATH
    ).load_data()

    chroma_client = chromadb.PersistentClient(
        path="./chroma.db"
    )

    vector_store = ChromaVectorStore(
        chroma_client=chroma_client,
        collection_name="dorcas_db",
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    for document in documents:
        document.text_template = (
            "metadata:\n"
            "{metadata}\n"
            "----\n"
            "content:\n"
            "{content}"
        )

        if "page_label" in document.excluded_metadata_keys:
            document.excluded_embed_metadata_keys.append(
                "page_label"
            )

    text_splitter = SentenceSplitter(
        separator=" ",
        chunk_size=1024,
        chunk_overlap=128,
    )

    title_extractor = TitleExtractor(
        llm=llm,
        nodes=5,
    )

    qa_extractor = QuestionsAnsweredExtractor(
        llm=llm,
        questions=3,
    )

    pipeline = IngestionPipeline(
        transformations=[
            text_splitter,
            title_extractor,
            qa_extractor,
        ]
    )

    nodes = pipeline.run(
        documents=documents,
        in_place=True,
        show_progress=True,
    )

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    _retriever = index.as_retriever(
        similarity_top_k=5
    )

    return _retriever