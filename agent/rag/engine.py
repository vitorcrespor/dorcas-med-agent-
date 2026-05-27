from dotenv import load_dotenv
import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


load_dotenv()
DB_PATH = os.getenv("DB_PATH")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

_retriever= None

def get_retriever():
    global _retriever

    if _retriever is not None:
        return _retriever
    if DB_PATH is None:
        raise ValueError("DB_PATH is not defined in .env")

    llm= Ollama(model='llama3.2:3b')
    embed_model= HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    documents= SimpleDirectoryReader(input_dir= DB_PATH).load_data()
    chroma_client= chromadb.PersistentClient(path="./chroma.db")
    vector_store= ChromaVectorStore(chroma_client= chroma_client, 
                                    collection_name='dorcas_db')
    storage_context= StorageContext.from_defaults(vector_store= vector_store)
    Settings.embed_model= embed_model
    Settings.llm= llm
    
    for doc in documents:
        doc.text_template= "metadata: \n{metadata}\n----\ncontent:\n{content}"
        if "page_label" in doc.excluded_metadata_keys:
            doc.excluded_embed_metadata_keys.append("page_label")
    
    text_splitter = SentenceSplitter(
        separator=" ",
        chunk_size=1024,
        chunk_overlap=128,)
    
    title_extractor= TitleExtractor(llm=llm, 
                               nodes= 5)
    qa_extractor= QuestionsAnsweredExtractor(llm= llm, 
                                        questions=3)
    pipeline= IngestionPipeline(transformations=[
                            text_splitter,
                            title_extractor,
                            qa_extractor])

    nodes= pipeline.run(
        documents=documents,
        in_place=True,
        show_progress=True,)
    
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    _retriever = index.as_retriever(
    similarity_top_k=5
)

    return _retriever