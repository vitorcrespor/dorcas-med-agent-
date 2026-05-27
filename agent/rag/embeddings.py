from langchain_ollama import ChatOllama
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama
from IPython.display import Markdown, display
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
DB_PATH= os.getenv('DB_PATH')
EMBED_MODEL_NAME= os.getenv('EMBED_MODEL_NAME')

embed_model= HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

documents= SimpleDirectoryReader(input_dir= DB_PATH).load_data()

chroma_client= chromadb.Client()
vector_store= ChromaVectorStore(chroma_client= chroma_client, 
                                collection_name='dorcas_db')

storage_context= StorageContext.from_defaults(vector_store= vector_store)
service_context= ServiceContext.from_defaults(embed_model= embed_model)
