import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import EMBEDDING_MODEL

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portfolio-knowledge-base")
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

def init_retriever():
    # Connect directly to the existing cloud vector store
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 4})

retriver = init_retriever()