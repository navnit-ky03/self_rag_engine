
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import (
    EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)


def init_retriever():
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


retriver = init_retriever()
