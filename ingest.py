
import hashlib
import os
import argparse

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import (
    DATA_FILE_PATH,
    EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)

load_dotenv()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    add_start_index=True,
)


def stable_id(text: str, metadata: dict) -> str:
    """Generate a repeatable ID for identical content and metadata."""
    identity = f"{metadata}|{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def parse_document(docs):
    """Create searchable chunks while preserving document context."""
    chunks = []

    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        # Preserve the current document safely until the explicit,
        # section-aware parser is added for the new knowledge-base format.
        for chunk in splitter.split_text(text):
            if not chunk.strip():
                continue

            metadata = {
                "section": "general",
                "source": os.path.basename(DATA_FILE_PATH),
                "page": int(doc.metadata.get("page", 0)),
            }

            chunks.append({
                "text": chunk,
                "metadata": metadata,
            })

    if not chunks:
        raise ValueError(
            "No searchable text was extracted from the PDF."
        )

    return chunks


def upload_knowledge_base(replace_namespace=False, confirm_namespace=None):
    # Validate the input before making any changes to the vector store.
    if not os.path.isfile(DATA_FILE_PATH):
        raise FileNotFoundError(
            f"Knowledge-base PDF not found: {DATA_FILE_PATH}"
        )

    print(f"Loading PDF: {DATA_FILE_PATH}")
    docs = PyPDFLoader(DATA_FILE_PATH).load()

    if not docs or not any(d.page_content.strip() for d in docs):
        raise ValueError("The PDF contains no extractable text.")

    chunks = parse_document(docs)
    texts = [item["text"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]
    ids = [
        stable_id(item["text"], item["metadata"])
        for item in chunks
    ]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs detected.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    print(
        f"Uploading {len(chunks)} chunks to "
        f"{PINECONE_INDEX_NAME}/{PINECONE_NAMESPACE}"
    )

    store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    if replace_namespace:
        if confirm_namespace != PINECONE_NAMESPACE:
            raise ValueError("Namespace confirmation must exactly match configured namespace")
        if not PINECONE_NAMESPACE or PINECONE_NAMESPACE.lower() in ("default", "__default__"):
            raise ValueError("Refusing to clear blank or default namespace")
        print(f"WARNING: CLEARING ALL VECTORS IN NAMESPACE {PINECONE_NAMESPACE!r} OF INDEX {PINECONE_INDEX_NAME!r}!")
        # Only the selected namespace is cleared, never the whole index.
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        pc.Index(PINECONE_INDEX_NAME).delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print("Cleared selected namespace. Uploading replacement chunks now...")

    store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )

    print("Upload completed in configured namespace.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load the current PDF into the configured Pinecone namespace.")
    parser.add_argument("--replace-namespace", action="store_true", help="DESTRUCTIVE: delete all vectors in target namespace before uploading")
    parser.add_argument("--confirm-namespace", help="Exact target namespace; required for --replace-namespace")
    args = parser.parse_args()
    upload_knowledge_base(args.replace_namespace, args.confirm_namespace)
