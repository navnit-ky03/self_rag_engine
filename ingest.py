import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portfolio-knowledge-base")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "Navnit_Kumar_Yadav_Master_Knowledge_Base.pdf")

def clear_pinecone_index(index_name):
    """Deletes all existing vectors in the Pinecone index to avoid duplicates."""
    pc = Pinecone()
    index = pc.Index(index_name)
    print(f"Clearing existing vectors in Pinecone index '{index_name}'...")
    index.delete(delete_all=True)
    print("Index cleared successfully.")

def parse_document_dynamically(docs, max_chunk_size=1200):
    full_text = "\n".join([doc.page_content for doc in docs])
    pattern = r"(?:^|\n)(\d+)\.\s+([^\n]+)"
    matches = list(re.finditer(pattern, full_text))
    chunks = []

    if not matches:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        return [{"text": doc.page_content, "metadata": {"section": "general"}} for doc in splitter.split_documents(docs)]

    # Profile Section
    first_match = matches[0]
    profile_text = full_text[:first_match.start()].strip()
    if profile_text:
        chunks.append({"text": profile_text, "metadata": {"section": "profile"}})

    # Dynamic Project Extraction
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    for i in range(len(matches)):
        current_match = matches[i]
        start_idx = current_match.start()
        end_idx = matches[i + 1].start() if (i + 1 < len(matches)) else len(full_text)
        project_raw_text = full_text[start_idx:end_idx].strip()
        
        raw_header = current_match.group(2).strip()
        clean_title = re.split(r'[:\(\-\|]', raw_header)[0].strip()
        proj_number = int(current_match.group(1))

        metadata = {
            "section": "project",
            "project": clean_title,
            "project_number": proj_number
        }

        if len(project_raw_text) <= max_chunk_size:
            chunks.append({"text": project_raw_text, "metadata": metadata})
        else:
            sub_texts = splitter.split_text(project_raw_text)
            for sub_t in sub_texts:
                chunks.append({"text": sub_t, "metadata": metadata.copy()})

    return chunks

def upload_knowledge_base():
    # 1. Wipe existing vectors first to prevent duplicates
    clear_pinecone_index(INDEX_NAME)

    print(f"Loading document from '{DATA_FILE_PATH}'...")
    loader = PyPDFLoader(DATA_FILE_PATH)
    docs = loader.load()

    print("Parsing document dynamically...")
    chunks = parse_document_dynamically(docs)

    print("\nInitializing Google Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print(f"Upserting fresh vectors to Pinecone index '{INDEX_NAME}'...")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    vectorstore.add_texts(
        texts=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

    print("\nSuccessfully replaced Pinecone vector store with fresh data!")

if __name__ == "__main__":
    upload_knowledge_base()