from PyPDF2 import PdfReader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import io

load_dotenv()

#this was the main error causing code , QdrantVectorStore internally itself creates a QdrantClient , so instead of passing client again to it simply pass url and apikey directly to QdrantVectorStore. Same thing in main.py

# cloud_client = QdrantClient(
#     url=os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY")
# )

def ingest(uploaded,collection_name):
    pdf_bytes = uploaded.read()
    pdf_buffer = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_buffer)
    pages = reader.pages

    docs = []

    for i, page in enumerate(pages):
        raw_text = page.extract_text() or ""
        if raw_text.strip():
            docs.append(
                Document(
                    page_content=raw_text,
                    metadata={"page_number": i + 1}
                )
            )

    # Split each page's content into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )

    split_docs = splitter.split_documents(documents=docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        collection_name=collection_name,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )