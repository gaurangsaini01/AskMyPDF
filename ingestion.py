from PyPDF2 import PdfReader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
load_dotenv()
def ingest(uploaded):
    reader = PdfReader(uploaded)
    # print(reader.pages[10].extract_text())
    full_text = ""
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
        url="http://localhost:6333",
        collection_name="self_prac",
        embedding=embeddings
    )
    print("INGESTION DONE...")