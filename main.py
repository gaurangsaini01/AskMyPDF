import streamlit as st
from ingestion import ingest
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
import re
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import json
import os


# Load OpenAI API Key
load_dotenv()
client = OpenAI()

st.markdown(" <h1 style='color:#803df5;'>📑 AskMyPDF</h1> ", unsafe_allow_html=True)
st.markdown("<h3>PDF-based RAG Chatbot</h3>", unsafe_allow_html=True)


# Step 1: Upload PDF
uploaded = st.file_uploader(
    label="Upload your PDF", 
    type="pdf", 
    help="Upload your PDF to begin QnA.",
    accept_multiple_files=False
)
# Step 2: Ingest on upload
if uploaded is not None and "ingested" not in st.session_state:
    with st.spinner("📄 Loading and processing your PDF..."):
        clean_name = re.sub(r'\W+', '_', uploaded.name.split('.')[0])
        collection_name = f"{clean_name}_{uploaded.size}"
        
        # Store collection name in session
        st.session_state.collection_name = collection_name
        
        ingest(uploaded, collection_name)
        st.session_state.ingested = True
    st.success("✅ PDF processed! Now you can chat.")
    # st.write("Please reload before inserting a new PDF 😃")

# Step 3: Chat input
if st.session_state.get("ingested"):
    query = st.chat_input("Ask your question...")
    if query:
        st.chat_message("user").write(query)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        # Connect to Qdrant
        vector_db = QdrantVectorStore.from_existing_collection(
            collection_name=st.session_state.collection_name,
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        # Search top results
        results = vector_db.similarity_search(query=query)

        # Build context
        context = []
        for result in results:
            context.append({
                "page_content": result.page_content,
                "page_number": result.metadata.get('page_number')
            })

        stringified_context = json.dumps(context)

        SYSTEM_PROMPT = f"""
        You are an AI agent who answers user queries based only on the context below, including page numbers.

        Never use your own knowledge. Always answer based strictly on the context. After every answer, mention the page number like: "for more reference visit page number".

        Context:
        {stringified_context}
        """
        
        # Get response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {'role': "system", "content": SYSTEM_PROMPT},
                {'role': "user", "content": query}
            ]
        )

        # Display the assistant's response
        st.chat_message("assistant").write(response.choices[0].message.content)
