import sys
import os
import numpy as np
import chromadb
import asyncio  # Add asyncio import here

# Fix for asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import Chroma

print("Model imported")


# Step 1: Upload and Load PDF
def load_pdf(file):
    """Load and extract text from a PDF file."""
    try:
        reader = PdfReader(file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        print(f"✅ Extracted text from PDF with {len(reader.pages)} pages.")
        return text
    except Exception as e:
        print(f"⚠️ Error reading PDF: {str(e)}")
        return ""

print("pdf load done")

# Upload your PDF file (Replace with the actual file path)
pdf_file_path = r"C:\RAHUL PAKHARE ROOT FOLDER\Buiding own AI Agent\GIT_AI_Agent\ai-document-chatbot\Rahul AI Clone.pdf"
pdf_text = load_pdf(pdf_file_path)

# Step 2: Chunk the Text
def chunk_text(text):
    """Split the extracted text into smaller chunks dynamically."""
    chunk_size = 600
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"✅ Text chunked into {len(chunks)} chunks.")
    return chunks

chunks = chunk_text(pdf_text) if pdf_text.strip() else []






# Step 3: Initialize ChromaDB and Store Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

def store_embeddings(chunks, collection, embedding_model):
    existing_docs = set(collection.get().get("documents", []))
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]
    
    if new_chunks:
        embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
        collection.add(
            ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
            documents=new_chunks,
            embeddings=embeddings
        )
        print("✅ New embeddings stored in ChromaDB!")
    else:
        print("✅ No new chunks to store.")

store_embeddings(chunks, collection, embedding_model)

vectorstore = Chroma(persist_directory="db", embedding_function=embedding_model)
print(f"✅ Total documents in ChromaDB: {len(vectorstore.get())}")
print("alldone")

