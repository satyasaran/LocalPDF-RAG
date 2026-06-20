"""
ingest.py
Loads all PDFs from ./pdfs, splits them into chunks, embeds them locally
using Ollama's nomic-embed-text model, and stores them in a local
Chroma vector database (./chroma_db).

Run this once whenever you add/change PDFs.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PDF_DIR = "pdfs"
DB_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"


def load_pdfs(pdf_dir: str):
    """Load every PDF in pdf_dir and return a list of Document objects."""
    all_docs = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in '{pdf_dir}/'. Add some PDFs and re-run.")
        return all_docs

    for fname in pdf_files:
        path = os.path.join(pdf_dir, fname)
        print(f"Loading: {fname}")
        loader = PyPDFLoader(path)
        docs = loader.load()  # one Document per page
        all_docs.extend(docs)

    print(f"Loaded {len(pdf_files)} PDF(s), {len(all_docs)} page(s) total.")
    return all_docs


def chunk_documents(docs):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks locally and persist them to a Chroma DB on disk."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print("Embedding chunks (this calls your local Ollama model)...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"Vector store saved to ./{DB_DIR}")
    return vectordb


def main():
    if not os.path.isdir(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"Created '{PDF_DIR}/' folder. Put your PDFs there and re-run this script.")
        return

    docs = load_pdfs(PDF_DIR)
    if not docs:
        return

    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("\nDone. You can now run: python ask.py")


if __name__ == "__main__":
    main()
