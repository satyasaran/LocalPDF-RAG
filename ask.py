"""
ask.py
Interactive command-line RAG chat. Retrieves relevant chunks from the local
Chroma vector store and asks your local Ollama model (mistral) to answer
based on that retrieved context.

Run: python ask.py
Type 'exit' or 'quit' to stop.
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
TOP_K = 4  # number of chunks to retrieve per question

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the
context provided below, which comes from the user's own PDF documents.

If the answer is not contained in the context, say you don't know based on
the provided documents — do not make things up.

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectordb


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} | {source} | page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def main():
    print("Loading local vector store...")
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    print("Loading local LLM (Ollama: mistral)...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

    print("\nReady. Ask questions about your PDFs. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if not question:
            continue

        # 1. Retrieve relevant chunks
        docs = retriever.invoke(question)
        if not docs:
            print("Assistant: No relevant content found in your PDFs.\n")
            continue

        context = format_context(docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # 2. Ask the local LLM, grounded in retrieved context
        response = llm.invoke(prompt)

        print(f"\nAssistant: {response.content}\n")

        # Show sources used (helpful for trust/debugging)
        sources = {(d.metadata.get("source", "?"), d.metadata.get("page", "?")) for d in docs}
        src_str = ", ".join(f"{s[0]} (p.{s[1]})" for s in sources)
        print(f"[Sources: {src_str}]\n")


if __name__ == "__main__":
    main()
