# Local PDF RAG (Ollama)

A simple command-line RAG (Retrieval-Augmented Generation) app that lets you
ask questions about your own PDF files — fully local, no API keys, no cost.

It uses:
- **[Ollama](https://ollama.com)** to run LLMs and embedding models locally
- **[Chroma](https://www.trychroma.com/)** as a local vector database
- **[LangChain](https://www.langchain.com/)** to glue retrieval + generation together

## How it works

1. `ingest.py` reads every PDF in `pdfs/`, splits the text into chunks,
   embeds each chunk locally, and stores the embeddings in a Chroma
   vector database (`chroma_db/`).
2. `ask.py` starts an interactive chat. For each question, it retrieves the
   most relevant chunks from `chroma_db/` and asks a local LLM to answer
   using only that context.

Everything runs on your machine — your PDFs and questions never leave it.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- macOS, Linux, or Windows (WSL recommended on Windows)

## Installation

### 1. Clone this repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the required Ollama models

```bash
ollama pull nomic-embed-text   # embedding model
ollama pull mistral            # generation model (or use any model you already have)
```

Check what you already have with:

```bash
ollama list
```

If you'd rather use a different model you already have locally (e.g.
`deepseek-r1:8b` or `gemma3:4b`), see [Configuration](#configuration) below.

## Usage

### 1. Add your PDFs

Place the PDF files you want to query into the `pdfs/` folder:

```bash
cp /path/to/your/file.pdf pdfs/
```

### 2. Build the vector index

Run this once, and again any time you add or change PDFs:

```bash
python3 ingest.py
```

This creates/updates a local `chroma_db/` folder containing the embeddings.

### 3. Ask questions

```bash
python3 ask.py
```

Type your questions at the `You:` prompt. Type `exit` or `quit` to stop.

Example session:

```
You: What is this document about?
Assistant: ...
[Sources: file.pdf (p.1), file.pdf (p.2)]
```

Each answer also shows which PDF page(s) it was drawn from.

## Configuration

Both scripts have a few constants near the top you can edit directly:

**`ingest.py`**
| Variable | Default | Description |
|---|---|---|
| `PDF_DIR` | `"pdfs"` | Folder to read PDFs from |
| `DB_DIR` | `"chroma_db"` | Where the vector store is saved |
| `EMBED_MODEL` | `"nomic-embed-text"` | Ollama embedding model |

**`ask.py`**
| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `"nomic-embed-text"` | Must match the model used in `ingest.py` |
| `LLM_MODEL` | `"mistral"` | Ollama chat model used to generate answers |
| `TOP_K` | `4` | Number of chunks retrieved per question |

To switch the generation model, change `LLM_MODEL` to any model from
`ollama list`, e.g. `"deepseek-r1:8b"` or `"gemma3:4b"`.

## Project structure

```
.
├── ingest.py          # Builds the vector store from PDFs
├── ask.py             # Interactive Q&A over the vector store
├── requirements.txt   # Python dependencies
├── pdfs/              # Put your PDF files here (gitignored)
├── chroma_db/          # Generated vector store (gitignored)
└── README.md
```

## Notes & limitations

- This app currently has **no conversation memory** — each question is
  answered independently, without awareness of previous questions in the
  session.
- Answers are grounded only in the retrieved PDF chunks; if the answer
  isn't in your PDFs, the model is instructed to say it doesn't know.
- Embeddings and generation both run through Ollama, so response speed
  depends on your machine's CPU/GPU.

## License

Add a license of your choice (e.g. MIT) if you plan to make this repo public.
