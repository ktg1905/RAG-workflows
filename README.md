# Basic RAG Pipeline using LangChain, ChromaDB & Groq

A simple Retrieval-Augmented Generation (RAG) pipeline built using **LangChain**, **Hugging Face Embeddings**, **ChromaDB**, and **Groq LLM**. The project ingests text documents, creates vector embeddings, stores them in a Chroma vector database, and answers user queries using retrieved context.

---

## Features

- Document ingestion from local `.txt` files
- Recursive text chunking
- Hugging Face embeddings (`BAAI/bge-small-en-v1.5`)
- ChromaDB vector store
- Semantic similarity search
- Groq-powered LLM for response generation
- Environment variable support using `.env`

---

## Tech Stack

- Python
- LangChain
- Hugging Face Embeddings
- ChromaDB
- Groq
- python-dotenv

---

## Project Structure

```text
RAG/
│
├── docs/                    # Input documents
├── db/                      # Chroma vector database
├── ingestion_pipeline.py    # Creates vector database
├── retrieval_pipeline.py    # Retrieves relevant chunks and generates answers
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your_username>/<repo_name>.git
cd <repo_name>
```

Create a virtual environment

Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```text
GROQ_API_KEY=your_groq_api_key
```

---

## Ingest Documents

Place your `.txt` documents inside the `docs/` folder.

Run

```bash
python ingestion_pipeline.py
```

This will

- Load documents
- Split them into chunks
- Generate embeddings
- Store them inside ChromaDB

---

## Query the Vector Database

Run

```bash
python retrieval_pipeline.py
```

Enter a question related to your uploaded documents.

The pipeline

- Retrieves relevant chunks
- Passes them to the Groq LLM
- Generates an answer grounded in the retrieved context

---

## Workflow

```text
Documents
     │
     ▼
Document Loader
     │
     ▼
Text Splitter
     │
     ▼
Hugging Face Embeddings
     │
     ▼
Chroma Vector Database
     │
User Query
     │
     ▼
Similarity Search
     │
     ▼
Relevant Chunks
     │
     ▼
Groq LLM
     │
     ▼
Final Answer
```

---

