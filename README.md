# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for medical question answering, using LangChain, ChromaDB, and the Llama-3.1-8b-instant model via Groq for answer generation.

## Features
- Ingest PDFs/TXT/MD from `data/documents`
- Chunk, embed (Sentence Transformers), and store in ChromaDB
- Retrieve relevant chunks and generate answers using Llama-3.1-8b-instant (Groq)
- Strictly grounded medical answers (no outside knowledge)
- CLI chat interface

## Quickstart

1. Install dependencies
```
pip install -r requirements.txt
```

2. Ingest data (first time or when docs change)
```
python main.py --ingest
```

3. Run the chatbot
```
python main.py
```

## Configuration
- Update model and paths in `config.py` and `.env` if needed.
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Default generator model: `Llama-3.1-8b-instant` via Groq (set in `.env` as `GROQ_MODEL_NAME`).

## Project Structure
```
rag-chatbot/
├── data/
│   └── documents/
├── embeddings/
│   └── embedding_model.py
├── vectordb/
│   └── chroma_store.py
├── ingestion/
│   └── ingest_data.py
├── retriever/
│   └── retriever.py
├── chains/
│   └── rag_chain.py
├── chatbot/
│   └── chat_interface.py
├── utils/
│   └── document_loader.py
├── config.py
├── requirements.txt
├── main.py
├── .env
└── README.md
```
