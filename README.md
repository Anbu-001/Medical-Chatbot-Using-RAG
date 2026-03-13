<<<<<<< HEAD
# Medical-Chatbot-Using-RAG
=======
# RAG Chatbot

A simple Retrieval-Augmented Generation chatbot using LangChain, ChromaDB, and a Hugging Face open-source model (FLAN-T5) for generation.

## Features
- Ingest PDFs/TXT/MD from `data/documents`
- Chunk, embed (Sentence Transformers), and store in ChromaDB
- Retrieve relevant chunks and generate answers
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
- Update model and paths in `config.py` if needed.
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Default generator model: `google/flan-t5-base` (downloads on first run).

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
└── README.md
```
>>>>>>> 30cc931 (Initial commit)
