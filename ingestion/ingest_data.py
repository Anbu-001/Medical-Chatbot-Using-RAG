from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.document_loader import load_documents
from embeddings.embedding_model import get_embedding_model
from vectordb.chroma_store import add_documents_to_store
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def ingest():
    raw_docs = load_documents(DATA_DIR)
    if not raw_docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(raw_docs)

    texts = [d.page_content for d in splits]
    metadatas = [d.metadata for d in splits]

    embeddings = get_embedding_model()
    add_documents_to_store(texts, metadatas, embeddings)
    return len(splits)
