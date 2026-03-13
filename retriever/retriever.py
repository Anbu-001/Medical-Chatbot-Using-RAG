from embeddings.embedding_model import get_embedding_model
from vectordb.chroma_store import get_vectorstore
from config import TOP_K


def get_retriever():
    embeddings = get_embedding_model()
    vs = get_vectorstore(embeddings)
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
