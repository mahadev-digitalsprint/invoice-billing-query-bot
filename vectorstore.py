import os
from langchain_community.vectorstores import FAISS
from embeddings import get_embeddings
from splitter import split_documents

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

def build_vectorstore():
    chunks = split_documents()
    embeddings = get_embeddings()

    print("Building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so we don't re-embed every run
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"Vector store saved to: {FAISS_INDEX_PATH}")
    return vector_store

def load_vectorstore():
    embeddings = get_embeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    else:
        print("No existing index found. Building new one...")
        return build_vectorstore()

def get_retriever(k=50):
    vector_store = load_vectorstore()
    return vector_store.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    vs = build_vectorstore()
    results = vs.similarity_search("Which invoices are overdue?", k=10)
    print("\nTop matching chunks:")
    for i, r in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(r.page_content[:300])
