from langchain_community.document_loaders import PyPDFLoader
import os

PDF_PATH = os.path.join(os.path.dirname(__file__), "Invoice And Billing_RAG.pdf")

def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF.")
    return docs

if __name__ == "__main__":
    docs = load_documents()
    print(docs[0].page_content[:300])
