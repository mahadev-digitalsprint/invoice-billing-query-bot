from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_documents

def split_documents():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    chunks = split_documents()
    print("\nSample chunk:")
    print(chunks[0].page_content)
    print("\nMetadata:", chunks[0].metadata)
