import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return embeddings

if __name__ == "__main__":
    embeddings = get_embeddings()
    test = embeddings.embed_query("What is the overdue invoice?")
    print(f"Embedding dimension: {len(test)}")
