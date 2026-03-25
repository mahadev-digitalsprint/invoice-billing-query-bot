import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vectorstore import get_retriever

load_dotenv()

# ── Prompt Template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Invoice & Billing Query Assistant.
Answer the user's question using ONLY the context provided below.
Be concise and accurate. If the answer is not in the context, say:
"I don't have enough information to answer that from the billing data."

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

# ── Gemini LLM ───────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)

# ── Helper: format retrieved docs into a single string ───────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── RAG Chain (LCEL) ─────────────────────────────────────────────────────────
def build_rag_chain():
    retriever = get_retriever(k=50)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    chain = build_rag_chain()
    answer = chain.invoke("Which invoices are overdue?")
    print("Answer:", answer)
