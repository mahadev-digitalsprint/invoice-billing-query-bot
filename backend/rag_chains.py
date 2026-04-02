import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from backend.rag_prompts import ANSWER_PROMPT, REWRITE_PROMPT

load_dotenv()


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Creates and caches the Gemini embedding model used for FAISS indexing."""

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Creates and caches the Gemini chat model used for extraction and answers."""

    return ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_rewriter_chain():
    """Builds the chain that rewrites follow-up questions into standalone questions."""

    return REWRITE_PROMPT | get_llm() | StrOutputParser()


@lru_cache(maxsize=1)
def get_answer_chain():
    """Builds the chain that turns retrieved context into a grounded answer."""

    return ANSWER_PROMPT | get_llm() | StrOutputParser()
