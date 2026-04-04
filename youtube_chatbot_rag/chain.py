"""Runnable chain assembly for the YouTube RAG chatbot."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import DEFAULT_CHAT_MODEL, DEFAULT_TOP_K
from .prompts import build_prompt
from .timestamps import format_context_documents


def build_retriever(vector_store, top_k: int = DEFAULT_TOP_K):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def build_llm(model: str = DEFAULT_CHAT_MODEL, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def build_chain(vector_store, model: str = DEFAULT_CHAT_MODEL, top_k: int = DEFAULT_TOP_K):
    retriever = build_retriever(vector_store, top_k=top_k)
    llm = build_llm(model=model)
    prompt = build_prompt()

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_context_documents),
            "question": RunnablePassthrough(),
        }
    )

    return parallel_chain | prompt | llm | StrOutputParser()
