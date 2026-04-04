"""Runnable chain assembly for the YouTube RAG chatbot."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from .config import DEFAULT_CHAT_MODEL, DEFAULT_TOP_K
from .prompts import build_prompt
from .timestamps import format_context_documents


def build_retriever(vector_store, top_k: int = 10):
    return vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": top_k, "fetch_k": top_k * 4}
    )


def build_llm(model: str = DEFAULT_CHAT_MODEL, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)

# 1. Define the Hugging Face Serverless endpoint
hf_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False,
)


def build_chain(vector_store, model: str = DEFAULT_CHAT_MODEL, top_k: int = 10):
    retriever = build_retriever(vector_store, top_k=top_k)
    # llm = build_llm(model=model)
    llm = ChatHuggingFace(llm = hf_llm)

    prompt = build_prompt()

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_context_documents),
            "question": RunnablePassthrough(),
        }
    )

    return parallel_chain | prompt | llm | StrOutputParser()
