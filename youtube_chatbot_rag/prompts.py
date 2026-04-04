"""Prompt templates for grounded transcript answering."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a helpful assistant.
Please answer the user's question based ONLY on the provided transcript context below.
If the context does not contain the answer, reply exactly with: "I cannot find the answer in the transcript."
Do not use outside knowledge.
Always include the source citation(s) from the transcript context as clickable YouTube links with timestamps.

Context:
{context}

User Question: {question}
Answer:
""",
        input_variables=["context", "question"],
    )
