"""Prompt templates for grounded transcript answering."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.
When you answer, include the source citation(s) from the transcript context as clickable YouTube links with timestamps.
If multiple videos are relevant, include a citation for each video.

{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )
