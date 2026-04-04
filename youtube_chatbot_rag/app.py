"""End-to-end orchestration for the YouTube RAG chatbot demo."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

from .chain import build_chain
from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_VIDEO_SOURCE,
)
from .sources import resolve_sources
from .transcript import fetch_transcripts, transcript_preview, transcripts_to_documents
from .vectorstore import (
    build_corpus_key,
    build_vector_store,
    save_corpus_manifest,
    load_vector_store,
    save_vector_store,
    split_documents,
    vector_store_exists,
)


@dataclass
class QASession:
    chain: object
    video_ids: list[str]


def get_vector_store(corpus_key: str, chunks, index_dir: str = DEFAULT_INDEX_DIR):
    if vector_store_exists(corpus_key, index_dir):
        print(f"Loading FAISS index from {index_dir}/{corpus_key}")
        return load_vector_store(corpus_key, index_dir)

    print(f"Building FAISS index and saving to {index_dir}/{corpus_key}")
    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store, corpus_key, index_dir)
    return vector_store


def normalize_source_inputs(sources: str | Iterable[str] | None) -> list[str]:
    if sources is None:
        return [DEFAULT_VIDEO_SOURCE]
    if isinstance(sources, str):
        return [sources]
    return list(sources)


def build_session(sources: str | Iterable[str] | None = None, index_dir: str = DEFAULT_INDEX_DIR) -> QASession:
    source_inputs = normalize_source_inputs(sources)
    video_ids = resolve_sources(source_inputs)
    if not video_ids:
        raise ValueError("No valid video IDs could be resolved from the provided sources.")

    transcripts, failed_video_errors = fetch_transcripts(video_ids)
    if failed_video_errors:
        print("Skipped videos without transcripts:")
        for failed_video_id, error_detail in failed_video_errors.items():
            print(f"  - {failed_video_id}: {error_detail}")

    if not transcripts:
        raise ValueError(
            "No transcripts could be loaded from the provided sources. "
            "If you are running from a blocked IP, set YOUTUBE_TRANSCRIPT_PROXY_HTTP and/or "
            "YOUTUBE_TRANSCRIPT_PROXY_HTTPS (or HTTP_PROXY/HTTPS_PROXY) and retry."
        )

    loaded_video_ids = [video_id for video_id in video_ids if video_id in transcripts]

    documents = transcripts_to_documents(transcripts)
    chunks = split_documents(documents)
    corpus_key = build_corpus_key(
        loaded_video_ids,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    vector_store = get_vector_store(corpus_key, chunks, index_dir=index_dir)
    save_corpus_manifest(
        corpus_key,
        index_dir,
        loaded_video_ids,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    )
    first_video_id = loaded_video_ids[0]
    first_segments = transcripts[first_video_id]
    print(json.dumps([segment.__dict__ for segment in first_segments[:2]], indent=2))
    print("\nTranscript preview:", transcript_preview(first_segments))
    print("\nResolved sources:", ", ".join(loaded_video_ids))

    chain = build_chain(vector_store)
    return QASession(chain=chain, video_ids=loaded_video_ids)


def ask_question(session: QASession, question: str) -> str:
    answer = session.chain.invoke(question)
    return answer


def run_demo(sources: str | Iterable[str] | None = None, question: str | None = None, index_dir: str = DEFAULT_INDEX_DIR):
    if question is None:
        question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"

    session = build_session(sources=sources, index_dir=index_dir)
    answer = ask_question(session, question)
    print("\nAnswer:\n", answer)


def run_interactive_chat(sources: str | Iterable[str] | None = None, index_dir: str = DEFAULT_INDEX_DIR):
    session = build_session(sources=sources, index_dir=index_dir)
    print("\nChat ready. Ask questions about the loaded source(s). Type 'exit' to quit.")

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        answer = ask_question(session, question)
        print("\nAnswer:\n", answer)


def main():
    os.environ.setdefault("OPENAI_API_KEY", "Paste API HERE")
    run_demo()


if __name__ == "__main__":
    main()
