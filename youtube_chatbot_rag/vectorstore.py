"""Chunking, embedding, and vector store helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_EMBEDDING_MODEL


def split_documents(documents, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def build_vector_store(documents, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.from_documents(documents, embeddings)


def build_corpus_key(
    video_ids: list[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> str:
    normalized_video_ids = sorted(video_ids)
    payload = json.dumps(
        {
            "video_ids": normalized_video_ids,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def get_vector_store_path(corpus_key: str, index_dir: str) -> Path:
    return Path(index_dir) / corpus_key


def vector_store_exists(corpus_key: str, index_dir: str) -> bool:
    store_path = get_vector_store_path(corpus_key, index_dir)
    return (store_path / "index.faiss").exists() and (store_path / "index.pkl").exists()


def save_vector_store(vector_store: FAISS, corpus_key: str, index_dir: str) -> Path:
    store_path = get_vector_store_path(corpus_key, index_dir)
    store_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_path))
    return store_path


def load_vector_store(corpus_key: str, index_dir: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> FAISS:
    store_path = get_vector_store_path(corpus_key, index_dir)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def corpus_manifest_path(corpus_key: str, index_dir: str) -> Path:
    return get_vector_store_path(corpus_key, index_dir) / "manifest.json"


def save_corpus_manifest(
    corpus_key: str,
    index_dir: str,
    video_ids: list[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Path:
    manifest_path = corpus_manifest_path(corpus_key, index_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "video_ids": video_ids,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embedding_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path
