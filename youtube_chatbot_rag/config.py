"""Central configuration for the YouTube RAG chatbot."""

from __future__ import annotations

DEFAULT_VIDEO_SOURCE = "https://www.youtube.com/watch?v=y1fGlAECNFM&list=PLXV9Vh2jYcjbnv67sXNDJiO8MWLA3ZJKR"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP =400
DEFAULT_TOP_K = 8
DEFAULT_INDEX_DIR = ".vectorstores/faiss"
