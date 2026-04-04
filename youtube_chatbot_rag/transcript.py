"""Transcript loading and formatting helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float
    video_id: str


def get_proxy_config_from_env() -> GenericProxyConfig | None:
    """Build proxy config from environment variables when available."""
    http_proxy = (
        os.getenv("YOUTUBE_TRANSCRIPT_PROXY_HTTP")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )
    https_proxy = (
        os.getenv("YOUTUBE_TRANSCRIPT_PROXY_HTTPS")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
    )

    if not http_proxy and not https_proxy:
        return None

    return GenericProxyConfig(http_url=http_proxy, https_url=https_proxy)


def fetch_transcript(video_id: str) -> list[TranscriptSegment]:
    """Fetch transcript segments for a YouTube video."""
    api = YouTubeTranscriptApi(proxy_config=get_proxy_config_from_env())
    transcript_data = api.fetch(video_id)
    return [
        TranscriptSegment(text=segment.text, start=segment.start, duration=segment.duration, video_id=video_id)
        for segment in transcript_data
    ]


def fetch_transcripts(video_ids: list[str]) -> tuple[dict[str, list[TranscriptSegment]], dict[str, str]]:
    """Fetch transcripts for multiple videos and return per-video failure details."""
    transcripts: dict[str, list[TranscriptSegment]] = {}
    failed_video_errors: dict[str, str] = {}

    for video_id in video_ids:
        try:
            transcripts[video_id] = fetch_transcript(video_id)
        except Exception as exc:
            summary = str(exc).strip().splitlines()[0] if str(exc).strip() else "Unknown error"
            failed_video_errors[video_id] = f"{exc.__class__.__name__}: {summary}"

    return transcripts, failed_video_errors


# def transcript_to_documents(segments: list[TranscriptSegment]) -> list[Document]:
#     """Convert transcript segments into LangChain documents with timestamp metadata."""
#     return [
#         Document(
#             page_content=segment.text,
#             metadata={"start": segment.start, "duration": segment.duration, "video_id": segment.video_id},
#         )
#         for segment in segments
#     ]


def transcripts_to_documents(transcripts):
    documents = []
    for video_id, segments in transcripts.items():
        paragraph_text = ""
        start_time = None
        
        for segment in segments:
            if start_time is None:
                start_time = segment.start
                
            paragraph_text += (segment.text + " ")
            
            # Create a paragraph if we hit punctuation after a reasonable length, or if it gets too long (fallback for missing punctuation)
            if (len(paragraph_text) > 500 and paragraph_text.strip().endswith((".", "?", "!"))) or len(paragraph_text) > 1000:
                duration = (segment.start + segment.duration) - start_time
                documents.append(
                    Document(
                        page_content=paragraph_text.strip(),
                        metadata={"video_id": video_id, "start": start_time, "duration": duration}
                    )
                )
                paragraph_text = ""
                start_time = None
                
        if paragraph_text.strip():
            duration = (segments[-1].start + segments[-1].duration) - start_time if start_time is not None else 0
            documents.append(
                Document(
                    page_content=paragraph_text.strip(),
                    metadata={"video_id": video_id, "start": start_time, "duration": duration}
                )
            )
    return documents


def transcript_preview(segments: list[TranscriptSegment], max_chars: int = 100) -> str:
    """Create a short preview string for debugging or display."""
    text = " ".join(segment.text for segment in segments)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")
