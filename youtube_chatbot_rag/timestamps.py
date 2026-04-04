"""Timestamp formatting helpers."""

from __future__ import annotations


def build_youtube_watch_url(video_id: str, start_seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s"


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def format_context_documents(documents) -> str:
    """Render retrieved documents with their source timestamps."""
    context_lines = []
    for document in documents:
        start = document.metadata.get("start", 0)
        duration = document.metadata.get("duration", 0)
        end = start + duration
        video_id = document.metadata.get("video_id")
        if video_id:
            source_label = f"{video_id} {format_timestamp(start)}-{format_timestamp(end)}"
            source_link = build_youtube_watch_url(video_id, start)
            source_prefix = f"[{source_label}]({source_link})"
        else:
            source_prefix = f"[source: {format_timestamp(start)}-{format_timestamp(end)}]"
        context_lines.append(
            f"{source_prefix} {document.page_content}"
        )
    return "\n\n".join(context_lines)
