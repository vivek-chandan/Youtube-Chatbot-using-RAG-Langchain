"""Helpers for resolving YouTube video IDs and playlists."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from collections.abc import Iterable


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
PLAYLIST_VIDEO_ID_PATTERN = re.compile(r"watch\?v=([A-Za-z0-9_-]{11})")
PLAYLIST_VIDEO_ID_JSON_PATTERN = re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"')


def is_video_id(value: str) -> bool:
    return bool(VIDEO_ID_PATTERN.match(value))


def extract_video_id(value: str) -> str | None:
    if is_video_id(value):
        return value

    parsed = urllib.parse.urlparse(value)

    # Support short YouTube links such as https://youtu.be/<video_id>?si=...
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            candidate = path_parts[0]
            if is_video_id(candidate):
                return candidate

    query_params = urllib.parse.parse_qs(parsed.query)
    if "v" in query_params and query_params["v"]:
        candidate = query_params["v"][0]
        if is_video_id(candidate):
            return candidate

    match = re.search(r"(?:v=|/shorts/)([A-Za-z0-9_-]{11})", value)
    if match:
        return match.group(1)

    return None


def fetch_playlist_video_ids(playlist_url: str, timeout: int = 30) -> list[str]:
    """Extract video IDs from a public YouTube playlist page."""
    request = urllib.request.Request(playlist_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="ignore")

    seen: set[str] = set()
    video_ids: list[str] = []
    for pattern in (PLAYLIST_VIDEO_ID_PATTERN, PLAYLIST_VIDEO_ID_JSON_PATTERN):
        for match in pattern.finditer(html):
            video_id = match.group(1)
            if video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)
    return video_ids


def is_playlist_source(source: str) -> bool:
    parsed = urllib.parse.urlparse(source)
    query_params = urllib.parse.parse_qs(parsed.query)
    return "list" in query_params or "playlist" in source


def resolve_sources(sources: str | Iterable[str]) -> list[str]:
    """Resolve a mix of video IDs, video URLs, and playlist URLs into video IDs."""
    if isinstance(sources, str):
        source_values = [sources]
    else:
        source_values = list(sources)

    resolved_video_ids: list[str] = []
    seen: set[str] = set()

    for source in source_values:
        if is_playlist_source(source):
            try:
                playlist_video_ids = fetch_playlist_video_ids(source)
            except Exception:
                playlist_video_ids = []

            if playlist_video_ids:
                for video_id in playlist_video_ids:
                    if video_id not in seen:
                        seen.add(video_id)
                        resolved_video_ids.append(video_id)
                continue

        normalized_video_id = extract_video_id(source)
        if normalized_video_id:
            if normalized_video_id not in seen:
                seen.add(normalized_video_id)
                resolved_video_ids.append(normalized_video_id)
            continue

    return resolved_video_ids
