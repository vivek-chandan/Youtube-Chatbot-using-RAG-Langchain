"""Convenience entrypoint for the modularized YouTube RAG chatbot."""

from __future__ import annotations

import argparse
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from youtube_chatbot_rag.app import run_demo, run_interactive_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the YouTube RAG chatbot over one or more videos or playlists.")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="YouTube video ID, video URL, or playlist URL. Repeat to add more sources.",
    )
    parser.add_argument(
        "--video-link",
        action="append",
        dest="video_links",
        help="Full YouTube video link. Repeat to add multiple videos.",
    )
    parser.add_argument(
        "--question",
        help="Question to ask over the combined transcript corpus.",
    )
    parser.add_argument(
        "--index-dir",
        default=".vectorstores/faiss",
        help="Directory where FAISS indexes are stored.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep chat running to ask multiple questions on the same loaded sources.",
    )
    return parser.parse_args()


def prompt_for_sources() -> list[str] | None:
    try:
        raw_value = input(
            "Paste YouTube video/playlist link(s), comma-separated (leave empty to use default): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not raw_value:
        return None

    sources = [value.strip() for value in raw_value.split(",") if value.strip()]
    return sources or None


def should_retry_after_error() -> bool:
    try:
        choice = input("Try another video/playlist link? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return choice in {"y", "yes"}


def main() -> None:
    args = parse_args()
    merged_sources: list[str] = []
    if args.sources:
        merged_sources.extend(args.sources)
    if args.video_links:
        merged_sources.extend(args.video_links)

    if not merged_sources:
        prompted_sources = prompt_for_sources()
        if prompted_sources:
            merged_sources.extend(prompted_sources)

    interactive_mode = args.interactive or args.question is None

    if interactive_mode:
        while True:
            try:
                run_interactive_chat(
                    sources=merged_sources or None,
                    index_dir=args.index_dir,
                )
                return
            except ValueError as exc:
                print(f"Error: {exc}")
                if not should_retry_after_error():
                    sys.exit(1)

                prompted_sources = prompt_for_sources()
                if prompted_sources:
                    merged_sources = prompted_sources
                else:
                    print("No new source provided. Exiting.")
                    sys.exit(1)

    try:
        run_demo(
            sources=merged_sources or None,
            question=args.question,
            index_dir=args.index_dir,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
