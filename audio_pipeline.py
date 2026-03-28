import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def concat_chunks_to_chapter(
    chunks: list[bytes], output_path: str, workdir: str, chunk_ext: str = "mp3"
) -> float:
    """Concat audio chunks and transcode to AAC. Returns duration in seconds."""
    chunk_files: list[str] = []

    # Write each chunk to a temp file
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(workdir, f"chunk_{i:04d}.{chunk_ext}")
        with open(chunk_path, "wb") as f:
            f.write(chunk)
        chunk_files.append(chunk_path)

    if not chunk_files:
        raise ValueError("No audio chunks to concat")

    # Build ffmpeg concat list file
    concat_list_path = os.path.join(workdir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for p in chunk_files:
            f.write(f"file '{p}'\n")

    # Concat and transcode to AAC 64kbps 24kHz mono
    _run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c:a", "aac",
        "-b:a", "64k",
        "-ar", "24000",
        "-ac", "1",
        "-y",
        output_path,
    ])

    # Get duration via ffprobe
    duration = _get_duration(output_path)
    return duration


def build_m4b(
    chapters: list[tuple[str, str, float]],  # (title, aac_path, duration_secs)
    output_path: str,
    workdir: str,
    book_title: str = "Audiobook",
) -> None:
    """Combine chapter AAC files into a single M4B with chapter markers."""
    if not chapters:
        raise ValueError("No chapters to combine")

    # Build concat list for final merge
    concat_list_path = os.path.join(workdir, "chapters_concat.txt")
    with open(concat_list_path, "w") as f:
        for _, aac_path, _ in chapters:
            f.write(f"file '{aac_path}'\n")

    # Concat all chapters into one AAC
    merged_aac = os.path.join(workdir, "merged.aac")
    _run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        "-y",
        merged_aac,
    ])

    # Build FFMETADATA file with chapter markers
    metadata_path = os.path.join(workdir, "metadata.txt")
    _write_ffmetadata(metadata_path, chapters, book_title)

    # Combine merged audio + metadata -> M4B
    _run_ffmpeg([
        "-i", merged_aac,
        "-i", metadata_path,
        "-map_metadata", "1",
        "-c", "copy",
        "-f", "ipod",
        "-movflags", "+faststart",
        "-y",
        output_path,
    ])


def _write_ffmetadata(
    path: str,
    chapters: list[tuple[str, str, float]],
    book_title: str,
) -> None:
    lines = [";FFMETADATA1\n", f"title={book_title}\n\n"]

    current_ms = 0
    for title, _, duration in chapters:
        start_ms = current_ms
        end_ms = current_ms + int(duration * 1000)
        safe_title = title.replace("=", r"\=").replace(";", r"\;").replace("#", r"\#").replace("\\", "\\\\")
        lines.append("[CHAPTER]\n")
        lines.append("TIMEBASE=1/1000\n")
        lines.append(f"START={start_ms}\n")
        lines.append(f"END={end_ms}\n")
        lines.append(f"title={safe_title}\n\n")
        current_ms = end_ms

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-loglevel", "error"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def _get_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])
