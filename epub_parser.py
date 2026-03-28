import re
from dataclasses import dataclass, field
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


@dataclass
class Chapter:
    title: str
    text: str
    chunks: list[str] = field(default_factory=list)


def parse_epub(file_path: str) -> list[Chapter]:
    book = epub.read_epub(file_path, options={"ignore_ncx": True})

    chapters: list[Chapter] = []

    # Get spine order (reading order)
    spine_ids = {item_id for item_id, _ in book.spine}

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        if item.get_id() not in spine_ids:
            continue

        soup = BeautifulSoup(item.get_content(), "lxml-xml")

        # Try to extract chapter title from headings
        title = ""
        for tag in ("h1", "h2", "h3"):
            heading = soup.find(tag)
            if heading:
                title = heading.get_text(strip=True)
                break

        if not title:
            title = item.get_name().replace("/", " ").replace(".xhtml", "").replace(".html", "").strip()

        # Remove nav, aside, script, style elements
        for tag in soup(["nav", "aside", "script", "style", "head"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = _clean_text(text)

        if len(text) < 50:
            continue

        chapter = Chapter(title=title, text=text)
        chapter.chunks = chunk_text(text)
        chapters.append(chapter)

    return chapters


def _clean_text(text: str) -> str:
    # Collapse multiple whitespace/newlines
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable chars except standard punctuation
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 480) -> list[str]:
    """Pack sentences into chunks up to max_chars.

    Break priority (never splits mid-word or mid-sentence):
      1. Sentence boundary  (.  !  ?)
      2. Clause boundary    (,  ;  :)
      3. Word boundary      (space)
    """
    if not text:
        return []

    chunks: list[str] = []
    current = ""

    for sentence in _iter_sentences(text):
        # Sentence fits alongside what we have — keep accumulating
        candidate = (current + " " + sentence) if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        # Flush current chunk before handling this sentence
        if current:
            chunks.append(current)
            current = ""

        # Sentence fits on its own
        if len(sentence) <= max_chars:
            current = sentence
            continue

        # Sentence exceeds limit — split at clause boundaries
        chunks.extend(_split_at_clauses(sentence, max_chars))

    if current:
        chunks.append(current)

    return chunks


def _iter_sentences(text: str):
    """Yield sentences split at .  !  ? followed by whitespace."""
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if s:
            yield s


def _split_at_clauses(text: str, max_chars: int) -> list[str]:
    """Split an oversized sentence at clause boundaries (,  ;  :)."""
    chunks: list[str] = []
    current = ""

    for clause in re.split(r"(?<=[,;:])\s+", text):
        clause = clause.strip()
        if not clause:
            continue

        candidate = (current + " " + clause) if current else clause
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(clause) <= max_chars:
            current = clause
        else:
            # Clause still too long — last resort: word boundaries
            chunks.extend(_split_at_words(clause, max_chars))

    if current:
        chunks.append(current)

    return chunks


def _split_at_words(text: str, max_chars: int) -> list[str]:
    """Split at word boundaries. A single word longer than max_chars is
    kept intact rather than cut mid-character."""
    chunks: list[str] = []
    current = ""

    for word in text.split():
        candidate = (current + " " + word) if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word  # keep oversized single word intact

    if current:
        chunks.append(current)

    return chunks


def total_chars(chapters: list[Chapter]) -> int:
    return sum(len(chunk) for ch in chapters for chunk in ch.chunks)
