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
    boilerplate: bool = False


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

        # Remove internal anchor links (footnote/endnote citations like <a href="#fn13">13</a>)
        for a in soup.find_all("a", href=True):
            if a["href"].startswith("#"):
                a.decompose()

        # Replace heading tags with their text as a sentence so the narrator
        # gets a natural pause before body text begins. Convert Roman numerals
        # within headings to spoken words.
        for htag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            heading_text = htag.get_text(separator=" ", strip=True)
            if heading_text:
                heading_text = re.sub(
                    r"\b(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\b",
                    lambda m: _roman_to_words(m.group(1)),
                    heading_text,
                )
                htag.replace_with(f"{heading_text}. ")
            else:
                htag.decompose()

        # Add a space after drop-cap spans so "IHAVE" → "I HAVE"
        for span in soup.find_all("span"):
            span_text = span.get_text()
            if len(span_text) == 1 and span_text.isupper():
                span.replace_with(span_text + " ")

        text = soup.get_text(separator=" ", strip=True)
        text = _clean_text(text)

        if len(text) < 100:
            continue

        is_bp = _is_boilerplate(text)
        chapter = Chapter(title=title, text=text, boilerplate=is_bp)
        chapter.chunks = chunk_text(text)
        chapters.append(chapter)

    return chapters


def _is_boilerplate(text: str) -> bool:
    """Return True for chapters that should be silently skipped: TOC, copyright,
    bibliography/endnotes, and other non-narrative pages."""
    sample = text[:400].lower()

    # Table of contents
    if "table of contents" in sample:
        return True

    # Copyright / publication info page
    copyright_signals = ["copyright ©", "all rights reserved", "published by", "isbn ", "library of congress"]
    if sum(1 for s in copyright_signals if s in sample) >= 2:
        return True

    # Bibliography / endnotes — short page that's mostly citations
    # Heuristic: high density of patterns like "p. 123" or "ibid." or "op. cit."
    citation_hits = len(re.findall(r"\bp\.\s*\d+|ibid\.|op\. cit\.|see [A-Z]", text))
    if citation_hits >= 3 and len(text) < 2000:
        return True

    return False


def _clean_text(text: str) -> str:
    # Remove citation markers: {13}, [13], superscript Unicode digits (¹²³ etc.)
    text = re.sub(r"\{\d+\}", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[\u00B2\u00B3\u00B9\u2070-\u2079]+", "", text)
    # Collapse multiple whitespace/newlines
    text = re.sub(r"\s+", " ", text)
    # Rejoin small-caps / drop-cap artifacts
    # "F I N A L" or "A T" → collapse all-caps sequences separated by single spaces
    text = re.sub(r"\b([A-Z](?: [A-Z])+)\b", lambda m: m.group(0).replace(" ", ""), text)
    # "F INALLY" → "FINALLY" (first letter separated from rest of uppercase word)
    text = re.sub(r"\b([A-Z]) ([A-Z]{2,})\b", r"\1\2", text)
    # Drop-cap concatenation: single one-letter word glued to next all-caps word
    # "IHAVE" → "I HAVE", "ACLERK" → "A CLERK"
    # Only split on "I" (pronoun) and "A" (article) — the only one-letter English words
    text = re.sub(r"\b(I|A)([A-Z]{2,})\b", r"\1 \2", text)
    # Convert Roman numeral section markers at sentence/paragraph boundaries
    # e.g. "III FINALLY" or ". II Some" — Roman numeral standing alone before prose
    text = re.sub(
        r"(?:^|(?<=\. )|(?<=\? )|(?<=! ))"
        r"(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{2,3}|V))\s+(?=[A-Z])",
        lambda m: _roman_to_words(m.group(1)) + ". ",
        text,
    )
    # Remove non-printable chars except standard punctuation
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


_ROMAN_MAP = [
    (1000, "one thousand"), (900, "nine hundred"), (500, "five hundred"),
    (400, "four hundred"), (100, "one hundred"), (90, "ninety"),
    (50, "fifty"), (40, "forty"), (10, "ten"), (9, "nine"),
    (8, "eight"), (7, "seven"), (6, "six"), (5, "five"),
    (4, "four"), (3, "three"), (2, "two"), (1, "one"),
]

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        total += v if v >= prev else -v
        prev = v
    return total


def _int_to_words(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + (" " + _int_to_words(rest) if rest else "")
    rest = n % 1000
    return _int_to_words(n // 1000) + " thousand" + (" " + _int_to_words(rest) if rest else "")


def _roman_to_words(s: str) -> str:
    """Convert a Roman numeral string to its spoken word form.
    Returns the original string unchanged if it doesn't parse as a valid
    non-zero Roman numeral, or if it's a single ambiguous letter (I, V, X)."""
    # Leave single ambiguous letters — "I" is usually a pronoun, "V"/"X" abbreviations
    if s.upper() in ("I", "V", "X"):
        return s
    n = _roman_to_int(s)
    if n == 0:
        return s
    return _int_to_words(n)


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
