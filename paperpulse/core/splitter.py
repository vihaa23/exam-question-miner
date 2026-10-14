"""
splitter.py — Convert raw page text into individual question records.

Design decisions
----------------
Sub-parts policy
    A main question ("Q3.") followed by sub-parts ("(a)", "(b)") is kept as
    a single question record with the sub-parts concatenated.  Rationale: the
    topic similarity of a question is determined by its full content; splitting
    sub-parts would create too-short fragments that embed poorly.  The
    `question_text` field therefore contains the parent stem + all sub-part
    text joined by newlines.

Boilerplate filtering
    Lines that match common exam boilerplate patterns (page numbers, headers,
    "Attempt any three", "All questions carry equal marks", etc.) are stripped
    before splitting.  This prevents false question boundaries.

Marks extraction
    Marks appear in many forms:  [5], (5 marks), (5M), [5 Marks], 5M.
    The extractor tries each pattern and returns the *last* match on the
    question line (most reliable position).  Unknown marks -> None.
    None is NOT treated as zero in downstream scoring.

Output DataFrame columns
    question_text  : str   — cleaned, joined question text
    year           : int|None
    marks          : float|None
    source_file    : str
    question_no    : str   — raw label found in text, e.g. "Q3" or "1"
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from paperpulse.core.extractor import ExtractionResult, PageRecord

# ─────────────────────────────────────────────────────────────────
# Boilerplate patterns — lines matching any of these are removed
# ─────────────────────────────────────────────────────────────────

_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    # Page numbers (standalone digits or "Page N of M")
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.I),
    # Instruction lines
    re.compile(r"attempt\s+(any|all)\s+\w+", re.I),
    re.compile(r"all\s+questions\s+(carry|are)\s+", re.I),
    re.compile(r"time\s*:\s*\d+\s*h(ours?)?", re.I),
    re.compile(r"maximum\s+marks?\s*:\s*\d+", re.I),
    re.compile(r"total\s+marks?\s*:\s*\d+", re.I),
    re.compile(r"note\s*:\s*", re.I),
    re.compile(r"answer\s+(all|any)\s+\w+\s+questions?", re.I),
    re.compile(r"use\s+separate\s+answer", re.I),
    re.compile(r"this\s+question\s+paper\s+contains", re.I),
    # University / course header fragments (generic)
    re.compile(r"(university|college|institute|department)\s+of", re.I),
    re.compile(r"(end\s*[\-–]\s*semester|mid\s*[\-–]?\s*term|annual)\s+exam", re.I),
    re.compile(r"roll\s+no", re.I),
    re.compile(r"exam(ination)?\s+date", re.I),
    re.compile(r"course\s+(code|name)\s*:", re.I),
]

# ─────────────────────────────────────────────────────────────────
# Question boundary patterns  (trigger a new question)
# ─────────────────────────────────────────────────────────────────
# Ordered from most specific to least specific.
# Groups:
#   main_q   — top-level question  (Q.1, Q1., Q1 , 1. , 1) )
#   sub_q    — sub-part            ((a), (i), a) , i) )

_MAIN_Q_RE = re.compile(
    r"^"
    r"(?:"
    r"Q\.?\s*(\d{1,2})"          # Q1 / Q.1 / Q 1
    r"|(\d{1,2})\s*[.)\-:]"      # 1. / 1) / 1-
    r")"
    r"\s+",
    re.IGNORECASE,
)

_SUB_Q_RE = re.compile(
    r"^"
    r"(?:"
    r"\(([a-z]|[ivx]+)\)"        # (a) / (ii)
    r"|([a-z]|[ivx]+)\)"         # a) / ii)
    r")\s+",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────
# Marks extraction patterns
# ─────────────────────────────────────────────────────────────────

_MARKS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\[(\d+(?:\.\d+)?)\s*[Mm]arks?\]"),   # [5 marks]
    re.compile(r"\((\d+(?:\.\d+)?)\s*[Mm]arks?\)"),   # (5 marks)
    re.compile(r"\[(\d+(?:\.\d+)?)\]"),                # [5]
    re.compile(r"\((\d+(?:\.\d+)?)\)(?!\s*\w)"),       # (5) not followed by text (avoids question labels)
    re.compile(r"\b(\d+(?:\.\d+)?)\s*M\b"),            # 5M / 10M
]


def _extract_marks(text: str) -> Optional[float]:
    """Return the numeric mark value from a question line, or None."""
    for pat in _MARKS_PATTERNS:
        matches = pat.findall(text)
        if matches:
            # Return the last match (closest to end of line, most reliable)
            return float(matches[-1])
    return None


# ─────────────────────────────────────────────────────────────────
# Text cleaning helpers
# ─────────────────────────────────────────────────────────────────

def _is_boilerplate(line: str) -> bool:
    """Return True if the line matches any boilerplate pattern."""
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(line.strip()):
            return True
    return False


def _clean_lines(text: str) -> list[str]:
    """Split text into lines and remove boilerplate and empty lines."""
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_boilerplate(stripped):
            continue
        cleaned.append(stripped)
    return cleaned


# ─────────────────────────────────────────────────────────────────
# Core splitter
# ─────────────────────────────────────────────────────────────────

def _label_from_match(m: re.Match[str]) -> str:
    """Return a clean question label string from a main-question regex match."""
    g1, g2 = m.group(1), m.group(2)
    if g1:
        return f"Q{g1}"
    if g2:
        return g2
    return "Q?"


def split_page(page: PageRecord) -> list[dict]:
    """Split a single page's text into question records.

    Returns a list of dicts (later assembled into a DataFrame row).
    Each dict has keys: question_text, year, marks, source_file, question_no.
    """
    lines = _clean_lines(page.text)
    if not lines:
        return []

    records: list[dict] = []

    # State machine: accumulate lines per question
    current_label: Optional[str] = None
    current_lines: list[str] = []
    in_sub_part = False

    def _flush() -> None:
        """Save the current accumulated question to records."""
        nonlocal current_label, current_lines, in_sub_part
        if current_label is None or not current_lines:
            return
        full_text = " ".join(current_lines)
        marks = _extract_marks(full_text)
        records.append(
            {
                "question_text": full_text.strip(),
                "year": page.year,
                "marks": marks,
                "source_file": page.source_file,
                "question_no": current_label,
            }
        )
        current_label = None
        current_lines = []
        in_sub_part = False

    for line in lines:
        main_m = _MAIN_Q_RE.match(line)
        sub_m = _SUB_Q_RE.match(line)

        if main_m:
            # New top-level question — flush previous
            _flush()
            current_label = _label_from_match(main_m)
            # Include the full line (label + body) as part of the text
            current_lines = [line]
            in_sub_part = False

        elif sub_m and current_label is not None:
            # Sub-part of the current question — append to same record
            current_lines.append(line)
            in_sub_part = True

        else:
            # Continuation line — append to current question if we have one
            if current_label is not None:
                current_lines.append(line)
            # else: text before any question label (e.g. section header) — skip

    _flush()  # Don't forget the last question on the page
    return records


def split_pages(pages: list[PageRecord]) -> list[dict]:
    """Split text from multiple pages into question records."""
    all_records: list[dict] = []
    for page in pages:
        all_records.extend(split_page(page))
    return all_records


def split_results(results: list[ExtractionResult]) -> pd.DataFrame:
    """Convert a list of ExtractionResults into a question DataFrame.

    Parameters
    ----------
    results:
        Output of ``extractor.extract_pdfs``.

    Returns
    -------
    DataFrame with columns:
        question_text, year, marks, source_file, question_no
    """
    all_records: list[dict] = []
    for result in results:
        all_records.extend(split_pages(result.pages))

    if not all_records:
        return pd.DataFrame(
            columns=["question_text", "year", "marks", "source_file", "question_no"]
        )

    df = pd.DataFrame(all_records)

    # Post-processing
    df["question_text"] = df["question_text"].str.strip()
    df = df[df["question_text"].str.len() > 10].reset_index(drop=True)  # drop noise

    # Normalise types
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["marks"] = pd.to_numeric(df["marks"], errors="coerce")

    return df[["question_text", "year", "marks", "source_file", "question_no"]]
