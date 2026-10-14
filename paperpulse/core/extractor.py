"""
extractor.py — PDF text extraction with year parsing and optional OCR fallback.

Strategy
--------
1. Try pdfplumber first (best for digital/born-digital PDFs).
2. If a page yields no text, try PyMuPDF (fitz) as a second pass.
3. If a page still has no text AND pytesseract is installed, OCR the page image.
4. If Tesseract is missing, log a clear warning for that page and continue.
5. Never raise an unhandled exception — collect errors per file and return them.

Year extraction (from filename)
--------------------------------
Patterns tried in order:
  - A 4-digit year 2000-2099 anywhere in the stem: e.g. "CS101_2023.pdf" -> 2023
  - "may-2019" / "nov2018" style prefixes are captured by the same pattern.
  - If no year is found, year is returned as None and the UI lets the user type it.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
# Optional imports (graceful fallbacks)
# ──────────────────────────────────────────────

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF  # type: ignore
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

if not _HAS_PDFPLUMBER and not _HAS_PYMUPDF:
    raise ImportError(
        "At least one PDF library is required. "
        "Run: pip install pdfplumber  OR  pip install PyMuPDF"
    )

# ──────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────

@dataclass
class PageRecord:
    """Text extracted from a single PDF page."""
    source_file: str
    year: Optional[int]
    page_number: int          # 1-indexed
    text: str
    ocr_used: bool = False


@dataclass
class ExtractionResult:
    """Result of extracting an entire PDF file."""
    source_file: str
    year: Optional[int]
    pages: list[PageRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)   # per-page / per-file messages
    skipped_pages: list[int] = field(default_factory=list)


# ──────────────────────────────────────────────
# Year parsing
# ──────────────────────────────────────────────

_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def parse_year_from_filename(path: str | Path) -> Optional[int]:
    """Extract a 4-digit year (1900-2099) from the file stem.

    Returns the *last* match so "may-2019-paper" -> 2019.
    Returns None if no year is found.
    """
    stem = Path(path).stem
    matches = _YEAR_RE.findall(stem)
    if matches:
        return int(matches[-1])
    return None


# ──────────────────────────────────────────────
# Per-page text extraction helpers
# ──────────────────────────────────────────────

def _extract_page_pdfplumber(pdf_path: str, page_index: int) -> str:
    """Extract text from a single page using pdfplumber (0-indexed page)."""
    if not _HAS_PDFPLUMBER:
        return ""
    with pdfplumber.open(pdf_path) as pdf:
        if page_index >= len(pdf.pages):
            return ""
        page = pdf.pages[page_index]
        return page.extract_text() or ""


def _extract_page_pymupdf(pdf_path: str, page_index: int) -> str:
    """Extract text from a single page using PyMuPDF (0-indexed page)."""
    if not _HAS_PYMUPDF:
        return ""
    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        doc.close()
        return ""
    page = doc[page_index]
    text = page.get_text("text")  # type: ignore[attr-defined]
    doc.close()
    return text or ""


def _ocr_page_pymupdf(pdf_path: str, page_index: int, dpi: int = 200) -> str:
    """Render a page to an image via PyMuPDF, then OCR it with pytesseract."""
    if not _HAS_PYMUPDF or not _HAS_TESSERACT:
        return ""
    doc = fitz.open(pdf_path)
    if page_index >= len(doc):
        doc.close()
        return ""
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)  # type: ignore[attr-defined]
    doc.close()

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text: str = pytesseract.image_to_string(img)
    return text


def _page_has_content(text: str, min_chars: int = 30) -> bool:
    """Return True if the extracted text has enough characters to be meaningful."""
    stripped = re.sub(r"\s+", "", text)
    return len(stripped) >= min_chars


# ──────────────────────────────────────────────
# Main extraction function
# ──────────────────────────────────────────────

def extract_pdf(
    pdf_path: str | Path,
    year_override: Optional[int] = None,
    use_ocr: bool = True,
    ocr_dpi: int = 200,
    min_chars_per_page: int = 30,
) -> ExtractionResult:
    """Extract text from every page of a PDF file.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    year_override:
        If provided, use this year instead of parsing from the filename.
    use_ocr:
        Whether to attempt OCR on pages with no extractable text.
        If True but Tesseract is not installed, a warning is shown per page
        and the page is skipped (never raises).
    ocr_dpi:
        Resolution for rendering pages before OCR.
    min_chars_per_page:
        Pages with fewer non-whitespace characters are treated as empty.

    Returns
    -------
    ExtractionResult with pages and any error messages.
    """
    pdf_path = Path(pdf_path)
    year = year_override if year_override is not None else parse_year_from_filename(pdf_path)
    result = ExtractionResult(source_file=str(pdf_path), year=year)

    # Determine total page count
    n_pages: int = 0
    try:
        if _HAS_PDFPLUMBER:
            with pdfplumber.open(str(pdf_path)) as pdf:
                n_pages = len(pdf.pages)
        elif _HAS_PYMUPDF:
            doc = fitz.open(str(pdf_path))
            n_pages = len(doc)
            doc.close()
    except Exception as exc:
        result.errors.append(f"Could not open {pdf_path.name}: {exc}")
        return result

    for page_idx in range(n_pages):
        page_no = page_idx + 1
        text = ""
        ocr_used = False

        # Primary: pdfplumber
        try:
            text = _extract_page_pdfplumber(str(pdf_path), page_idx)
        except Exception as exc:
            result.errors.append(f"pdfplumber error on page {page_no}: {exc}")

        # Secondary: PyMuPDF if pdfplumber gave nothing
        if not _page_has_content(text, min_chars_per_page) and _HAS_PYMUPDF:
            try:
                text = _extract_page_pymupdf(str(pdf_path), page_idx)
            except Exception as exc:
                result.errors.append(f"PyMuPDF error on page {page_no}: {exc}")

        # Tertiary: OCR
        if not _page_has_content(text, min_chars_per_page) and use_ocr:
            if _HAS_TESSERACT:
                try:
                    text = _ocr_page_pymupdf(str(pdf_path), page_idx, dpi=ocr_dpi)
                    ocr_used = True
                except Exception as exc:
                    result.errors.append(f"OCR error on page {page_no}: {exc}")
            else:
                msg = (
                    f"Page {page_no} of '{pdf_path.name}' appears to be scanned "
                    f"and Tesseract OCR is not installed. Skipping this page. "
                    f"Install Tesseract + pytesseract to process scanned pages."
                )
                warnings.warn(msg, stacklevel=2)
                result.errors.append(msg)
                result.skipped_pages.append(page_no)
                continue

        # Still nothing after all attempts
        if not _page_has_content(text, min_chars_per_page):
            result.skipped_pages.append(page_no)
            continue

        result.pages.append(
            PageRecord(
                source_file=str(pdf_path),
                year=year,
                page_number=page_no,
                text=text.strip(),
                ocr_used=ocr_used,
            )
        )

    return result


def extract_pdfs(
    pdf_paths: list[str | Path],
    year_overrides: Optional[dict[str, int]] = None,
    use_ocr: bool = True,
) -> list[ExtractionResult]:
    """Extract text from multiple PDF files.

    Parameters
    ----------
    pdf_paths:
        List of paths to PDF files.
    year_overrides:
        Mapping from filename (stem or full name) to year, e.g.
        {"CS101_paper.pdf": 2023, "CS101_paper": 2022}.
        If a file's name or stem is in the map, it overrides auto-detection.
    use_ocr:
        Passed through to ``extract_pdf``.

    Returns
    -------
    List of ExtractionResult objects, one per file (including failed ones).
    """
    year_overrides = year_overrides or {}
    results: list[ExtractionResult] = []

    for path in pdf_paths:
        p = Path(path)
        year = year_overrides.get(p.name) or year_overrides.get(p.stem)
        results.append(extract_pdf(p, year_override=year, use_ocr=use_ocr))

    return results
