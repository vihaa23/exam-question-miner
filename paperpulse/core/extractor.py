"""
extractor.py — PDF & image text extraction with year parsing and robust Tesseract OCR.

Strategy
--------
1. For PDFs:
   a. Try pdfplumber first (best for digital/born-digital PDFs).
   b. If a page yields no text, try PyMuPDF (fitz) as a second pass.
   c. If a page still has no text, render to image and OCR with pytesseract.
2. For Images (PNG, JPG, JPEG, WEBP, BMP, TIFF):
   a. Preprocess with grayscale, resizing, and contrast enhancement for optimal OCR.
   b. Run pytesseract OCR directly to extract question text.
3. Auto-detects Tesseract binary on Windows / Linux / macOS.
4. Allows overriding or manually configuring Tesseract path via UI or API.
5. Collects warnings and errors gracefully without crashing.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

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
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # type: ignore
    _HAS_TESSERACT_PKG = True
except ImportError:
    _HAS_TESSERACT_PKG = False

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# ──────────────────────────────────────────────
# Tesseract Binary Auto-detection
# ──────────────────────────────────────────────

def find_tesseract_binary() -> Optional[str]:
    """Search for the tesseract executable in standard system locations."""
    # 1. Check if already configured in pytesseract
    if _HAS_TESSERACT_PKG:
        cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        if cmd and cmd != "tesseract" and Path(cmd).is_file():
            return str(Path(cmd).resolve())

    # 2. Check PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return str(Path(which_path).resolve())

    # 3. Common Windows locations
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    user_profile = os.environ.get("USERPROFILE", "")

    candidate_paths = [
        Path(program_files) / "Tesseract-OCR" / "tesseract.exe",
        Path(program_files_x86) / "Tesseract-OCR" / "tesseract.exe",
        Path(local_app_data) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        Path(user_profile) / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]

    for p in candidate_paths:
        if p.is_file():
            return str(p.resolve())

    return None


def init_tesseract() -> tuple[bool, str]:
    """Initialize pytesseract with the discovered executable path."""
    if not _HAS_TESSERACT_PKG:
        return False, "pytesseract Python package is not installed (run `pip install pytesseract Pillow`)."

    binary = find_tesseract_binary()
    if binary:
        try:
            pytesseract.pytesseract.tesseract_cmd = binary
            ver = pytesseract.get_tesseract_version()
            return True, f"Tesseract v{ver} ({binary})"
        except Exception as exc:
            return False, f"Tesseract executable found at {binary} but failed to run: {exc}"

    return False, "Tesseract OCR binary not found in PATH or standard install directories."


def set_tesseract_cmd(path: str) -> tuple[bool, str]:
    """Manually point pytesseract to a specific executable."""
    if not _HAS_TESSERACT_PKG:
        return False, "pytesseract library is not installed."
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return False, f"File does not exist: {path}"
    try:
        pytesseract.pytesseract.tesseract_cmd = str(p)
        ver = pytesseract.get_tesseract_version()
        return True, f"Tesseract v{ver} configured from {p}"
    except Exception as exc:
        return False, f"Failed executing {p}: {exc}"


# Run initial detection
_HAS_TESSERACT, _TESSERACT_STATUS = init_tesseract()


# ──────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────

@dataclass
class PageRecord:
    """Text extracted from a single PDF page or image file."""
    source_file: str
    year: Optional[int]
    page_number: int          # 1-indexed
    text: str
    ocr_used: bool = False


@dataclass
class ExtractionResult:
    """Result of extracting an entire PDF or image file."""
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

    Returns the *last* match so 'may-2019-paper' -> 2019.
    Returns None if no year is found.
    """
    stem = Path(path).stem
    matches = _YEAR_RE.findall(stem)
    if matches:
        return int(matches[-1])
    return None


# ──────────────────────────────────────────────
# Image Preprocessing & OCR
# ──────────────────────────────────────────────

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Enhance image quality before passing to Tesseract OCR."""
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if resolution is low (upscale small scans/photos for clear character strokes)
    w, h = img.size
    if w < 1600:
        scale = 1600.0 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    # Grayscale
    gray = ImageOps.grayscale(img)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(gray)
    contrasted = enhancer.enhance(1.8)

    # Enhance sharpness
    sharpener = ImageEnhance.Sharpness(contrasted)
    sharpened = sharpener.enhance(1.4)

    return sharpened


def ocr_image(img: Image.Image) -> str:
    """Run Tesseract OCR on a PIL Image with image preprocessing."""
    global _HAS_TESSERACT
    # Re-check binary in case it was installed recently
    if not _HAS_TESSERACT:
        _HAS_TESSERACT, _ = init_tesseract()

    if not _HAS_TESSERACT:
        raise RuntimeError(
            "Tesseract OCR is not available. Please install Tesseract-OCR and ensure it is in PATH."
        )

    processed = preprocess_image_for_ocr(img)
    # PSM 3: Fully automatic page segmentation (standard for exam question pages)
    config = "--oem 3 --psm 3"
    text: str = pytesseract.image_to_string(processed, config=config)
    return text.strip()


def extract_image(
    image_path_or_bytes: Union[str, Path, bytes],
    year_override: Optional[int] = None,
    filename: Optional[str] = None,
) -> ExtractionResult:
    """Extract question text from an exam paper image using Tesseract OCR."""
    if isinstance(image_path_or_bytes, (str, Path)):
        path = Path(image_path_or_bytes)
        source_name = path.name
        year = year_override if year_override is not None else parse_year_from_filename(path)
        result = ExtractionResult(source_file=str(path), year=year)
        try:
            with Image.open(path) as img:
                text = ocr_image(img)
        except Exception as exc:
            result.errors.append(f"OCR failed for {source_name}: {exc}")
            return result
    else:
        import io
        source_name = filename or "image_upload.png"
        year = year_override if year_override is not None else parse_year_from_filename(source_name)
        result = ExtractionResult(source_file=source_name, year=year)
        try:
            with Image.open(io.BytesIO(image_path_or_bytes)) as img:
                text = ocr_image(img)
        except Exception as exc:
            result.errors.append(f"OCR failed for {source_name}: {exc}")
            return result

    if not _page_has_content(text, min_chars=15):
        result.errors.append(f"OCR produced insufficient text from '{source_name}'.")
        result.skipped_pages.append(1)
        return result

    result.pages.append(
        PageRecord(
            source_file=result.source_file,
            year=year,
            page_number=1,
            text=text,
            ocr_used=True,
        )
    )
    return result


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
    if not _HAS_PYMUPDF or not _HAS_TESSERACT_PKG:
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
    return ocr_image(img)


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
    """Extract text from every page of a PDF file."""
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
            global _HAS_TESSERACT
            if not _HAS_TESSERACT:
                _HAS_TESSERACT, _ = init_tesseract()

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


def extract_file(
    file_path: str | Path,
    year_override: Optional[int] = None,
    use_ocr: bool = True,
) -> ExtractionResult:
    """Extract text from a file (PDF, Image, or plain text)."""
    p = Path(file_path)
    suffix = p.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return extract_image(p, year_override=year_override)
    elif suffix == ".pdf":
        return extract_pdf(p, year_override=year_override, use_ocr=use_ocr)
    elif suffix in {".txt", ".text"}:
        year = year_override if year_override is not None else parse_year_from_filename(p)
        text = p.read_text(encoding="utf-8", errors="replace")
        page = PageRecord(source_file=str(p), year=year, page_number=1, text=text, ocr_used=False)
        return ExtractionResult(source_file=str(p), year=year, pages=[page])
    else:
        # Default try PDF
        return extract_pdf(p, year_override=year_override, use_ocr=use_ocr)


def extract_files(
    file_paths: list[str | Path],
    year_overrides: Optional[dict[str, int]] = None,
    use_ocr: bool = True,
) -> list[ExtractionResult]:
    """Extract text from multiple files (PDFs, Images, or TXT)."""
    year_overrides = year_overrides or {}
    results: list[ExtractionResult] = []

    for path in file_paths:
        p = Path(path)
        year = year_overrides.get(p.name) or year_overrides.get(p.stem)
        results.append(extract_file(p, year_override=year, use_ocr=use_ocr))

    return results


# Backward compatibility alias
extract_pdfs = extract_files
