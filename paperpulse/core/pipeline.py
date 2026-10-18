"""
pipeline.py — Orchestrates the full PaperPulse processing pipeline.

Stages in order:
  1. Extract  — PDF pages -> PageRecord list
  2. Split    — PageRecord list -> question DataFrame
  3. Embed    — question texts -> embedding matrix
  4. Cluster  — embeddings -> cluster labels added to DataFrame
  5. Label    — clusters -> human-readable names added to DataFrame
  6. Rank     — labelled clusters -> sorted RankedCluster list

The pipeline is designed so each stage can also be called independently
for debugging or testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from paperpulse.core.clusterer import DEFAULT_THRESHOLD, assign_clusters
from paperpulse.core.embedder import DEFAULT_CACHE_DIR, Embedder
from paperpulse.core.extractor import ExtractionResult, extract_pdfs
from paperpulse.core.labeler import label_clusters
from paperpulse.core.ranker import RankedCluster, rank_clusters
from paperpulse.core.splitter import split_results

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Pipeline result container
# ─────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Holds the outputs of every pipeline stage."""
    # Extraction
    extraction_results: list[ExtractionResult] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)

    # Questions
    questions_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Embeddings
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    # Ranked clusters
    ranked_clusters: list[RankedCluster] = field(default_factory=list)

    # Convenience counts
    @property
    def n_papers(self) -> int:
        return len(self.extraction_results)

    @property
    def n_questions(self) -> int:
        return len(self.questions_df)

    @property
    def n_clusters(self) -> int:
        return len(self.ranked_clusters)


# ─────────────────────────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────────────────────────

def run_pipeline(
    file_paths: Optional[list[str | Path]] = None,
    pdf_paths: Optional[list[str | Path]] = None,
    year_overrides: Optional[dict[str, int]] = None,
    use_ocr: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    cluster_method: str = "embedding",
    embedder_model: str = "all-MiniLM-L6-v2",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    weight_frequency: float = 0.5,
    weight_recency: float = 0.3,
    weight_marks: float = 0.2,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> PipelineResult:
    """Run the full PaperPulse pipeline on a list of PDF files.

    Parameters
    ----------
    pdf_paths:
        Paths to uploaded PDF exam papers.
    year_overrides:
        Dict mapping filename -> year for UI-supplied years.
    use_ocr:
        Attempt OCR on scanned pages (requires Tesseract).
    threshold:
        Clustering distance threshold (exposed as slider in UI).
    cluster_method:
        'embedding' (semantic) or 'tfidf' (baseline).
    embedder_model:
        Sentence-transformers model name.
    cache_dir:
        Directory for embedding cache.
    weight_frequency, weight_recency, weight_marks:
        Ranker scoring weights.
    progress_callback:
        Optional callable(message: str, fraction: float) for UI progress bar.

    Returns
    -------
    PipelineResult with all intermediate outputs populated.
    """
    result = PipelineResult()

    def _progress(msg: str, frac: float) -> None:
        logger.info("[%.0f%%] %s", frac * 100, msg)
        if progress_callback:
            progress_callback(msg, frac)

    # ── Stage 1: Extract ─────────────────────────────────────────
    input_paths = file_paths if file_paths is not None else (pdf_paths or [])
    _progress("Extracting text from exam papers…", 0.0)
    extraction_results = extract_pdfs(
        input_paths,
        year_overrides=year_overrides,
        use_ocr=use_ocr,
    )
    result.extraction_results = extraction_results

    # Collect all errors from extraction
    for er in extraction_results:
        result.extraction_errors.extend(er.errors)

    total_pages = sum(len(er.pages) for er in extraction_results)
    logger.info("Extracted %d pages across %d files.", total_pages, len(extraction_results))

    # ── Stage 2: Split ───────────────────────────────────────────
    _progress("Splitting pages into questions…", 0.2)
    questions_df = split_results(extraction_results)
    result.questions_df = questions_df
    logger.info("Found %d questions.", len(questions_df))

    if questions_df.empty:
        _progress("No questions found. Check PDF quality.", 1.0)
        return result

    # ── Stage 3: Embed ───────────────────────────────────────────
    _progress("Computing semantic embeddings…", 0.4)
    embedder = Embedder(model_name=embedder_model, cache_dir=Path(cache_dir))
    embeddings = embedder.encode_dataframe(questions_df, show_progress=False)
    result.embeddings = embeddings

    # ── Stage 4: Cluster ─────────────────────────────────────────
    _progress("Clustering similar questions…", 0.6)
    clustered_df = assign_clusters(
        questions_df,
        embeddings=embeddings,
        threshold=threshold,
        method=cluster_method,
    )

    # ── Stage 5: Label ───────────────────────────────────────────
    _progress("Labelling topic clusters…", 0.75)
    labelled_df = label_clusters(clustered_df, embeddings=embeddings)
    result.questions_df = labelled_df

    # ── Stage 6: Rank ────────────────────────────────────────────
    _progress("Ranking by study priority…", 0.9)
    ranked = rank_clusters(
        labelled_df,
        weight_frequency=weight_frequency,
        weight_recency=weight_recency,
        weight_marks=weight_marks,
    )
    result.ranked_clusters = ranked

    _progress("Done!", 1.0)
    return result
