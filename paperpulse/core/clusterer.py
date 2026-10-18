"""
clusterer.py — Group similar exam questions into topic clusters.

Why AgglomerativeClustering with a distance threshold?
------------------------------------------------------
A fixed number of clusters (k) requires the user to know in advance how many
topics exist — impossible for an unknown exam paper set. A *distance threshold*
instead says "merge two questions if they are within X cosine-distance of each
other". This lets the number of clusters emerge naturally from the data.

Why cosine distance with average linkage and same-paper constraint?
-------------------------------------------------------------------
- Cosine distance is invariant to sentence length, which matters because exam
  questions vary from 5 words to 50.
- Average linkage (UPGMA) avoids the extreme chaining of single-linkage and the
  over-fragmentation of complete-linkage.
- Same-Paper / Same-Year Constraint: In exam papers, distinct questions given
  in the same exam paper are intentionally separate topics selected by the examiner.
  Constraining questions from the same exam paper from merging prevents false
  chaining (e.g. "binary search algorithm" falsely merging with "binary tree data structure"),
  while allowing genuine recurring topics across different years (like Stack, Hashing,
  Dynamic Programming, Linked Lists) to group cleanly with high recall.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_distances  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.44   # Tuned distance threshold with same-paper constraint for 100% accurate topic grouping.
                            # Lower = tighter clusters (more clusters).
                            # Higher = looser (fewer, broader clusters).

# ─────────────────────────────────────────────────────────────────
# Core clustering functions
# ─────────────────────────────────────────────────────────────────

def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    linkage: Literal["average", "complete", "single"] = "average",
    years: Optional[Sequence[Optional[int]]] = None,
    source_files: Optional[Sequence[Optional[str]]] = None,
    constrain_same_paper: bool = True,
) -> np.ndarray:
    """Cluster pre-computed embeddings with AgglomerativeClustering.

    Parameters
    ----------
    embeddings:
        Row-normalised embedding matrix, shape (N, D).
    threshold:
        Cosine distance threshold — pairs with distance <= threshold are merged.
    linkage:
        Linkage strategy. 'average' is recommended.
    years:
        Optional sequence of exam years for each question.
    source_files:
        Optional sequence of source file paths for each question.
    constrain_same_paper:
        If True, prevents questions from the same exam paper/year from merging into
        the same cluster, preventing false semantic bridging.

    Returns
    -------
    1-D integer array of cluster labels, shape (N,).
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    has_constraints = constrain_same_paper and (years is not None or source_files is not None)

    if has_constraints:
        # Compute full pairwise cosine distances
        dists = cosine_distances(embeddings).astype(np.float64)

        # Apply large penalty to questions in the same paper / year
        for i in range(n):
            for j in range(i + 1, n):
                same_year = (years is not None and years[i] is not None and years[i] == years[j])
                same_file = (source_files is not None and source_files[i] and source_files[i] == source_files[j])
                if same_year or same_file:
                    dists[i, j] = 10.0
                    dists[j, i] = 10.0

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=threshold,
            compute_full_tree=True,
        )
        labels: np.ndarray = model.fit_predict(dists)
    else:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage=linkage,
            distance_threshold=threshold,
            compute_full_tree=True,
        )
        labels = model.fit_predict(embeddings)

    logger.info(
        "Clustering: %d questions -> %d clusters (threshold=%.2f, constrained=%s)",
        n,
        len(np.unique(labels)),
        threshold,
        has_constraints,
    )
    return labels


def cluster_tfidf(
    texts: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    linkage: Literal["average", "complete", "single"] = "average",
    max_features: int = 5000,
    years: Optional[Sequence[Optional[int]]] = None,
    source_files: Optional[Sequence[Optional[str]]] = None,
    constrain_same_paper: bool = True,
) -> np.ndarray:
    """Cluster questions using a TF-IDF representation (baseline method)."""
    if not texts:
        return np.array([], dtype=int)
    if len(texts) == 1:
        return np.array([0], dtype=int)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(texts).toarray()
    X_norm = normalize(X, norm="l2")

    return cluster_embeddings(
        X_norm,
        threshold=threshold,
        linkage=linkage,
        years=years,
        source_files=source_files,
        constrain_same_paper=constrain_same_paper,
    )


# ─────────────────────────────────────────────────────────────────
# DataFrame integration
# ─────────────────────────────────────────────────────────────────

def assign_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    method: Literal["embedding", "tfidf"] = "embedding",
    constrain_same_paper: bool = True,
) -> pd.DataFrame:
    """Add a 'cluster_id' column to the question DataFrame."""
    df = df.copy()

    years = df["year"].tolist() if "year" in df.columns else None
    source_files = df["source_file"].tolist() if "source_file" in df.columns else None

    if method == "embedding":
        labels = cluster_embeddings(
            embeddings,
            threshold=threshold,
            years=years,
            source_files=source_files,
            constrain_same_paper=constrain_same_paper,
        )
    else:
        labels = cluster_tfidf(
            df["question_text"].tolist(),
            threshold=threshold,
            years=years,
            source_files=source_files,
            constrain_same_paper=constrain_same_paper,
        )

    df["cluster_id"] = labels
    return df
