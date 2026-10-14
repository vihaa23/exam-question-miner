"""
clusterer.py — Group similar exam questions into topic clusters.

Why AgglomerativeClustering with a distance threshold?
------------------------------------------------------
A fixed number of clusters (k) requires the user to know in advance how many
topics exist — impossible for an unknown exam paper set.  A *distance threshold*
instead says "merge two questions if they are within X cosine-distance of each
other".  This lets the number of clusters emerge naturally from the data.

Why cosine distance with average linkage?
-----------------------------------------
- Cosine distance is invariant to sentence length, which matters because exam
  questions vary from 5 words to 50.
- Average linkage (UPGMA) is more robust than single-linkage (chaining) and
  complete-linkage (outlier sensitivity) for medium-sized question sets.

TF-IDF baseline
---------------
Provided so the UI and evaluation can compare keyword matching vs. semantic
embeddings.  It uses the same AgglomerativeClustering algorithm, just with
TF-IDF + cosine distance instead of sentence-transformer embeddings.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.35   # Cosine distance; tuned from evaluation results.
                            # Lower = tighter clusters (more clusters).
                            # Higher = looser (fewer, broader clusters).

# ─────────────────────────────────────────────────────────────────
# Core clustering functions
# ─────────────────────────────────────────────────────────────────

def cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    linkage: Literal["average", "complete", "single"] = "average",
) -> np.ndarray:
    """Cluster pre-computed embeddings with AgglomerativeClustering.

    Parameters
    ----------
    embeddings:
        Row-normalised embedding matrix, shape (N, D).
    threshold:
        Cosine distance threshold — pairs with distance ≤ threshold are merged.
        Exposed as a slider in the Streamlit UI.
    linkage:
        Linkage strategy. 'average' is recommended.

    Returns
    -------
    1-D integer array of cluster labels, shape (N,).
    """
    if len(embeddings) == 0:
        return np.array([], dtype=int)

    if len(embeddings) == 1:
        return np.array([0], dtype=int)

    # sklearn's AgglomerativeClustering with metric='cosine' requires
    # connectivity=None and distance_threshold to be used together.
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=linkage,
        distance_threshold=threshold,
        compute_full_tree=True,
    )
    labels: np.ndarray = model.fit_predict(embeddings)
    logger.info(
        "Clustering: %d questions -> %d clusters (threshold=%.2f)",
        len(embeddings),
        len(np.unique(labels)),
        threshold,
    )
    return labels


def cluster_tfidf(
    texts: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    linkage: Literal["average", "complete", "single"] = "average",
    max_features: int = 5000,
) -> np.ndarray:
    """Cluster questions using a TF-IDF representation (baseline method).

    Parameters
    ----------
    texts:
        Raw question strings.
    threshold:
        Cosine distance threshold (same scale as embedding threshold).
    linkage:
        Linkage strategy.
    max_features:
        TF-IDF vocabulary size cap.

    Returns
    -------
    1-D integer array of cluster labels, shape (N,).
    """
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

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=linkage,
        distance_threshold=threshold,
        compute_full_tree=True,
    )
    labels: np.ndarray = model.fit_predict(X_norm)
    logger.info(
        "TF-IDF baseline: %d questions -> %d clusters (threshold=%.2f)",
        len(texts),
        len(np.unique(labels)),
        threshold,
    )
    return labels


# ─────────────────────────────────────────────────────────────────
# DataFrame integration
# ─────────────────────────────────────────────────────────────────

def assign_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    method: Literal["embedding", "tfidf"] = "embedding",
) -> pd.DataFrame:
    """Add a 'cluster_id' column to the question DataFrame.

    Parameters
    ----------
    df:
        Question DataFrame from splitter.
    embeddings:
        Embedding matrix aligned with df rows (required if method='embedding').
    threshold:
        Distance threshold for clustering.
    method:
        'embedding' uses sentence-transformer vectors; 'tfidf' uses TF-IDF.

    Returns
    -------
    Copy of df with 'cluster_id' column added.
    """
    df = df.copy()

    if method == "embedding":
        labels = cluster_embeddings(embeddings, threshold=threshold)
    else:
        labels = cluster_tfidf(df["question_text"].tolist(), threshold=threshold)

    df["cluster_id"] = labels
    return df
