"""
labeler.py — Assign a human-readable name to each question cluster.

Strategy (deterministic, no LLM required)
------------------------------------------
1. Compute the cluster centroid (mean of normalised embeddings).
2. Find the *medoid*: the question whose embedding is closest to the centroid.
3. Trim that question to a short phrase (≤ 12 words, stopping at punctuation).
4. Append top-2 TF-IDF keywords from the cluster to improve specificity.

Output example:
    "Explain the working of a stack  [stack, operations]"

Why not just TF-IDF keywords?
------------------------------
Pure keyword labels ("stack, push, pop") lose grammatical context.
Using the medoid sentence gives a natural phrase; keywords catch important
terms that the medoid sentence may not have started with.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _clean_question_for_label(text: str) -> str:
    """Strip leading question numbering, marks notations, and noisy prefixes."""
    # Strip leading Q.1, Q1., Q1, 1., 1), etc.
    text = re.sub(r"^(?:Q\.?\s*\d+|\d+\s*[.)\-:]?)\s*", "", text, flags=re.IGNORECASE).strip()
    # Strip marks notations like [10 marks], (5 marks), [10], etc.
    text = re.sub(r"\[\s*\d+(?:\.\d+)?\s*(?:marks?|m)?\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\d+(?:\.\d+)?\s*(?:marks?|m)?\s*\)", "", text, flags=re.IGNORECASE)
    return text.strip()


def _trim_to_phrase(text: str, max_words: int = 10) -> str:
    """Return a short clean phrase from a question string."""
    clean_text = _clean_question_for_label(text)
    if not clean_text:
        clean_text = text.strip()

    # Cut at question mark, sentence boundary, or newline
    cut = re.search(r"(\?|\. |\:|\n)", clean_text)
    if cut:
        clean_text = clean_text[: cut.start()].strip()

    words = clean_text.split()
    if len(words) > max_words:
        clean_text = " ".join(words[:max_words]) + "..."

    return clean_text.strip()


def _top_tfidf_keywords(texts: list[str], n: int = 2, max_features: int = 200) -> list[str]:
    """Extract top-n TF-IDF keywords from question strings, ignoring numbers and common exam filler."""
    if not texts:
        return []
    try:
        exam_fillers = {
            "explain", "describe", "write", "define", "differentiate", "distinguish",
            "marks", "mark", "short", "notes", "note", "example", "examples",
            "compare", "contrast", "briefly", "detail", "details", "discuss",
            "algorithm", "step", "steps", "give", "state"
        }
        all_stops = list(exam_fillers.union(TfidfVectorizer(stop_words="english").get_stop_words() or []))
        vec = TfidfVectorizer(
            stop_words=all_stops,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # Only alphabetical words >= 3 letters (excludes marks like 10)
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        X = vec.fit_transform(texts)
        mean_scores = X.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:n]
        vocab = vec.get_feature_names_out()
        return [vocab[i] for i in top_indices]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────
# Main labeling function
# ─────────────────────────────────────────────────────────────────

def label_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    max_phrase_words: int = 10,
    n_keywords: int = 2,
) -> pd.DataFrame:
    """Assign a human-readable label to each cluster.

    Parameters
    ----------
    df:
        Question DataFrame with a 'cluster_id' column (from clusterer).
    embeddings:
        Embedding matrix aligned with df rows.
    max_phrase_words:
        Maximum words in the medoid phrase.
    n_keywords:
        Number of TF-IDF keywords to append to the label.

    Returns
    -------
    DataFrame with two new columns:
        cluster_label  : str  — human-readable cluster name
        medoid_index   : int  — row index of the medoid question
    """
    df = df.copy()
    cluster_labels: dict[int, str] = {}
    medoid_indices: dict[int, int] = {}

    for cluster_id, group in df.groupby("cluster_id"):
        indices = group.index.tolist()
        texts = group["question_text"].tolist()

        if len(indices) == 1:
            # Singleton cluster
            label = _trim_to_phrase(texts[0], max_phrase_words)
            cluster_labels[cluster_id] = label
            medoid_indices[cluster_id] = indices[0]
            continue

        # Centroid of normalised embeddings
        cluster_embs = embeddings[indices]          # (k, D)
        centroid = cluster_embs.mean(axis=0)        # (D,)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)

        # Medoid: closest embedding to the centroid
        cosine_sims = cluster_embs @ centroid_norm  # (k,)
        best_local = int(np.argmax(cosine_sims))
        medoid_global_idx = indices[best_local]
        medoid_text = texts[best_local]

        phrase = _trim_to_phrase(medoid_text, max_phrase_words)
        keywords = _top_tfidf_keywords(texts, n=n_keywords)
        # Avoid repeating keywords if they already exist in phrase
        additional_kw = [kw for kw in keywords if kw.lower() not in phrase.lower()]
        if additional_kw:
            label = f"{phrase} ({', '.join(additional_kw)})"
        else:
            label = phrase

        cluster_labels[cluster_id] = label
        medoid_indices[cluster_id] = medoid_global_idx

    df["cluster_label"] = df["cluster_id"].map(cluster_labels)
    df["medoid_index"] = df["cluster_id"].map(medoid_indices)
    return df
