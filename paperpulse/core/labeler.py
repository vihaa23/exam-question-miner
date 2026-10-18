"""
labeler.py — Assign a human-readable name to each question cluster.

Strategy (deterministic, no LLM required)
------------------------------------------
1. Compute the cluster centroid (mean of normalised embeddings).
2. Find the *medoid*: the question whose embedding is closest to the centroid.
3. Clean the question text: strip question numbers (Q.1, 1.), marks notations ([10], (5M)),
   and question-prompt phrases ("What is", "Explain the concept of", "Describe").
4. Extract the core concept title cleanly without awkward word-boundary cutoffs.
5. If top TF-IDF keywords add valuable clarity, integrate them naturally without duplicates.

Output examples:
    - "Stack: Working and Array Operations"
    - "Dynamic Programming: Knapsack & Subsequence"
    - "Binary Search Tree (BST)"
    - "Hashing: Collision Resolution Techniques"
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

_PROMPT_PREFIX_RE = re.compile(
    r"^(?:(?:Q\.?\s*\d+|\d+\s*[.)\-:]?)\s*)?"
    r"(?:what\s+is\s+a\b|what\s+is\b|what\s+are\b|"
    r"explain\s+the\s+concept\s+of\b|explain\s+the\b|explain\b|"
    r"describe\s+the\s+working\s+of\b|describe\s+the\s+concept\s+of\b|describe\b|"
    r"define\s+a\b|define\b|write\s+an\s+algorithm\s+for\b|write\s+and\s+explain\b|"
    r"differentiate\s+between\b|discuss\b)\s*",
    re.IGNORECASE,
)


def _clean_question_for_label(text: str) -> str:
    """Strip leading numbering, marks notations, and boilerplate prompts."""
    # Strip marks notations like [10 marks], (5 marks), [10], etc.
    text = re.sub(r"\[\s*\d+(?:\.\d+)?\s*(?:marks?|m)?\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\d+(?:\.\d+)?\s*(?:marks?|m)?\s*\)", "", text, flags=re.IGNORECASE)

    # Strip Q.1, 1., etc. and common prompt openers
    text = _PROMPT_PREFIX_RE.sub("", text.strip()).strip()

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    return text.strip()


def _trim_to_phrase(text: str, max_words: int = 10) -> str:
    """Return a clean, readable topic phrase from a question string."""
    clean_text = _clean_question_for_label(text)
    if not clean_text:
        clean_text = text.strip()

    # Cut at question mark, sentence boundary, or newline
    cut = re.search(r"(\?|\. |\:|\n)", clean_text)
    if cut:
        first_segment = clean_text[: cut.start()].strip()
        if len(first_segment) >= 4:
            clean_text = first_segment
        else:
            # Skip past punctuation and take next part
            clean_text = clean_text[cut.end():].strip()

    words = clean_text.split()
    if len(words) > max_words:
        clean_text = " ".join(words[:max_words])

    # Clean any trailing punctuation or dangling symbols
    clean_text = re.sub(r"[\s,\.\-\?:]+$", "", clean_text).strip()
    return clean_text


def _top_tfidf_keywords(texts: list[str], n: int = 2, max_features: int = 200) -> list[str]:
    """Extract top-n TF-IDF keywords from question strings, ignoring stop words and filler."""
    if not texts:
        return []
    try:
        exam_fillers = {
            "explain", "describe", "write", "define", "differentiate", "distinguish",
            "marks", "mark", "short", "notes", "note", "example", "examples",
            "compare", "contrast", "briefly", "detail", "details", "discuss",
            "algorithm", "step", "steps", "give", "state", "used", "help",
            "suitable", "different", "various"
        }
        all_stops = list(exam_fillers.union(TfidfVectorizer(stop_words="english").get_stop_words() or []))
        vec = TfidfVectorizer(
            stop_words=all_stops,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # Only alphabetical words >= 3 letters
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
    max_phrase_words: int = 8,
    n_keywords: int = 2,
) -> pd.DataFrame:
    """Assign a human-readable label to each cluster."""
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

        # Append clean keyword if it adds useful context and isn't already present
        clean_phrase_lower = phrase.lower()
        additional_kw = [
            kw.title() for kw in keywords
            if kw.lower() not in clean_phrase_lower and len(kw) > 3
        ]

        if additional_kw:
            label = f"{phrase} [{', '.join(additional_kw[:2])}]"
        else:
            label = phrase

        cluster_labels[cluster_id] = label
        medoid_indices[cluster_id] = medoid_global_idx

    df["cluster_label"] = df["cluster_id"].map(cluster_labels)
    df["medoid_index"] = df["cluster_id"].map(medoid_indices)
    return df
