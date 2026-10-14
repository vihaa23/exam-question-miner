"""
embedder.py — Encode question text into dense vector embeddings.

Model choice
-----------
all-MiniLM-L6-v2 is a sentence-transformers model fine-tuned for semantic
similarity.  It is fast (≈14k sentences/s on CPU), small (80 MB), and
performs well on short technical text like exam questions.

Caching strategy
----------------
Embeddings are expensive to recompute across sessions.  We cache on disk using
a pickle file keyed by a SHA-256 hash of the (sorted) unique question texts.
If the question set changes, the cache is invalidated automatically.

Why embeddings over keyword matching?
--------------------------------------
Two exam questions can ask the same thing with completely different words:
  "Explain the working of a stack." vs "Describe the LIFO data structure."
TF-IDF misses this; sentence embeddings capture it because they encode
semantic meaning, not surface tokens.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Default settings
# ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path(".embedding_cache")
_CACHE_FILE_NAME = "embeddings.pkl"


# ─────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────

def _texts_hash(texts: list[str]) -> str:
    """Produce a deterministic SHA-256 hex digest for a list of texts."""
    joined = "\n".join(sorted(set(texts)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, texts_hash: str, model_name: str) -> Path:
    safe_model = model_name.replace("/", "_")
    return cache_dir / f"{safe_model}_{texts_hash[:16]}.pkl"


def _load_cache(path: Path) -> Optional[dict[str, np.ndarray]]:
    """Load a cached embedding dict from disk. Returns None if unavailable."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301
    except (FileNotFoundError, Exception) as exc:
        logger.debug("Cache miss or error loading cache: %s", exc)
        return None


def _save_cache(path: Path, data: dict[str, np.ndarray]) -> None:
    """Persist embedding dict to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ─────────────────────────────────────────────────────────────────
# Embedder class
# ─────────────────────────────────────────────────────────────────

class Embedder:
    """Wraps a sentence-transformers model with disk-based caching.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or local path.
    cache_dir:
        Directory to store/read embedding cache files.
    batch_size:
        Number of sentences per encoding batch.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self._model = None  # Lazy load

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                logger.info("Loading model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Run: pip install sentence-transformers"
                ) from exc
        return self._model

    def encode(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into an embedding matrix.

        Parameters
        ----------
        texts:
            List of question strings.
        show_progress:
            If True, show a tqdm progress bar during encoding.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        texts_hash = _texts_hash(texts)
        cache_file = _cache_path(self.cache_dir, texts_hash, self.model_name)

        cached = _load_cache(cache_file)
        if cached is not None:
            logger.info("Loaded embeddings from cache: %s", cache_file)
            # Re-order to match the input list order
            try:
                return np.stack([cached[t] for t in texts])
            except KeyError:
                logger.warning("Cache key mismatch — re-encoding.")

        model = self._load_model()
        logger.info("Encoding %d questions with %s …", len(texts), self.model_name)
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit-norm for cosine similarity
        )

        # Cache: map text -> embedding vector
        cache_data = {t: embeddings[i] for i, t in enumerate(texts)}
        _save_cache(cache_file, cache_data)
        logger.info("Saved embeddings to cache: %s", cache_file)

        return embeddings

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "question_text",
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode the question_text column of a DataFrame.

        Parameters
        ----------
        df:
            Question DataFrame (from splitter).
        text_col:
            Column containing question strings.
        show_progress:
            Show tqdm bar.

        Returns
        -------
        np.ndarray of shape (len(df), embedding_dim).
        """
        texts = df[text_col].tolist()
        return self.encode(texts, show_progress=show_progress)
