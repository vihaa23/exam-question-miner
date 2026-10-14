"""Tests for paperpulse.core.clusterer — synthetic question clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paperpulse.core.clusterer import (
    assign_clusters,
    cluster_embeddings,
    cluster_tfidf,
    DEFAULT_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────
# Synthetic embeddings helpers
# ─────────────────────────────────────────────────────────────────

def _unit(v: list[float]) -> np.ndarray:
    """Return a unit-normalised vector."""
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _make_embeddings() -> np.ndarray:
    """Three clearly distinct groups of embeddings."""
    # Group A: stack questions (should cluster together)
    a1 = _unit([1, 0, 0, 0, 0])
    a2 = _unit([0.98, 0.1, 0.05, 0, 0])
    a3 = _unit([0.96, 0.15, 0.02, 0, 0])
    # Group B: graph questions (clearly different direction)
    b1 = _unit([0, 1, 0, 0, 0])
    b2 = _unit([0.05, 0.97, 0.1, 0, 0])
    # Group C: sorting (different direction again)
    c1 = _unit([0, 0, 0, 1, 0])
    c2 = _unit([0.02, 0, 0.05, 0.98, 0])

    return np.stack([a1, a2, a3, b1, b2, c1, c2])


# ─────────────────────────────────────────────────────────────────
# cluster_embeddings unit tests
# ─────────────────────────────────────────────────────────────────

class TestClusterEmbeddings:
    def test_returns_array_of_correct_length(self):
        embs = _make_embeddings()
        labels = cluster_embeddings(embs, threshold=0.3)
        assert len(labels) == len(embs)

    def test_three_distinct_groups_found(self):
        embs = _make_embeddings()
        labels = cluster_embeddings(embs, threshold=0.15)
        n_clusters = len(np.unique(labels))
        assert n_clusters == 3

    def test_same_direction_embeddings_cluster_together(self):
        """The first 3 embeddings (group A) should all share the same label."""
        embs = _make_embeddings()
        labels = cluster_embeddings(embs, threshold=0.15)
        assert labels[0] == labels[1] == labels[2]

    def test_different_groups_have_different_labels(self):
        embs = _make_embeddings()
        labels = cluster_embeddings(embs, threshold=0.15)
        # Group A [0,1,2] vs Group B [3,4] vs Group C [5,6]
        assert labels[0] != labels[3]
        assert labels[0] != labels[5]
        assert labels[3] != labels[5]

    def test_high_threshold_merges_all(self):
        embs = _make_embeddings()
        # Threshold of 1.0 = max cosine distance → all merged into 1
        labels = cluster_embeddings(embs, threshold=1.0)
        assert len(np.unique(labels)) == 1

    def test_very_low_threshold_all_singletons(self):
        embs = _make_embeddings()
        # Threshold of 0.0 → no two embeddings are identical → all singletons
        labels = cluster_embeddings(embs, threshold=0.0)
        assert len(np.unique(labels)) == len(embs)

    def test_single_item_returns_label_zero(self):
        emb = _unit([1, 0, 0, 0, 0]).reshape(1, -1)
        labels = cluster_embeddings(emb, threshold=0.3)
        assert labels.tolist() == [0]

    def test_empty_input_returns_empty(self):
        labels = cluster_embeddings(np.empty((0, 5)), threshold=0.3)
        assert len(labels) == 0


# ─────────────────────────────────────────────────────────────────
# cluster_tfidf unit tests
# ─────────────────────────────────────────────────────────────────

class TestClusterTFIDF:
    # TF-IDF clusters on shared vocabulary, NOT semantic meaning.
    # Questions that share the same keywords will cluster; paraphrased
    # questions with different words will not — that is embeddings' job.
    # These questions are chosen to have strong lexical overlap.
    STACK_QUESTIONS = [
        "Explain the stack data structure with push pop and peek operations.",
        "Describe the stack data structure and explain push pop peek.",
        "What is a stack? Explain stack operations push pop and peek with examples.",
    ]
    GRAPH_QUESTIONS = [
        "Explain BFS breadth first search and DFS depth first search graph traversal.",
        "Describe BFS and DFS traversal of a graph with examples.",
    ]
    ALL_QUESTIONS = STACK_QUESTIONS + GRAPH_QUESTIONS

    def test_returns_correct_length(self):
        labels = cluster_tfidf(self.ALL_QUESTIONS, threshold=0.5)
        assert len(labels) == len(self.ALL_QUESTIONS)

    def test_stack_questions_cluster_together(self):
        labels = cluster_tfidf(self.ALL_QUESTIONS, threshold=0.65)
        # First 3 (stack) should share a label
        assert labels[0] == labels[1] == labels[2]

    def test_stack_and_graph_have_different_labels(self):
        labels = cluster_tfidf(self.ALL_QUESTIONS, threshold=0.5)
        assert labels[0] != labels[3]

    def test_single_text(self):
        labels = cluster_tfidf(["What is a queue?"], threshold=0.3)
        assert labels.tolist() == [0]

    def test_empty_input(self):
        labels = cluster_tfidf([], threshold=0.3)
        assert len(labels) == 0


# ─────────────────────────────────────────────────────────────────
# assign_clusters (DataFrame integration)
# ─────────────────────────────────────────────────────────────────

class TestAssignClusters:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "question_text": [
                "Explain the stack and its push pop operations.",
                "Describe how a stack works using LIFO principle.",
                "What is a graph? Explain BFS and DFS.",
                "Explain breadth first and depth first traversal of graphs.",
            ],
            "year": [2021, 2022, 2021, 2022],
            "marks": [10.0, 10.0, 10.0, 10.0],
            "source_file": ["a.pdf", "b.pdf", "a.pdf", "b.pdf"],
            "question_no": ["Q1", "Q1", "Q2", "Q2"],
        })

    def test_cluster_id_column_added(self):
        df = self._make_df()
        embs = _make_embeddings()[:4]  # reuse 4 synthetic embeddings
        result = assign_clusters(df, embs, threshold=DEFAULT_THRESHOLD, method="embedding")
        assert "cluster_id" in result.columns

    def test_original_df_not_mutated(self):
        df = self._make_df()
        embs = _make_embeddings()[:4]
        assign_clusters(df, embs, threshold=DEFAULT_THRESHOLD, method="embedding")
        assert "cluster_id" not in df.columns  # original untouched

    def test_tfidf_method_works(self):
        df = self._make_df()
        result = assign_clusters(df, np.empty((4, 5)), threshold=0.5, method="tfidf")
        assert "cluster_id" in result.columns
        assert len(result) == len(df)

    def test_cluster_ids_are_integers(self):
        df = self._make_df()
        embs = _make_embeddings()[:4]
        result = assign_clusters(df, embs, threshold=DEFAULT_THRESHOLD, method="embedding")
        assert result["cluster_id"].dtype in [np.int32, np.int64, int]
