"""Tests for paperpulse.core.ranker — scoring, hints, and study planner."""

from __future__ import annotations

import pandas as pd
import pytest

from paperpulse.core.ranker import (
    _alternate_year_hint,
    _recency_score,
    generate_study_plan,
    rank_clusters,
    ranked_to_dataframe,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal question DataFrame for ranker tests."""
    defaults = {"question_text": "Dummy question", "source_file": "test.pdf", "question_no": "Q1"}
    full_rows = []
    for row in rows:
        r = dict(defaults)
        r.update(row)
        full_rows.append(r)
    return pd.DataFrame(full_rows)


# ─────────────────────────────────────────────────────────────────
# Recency score unit tests
# ─────────────────────────────────────────────────────────────────

class TestRecencyScore:
    def test_appeared_in_all_years(self):
        all_years = [2019, 2020, 2021, 2022]
        score = _recency_score([2019, 2020, 2021, 2022], all_years)
        assert score == pytest.approx(1.0)

    def test_appeared_only_in_most_recent(self):
        all_years = [2019, 2020, 2021, 2022]
        # Weight 2022 = 4, 2019=1, 2020=2, 2021=3 → total = 10
        score = _recency_score([2022], all_years)
        assert score == pytest.approx(4 / 10)

    def test_appeared_only_in_oldest(self):
        all_years = [2019, 2020, 2021, 2022]
        score = _recency_score([2019], all_years)
        assert score == pytest.approx(1 / 10)

    def test_empty_years(self):
        assert _recency_score([], [2019, 2020]) == 0.0

    def test_single_year_dataset(self):
        # Only one year in dataset → weight = 1, total = 1
        score = _recency_score([2023], [2023])
        assert score == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────
# Alternate-year pattern hint
# ─────────────────────────────────────────────────────────────────

class TestAlternateYearHint:
    def test_alternating_pattern_skipped_last(self):
        # 2019, 2021, 2023 — skipped 2024
        hint = _alternate_year_hint([2019, 2021, 2023], [2019, 2020, 2021, 2022, 2023, 2024])
        assert hint is not None
        assert "Pattern hint" in hint
        assert "2024" in hint

    def test_no_hint_if_appeared_last_year(self):
        hint = _alternate_year_hint([2019, 2021, 2023, 2024], [2019, 2020, 2021, 2022, 2023, 2024])
        assert hint is None

    def test_no_hint_if_too_few_years(self):
        hint = _alternate_year_hint([2021, 2023], [2021, 2022, 2023, 2024])
        assert hint is None  # < 3 appearances

    def test_no_hint_if_not_alternating(self):
        # Three consecutive years — not alternating
        hint = _alternate_year_hint([2019, 2020, 2021], [2019, 2020, 2021, 2022, 2024])
        assert hint is None


# ─────────────────────────────────────────────────────────────────
# Priority scoring
# ─────────────────────────────────────────────────────────────────

class TestRankClusters:
    def _build_df(self) -> pd.DataFrame:
        """Build a 3-cluster DataFrame for scoring tests."""
        return _make_df([
            # Cluster 0: appeared every year (2019-2024), high marks
            {"cluster_id": 0, "cluster_label": "Stacks", "year": 2019, "marks": 10.0},
            {"cluster_id": 0, "cluster_label": "Stacks", "year": 2020, "marks": 10.0},
            {"cluster_id": 0, "cluster_label": "Stacks", "year": 2021, "marks": 10.0},
            {"cluster_id": 0, "cluster_label": "Stacks", "year": 2022, "marks": 10.0},
            {"cluster_id": 0, "cluster_label": "Stacks", "year": 2023, "marks": 10.0},
            {"cluster_id": 0, "cluster_label": "Stacks", "year": 2024, "marks": 10.0},
            # Cluster 1: appeared only in oldest year, low marks
            {"cluster_id": 1, "cluster_label": "Bubble Sort", "year": 2019, "marks": 5.0},
            # Cluster 2: appeared in most recent year only
            {"cluster_id": 2, "cluster_label": "Dynamic Programming", "year": 2024, "marks": 8.0},
        ])

    def test_high_frequency_ranks_first(self):
        df = self._build_df()
        ranked = rank_clusters(df)
        assert ranked[0].cluster_label == "Stacks"

    def test_all_clusters_returned(self):
        df = self._build_df()
        ranked = rank_clusters(df)
        assert len(ranked) == 3

    def test_scores_in_valid_range(self):
        df = self._build_df()
        ranked = rank_clusters(df)
        for c in ranked:
            assert 0.0 <= c.priority_score <= 1.0

    def test_oldest_only_ranks_last(self):
        df = self._build_df()
        ranked = rank_clusters(df)
        labels = [c.cluster_label for c in ranked]
        assert labels.index("Bubble Sort") > labels.index("Stacks")

    def test_distinct_years_counted_not_instances(self):
        """Duplicate questions in the same year should not inflate distinct_years."""
        df = _make_df([
            {"cluster_id": 0, "cluster_label": "Hashing", "year": 2022, "marks": 10.0},
            {"cluster_id": 0, "cluster_label": "Hashing", "year": 2022, "marks": 10.0},  # same year
            {"cluster_id": 0, "cluster_label": "Hashing", "year": 2023, "marks": 10.0},
        ])
        ranked = rank_clusters(df)
        assert ranked[0].distinct_years == 2   # NOT 3

    def test_unknown_marks_uses_neutral(self):
        """Clusters with unknown marks should still get a score."""
        df = _make_df([
            {"cluster_id": 0, "cluster_label": "AVL Trees", "year": 2023, "marks": None},
        ])
        ranked = rank_clusters(df)
        assert ranked[0].priority_score > 0

    def test_reason_string_populated(self):
        df = self._build_df()
        ranked = rank_clusters(df)
        for c in ranked:
            assert isinstance(c.reason, str)
            assert len(c.reason) > 0

    def test_ranked_to_dataframe_shape(self):
        df = self._build_df()
        ranked = rank_clusters(df)
        summary = ranked_to_dataframe(ranked)
        assert len(summary) == 3
        assert "Topic" in summary.columns
        assert "Priority Score" in summary.columns


# ─────────────────────────────────────────────────────────────────
# Study planner
# ─────────────────────────────────────────────────────────────────

class TestStudyPlanner:
    def _ranked(self):
        df = _make_df([
            {"cluster_id": i, "cluster_label": f"Topic {i}", "year": 2020 + i, "marks": 10.0}
            for i in range(6)
        ])
        return rank_clusters(df)

    def test_plan_respects_days(self):
        ranked = self._ranked()
        plan = generate_study_plan(ranked, available_days=2, topics_per_day=2)
        assert len(plan) <= 4

    def test_plan_ordered_by_priority(self):
        ranked = self._ranked()
        plan = generate_study_plan(ranked, available_days=3)
        scores = plan["Priority Score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_plan_columns_present(self):
        ranked = self._ranked()
        plan = generate_study_plan(ranked, available_days=3)
        for col in ["Rank", "Topic", "Priority Score", "Why Study This"]:
            assert col in plan.columns

    def test_plan_returns_dataframe(self):
        ranked = self._ranked()
        plan = generate_study_plan(ranked, available_days=1)
        assert isinstance(plan, pd.DataFrame)
