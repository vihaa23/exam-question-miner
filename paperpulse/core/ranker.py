"""
ranker.py — Score and rank topic clusters by study priority.

Priority Score Formula
----------------------
Priority = w_freq * Frequency + w_rec * Recency + w_marks * MarksScore

Where:
  Frequency   = distinct_years_appeared / total_years_in_dataset
                (how often the topic appeared, as a fraction)

  Recency     = Σ (year_weight_i for each year topic appeared)
                year_weight_i = (year - min_year + 1) / Σ(1..n_years)
                (topics in recent years get more weight)

  MarksScore  = avg_marks / max_avg_marks_in_dataset
                (normalised average marks; unknown marks use NEUTRAL_MARKS_DEFAULT)

Default weights (configurable — change here to tune globally):
  w_freq  = 0.5   # half the weight on frequency
  w_rec   = 0.3   # 30% on recency
  w_marks = 0.2   # 20% on marks

Why distinct years (not raw question count)?
--------------------------------------------
A paper that repeats the same question 3 times in one year should not rank
higher than a topic that appeared in 3 different years.  Distinct year count
measures *consistent exam relevance*, not accidental repetition within one paper.

Alternate-year pattern hint
----------------------------
If a topic appeared in alternating years and did NOT appear in the most recent
year of the dataset, we flag it with "⚡ Pattern hint: skipped last year —
may appear next".  This is labelled clearly as a hint, never a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Configurable weights  (single place to tune)
# ─────────────────────────────────────────────────────────────────

WEIGHT_FREQUENCY: float = 0.5
WEIGHT_RECENCY: float = 0.3
WEIGHT_MARKS: float = 0.2

NEUTRAL_MARKS_DEFAULT: float = 5.0   # Used when marks are unknown

# ─────────────────────────────────────────────────────────────────
# Data container for a ranked cluster
# ─────────────────────────────────────────────────────────────────

@dataclass
class RankedCluster:
    """All ranking information for one question cluster."""
    cluster_id: int
    cluster_label: str
    distinct_years: int
    total_years: int
    years_list: list[int]
    avg_marks: Optional[float]
    priority_score: float
    frequency_score: float
    recency_score: float
    marks_score: float
    reason: str
    pattern_hint: Optional[str]
    questions: pd.DataFrame    # subset of question DataFrame for this cluster


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _recency_score(years_appeared: list[int], all_years: list[int]) -> float:
    """Compute a normalised recency score for a cluster.

    Years are weighted linearly (most recent year gets weight n, oldest gets 1).
    The score is the sum of weights for the years the cluster appeared in,
    normalised so a cluster that appeared in ALL years scores 1.0.
    """
    if not all_years or not years_appeared:
        return 0.0

    min_year = min(all_years)
    weights = {y: (y - min_year + 1) for y in all_years}
    total_weight = sum(weights.values()) or 1.0

    cluster_weight = sum(weights.get(y, 0) for y in years_appeared)
    return cluster_weight / total_weight


def _alternate_year_hint(
    years_appeared: list[int],
    all_years: list[int],
) -> Optional[str]:
    """Return a pattern hint string if the alternating-year pattern is detected.

    Detection criteria:
    1. The topic appeared in ≥ 3 years.
    2. Each consecutive pair of appearing years has a gap of exactly 2
       (alternating pattern) — at least half the gaps must be 2.
    3. The topic did NOT appear in the most recent year of the dataset.
    """
    if len(years_appeared) < 3 or not all_years:
        return None

    max_year = max(all_years)
    if max_year in years_appeared:
        return None   # Appeared in last year — no "skip" to hint about

    sorted_years = sorted(years_appeared)
    gaps = [sorted_years[i + 1] - sorted_years[i] for i in range(len(sorted_years) - 1)]
    alternating_gaps = sum(1 for g in gaps if g == 2)

    if alternating_gaps >= len(gaps) / 2:
        return (
            f"Pattern hint: this topic appeared in alternating years "
            f"({', '.join(str(y) for y in sorted_years)}) and was absent in {max_year}. "
            f"This is a pattern observation only, not a prediction."
        )
    return None


def _build_reason(
    cluster_label: str,
    distinct_years: int,
    total_years: int,
    years_list: list[int],
    avg_marks: Optional[float],
) -> str:
    """Build a human-readable reason string for a cluster's priority."""
    parts: list[str] = []
    if total_years > 0:
        parts.append(f"Appeared in {distinct_years} of {total_years} years")
    else:
        parts.append(f"Appeared in {distinct_years} papers")
    if years_list:
        last_year = max(years_list)
        parts.append(f"last seen in {last_year}")
    if avg_marks is not None:
        parts.append(f"usually {avg_marks:.0f} marks")
    else:
        parts.append("marks unspecified")
    return ", ".join(parts)


# ─────────────────────────────────────────────────────────────────
# Main ranking function
# ─────────────────────────────────────────────────────────────────

def rank_clusters(
    df: pd.DataFrame,
    weight_frequency: float = WEIGHT_FREQUENCY,
    weight_recency: float = WEIGHT_RECENCY,
    weight_marks: float = WEIGHT_MARKS,
    neutral_marks: float = NEUTRAL_MARKS_DEFAULT,
) -> list[RankedCluster]:
    """Score and rank all clusters by study priority.

    Parameters
    ----------
    df:
        Question DataFrame with columns: question_text, year, marks,
        source_file, question_no, cluster_id, cluster_label.
    weight_frequency, weight_recency, weight_marks:
        Scoring weights (must sum to 1.0 for a 0-1 score range).
    neutral_marks:
        Default marks value used when marks are unknown.

    Returns
    -------
    List of RankedCluster objects, sorted by priority_score descending.
    """
    all_years_series = df["year"].dropna().unique()
    all_years: list[int] = sorted(int(y) for y in all_years_series)
    total_years = len(all_years)

    # Compute max *per-cluster* average marks for normalisation.
    # Using per-cluster avg (not global mean) ensures marks_score is always in [0, 1].
    per_cluster_avg: list[float] = []
    for _, grp in df.groupby("cluster_id"):
        known = grp["marks"].dropna()
        if not known.empty:
            per_cluster_avg.append(float(known.mean()))
        else:
            per_cluster_avg.append(neutral_marks)
    max_avg_marks = max(per_cluster_avg) if per_cluster_avg else neutral_marks
    max_avg_marks = max(max_avg_marks, 1.0)  # safety guard

    ranked: list[RankedCluster] = []

    for cluster_id, group in df.groupby("cluster_id"):
        cluster_label = group["cluster_label"].iloc[0]

        # ── Distinct years ──────────────────────────────────────────
        years_in_cluster = sorted(
            int(y) for y in group["year"].dropna().unique()
        )
        distinct_years = len(years_in_cluster)

        # ── Average marks ────────────────────────────────────────────
        known_marks = group["marks"].dropna()
        avg_marks: Optional[float] = (
            float(known_marks.mean()) if not known_marks.empty else None
        )
        marks_for_score = avg_marks if avg_marks is not None else neutral_marks

        # ── Component scores ─────────────────────────────────────────
        freq_score = distinct_years / total_years if total_years > 0 else 0.0
        rec_score = _recency_score(years_in_cluster, all_years)
        marks_score = marks_for_score / max_avg_marks

        # ── Weighted priority ─────────────────────────────────────────
        priority = (
            weight_frequency * freq_score
            + weight_recency * rec_score
            + weight_marks * marks_score
        )

        reason = _build_reason(
            cluster_label, distinct_years, total_years, years_in_cluster, avg_marks
        )
        hint = _alternate_year_hint(years_in_cluster, all_years)

        ranked.append(
            RankedCluster(
                cluster_id=int(cluster_id),
                cluster_label=cluster_label,
                distinct_years=distinct_years,
                total_years=total_years,
                years_list=years_in_cluster,
                avg_marks=avg_marks,
                priority_score=round(priority, 4),
                frequency_score=round(freq_score, 4),
                recency_score=round(rec_score, 4),
                marks_score=round(marks_score, 4),
                reason=reason,
                pattern_hint=hint,
                questions=group.reset_index(drop=True),
            )
        )

    ranked.sort(key=lambda c: c.priority_score, reverse=True)
    return ranked


# ─────────────────────────────────────────────────────────────────
# Study planner
# ─────────────────────────────────────────────────────────────────

def generate_study_plan(
    ranked: list[RankedCluster],
    available_days: int,
    topics_per_day: int = 2,
) -> pd.DataFrame:
    """Generate an ordered study plan given available days.

    Parameters
    ----------
    ranked:
        Output of rank_clusters(), already sorted by priority.
    available_days:
        Number of study days available.
    topics_per_day:
        Estimated number of topics a student can cover per day.

    Returns
    -------
    DataFrame with columns: rank, cluster_label, priority_score,
    years_appeared, avg_marks, reason, pattern_hint.
    """
    max_topics = available_days * topics_per_day
    top = ranked[:max_topics]

    rows = []
    for i, c in enumerate(top, start=1):
        rows.append(
            {
                "Rank": i,
                "Topic": c.cluster_label,
                "Priority Score": c.priority_score,
                "Years Appeared": len(c.years_list),
                "Years": ", ".join(str(y) for y in c.years_list),
                "Avg Marks": f"{c.avg_marks:.0f}" if c.avg_marks is not None else "Unknown",
                "Why Study This": c.reason,
                "Pattern Hint": c.pattern_hint or "",
            }
        )

    return pd.DataFrame(rows)


def ranked_to_dataframe(ranked: list[RankedCluster]) -> pd.DataFrame:
    """Convert the ranked cluster list to a flat summary DataFrame for display/export."""
    rows = []
    for c in ranked:
        rows.append(
            {
                "cluster_id": c.cluster_id,
                "Topic": c.cluster_label,
                "Priority Score": c.priority_score,
                "Frequency Score": c.frequency_score,
                "Recency Score": c.recency_score,
                "Marks Score": c.marks_score,
                "Years Appeared": c.distinct_years,
                "Years": ", ".join(str(y) for y in c.years_list),
                "Avg Marks": c.avg_marks,
                "Why": c.reason,
                "Pattern Hint": c.pattern_hint or "",
            }
        )
    return pd.DataFrame(rows)
