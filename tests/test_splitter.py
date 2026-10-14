"""Tests for paperpulse.core.splitter — covers many question-numbering formats."""

from __future__ import annotations

import pandas as pd
import pytest

from paperpulse.core.extractor import PageRecord
from paperpulse.core.splitter import (
    _extract_marks,
    _is_boilerplate,
    split_page,
    split_results,
)
from paperpulse.core.extractor import ExtractionResult


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_page(text: str, year: int = 2023, source: str = "test.pdf") -> PageRecord:
    return PageRecord(source_file=source, year=year, page_number=1, text=text)


def _make_result(text: str, year: int = 2023) -> ExtractionResult:
    page = _make_page(text, year)
    er = ExtractionResult(source_file="test.pdf", year=year, pages=[page])
    return er


# ─────────────────────────────────────────────────────────────────
# Marks extraction
# ─────────────────────────────────────────────────────────────────

class TestExtractMarks:
    def test_bracket_marks(self):
        assert _extract_marks("Explain recursion. [5]") == 5.0

    def test_bracket_with_word(self):
        assert _extract_marks("Write an algorithm. [10 marks]") == 10.0

    def test_parens_marks(self):
        assert _extract_marks("Describe BFS. (5 marks)") == 5.0

    def test_M_suffix(self):
        assert _extract_marks("What is a stack? 7M") == 7.0

    def test_float_marks(self):
        assert _extract_marks("Explain the heap. [2.5 Marks]") == 2.5

    def test_no_marks(self):
        assert _extract_marks("What is a queue?") is None

    def test_returns_last_match(self):
        # If two mark patterns exist, return the last
        result = _extract_marks("Write [5] programs. [10]")
        assert result == 10.0


# ─────────────────────────────────────────────────────────────────
# Boilerplate detection
# ─────────────────────────────────────────────────────────────────

class TestBoilerplate:
    def test_standalone_page_number(self):
        assert _is_boilerplate("3")

    def test_page_n_of_m(self):
        assert _is_boilerplate("Page 2 of 10")

    def test_attempt_any(self):
        assert _is_boilerplate("Attempt any THREE of the following")

    def test_time_line(self):
        assert _is_boilerplate("Time: 3 Hours")

    def test_max_marks(self):
        assert _is_boilerplate("Maximum Marks: 100")

    def test_real_question_not_boilerplate(self):
        assert not _is_boilerplate("Q.1 Explain the concept of recursion.")

    def test_roll_no(self):
        assert _is_boilerplate("Roll No: ___________")


# ─────────────────────────────────────────────────────────────────
# Question splitting — format variations
# ─────────────────────────────────────────────────────────────────

class TestSplitPage:
    def test_q_dot_format(self):
        text = (
            "Q.1 Explain the stack data structure.\n"
            "Q.2 What is a queue?\n"
            "Q.3 Describe binary search."
        )
        records = split_page(_make_page(text))
        assert len(records) == 3
        assert records[0]["question_no"] == "Q1"
        assert records[1]["question_no"] == "Q2"

    def test_numbered_dot_format(self):
        text = (
            "1. What is a linked list?\n"
            "2. Explain hashing.\n"
            "3. Describe AVL trees."
        )
        records = split_page(_make_page(text))
        assert len(records) == 3
        assert records[0]["question_no"] == "1"

    def test_numbered_paren_format(self):
        text = (
            "1) Define recursion.\n"
            "2) Explain quick sort.\n"
        )
        records = split_page(_make_page(text))
        assert len(records) == 2

    def test_Q_capital_no_dot(self):
        text = "Q1 Explain BFS.\nQ2 Explain DFS."
        records = split_page(_make_page(text))
        assert len(records) == 2

    def test_sub_parts_merged_into_parent(self):
        """Sub-parts (a), (b) must be merged into the parent question."""
        text = (
            "Q.1 Write notes on the following:\n"
            "(a) Stack operations\n"
            "(b) Queue applications\n"
            "Q.2 Explain hashing.\n"
        )
        records = split_page(_make_page(text))
        # Q.1 + its sub-parts = 1 record, Q.2 = 1 record
        assert len(records) == 2
        combined_text = records[0]["question_text"]
        assert "Stack operations" in combined_text
        assert "Queue applications" in combined_text

    def test_lowercase_sub_part_merged(self):
        text = (
            "Q.3 Explain sorting:\n"
            "a) Bubble Sort\n"
            "b) Merge Sort\n"
            "Q.4 What is BFS?\n"
        )
        records = split_page(_make_page(text))
        assert len(records) == 2
        assert "Bubble Sort" in records[0]["question_text"]

    def test_boilerplate_filtered_before_questions(self):
        text = (
            "University of Technology\n"
            "Time: 3 Hours     Maximum Marks: 100\n"
            "Attempt any FIVE questions.\n"
            "Q.1 Explain recursion.\n"
            "Q.2 What is a heap?\n"
        )
        records = split_page(_make_page(text))
        assert len(records) == 2

    def test_marks_extracted(self):
        text = "Q.1 Explain the stack. [10]\nQ.2 What is hashing? [5]"
        records = split_page(_make_page(text))
        assert records[0]["marks"] == 10.0
        assert records[1]["marks"] == 5.0

    def test_marks_none_when_absent(self):
        text = "Q.1 Explain the stack.\nQ.2 What is hashing?"
        records = split_page(_make_page(text))
        assert records[0]["marks"] is None

    def test_year_preserved(self):
        text = "Q.1 Explain recursion."
        records = split_page(_make_page(text, year=2021))
        assert records[0]["year"] == 2021

    def test_source_file_preserved(self):
        text = "Q.1 Explain recursion."
        records = split_page(_make_page(text, source="myexam.pdf"))
        assert records[0]["source_file"] == "myexam.pdf"

    def test_empty_page_returns_empty_list(self):
        records = split_page(_make_page(""))
        assert records == []

    def test_only_boilerplate_returns_empty(self):
        text = "Time: 3 Hours\nMaximum Marks: 100\nAttempt any FIVE."
        records = split_page(_make_page(text))
        assert records == []

    def test_roman_numeral_subpart(self):
        """(i), (ii) sub-parts should be merged."""
        text = (
            "Q.2 Explain the following:\n"
            "(i) Inorder traversal\n"
            "(ii) Postorder traversal\n"
            "Q.3 What is a heap?\n"
        )
        records = split_page(_make_page(text))
        assert len(records) == 2
        assert "Inorder traversal" in records[0]["question_text"]

    def test_multi_digit_question_number(self):
        text = "Q.10 Explain heap sort.\nQ.11 Describe quick sort."
        records = split_page(_make_page(text))
        assert len(records) == 2
        assert records[0]["question_no"] == "Q10"

    def test_M_suffix_marks(self):
        text = "Q.1 Describe the queue data structure. 10M\nQ.2 What is hashing? 5M"
        records = split_page(_make_page(text))
        assert records[0]["marks"] == 10.0
        assert records[1]["marks"] == 5.0


# ─────────────────────────────────────────────────────────────────
# split_results — DataFrame output
# ─────────────────────────────────────────────────────────────────

class TestSplitResults:
    def test_returns_dataframe(self):
        er = _make_result("Q.1 Explain hashing.\nQ.2 What is a stack?")
        df = split_results([er])
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        er = _make_result("Q.1 Explain hashing.")
        df = split_results([er])
        for col in ["question_text", "year", "marks", "source_file", "question_no"]:
            assert col in df.columns

    def test_multiple_files_combined(self):
        er1 = _make_result("Q.1 Explain hashing.\nQ.2 What is a stack?", year=2022)
        er2 = _make_result("Q.1 Describe recursion.\nQ.2 Explain BFS.", year=2023)
        df = split_results([er1, er2])
        assert len(df) == 4
        assert set(df["year"].unique()) == {2022, 2023}

    def test_empty_results_returns_empty_df(self):
        df = split_results([])
        assert df.empty
        assert "question_text" in df.columns

    def test_noise_rows_filtered(self):
        """Very short extracted texts (≤10 chars) should be filtered out."""
        er = _make_result("Q.1 OK\nQ.2 What is a binary search tree and how does it work?")
        df = split_results([er])
        # "Q.1 OK" -> "OK" after stripping label -> ≤10 chars -> filtered
        texts = df["question_text"].tolist()
        assert all(len(t) > 10 for t in texts)
