# PaperPulse 📚 — Exam Question Miner

> Upload past exam papers → discover repeated topics → get a study plan.

A small, resume-worthy ML project that helps students study smarter by mining
patterns from 5–8 years of past exam papers for any subject.

---

## Pipeline Overview

```
PDFs
 │
 ▼
[1] extractor.py  ──→  page text + year per file
 │
 ▼
[2] splitter.py   ──→  question DataFrame
                        (question_text, year, marks, source_file, question_no)
 │
 ▼
[3] embedder.py   ──→  embedding matrix  (all-MiniLM-L6-v2, cached to disk)
 │
 ▼
[4] clusterer.py  ──→  cluster_id per question  (AgglomerativeClustering)
 │
 ▼
[5] labeler.py    ──→  cluster_label per cluster (medoid phrase + keywords)
 │
 ▼
[6] ranker.py     ──→  priority-ranked list of topics + study planner
 │
 ▼
[7] app.py        ──→  Streamlit UI
```

---

## Why Each Step Exists

### Why sentence embeddings over keyword matching?

Two exam questions can cover the same topic with zero shared vocabulary:

> "Explain the working of a stack." vs "Describe the LIFO data structure."

TF-IDF assigns these a cosine similarity near 0 because they share no keywords.
`all-MiniLM-L6-v2` was fine-tuned for semantic similarity and correctly scores
these as near-identical.  The TF-IDF baseline is kept for comparison (see
`evaluation/evaluate.py`).

### Why a distance threshold instead of a fixed number of clusters (k)?

Fixing k forces the user to guess how many topics exist in advance — impossible
for an unfamiliar exam.  A distance threshold says "merge questions if they are
within X cosine distance of each other," letting the number of clusters emerge
naturally from the data.  This is exposed as a slider in the UI so students can
tune it for their subject.

### Why count distinct years instead of raw question occurrences?

A paper that prints the same question twice in one year should not rank higher
than a topic that appeared in three different years.  Distinct year count
measures *consistent exam relevance*, not formatting accidents within one paper.

### Why average linkage?

- **Single linkage** chains: one bridge question pulls two unrelated groups
  together.
- **Complete linkage** is outlier-sensitive: one far question inflates the
  cluster diameter.
- **Average linkage (UPGMA)** balances both — it is the standard choice for
  clustering short text where lengths vary.

---

## Priority Score Formula

```
Priority = w_freq × Frequency + w_rec × Recency + w_marks × MarksScore

where:
  Frequency   = distinct_years_appeared / total_years_in_dataset
  Recency     = Σ year_weight_i / Σ all_year_weights
                (year_weight = year − min_year + 1; recent years weigh more)
  MarksScore  = avg_marks / max_avg_marks_in_dataset
                (unknown marks → NEUTRAL_MARKS_DEFAULT = 5.0)

Default weights (ranker.py, change in one place):
  w_freq  = 0.5
  w_rec   = 0.3
  w_marks = 0.2
```

---

## Alternate-Year Pattern Hint

If a topic appeared in alternating years (gaps of 2) **and** did not appear in
the most recent exam year, the UI shows:

> ⚡ Pattern hint: this topic appeared in alternating years (2019, 2021, 2023)
> and was absent in 2024. This is a pattern hint only — not a prediction.

This is clearly labelled as a hint.  The model makes no predictions.

---

## Evaluation Results

See [`docs/results.md`](docs/results.md) for real measured precision, recall,
and F1 scores comparing TF-IDF vs sentence embeddings at multiple thresholds.

> **The default threshold (0.35) was chosen from these evaluation results.**
> Run `python -m evaluation.evaluate` to reproduce them.

---

## Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| Scanned PDFs | Pages with no selectable text are skipped | Install Tesseract + pytesseract for OCR |
| Formatting differences between universities | Question-boundary regex may miss some formats | Adjust regex in `splitter.py` |
| Subjective similarity threshold | 0.35 suits "Data Structures"-style courses; adjust the slider | Tune with `evaluation/evaluate.py` on your papers |
| Sub-part merging | Sub-parts joined into the parent question; very long sub-parts may dilute the embedding | See `splitter.py` design notes |
| No language support beyond English | TF-IDF stop words and tokenisation are English-only | Replace with multilingual model |

---

## Project Structure

```
exam-question-miner/
├── paperpulse/
│   ├── app.py                 # Streamlit UI
│   └── core/
│       ├── extractor.py       # PDF → page text
│       ├── splitter.py        # text → question DataFrame
│       ├── embedder.py        # questions → embeddings (cached)
│       ├── clusterer.py       # AgglomerativeClustering + TF-IDF baseline
│       ├── labeler.py         # medoid phrase + TF-IDF keywords
│       ├── ranker.py          # priority scoring + study planner
│       └── pipeline.py        # orchestrator
├── evaluation/
│   ├── label_pairs.csv        # hand-labelled question pairs
│   └── evaluate.py            # P/R/F1 benchmark
├── tests/
│   ├── test_splitter.py
│   ├── test_ranker.py
│   └── test_clusterer.py
├── sample_data/
│   └── generate_sample_data.py
├── docs/
│   └── results.md
└── requirements.txt
```

---

## Setup & Run

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd exam-question-miner

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate sample data (optional, for the demo)
python sample_data/generate_sample_data.py

# 5. Run the app
streamlit run paperpulse/app.py
```

### Optional: OCR for scanned PDFs

1. Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add it to PATH.
2. Uncomment `pytesseract` and `Pillow` in `requirements.txt`.
3. `pip install pytesseract Pillow`

### Run tests

```bash
pytest tests/ -v
```

### Run evaluation

```bash
python -m evaluation.evaluate
```

Results are saved to `docs/results.md`.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| PDF extraction | pdfplumber, PyMuPDF |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Clustering | scikit-learn AgglomerativeClustering |
| Data | pandas, numpy |
| UI | Streamlit |
| Tests | pytest |
| OCR (optional) | pytesseract + Tesseract |

---

## Contributing

1. Add more question-number formats to `splitter.py` if your university uses
   different conventions.
2. Expand `evaluation/label_pairs.csv` with real pairs from your subject for
   better threshold calibration.
3. Adjust scoring weights in `ranker.py` (documented at the top of the file).