"""
app.py — PaperPulse Streamlit UI (redesigned).

Design goals:
- Looks like a real tool, not auto-generated
- Clean card-based layout with clear visual hierarchy
- Consistent colour system: deep navy header, white cards, slate accents
- Still plain and functional — no decorative clutter
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from paperpulse.core.clusterer import DEFAULT_THRESHOLD
from paperpulse.core.extractor import _HAS_TESSERACT, parse_year_from_filename
from paperpulse.core.pipeline import PipelineResult, run_pipeline
from paperpulse.core.ranker import generate_study_plan, ranked_to_dataframe

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperPulse: Exam Question Miner",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import font ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ─────────────────────────────── */
.hero {
    background: #0f172a;
    border-radius: 12px;
    padding: 32px 36px 28px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 18px;
}
.hero-icon { font-size: 42px; line-height: 1; }
.hero-title {
    color: #ffffff;
    font-size: 28px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #94a3b8;
    font-size: 14px;
    margin: 6px 0 0;
    line-height: 1.5;
}

/* ── Section card ─────────────────────────────── */
.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 15px;
    font-weight: 600;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.step-badge {
    background: #0f172a;
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* ── Metric cards ─────────────────────────────── */
.metric-row { display: flex; gap: 14px; margin-bottom: 24px; }
.metric-card {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.metric-value {
    font-size: 36px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 12px;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Priority pill ────────────────────────────── */
.priority-high { color: #dc2626; font-weight: 600; }
.priority-mid  { color: #d97706; font-weight: 600; }
.priority-low  { color: #16a34a; font-weight: 600; }

/* ── Hint banner ──────────────────────────────── */
.hint-banner {
    background: #fefce8;
    border: 1px solid #fde047;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: #713f12;
    margin-bottom: 12px;
    line-height: 1.5;
}

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
.sidebar-heading {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    margin: 20px 0 10px;
}

/* ── Warning / info badges ────────────────────── */
.badge-warn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    color: #9a3412;
    margin-bottom: 12px;
}
.badge-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    color: #166534;
    margin-bottom: 12px;
}

/* ── Table tweaks ─────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
thead tr th { background: #f1f5f9 !important; font-size: 12px !important; }

/* ── Buttons ──────────────────────────────────── */
button[kind="primary"] {
    background: #0f172a !important;
    border-color: #0f172a !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background: #1e293b !important;
    border-color: #1e293b !important;
}

/* ── Expander headers ─────────────────────────── */
[data-testid="stExpander"] summary {
    font-size: 14px;
    font-weight: 500;
    color: #0f172a;
}

/* ── Divider ──────────────────────────────────── */
.pp-divider {
    height: 1px;
    background: #e2e8f0;
    margin: 28px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "result": None,
        "pdf_paths": [],
        "year_map": {},
        "threshold": DEFAULT_THRESHOLD,
        "sample_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────
# Sample data helpers
# ─────────────────────────────────────────────────────────────────

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"

def _get_sample_txt_files() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(SAMPLE_DIR.glob("*.txt"))

def _sample_data_available() -> bool:
    return len(_get_sample_txt_files()) > 0

def _load_sample_as_text_extractions(txt_files: list[Path]) -> list:
    from paperpulse.core.extractor import ExtractionResult, PageRecord
    results = []
    for txt_path in txt_files:
        year = parse_year_from_filename(txt_path)
        text = txt_path.read_text(encoding="utf-8")
        page = PageRecord(source_file=str(txt_path), year=year, page_number=1, text=text, ocr_used=False)
        er = ExtractionResult(source_file=str(txt_path), year=year, pages=[page])
        results.append(er)
    return results

# ─────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────

def _run(pdf_paths: list[str], year_map: dict[str, int], threshold: float, sample_mode: bool) -> PipelineResult:
    from paperpulse.core.splitter import split_results
    from paperpulse.core.embedder import Embedder
    from paperpulse.core.clusterer import assign_clusters
    from paperpulse.core.labeler import label_clusters
    from paperpulse.core.ranker import rank_clusters

    prog = st.progress(0, text="Starting…")

    def _cb(msg: str, frac: float) -> None:
        prog.progress(int(frac * 100), text=msg)

    if sample_mode:
        txt_files = _get_sample_txt_files()
        extraction_results = _load_sample_as_text_extractions(txt_files)
        result = PipelineResult()
        result.extraction_results = extraction_results

        _cb("Splitting pages into questions…", 0.2)
        questions_df = split_results(extraction_results)
        result.questions_df = questions_df

        if questions_df.empty:
            _cb("No questions found.", 1.0)
            return result

        _cb("Computing semantic embeddings…", 0.4)
        embedder = Embedder()
        embeddings = embedder.encode_dataframe(questions_df)
        result.embeddings = embeddings

        _cb("Clustering similar questions…", 0.6)
        clustered_df = assign_clusters(questions_df, embeddings=embeddings, threshold=threshold)

        _cb("Labelling clusters…", 0.75)
        labelled_df = label_clusters(clustered_df, embeddings=embeddings)
        result.questions_df = labelled_df

        _cb("Ranking by priority…", 0.9)
        result.ranked_clusters = rank_clusters(labelled_df)

        _cb("Done!", 1.0)
        return result
    else:
        return run_pipeline(
            pdf_paths=pdf_paths,
            year_overrides=year_map,
            threshold=threshold,
            progress_callback=_cb,
        )

def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()

def _priority_label(score: float) -> str:
    if score >= 0.65:
        return f'<span class="priority-high">● High</span>'
    elif score >= 0.35:
        return f'<span class="priority-mid">● Medium</span>'
    else:
        return f'<span class="priority-low">● Low</span>'


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-heading">Analysis Settings</p>', unsafe_allow_html=True)

    threshold = st.slider(
        "Topic grouping sensitivity",
        min_value=0.10, max_value=0.70,
        value=float(st.session_state["threshold"]),
        step=0.05,
        help="Lower = more groups (stricter matching). Higher = fewer, broader groups. Default 0.35 is recommended.",
    )
    st.session_state["threshold"] = threshold

    st.markdown('<p class="sidebar-heading">Priority Score Formula</p>', unsafe_allow_html=True)
    st.code(
        "Priority =\n  0.5 × Frequency\n+ 0.3 × Recency\n+ 0.2 × Avg Marks",
        language=None,
    )

    st.markdown('<p class="sidebar-heading">System</p>', unsafe_allow_html=True)
    if not _HAS_TESSERACT:
        st.markdown(
            '<div class="badge-warn">⚠ Tesseract not installed: scanned pages will be skipped</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="badge-ok">✓ Tesseract OCR available</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-icon">📚</div>
  <div>
    <p class="hero-title">PaperPulse</p>
    <p class="hero-sub">Upload past exam papers, discover recurring question topics, and generate an exam preparation schedule.<br>
    Optimised for standard PDF exam papers. Scanned pages use Tesseract OCR fallback when installed.</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Step 1: Upload
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-card">
  <p class="section-title"><span class="step-badge">01</span>Upload Exam Papers</p>
</div>
""", unsafe_allow_html=True)

col_upload, col_gap, col_sample = st.columns([5, 0.3, 2])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload 5 to 8 years of past exam PDFs for the same subject",
        type=["pdf"],
        accept_multiple_files=True,
        help="Name files with the year (for example Physics_2022.pdf) for automatic year detection.",
        label_visibility="visible",
    )

with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    if _sample_data_available():
        if st.button("Use sample papers", use_container_width=True):
            st.session_state["sample_mode"] = True
            st.session_state["result"] = None
            st.rerun()
    else:
        st.info("Run `sample_data/generate_sample_data.py` to create demo papers.")

# Year confirmation
upload_dir = Path(__file__).parent.parent / ".cache" / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)
pdf_paths: list[str] = []
year_map: dict[str, int] = {}

if uploaded_files and not st.session_state.get("sample_mode"):
    st.session_state["sample_mode"] = False

    st.markdown("**Confirm the exam year for each file:**")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, uf in enumerate(uploaded_files):
        dest = upload_dir / uf.name
        dest.write_bytes(uf.getvalue())
        pdf_paths.append(str(dest))
        auto_year = parse_year_from_filename(uf.name)
        with cols[i % 4]:
            confirmed_year = st.number_input(
                label=uf.name,
                min_value=1990, max_value=2099,
                value=auto_year or 2024,
                step=1,
                key=f"year_{uf.name}",
            )
            year_map[uf.name] = int(confirmed_year)

    st.session_state["pdf_paths"] = pdf_paths
    st.session_state["year_map"] = year_map

# Show sample mode banner
if st.session_state.get("sample_mode"):
    st.info(
        "📄 **Sample mode active**: using 6 years of Data Structures exam papers (2019 to 2024). "
        "Click Analyse Papers below to see results, or upload your own PDFs above.",
        icon=None,
    )

# ─────────────────────────────────────────────────────────────────
# Step 2: Analyse
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-card">
  <p class="section-title"><span class="step-badge">02</span>Analyse</p>
</div>
""", unsafe_allow_html=True)

sample_mode = st.session_state.get("sample_mode", False)
has_input = bool(uploaded_files) or sample_mode

btn_col, reset_col, _ = st.columns([2, 1, 5])
with btn_col:
    analyze_btn = st.button(
        "Analyse Papers",
        disabled=not has_input,
        type="primary",
        use_container_width=True,
    )
with reset_col:
    if st.session_state.get("result") or sample_mode:
        if st.button("Reset", use_container_width=True):
            st.session_state["result"] = None
            st.session_state["sample_mode"] = False
            st.rerun()

if analyze_btn and has_input:
    with st.spinner("Processing… downloading model on first run (~80 MB, one-time only)."):
        try:
            result = _run(
                pdf_paths=st.session_state.get("pdf_paths", []),
                year_map=st.session_state.get("year_map", {}),
                threshold=st.session_state["threshold"],
                sample_mode=sample_mode,
            )
            st.session_state["result"] = result
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

# ─────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────

result: PipelineResult | None = st.session_state.get("result")

if result is not None:

    # Extraction warnings
    if result.extraction_errors:
        with st.expander(f"⚠ {len(result.extraction_errors)} extraction warning(s)", expanded=False):
            for err in result.extraction_errors:
                st.warning(err)

    # ── Summary metrics ─────────────────────────────────────────
    st.markdown('<div class="pp-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section-card">
  <p class="section-title">Results Overview</p>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{result.n_papers}</div>
  <div class="metric-label">Papers Analysed</div>
</div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{result.n_questions}</div>
  <div class="metric-label">Questions Extracted</div>
</div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{result.n_clusters}</div>
  <div class="metric-label">Topics Identified</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if result.n_questions == 0:
        st.warning(
            "No questions were extracted. Check that your PDFs contain selectable text. "
            "Scanned pages require Tesseract OCR (optional install)."
        )
        st.stop()

    # ── Priority table ───────────────────────────────────────────
    st.markdown("""
<div class="section-card">
  <p class="section-title">Repeated Topics and Study Priority</p>
</div>
""", unsafe_allow_html=True)
    st.caption("Topics are ranked by how often they appeared, how recently, and how many marks they typically carry.")

    summary_df = ranked_to_dataframe(result.ranked_clusters)
    display_df = summary_df[["Topic", "Priority Score", "Years Appeared", "Years", "Avg Marks", "Why", "Pattern Hint"]].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Rank"
    st.dataframe(display_df, use_container_width=True)

    dl1, dl2 = st.columns([1, 5])
    with dl1:
        st.download_button(
            "Download CSV",
            data=_to_csv_bytes(summary_df),
            file_name="paperpulse_results.csv",
            mime="text/csv",
        )

    # ── Topic expanders ──────────────────────────────────────────
    st.markdown('<div class="pp-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section-card">
  <p class="section-title">Question Variants by Topic</p>
</div>
""", unsafe_allow_html=True)
    st.caption("Expand a topic to see every variant across all years.")

    for i, cluster in enumerate(result.ranked_clusters, start=1):
        label_short = cluster.cluster_label[:90]
        years_str = f"{cluster.distinct_years} of {cluster.total_years} years"
        score_str = f"{cluster.priority_score:.2f}"

        with st.expander(f"#{i}: {label_short} ({years_str}, Priority: {score_str})"):
            if cluster.pattern_hint:
                st.markdown(f'<div class="hint-banner">{cluster.pattern_hint}</div>', unsafe_allow_html=True)

            q_display = cluster.questions[["year", "question_no", "question_text", "marks", "source_file"]].copy()
            q_display["source_file"] = q_display["source_file"].apply(lambda p: Path(p).name)
            q_display.columns = ["Year", "Q No.", "Question", "Marks", "Source"]
            q_display = q_display.sort_values("Year").reset_index(drop=True)
            st.dataframe(q_display, use_container_width=True)

    # ── Study planner ────────────────────────────────────────────
    st.markdown('<div class="pp-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="section-card">
  <p class="section-title">Study Planner</p>
</div>
""", unsafe_allow_html=True)
    st.caption("Tell us how many days you have and we'll show you the highest-priority topics to cover.")

    day_col, _ = st.columns([1, 4])
    with day_col:
        available_days = st.number_input("Days available", min_value=1, max_value=60, value=7, step=1)

    plan_df = generate_study_plan(result.ranked_clusters, available_days=int(available_days))

    if plan_df.empty:
        st.info("No topics to plan. Try increasing the number of days or loosening the grouping sensitivity.")
    else:
        st.dataframe(plan_df.set_index("Rank"), use_container_width=True)
        st.download_button(
            "⬇ Download study plan",
            data=_to_csv_bytes(plan_df),
            file_name="paperpulse_study_plan.csv",
            mime="text/csv",
        )

    # ── Raw data ─────────────────────────────────────────────────
    with st.expander("View / download all extracted questions"):
        raw_df = result.questions_df[
            ["question_text", "year", "marks", "source_file", "question_no", "cluster_id", "cluster_label"]
        ].copy()
        raw_df["source_file"] = raw_df["source_file"].apply(lambda p: Path(p).name)
        st.dataframe(raw_df, use_container_width=True)
        st.download_button(
            "⬇ Download all questions",
            data=_to_csv_bytes(raw_df),
            file_name="paperpulse_all_questions.csv",
            mime="text/csv",
        )
