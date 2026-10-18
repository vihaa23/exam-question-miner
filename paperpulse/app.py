"""
app.py — PaperPulse: Exam Question Intelligence & Topic Miner.

Key Features:
- Multi-format ingestion: PDFs and exam paper images (PNG, JPG, JPEG, WEBP, BMP, TIFF).
- Tesseract OCR engine auto-detection with custom path configuration and live test diagnostic.
- Topic clustering with semantic sentence embeddings and same-paper collision constraints.
- Rich Interactive Visual Analytics (Altair charts):
    1. Multi-Year Topic Recurrence Timeline / Matrix
    2. Priority Score & Composite Factor Breakdown (Frequency, Recency, Marks)
    3. Exam Marks Weightage Distribution
- Interactive study plan generator and raw question vault with CSV exports.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

# Allow running directly or via module
sys.path.insert(0, str(Path(__file__).parent.parent))

from paperpulse.core.clusterer import DEFAULT_THRESHOLD, assign_clusters
from paperpulse.core.embedder import Embedder
from paperpulse.core.extractor import (
    IMAGE_EXTENSIONS,
    _HAS_TESSERACT,
    _TESSERACT_STATUS,
    extract_files,
    find_tesseract_binary,
    init_tesseract,
    ocr_image,
    parse_year_from_filename,
    set_tesseract_cmd,
)
from paperpulse.core.labeler import label_clusters
from paperpulse.core.pipeline import PipelineResult, run_pipeline
from paperpulse.core.ranker import RankedCluster, generate_study_plan, rank_clusters, ranked_to_dataframe
from paperpulse.core.splitter import split_results

# ─────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperPulse · Exam Question Miner",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom Professional CSS Styling
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Top Header Banner ─────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    padding: 26px 32px;
    margin-bottom: 24px;
    color: #ffffff;
    border: 1px solid #334155;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}
.app-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 12px;
}
.app-subtitle {
    font-size: 14px;
    color: #94a3b8;
    margin: 8px 0 0 0;
    line-height: 1.5;
}
.header-badges {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.header-badge {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #cbd5e1;
}

/* ── Card Container ───────────────────────────── */
.card-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}
.card-header-title {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #334155;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.step-num {
    background: #0f172a;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 12px;
}

/* ── Metric Card ──────────────────────────────── */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.kpi-val {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ── Status Pills ─────────────────────────────── */
.pill-high {
    background: #fee2e2;
    color: #991b1b;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}
.pill-mid {
    background: #fef3c7;
    color: #92400e;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}
.pill-low {
    background: #e2e8f0;
    color: #334155;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}

/* ── Pattern Hint Box ─────────────────────────── */
.hint-callout {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
    color: #14532d;
    margin-bottom: 12px;
    line-height: 1.5;
}

/* ── Year Tags ────────────────────────────────── */
.year-tag {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
}

/* ── Table & Dataframe ────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Primary Button ───────────────────────────── */
button[kind="primary"] {
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "result": None,
        "uploaded_file_paths": [],
        "year_map": {},
        "threshold": DEFAULT_THRESHOLD,
        "sample_mode": False,
        "tesseract_custom_path": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────
# Sample Data Helpers
# ─────────────────────────────────────────────────────────────────

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"

def _get_sample_txt_files() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(SAMPLE_DIR.glob("*.txt"))

def _sample_data_available() -> bool:
    return len(_get_sample_txt_files()) > 0


# ─────────────────────────────────────────────────────────────────
# Sidebar Settings & OCR Diagnostics
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Analysis Settings")

    threshold = st.slider(
        "Clustering distance sensitivity",
        min_value=0.10,
        max_value=0.70,
        value=float(st.session_state["threshold"]),
        step=0.02,
        help="Controls semantic similarity threshold. 0.44 is calibrated for accurate multi-year grouping with same-paper collision protection.",
    )
    st.session_state["threshold"] = threshold

    st.markdown("---")
    st.markdown("### Priority Score Formula")
    st.markdown("""
    $$\\text{Score} = 0.5 \\times F + 0.3 \\times R + 0.2 \\times M$$
    - **Frequency (50%)**: How many distinct exam years the topic appeared
    - **Recency (30%)**: Weighted presence towards recent exam years
    - **Marks (20%)**: Typical marks weightage for the question
    """)

    st.markdown("---")
    st.markdown("### OCR Engine (Tesseract)")

    tess_available, tess_msg = init_tesseract()

    if tess_available:
        st.success(f"✓ {tess_msg}", icon=None)
    else:
        st.warning(f"⚠ {tess_msg}", icon=None)

    with st.expander("Configure Tesseract OCR Path", expanded=not tess_available):
        custom_p = st.text_input(
            "Executable path (tesseract.exe):",
            value=st.session_state.get("tesseract_custom_path", ""),
            placeholder=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            help="If Tesseract is installed in a custom location, paste the full path to tesseract.exe here.",
        )
        if custom_p:
            st.session_state["tesseract_custom_path"] = custom_p
            ok, msg = set_tesseract_cmd(custom_p)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        if st.button("Run OCR Diagnostic Test", use_container_width=True):
            try:
                from PIL import Image, ImageDraw
                test_img = Image.new("RGB", (300, 80), color=(255, 255, 255))
                d = ImageDraw.Draw(test_img)
                d.text((10, 25), "PaperPulse OCR Test 123", fill=(0, 0, 0))
                detected = ocr_image(test_img)
                if "PaperPulse" in detected or "Test" in detected or "123" in detected:
                    st.success(f"OCR Test Passed! Read: '{detected}'")
                else:
                    st.info(f"OCR ran but read: '{detected}'")
            except Exception as exc:
                st.error(f"OCR Test Failed: {exc}")

        if not tess_available:
            st.markdown("""
            **How to install Tesseract on Windows:**
            1. Download the installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
            2. Run the installer (defaults to `C:\\Program Files\\Tesseract-OCR`).
            3. Paste that path above or restart your terminal.
            """)


# ─────────────────────────────────────────────────────────────────
# Top Hero Header
# ─────────────────────────────────────────────────────────────────

ocr_badge = "✓ Tesseract OCR Active" if tess_available else "⚠ Tesseract Not Found (Scans Require Config)"

st.markdown(f"""
<div class="app-header">
  <h1 class="app-title">📚 PaperPulse · Exam Question Intelligence</h1>
  <p class="app-subtitle">
    Mine recurring question topics across multi-year exam papers, discover exam patterns,
    and generate a high-yield study schedule. Supports standard digital PDFs and scanned paper images.
  </p>
  <div class="header-badges">
    <span class="header-badge">AI Embeddings: all-MiniLM-L6-v2</span>
    <span class="header-badge">Same-Paper Collision Protection</span>
    <span class="header-badge">{ocr_badge}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Step 1: Upload Exam Papers or Use Sample Set
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="card-box">
  <div class="card-header-title"><span class="step-num">01</span> Upload Past Exam Papers</div>
</div>
""", unsafe_allow_html=True)

col_upload, col_sample = st.columns([5, 2])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload past exam papers (PDFs or paper images):",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
        help="Upload multiple years of papers for the same subject. Images of papers will be processed with Tesseract OCR.",
    )

with col_sample:
    st.markdown("<p style='font-size:12px; font-weight:600; color:#64748b; margin-top:6px;'>DEMO PAPERS</p>", unsafe_allow_html=True)
    if _sample_data_available():
        if st.button("Load 6-Year Sample Papers (Data Structures 2019–2024)", use_container_width=True):
            st.session_state["sample_mode"] = True
            st.session_state["result"] = None
            st.rerun()
    else:
        st.info("Run sample_data/generate_sample_data.py to create demo files.")

upload_dir = Path(__file__).parent.parent / ".cache" / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)
saved_paths: list[str] = []
year_map: dict[str, int] = {}

if uploaded_files and not st.session_state.get("sample_mode"):
    st.session_state["sample_mode"] = False
    st.markdown("**Confirm Exam Year for each uploaded file:**")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, uf in enumerate(uploaded_files):
        dest = upload_dir / uf.name
        dest.write_bytes(uf.getvalue())
        saved_paths.append(str(dest))
        auto_year = parse_year_from_filename(uf.name) or (2024 - (i % 6))
        with cols[i % 4]:
            confirmed_year = st.number_input(
                label=f"{uf.name[:25]}… ({uf.name.split('.')[-1].upper()})" if len(uf.name) > 28 else f"{uf.name}",
                min_value=1990, max_value=2099,
                value=int(auto_year),
                step=1,
                key=f"year_input_{uf.name}",
            )
            year_map[uf.name] = int(confirmed_year)

    st.session_state["uploaded_file_paths"] = saved_paths
    st.session_state["year_map"] = year_map

if st.session_state.get("sample_mode"):
    st.info(
        "📄 **Sample Mode Active**: Loaded 6 years of Data Structures exam papers (2019 to 2024). "
        "Click **Analyse Papers** below to examine the recurring questions and visual charts.",
        icon=None,
    )


# ─────────────────────────────────────────────────────────────────
# Step 2: Analyse Action
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="card-box">
  <div class="card-header-title"><span class="step-num">02</span> Run Extraction & Topic Mining</div>
</div>
""", unsafe_allow_html=True)

sample_mode = st.session_state.get("sample_mode", False)
has_input = bool(uploaded_files) or sample_mode

btn_run_col, btn_reset_col, _ = st.columns([2, 1, 4])
with btn_run_col:
    run_btn = st.button("Analyse Papers", disabled=not has_input, type="primary", use_container_width=True)

with btn_reset_col:
    if st.session_state.get("result") or sample_mode:
        if st.button("Reset", use_container_width=True):
            st.session_state["result"] = None
            st.session_state["sample_mode"] = False
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────────────────────────

def _execute_analysis(paths: list[str], ym: dict[str, int], th: float, is_sample: bool) -> PipelineResult:
    prog = st.progress(0, text="Initializing extraction…")

    def _cb(msg: str, frac: float) -> None:
        prog.progress(int(frac * 100), text=msg)

    if is_sample:
        txt_files = _get_sample_txt_files()
        _cb("Extracting text from sample papers…", 0.1)
        extraction_results = extract_files(txt_files)
        res = PipelineResult()
        res.extraction_results = extraction_results

        _cb("Splitting pages into individual questions…", 0.25)
        q_df = split_results(extraction_results)
        res.questions_df = q_df

        if q_df.empty:
            _cb("No questions found.", 1.0)
            return res

        _cb("Encoding semantic embeddings…", 0.45)
        embedder = Embedder()
        emb = embedder.encode_dataframe(q_df, show_progress=False)
        res.embeddings = emb

        _cb("Clustering recurring topics across years…", 0.65)
        c_df = assign_clusters(q_df, embeddings=emb, threshold=th, constrain_same_paper=True)

        _cb("Assigning clean topic labels…", 0.8)
        l_df = label_clusters(c_df, embeddings=emb)
        res.questions_df = l_df

        _cb("Scoring priority and ranking topics…", 0.9)
        res.ranked_clusters = rank_clusters(l_df)

        _cb("Analysis complete!", 1.0)
        return res
    else:
        return run_pipeline(
            file_paths=paths,
            year_overrides=ym,
            threshold=th,
            progress_callback=_cb,
        )


if run_btn and has_input:
    with st.spinner("Processing exam papers…"):
        try:
            analysis_result = _execute_analysis(
                paths=st.session_state.get("uploaded_file_paths", []),
                ym=st.session_state.get("year_map", {}),
                th=st.session_state["threshold"],
                is_sample=sample_mode,
            )
            st.session_state["result"] = analysis_result
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()


# ─────────────────────────────────────────────────────────────────
# Results Presentation
# ─────────────────────────────────────────────────────────────────

result: PipelineResult | None = st.session_state.get("result")

if result is not None:
    if result.extraction_errors:
        with st.expander(f"⚠ {len(result.extraction_errors)} extraction notice(s)", expanded=False):
            for err in result.extraction_errors:
                st.warning(err)

    if result.n_questions == 0:
        st.warning("No questions were extracted. Please verify that files contain text or configured Tesseract OCR.")
        st.stop()

    summary_df = ranked_to_dataframe(result.ranked_clusters)

    # ── KPI Metrics Overview ────────────────────────────────────
    st.markdown("""
    <div class="card-box">
      <div class="card-header-title">Results Overview</div>
    </div>
    """, unsafe_allow_html=True)

    high_yield_count = sum(1 for c in result.ranked_clusters if c.priority_score >= 0.60)
    top_topic = result.ranked_clusters[0].cluster_label if result.ranked_clusters else "None"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val">{result.n_papers}</div>
          <div class="kpi-label">Papers Analysed</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val">{result.n_questions}</div>
          <div class="kpi-label">Questions Extracted</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val">{result.n_clusters}</div>
          <div class="kpi-label">Topics Discovered</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val" style="color:#2563eb;">{high_yield_count}</div>
          <div class="kpi-label">High-Yield Topics</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # Visual Analytics & Charts Section
    # ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card-box">
      <div class="card-header-title">📊 Visual Topic Analytics & Exam Trends</div>
    </div>
    """, unsafe_allow_html=True)

    tab_recurrence, tab_breakdown, tab_marks = st.tabs([
        "Multi-Year Topic Timeline",
        "Priority Score Factors",
        "Marks Weightage Distribution",
    ])

    # 1. Topic Recurrence Timeline / Matrix Chart
    with tab_recurrence:
        st.caption("Visual matrix showing which exam years each topic appeared in. High-density rows indicate high exam repeat frequency.")
        all_dataset_years = sorted({y for c in result.ranked_clusters for y in c.years_list if y is not None})
        if not all_dataset_years:
            all_dataset_years = [2024]

        heatmap_rows = []
        for c in result.ranked_clusters[:15]:  # Top 15 topics
            t_short = c.cluster_label[:40]
            for y in all_dataset_years:
                appeared = 1 if y in c.years_list else 0
                heatmap_rows.append({
                    "Topic": t_short,
                    "Exam Year": str(y),
                    "Appeared": appeared,
                    "Status": "Appeared" if appeared else "Not Asked",
                    "Priority": c.priority_score,
                })

        hm_df = pd.DataFrame(heatmap_rows)
        if not hm_df.empty:
            hm_base = alt.Chart(hm_df).encode(
                x=alt.X("Exam Year:O", title="Exam Year", axis=alt.Axis(labelAngle=0, labelFontWeight="bold")),
                y=alt.Y("Topic:N", sort=alt.SortField("Priority", order="descending"), title="Topic (Ranked by Priority)"),
            )
            hm_rects = hm_base.mark_rect(cornerRadius=4, stroke="#ffffff", strokeWidth=2).encode(
                color=alt.Color(
                    "Appeared:Q",
                    scale=alt.Scale(domain=[0, 1], range=["#f1f5f9", "#2563eb"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Topic:N", title="Topic"),
                    alt.Tooltip("Exam Year:O", title="Year"),
                    alt.Tooltip("Status:N", title="Exam Status"),
                ],
            )
            hm_text = hm_base.mark_text(baseline="middle", fontSize=12, fontWeight="bold").encode(
                text=alt.condition(alt.datum.Appeared == 1, alt.value("✓"), alt.value("·")),
                color=alt.condition(alt.datum.Appeared == 1, alt.value("#ffffff"), alt.value("#94a3b8")),
            )
            chart_heatmap = (hm_rects + hm_text).properties(
                height=max(260, len(result.ranked_clusters[:15]) * 26),
                width="container",
            )
            st.altair_chart(chart_heatmap, use_container_width=True)

    # 2. Priority Score & Composite Factors Stacked Bar Chart
    with tab_breakdown:
        st.caption("Breakdown of study priority into Frequency (repeat occurrences), Recency (recent exam weight), and Typical Marks.")
        breakdown_rows = []
        for c in result.ranked_clusters[:12]:
            t_name = c.cluster_label[:36]
            breakdown_rows.extend([
                {"Topic": t_name, "Factor": "Frequency (50%)", "Value": round(c.frequency_score * 0.5, 3), "Priority": c.priority_score},
                {"Topic": t_name, "Factor": "Recency (30%)", "Value": round(c.recency_score * 0.3, 3), "Priority": c.priority_score},
                {"Topic": t_name, "Factor": "Marks (20%)", "Value": round(c.marks_score * 0.2, 3), "Priority": c.priority_score},
            ])

        bd_df = pd.DataFrame(breakdown_rows)
        if not bd_df.empty:
            chart_breakdown = alt.Chart(bd_df).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("Topic:N", sort=alt.SortField("Priority", order="descending"), title=None),
                x=alt.X("Value:Q", title="Priority Score Contribution (0.0 to 1.0)", stack="zero"),
                color=alt.Color(
                    "Factor:N",
                    scale=alt.Scale(
                        domain=["Frequency (50%)", "Recency (30%)", "Marks (20%)"],
                        range=["#2563eb", "#10b981", "#f59e0b"],
                    ),
                    legend=alt.Legend(title="Score Component", orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("Topic:N", title="Topic"),
                    alt.Tooltip("Factor:N", title="Component"),
                    alt.Tooltip("Value:Q", title="Weighted Contribution", format=".3f"),
                ],
            ).properties(
                height=max(260, len(result.ranked_clusters[:12]) * 28),
                width="container",
            )
            st.altair_chart(chart_breakdown, use_container_width=True)

    # 3. Marks Weightage Chart
    with tab_marks:
        st.caption("Average marks carried by questions under each topic.")
        marks_rows = []
        for c in result.ranked_clusters[:12]:
            marks_rows.append({
                "Topic": c.cluster_label[:36],
                "Avg Marks": c.avg_marks if c.avg_marks is not None else 5.0,
                "Priority Score": c.priority_score,
            })
        m_df = pd.DataFrame(marks_rows)
        if not m_df.empty:
            chart_marks = alt.Chart(m_df).mark_bar(cornerRadiusEnd=4, color="#6366f1").encode(
                y=alt.Y("Topic:N", sort=alt.SortField("Priority Score", order="descending"), title=None),
                x=alt.X("Avg Marks:Q", title="Average Marks per Question"),
                tooltip=["Topic:N", alt.Tooltip("Avg Marks:Q", format=".1f"), "Priority Score:Q"],
            ).properties(
                height=max(260, len(marks_rows) * 26),
                width="container",
            )
            st.altair_chart(chart_marks, use_container_width=True)

    # ─────────────────────────────────────────────────────────────
    # Priority Topics Table
    # ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card-box">
      <div class="card-header-title">📋 Recurring Question Topics & Study Priority</div>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input("Filter topics by keyword:", placeholder="e.g. stack, sort, binary, hashing, graph...")

    table_data = []
    for rank_idx, cluster in enumerate(result.ranked_clusters, start=1):
        if search_query and search_query.lower() not in cluster.cluster_label.lower():
            continue

        years_str = ", ".join(str(y) for y in sorted(cluster.years_list))
        tier = "High" if cluster.priority_score >= 0.65 else ("Medium" if cluster.priority_score >= 0.35 else "Low")
        table_data.append({
            "Rank": rank_idx,
            "Topic": cluster.cluster_label,
            "Priority Score": f"{cluster.priority_score:.2f}",
            "Priority Tier": tier,
            "Years Appeared": f"{cluster.distinct_years} of {cluster.total_years}",
            "Exam Years": years_str,
            "Avg Marks": f"{cluster.avg_marks:.0f}" if cluster.avg_marks is not None else "—",
            "Pattern Note": cluster.pattern_hint or cluster.reason,
        })

    display_df = pd.DataFrame(table_data)
    if not display_df.empty:
        st.dataframe(display_df.set_index("Rank"), use_container_width=True)
    else:
        st.info("No topics matched your search filter.")

    buf = io.BytesIO()
    summary_df.to_csv(buf, index=False, encoding="utf-8")
    st.download_button(
        "⬇ Download Analysis Summary (CSV)",
        data=buf.getvalue(),
        file_name="paperpulse_exam_topics.csv",
        mime="text/csv",
    )

    # ─────────────────────────────────────────────────────────────
    # Question Variants Deep Dive
    # ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card-box">
      <div class="card-header-title">🔍 Question Variants Across Exam Years</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Inspect how the question phrasing and marks evolved year-by-year for each topic.")

    for i, cluster in enumerate(result.ranked_clusters, start=1):
        t_label = cluster.cluster_label
        y_info = f"{cluster.distinct_years} of {cluster.total_years} years ({', '.join(str(y) for y in sorted(cluster.years_list))})"
        with st.expander(f"#{i} — {t_label} · [{y_info}]"):
            if cluster.pattern_hint:
                st.markdown(f'<div class="hint-callout">⚡ {cluster.pattern_hint}</div>', unsafe_allow_html=True)

            q_disp = cluster.questions[["year", "question_no", "question_text", "marks", "source_file"]].copy()
            q_disp["source_file"] = q_disp["source_file"].apply(lambda p: Path(p).name)
            q_disp.columns = ["Year", "Q No.", "Exact Exam Question", "Marks", "Source Paper"]
            q_disp = q_disp.sort_values("Year").reset_index(drop=True)
            st.dataframe(q_disp, use_container_width=True)

    # ─────────────────────────────────────────────────────────────
    # Study Plan Generator
    # ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card-box">
      <div class="card-header-title">🗓 Evidence-Based Exam Prep Plan</div>
    </div>
    """, unsafe_allow_html=True)

    plan_col, _ = st.columns([1, 3])
    with plan_col:
        prep_days = st.number_input("Days available before examination:", min_value=1, max_value=60, value=7, step=1)

    plan_df = generate_study_plan(result.ranked_clusters, available_days=int(prep_days))
    if not plan_df.empty:
        st.dataframe(plan_df.set_index("Rank"), use_container_width=True)
        buf_plan = io.BytesIO()
        plan_df.to_csv(buf_plan, index=False, encoding="utf-8")
        st.download_button(
            "⬇ Download Custom Study Plan (CSV)",
            data=buf_plan.getvalue(),
            file_name="paperpulse_study_plan.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────────────────────
    # Raw Question Vault
    # ─────────────────────────────────────────────────────────────
    with st.expander("Explore all extracted raw questions"):
        raw_df = result.questions_df[
            ["question_text", "year", "marks", "source_file", "question_no", "cluster_id", "cluster_label"]
        ].copy()
        raw_df["source_file"] = raw_df["source_file"].apply(lambda p: Path(p).name)
        st.dataframe(raw_df, use_container_width=True)
        buf_raw = io.BytesIO()
        raw_df.to_csv(buf_raw, index=False, encoding="utf-8")
        st.download_button(
            "⬇ Download All Extracted Questions (CSV)",
            data=buf_raw.getvalue(),
            file_name="paperpulse_all_questions.csv",
            mime="text/csv",
        )
