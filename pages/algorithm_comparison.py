"""
Page 4: Algorithm Comparison — Side-by-side execution of three search algorithms with physical record dereferencing.
Methods compared:
1. Binary Search (Whole array baseline)
2. Naive Learned Index (scikit-learn model.predict)
3. Optimized Learned Index (Direct arithmetic)
"""

import time
import streamlit as st
import numpy as np
import pandas as pd

from core.dataset import generate_synthetic_uniform
from core.learned_index_model import LearnedIndexModel
from components.metrics import page_header, metric_row, academic_callout
from components.diagrams import academic_comparison_table, search_steps_display


def _ensure_bundle_and_model():
    if "current_bundle" not in st.session_state:
        st.session_state.current_bundle = generate_synthetic_uniform(size=100_000, seed=21)
        st.session_state.bundle_cache_key = "default_uniform_100000_21"

    bundle = st.session_state.current_bundle

    if "model" not in st.session_state or st.session_state.model is None:
        model = LearnedIndexModel()
        model.train(bundle)
        st.session_state.model = model

    return bundle, st.session_state.model


def render():
    """Render the Algorithm Comparison page."""

    page_header(
        title="Algorithm Comparison",
        subtitle="Empirical Trace, Latency Profiling, and Overhead Analysis across Three Implementations",
        pills=[
            {"text": "Algorithm Evaluation", "type": "dark"},
            {"text": "Side-by-Side Trace", "type": "teal"}
        ]
    )

    bundle, model = _ensure_bundle_and_model()

    # Overview table
    st.markdown("### 1. Architectural Characteristics")
    academic_comparison_table()

    st.markdown("---")

    # Interactive Query Interface
    st.markdown("### 2. Multi-Method Live Query Execution")
    col1, col2 = st.columns([1, 2])

    with col1:
        query_type = st.radio(
            "Query Key Target",
            ["Present in Dataset", "Absent from Dataset", "Manual Entry"],
            index=0
        )

        if query_type == "Present in Dataset":
            pos_choice = st.slider("Index Position", 0, len(bundle.index_keys) - 1, len(bundle.index_keys) // 3)
            search_key = int(bundle.index_keys[pos_choice])
        elif query_type == "Absent from Dataset":
            # Deterministically find an absent key
            present_set = set(bundle.index_keys.tolist())
            candidate = int(bundle.index_keys[len(bundle.index_keys) // 2]) + 1
            while candidate in present_set:
                candidate += 1
            search_key = candidate
            st.markdown(f"Selected Absent Key: `Key = {search_key:,}`")
        else:
            search_key = st.number_input(
                "Manual Key Value",
                value=int(bundle.index_keys[len(bundle.index_keys) // 2]),
                step=1
            )

    with col2:
        st.markdown(f"**Target Search Key:** `Key = {search_key:,}` | Dataset: `{bundle.name}` ({len(bundle.index_keys):,} records)")

        # Run Method 1: Binary Search Baseline
        t0 = time.perf_counter()
        bs_res = model.search_binary_detailed(search_key)
        bs_time = (time.perf_counter() - t0) * 1e6

        # Run Method 2: Naive Learned Index (scikit-learn predict)
        t0 = time.perf_counter()
        naive_res = model.search_naive_detailed(search_key)
        naive_time = (time.perf_counter() - t0) * 1e6

        # Run Method 3: Optimized Learned Index (Direct arithmetic)
        t0 = time.perf_counter()
        opt_res = model.search_optimized_detailed(search_key)
        opt_time = (time.perf_counter() - t0) * 1e6

        # Verify semantic agreement
        all_agree = (bs_res["index"] == naive_res["index"] == opt_res["index"])
        if all_agree:
            status_text = "All 3 algorithms returned identical index positions and record references. Correctness verified."
            academic_callout(status_text, callout_type="teal", title="Correctness Check: PASS")
        else:
            academic_callout("Mismatch detected between algorithm results!", callout_type="warning", title="Correctness Check: FAIL")

    st.markdown("---")

    # Side-by-side results cards
    st.markdown("### 3. Execution Metrics per Method")
    col_bs, col_naive, col_opt = st.columns(3)

    with col_bs:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label" style="color: #315E91;">Method 1: Binary Search</div>
            <div class="metric-value" style="color: #315E91;">Baseline</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        - **Result:** {"FOUND" if bs_res["found"] else "NOT FOUND"}
        - **Index Position:** `{bs_res['index']}`
        - **Physical Row ID:** `Row {bs_res['row_id']}`
        - **Comparisons:** `{bs_res['comparisons']}`
        - **Search Scope:** `[0 ... {len(bundle.index_keys)-1:,}]`
        - **Query Latency:** `{bs_time:.2f} μs`
        """)

    with col_naive:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label" style="color: #D97706;">Method 2: Naive Learned Index</div>
            <div class="metric-value" style="color: #D97706;">sklearn.predict</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        - **Result:** {"FOUND" if naive_res["found"] else "NOT FOUND"}
        - **Index Position:** `{naive_res['index']}`
        - **Physical Row ID:** `Row {naive_res['row_id']}`
        - **Predicted Pos:** `{naive_res['predicted_pos']:,}`
        - **Local Comparisons:** `{naive_res['comparisons']}`
        - **Search Scope:** `[{naive_res['search_lo']:,} ... {naive_res['search_hi']:,}]`
        - **Query Latency:** `{naive_time:.2f} μs`
        """)

    with col_opt:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label" style="color: #368A83;">Method 3: Optimized Learned</div>
            <div class="metric-value" style="color: #368A83;">Direct Arithmetic</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        - **Result:** {"FOUND" if opt_res["found"] else "NOT FOUND"}
        - **Index Position:** `{opt_res['index']}`
        - **Physical Row ID:** `Row {opt_res['row_id']}`
        - **Predicted Pos:** `{opt_res['predicted_pos']:,}`
        - **Local Comparisons:** `{opt_res['comparisons']}`
        - **Search Scope:** `[{opt_res['search_lo']:,} ... {opt_res['search_hi']:,}]`
        - **Query Latency:** `{opt_time:.2f} μs`
        """)

    st.markdown("---")

    # Step-by-Step Search Traces
    st.markdown("### 4. Step-by-Step Search Traces")
    tab_bs, tab_naive, tab_opt = st.tabs(["Binary Search Trace", "Naive Learned Index Trace", "Optimized Learned Index Trace"])

    with tab_bs:
        st.markdown(f"**Binary Search Trace** — Whole Array Scope `[0 ... {len(bundle.index_keys)-1:,}]` ({len(bundle.index_keys):,} entries)")
        search_steps_display(bs_res.get("steps", []), search_key)

    with tab_naive:
        st.markdown(f"**Naive Learned Index Trace** — Predicted `Pos = {naive_res['predicted_pos']:,}` | Window `[{naive_res['search_lo']:,} ... {naive_res['search_hi']:,}]` ({naive_res['search_window']:,} entries)")
        search_steps_display(naive_res.get("steps", []), search_key)

    with tab_opt:
        st.markdown(f"**Optimized Learned Index Trace** — Predicted `Pos = {opt_res['predicted_pos']:,}` | Window `[{opt_res['search_lo']:,} ... {opt_res['search_hi']:,}]` ({opt_res['search_window']:,} entries)")
        search_steps_display(opt_res.get("steps", []), search_key)

    st.markdown("---")

    # Dereferenced Physical Record Preview
    if opt_res["found"] and opt_res["record"]:
        st.markdown("### 5. Dereferenced Physical Database Record")
        st.markdown(f"""
        <div class="academic-box teal">
            <div style="font-weight: 600; font-size: 0.82rem; text-transform: uppercase; margin-bottom: 4px;">Primary Key Matched: {search_key:,}</div>
            Row ID: <code>Row {opt_res['row_id']}</code><br>
            Payload Record: <code>{opt_res['record']}</code>
        </div>
        """, unsafe_allow_html=True)
