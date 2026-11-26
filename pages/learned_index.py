"""
Page 3: Learned Index Explained — Conceptual Foundation and Step-by-Step Walkthrough.
Features:
- "Why Must Index Keys Be Sorted?" interactive demonstration.
- Physical table vs. logical sorted index with record pointers.
- 4-step interactive learned index execution with full record dereference.
"""

import streamlit as st
import numpy as np
import pandas as pd

from core.dataset import generate_synthetic_uniform
from core.learned_index_model import LearnedIndexModel
from components.metrics import page_header, metric_row, academic_callout
from components.charts import regression_plot, search_window_comparison_plot
from components.diagrams import physical_vs_index_diagram, search_steps_display


def _ensure_bundle_and_model():
    """Ensure active dataset bundle and trained model exist."""
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
    """Render the Learned Index Explained page."""

    page_header(
        title="Learned Index Explained",
        subtitle="Theoretical Foundation, Monotonicity Requirements, and Inference Mechanics",
        pills=[
            {"text": "Algorithm Theory", "type": "dark"},
            {"text": "Interactive Trace", "type": "teal"}
        ]
    )

    bundle, model = _ensure_bundle_and_model()
    stats = model.get_prediction_stats()

    # =========================================================================
    # SECTION 1: WHY MUST INDEX KEYS BE SORTED? (SPECIFIC REQUIREMENT #4)
    # =========================================================================
    st.markdown("### Why Must Index Keys Be Sorted?")
    st.markdown("""
    In a relational database, records are written into **Heap Files** in arbitrary or arrival order. 
    A question frequently asked in database systems design is: *Why can a learned model not simply predict positions in the raw heap table?*
    
    The reason is rooted in **monotonicity** and **Cumulative Distribution Functions (CDF)**:
    """)

    # Visual demonstration of Unsorted Heap vs Sorted Primary Index
    sorted_pairs = list(zip(bundle.index_keys[:10], bundle.index_row_ids[:10]))
    physical_vs_index_diagram(bundle.physical_records[:10], sorted_pairs)

    st.markdown("""
    #### The Mathematical Principle
    1. **Arbitrary Order Destroys Predictability:** In an unsorted heap table, key values fluctuate randomly relative to row position. No low-degree polynomial or linear model can learn a random permutation function without massive overfitting (requiring $O(N)$ parameters).
    2. **Sorting Creates a Monotonic CDF:** When index entries are sorted by key in ascending order, the array position $y_i$ of key $x_i$ becomes a strictly non-decreasing step function:
       $$y_i = \\sum_{j=0}^{N-1} \\mathbb{I}(x_j \\le x_i) = N \\cdot F_X(x_i)$$
       where $F_X(x)$ is the empirical Cumulative Distribution Function.
    3. **Bounded Error Guarantee:** Because the mapping is monotonic, the residual error $e_i = |\\hat{y}_i - y_i|$ is bounded by a maximum error $\\epsilon_{\\max}$. This enables a guaranteed local search window $[\\hat{y} - \\epsilon_{\\max}, \\hat{y} + \\epsilon_{\\max}]$.
    4. **Record Association is Preserved:** While the index entries are sorted, the physical records remain untouched in heap storage. Each index entry maintains a physical pointer (`row_id`) that dereferences the original record upon completion.
    """)

    academic_callout(
        "<strong>Key Takeaway:</strong> The learned index does not sort keys independently of their records. "
        "It constructs a logical primary index where <code>(Key, Row_ID)</code> pairs are sorted by key. "
        "The machine learning model learns the mapping from <strong>Key &rarr; Index Position</strong>, and upon finding the "
        "matching entry, dereferences the pointer to retrieve the full database record.",
        callout_type="teal",
        title="Database Principle"
    )

    st.markdown("---")

    # =========================================================================
    # STEP 1: MODEL TRAINING
    # =========================================================================
    st.markdown("### Step 1: Model Training (CDF Regression)")
    st.markdown("""
    The learned index trains a simple Linear Regression model on the sorted index entries:
    $$\\hat{y} = m \\cdot x + c$$
    where $x$ represents the search key, and $\\hat{y}$ predicts the array position in the index.
    """)

    metric_row([
        {"label": "Learned Slope (m)", "value": f"{stats['slope']:.6f}", "description": "Index entries per key unit"},
        {"label": "Learned Intercept (c)", "value": f"{stats['intercept']:.2f}", "description": "Position offset bias"},
        {"label": "Goodness of Fit (R²)", "value": f"{stats['r2_score']:.6f}", "description": "Correlation metric"},
        {"label": "Training Latency", "value": f"{stats['training_time_ms']:.1f} ms", "description": "One-time training time"},
    ])

    st.markdown("")

    metric_row([
        {"label": "Mean Absolute Error", "value": f"{stats['mean_absolute_error']:.2f}", "description": "Average residual (positions)"},
        {"label": "Maximum Absolute Error", "value": f"{stats['max_absolute_error']:.0f}", "description": "Worst-case training error"},
        {"label": "Conservative Radius", "value": f"{stats['search_radius']:,}", "description": "max_error + 1"},
        {"label": "Local Window Size", "value": f"{stats['search_window_size']:,}", "description": "2 × radius + 1 elements"},
    ])

    # Plotly regression chart
    reg_data = model.get_regression_data(n_samples=2500)
    fig_reg = regression_plot(
        reg_data["keys"],
        reg_data["actual_positions"],
        reg_data["predicted_positions"],
        max_error=model.max_error,
        title=f"Learned Linear CDF Model Fit — {bundle.name}"
    )
    st.plotly_chart(fig_reg, width="stretch")

    st.markdown("---")

    # =========================================================================
    # STEP 2 & 3: KEY PREDICTION & LOCAL WINDOW SEARCH
    # =========================================================================
    st.markdown("### Steps 2 & 3: Key Prediction and Local Search Window")

    col_input, col_pred = st.columns([1, 1])

    with col_input:
        st.markdown("**Search Key Specification**")
        input_mode = st.radio("Selection Mode", ["Choose from Dataset", "Enter Manual Key"], horizontal=True)

        if input_mode == "Choose from Dataset":
            slider_pos = st.slider("Index Position", 0, len(bundle.index_keys) - 1, len(bundle.index_keys) // 2)
            query_key = int(bundle.index_keys[slider_pos])
            st.markdown(f"Selected Key: `Key = {query_key:,}` at index `Pos = {slider_pos:,}`")
        else:
            query_key = st.number_input(
                "Manual Key Value",
                value=int(bundle.index_keys[len(bundle.index_keys) // 2]),
                step=1
            )

    # Detailed search execution
    detailed_res = model.search_optimized_detailed(query_key)
    pred_pos = detailed_res["predicted_pos"]
    lo = detailed_res["search_lo"]
    hi = detailed_res["search_hi"]
    win_size = detailed_res["search_window"]
    is_found = detailed_res["found"]
    actual_idx = detailed_res["index"]
    row_id = detailed_res["row_id"]
    retrieved_record = detailed_res["record"]

    with col_pred:
        st.markdown("**Inference Output**")
        st.markdown(f"""
        <table class="academic-table" style="margin: 0;">
            <tr>
                <td style="color: #64748B;">Input Key (x)</td>
                <td style="font-family: monospace; font-weight: 600;">{query_key:,}</td>
            </tr>
            <tr>
                <td style="color: #64748B;">Predicted Position (ŷ = mx + c)</td>
                <td style="font-family: monospace; font-weight: 600; color: #D97706;">{pred_pos:,}</td>
            </tr>
            <tr>
                <td style="color: #64748B;">Search Window [lo ... hi]</td>
                <td style="font-family: monospace; font-weight: 600; color: #368A83;">[{lo:,} ... {hi:,}] ({win_size:,} rows)</td>
            </tr>
            <tr>
                <td style="color: #64748B;">Actual Index Position (y)</td>
                <td style="font-family: monospace; font-weight: 600; color: #315E91;">{actual_idx if actual_idx != -1 else 'KEY ABSENT'}</td>
            </tr>
            <tr>
                <td style="color: #64748B;">Observed Prediction Residual</td>
                <td style="font-family: monospace; font-weight: 600;">{detailed_res['prediction_error'] if detailed_res['prediction_error'] is not None else 'N/A'} positions</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("")
    # Visual search window bar
    fig_win = search_window_comparison_plot(pred_pos, lo, hi, actual_idx, len(bundle.index_keys))
    st.plotly_chart(fig_win, width="stretch")

    st.markdown("---")

    # =========================================================================
    # STEP 4: SEARCH RESULT & RECORD DEREFERENCING
    # =========================================================================
    st.markdown("### Step 4: Search Verification and Record Dereferencing")

    col_res1, col_res2 = st.columns([1, 1])

    with col_res1:
        if is_found:
            st.markdown(f"""
            <div class="academic-box teal" style="margin: 0;">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">Search Status: KEY FOUND</div>
                Index Position: <code>{actual_idx:,}</code><br>
                Dereferenced Row ID: <code>Row {row_id}</code><br>
                Local Comparisons: <code>{detailed_res['comparisons']}</code> (out of {len(bundle.index_keys):,} records)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="academic-box warning" style="margin: 0;">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">Search Status: KEY NOT FOUND</div>
                The key does not exist in the index window.<br>
                Local Comparisons: <code>{detailed_res['comparisons']}</code>
            </div>
            """, unsafe_allow_html=True)

    with col_res2:
        if is_found and retrieved_record:
            st.markdown(f"""
            <div class="academic-box" style="margin: 0;">
                <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 4px; text-transform: uppercase;">Retrieved Physical Record</div>
                <table style="width: 100%; font-size: 0.82rem;">
                    <tr><td style="color: #64748B;">Primary Key:</td><td style="font-family: monospace; font-weight: 600;">{retrieved_record.get('key')}</td></tr>
                    <tr><td style="color: #64748B;">Payload (Name):</td><td>{retrieved_record.get('name', retrieved_record.get('payload', 'Record'))}</td></tr>
                    <tr><td style="color: #64748B;">Department:</td><td>{retrieved_record.get('department', 'N/A')}</td></tr>
                    <tr><td style="color: #64748B;">Transaction Status:</td><td><code>{retrieved_record.get('status', 'COMMITTED')}</code></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="academic-box" style="margin: 0;">
                <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 4px; text-transform: uppercase;">No Physical Record Dereferenced</div>
                Database engine returns SQL <code>NOT FOUND</code> (Empty result set).
            </div>
            """, unsafe_allow_html=True)

    # Step-by-Step Search Progression
    st.markdown("#### Local Binary Search Progression")
    search_steps_display(detailed_res.get("steps", []), query_key)


if __name__ == "__main__":
    render()
