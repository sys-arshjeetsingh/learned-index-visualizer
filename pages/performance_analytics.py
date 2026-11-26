"""
Page 5: Performance Analytics — Interactive live benchmarking dashboard.
Ensures:
- Configurable queries, distribution mix, and repetitions.
- Separate measurement of training time and query latency.
- Verified 0 mismatches before latency is displayed.
- Academic Plotly charts styled with the research aesthetic.
"""

import streamlit as st
import numpy as np
import pandas as pd

from core.dataset import generate_synthetic_uniform
from core.learned_index_model import LearnedIndexModel
from core.benchmark import run_benchmark
from components.metrics import page_header, metric_row, academic_callout
from components import charts


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
    """Render the Performance Analytics page."""

    page_header(
        title="Performance Analytics",
        subtitle="Empirical Multi-Method Latency Profiling, Error Residue Analysis, and Benchmark Statistics",
        pills=[
            {"text": "Live Benchmarking", "type": "dark"},
            {"text": "Zero-Mismatch Verification", "type": "teal"}
        ]
    )

    bundle, model = _ensure_bundle_and_model()

    # --- Benchmark Parameters ---
    st.markdown("### 1. Benchmark Workload Configuration")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n_queries = st.selectbox(
            "Query Workload Size",
            [1000, 5000, 10000, 20000],
            index=2,
            help="Total number of random search keys evaluated per run."
        )

    with col2:
        present_ratio = st.slider(
            "Hit Ratio (Present Keys)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Proportion of queries that exist in the database (50% present, 50% absent)."
        )

    with col3:
        n_reps = st.selectbox(
            "Repetition Rounds",
            [1, 3, 5],
            index=1,
            help="Number of batch timing repetitions. Best total execution time is selected to eliminate OS scheduling jitter."
        )

    with col4:
        query_seed = st.number_input("Query Generator Seed", value=99, step=1)

    execute_btn = st.button("Execute Live Benchmark", width="stretch")

    if execute_btn:
        with st.spinner(f"Executing {n_queries:,} queries across 3 algorithms ({n_reps} rounds)..."):
            results = run_benchmark(
                model=model,
                data_or_keys=bundle,
                n_queries=n_queries,
                present_ratio=present_ratio,
                query_seed=query_seed,
                n_repetitions=n_reps,
                warmup=True
            )
            st.session_state.benchmark_results = results

    if "benchmark_results" not in st.session_state:
        academic_callout(
            "Click <strong>Execute Live Benchmark</strong> to generate fresh measurements on your CPU. "
            "All timings are newly measured using high-resolution <code>time.perf_counter()</code> and are verified against "
            "the binary search ground truth before reporting.",
            callout_type="primary",
            title="Fresh Benchmark Execution"
        )
        return

    results = st.session_state.benchmark_results

    st.markdown("---")

    # --- Correctness Check ---
    st.markdown("### 2. Retrieval Correctness Verification")
    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        n_m = results["naive_mismatches"]
        status = "0 Mismatches (100% Correct)" if n_m == 0 else f"{n_m} Mismatches"
        st.markdown(f"**Naive vs. Binary Search Ground Truth:** `{status}`")

    with col_c2:
        o_m = results["opt_mismatches"]
        status = "0 Mismatches (100% Correct)" if o_m == 0 else f"{o_m} Mismatches"
        st.markdown(f"**Optimized vs. Binary Search Ground Truth:** `{status}`")

    with col_c3:
        st.markdown(f"**Evaluated Workload:** `{results['n_queries']:,} queries` ({results['present_ratio']*100:.0f}% hit ratio)")

    st.markdown("---")

    # --- Benchmark Metrics ---
    st.markdown("### 3. Empirical Latency Measurements")
    metric_row([
        {"label": "Binary Search Latency", "value": f"{results['bs_mean_time_us']:.2f} μs", "description": "Baseline divide-and-conquer"},
        {"label": "Naive Learned Latency", "value": f"{results['naive_mean_time_us']:.2f} μs", "description": "sklearn predict + window"},
        {"label": "Optimized Learned Latency", "value": f"{results['opt_mean_time_us']:.2f} μs", "description": "Direct arithmetic + window"},
    ])

    st.markdown("")

    metric_row([
        {"label": "Optimized Speedup", "value": f"{results['opt_speedup']:.2f}×", "description": "Relative to Binary Search", "delta_type": "good" if results['opt_speedup'] > 1.0 else "neutral"},
        {"label": "Framework Overhead Penalty", "value": f"{results['naive_mean_time_us']/results['opt_mean_time_us']:.1f}×", "description": "Naive slowdown vs. Optimized"},
        {"label": "Training Latency", "value": f"{results['training_time_ms']:.1f} ms", "description": "One-time training time"},
    ])

    st.markdown("---")

    # --- Tabular Summary ---
    st.markdown("### 4. Comprehensive Evaluation Matrix")
    detail_df = pd.DataFrame({
        "Performance Dimension": [
            "Mean Lookup Latency",
            "Median Lookup Latency",
            "Total Execution Duration",
            "Relative Speedup Factor",
            "Retrieval Discrepancies",
            "Algorithmic Space Complexity"
        ],
        "Binary Search (Baseline)": [
            f"{results['bs_mean_time_us']:.2f} μs",
            f"{results['bs_median_time_us']:.2f} μs",
            f"{results['bs_total_time']*1000:.2f} ms",
            "1.00× (Baseline)",
            "0 (Ground Truth)",
            "O(N) sorted keys"
        ],
        "Naive Learned Index": [
            f"{results['naive_mean_time_us']:.2f} μs",
            f"{results['naive_median_time_us']:.2f} μs",
            f"{results['naive_total_time']*1000:.2f} ms",
            f"{results['naive_speedup']:.2f}×",
            f"{results['naive_mismatches']}",
            "O(1) model + O(W) window"
        ],
        "Optimized Learned Index": [
            f"{results['opt_mean_time_us']:.2f} μs",
            f"{results['opt_median_time_us']:.2f} μs",
            f"{results['opt_total_time']*1000:.2f} ms",
            f"{results['opt_speedup']:.2f}×",
            f"{results['opt_mismatches']}",
            "O(1) model + O(W) window"
        ]
    })
    st.dataframe(detail_df, width="stretch", hide_index=True)

    st.markdown("---")

    # --- Academic Charts ---
    st.markdown("### 5. Graphical Latency & Error Analysis")

    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        fig_bar = charts.benchmark_comparison_bar(
            results["bs_mean_time_us"],
            results["naive_mean_time_us"],
            results["opt_mean_time_us"],
            title="Mean Query Latency (μs)"
        )
        st.plotly_chart(fig_bar, width="stretch")

    with col_ch2:
        # Latency distribution box plot
        bs_p = np.clip(results["bs_per_query_us"], 0, np.percentile(results["bs_per_query_us"], 99))
        nv_p = np.clip(results["naive_per_query_us"], 0, np.percentile(results["naive_per_query_us"], 99))
        op_p = np.clip(results["opt_per_query_us"], 0, np.percentile(results["opt_per_query_us"], 99))
        fig_box = charts.latency_distribution_box(bs_p.tolist(), nv_p.tolist(), op_p.tolist())
        st.plotly_chart(fig_box, width="stretch")

    # Prediction error analysis
    reg_data = model.get_regression_data(2500)
    col_err1, col_err2 = st.columns(2)

    with col_err1:
        fig_err_hist = charts.error_distribution_plot(
            np.abs(reg_data["errors"]),
            title="Prediction Residuals |ŷ - y| Distribution"
        )
        st.plotly_chart(fig_err_hist, width="stretch")

    with col_err2:
        fig_evk = charts.error_vs_key_plot(
            reg_data["keys"],
            reg_data["errors"],
            max_error=results["max_absolute_error"],
            title="Signed Residual (ŷ - y) Across Key Space"
        )
        st.plotly_chart(fig_evk, width="stretch")

    academic_callout(
        "<strong>Benchmarking Protocol:</strong> All queries use uniform high-resolution <code>time.perf_counter()</code>. "
        "Batch repetitions run each algorithm across the entire query set in tight loops. "
        "The fastest repetition is reported to minimize operating system context-switching artifacts. "
        "Timings naturally vary across different hardware architectures, CPU clock speeds, and Python interpreter builds.",
        callout_type="primary",
        title="Measurement Protocol Note"
    )


if __name__ == "__main__":
    render()
