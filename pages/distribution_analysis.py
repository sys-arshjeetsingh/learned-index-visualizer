"""
Page: Distribution Analysis — Empirical Comparison Across Key Distributions.
Compares:
- Synthetic Uniform
- Synthetic Normal (Gaussian)
- Synthetic Log-Normal (Heavy Skew)
- Real-world SOSD Benchmarks (Wikipedia, Books, OSM, Facebook)

Measures and visualizes:
- Key distribution histogram.
- Regression fit & non-linear CDF deviation.
- Prediction error (MAE, Max Error).
- Search window size.
- Binary search latency vs. learned index latency.
- Academic explanation of CDF curvature limits on single linear models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import time

from core.dataset import (
    generate_synthetic_uniform,
    generate_synthetic_normal,
    generate_synthetic_lognormal,
    load_sosd_dataset
)
from core.learned_index_model import LearnedIndexModel
from core.binary_search import binary_search
from core.dataset import generate_queries
from components.metrics import page_header, metric_row, academic_callout
from components.charts import key_distribution_plot, regression_plot


DISTRIBUTION_PRESETS = [
    ("Synthetic Uniform", "uniform", "Uniform distribution U(0, Max). Constant key density."),
    ("Synthetic Normal", "normal", "Gaussian distribution N(μ, σ²). S-shaped CDF curve."),
    ("Synthetic Log-Normal", "lognormal", "Power-law right-skewed distribution. Steep CDF head and long flat tail."),
    ("Real-world SOSD Wikipedia", "wiki_ts", "Wikipedia edit timestamps. Multi-modal temporal bursts and daily waves."),
    ("Real-world SOSD Books", "books", "Amazon book sales ranks. Dense popularity head with long tail."),
    ("Real-world SOSD OpenStreetMap", "osm", "64-bit Morton spatial cell codes. Highly clustered and non-linear."),
]


@st.cache_data(show_spinner=False)
def benchmark_distribution_profile(dist_type: str, sample_size: int = 50_000, seed: int = 42):
    """Generate, train, and benchmark a distribution profile."""
    if dist_type == "uniform":
        bundle = generate_synthetic_uniform(size=sample_size, seed=seed)
    elif dist_type == "normal":
        bundle = generate_synthetic_normal(size=sample_size, seed=seed)
    elif dist_type == "lognormal":
        bundle = generate_synthetic_lognormal(size=sample_size, seed=seed)
    else:
        bundle = load_sosd_dataset(dist_type, sample_size=sample_size, seed=seed)

    model = LearnedIndexModel()
    train_stats = model.train(bundle)

    # Benchmark with 5,000 queries
    keys = bundle.index_keys
    queries = generate_queries(keys, n_queries=5000, present_ratio=0.5, seed=123)
    q_list = [int(q) for q in queries]

    # Time Binary Search
    t0 = time.perf_counter()
    for q in q_list:
        binary_search(keys, q)
    bs_time = (time.perf_counter() - t0) / len(q_list) * 1e6

    # Time Optimized Learned Index
    t0 = time.perf_counter()
    for q in q_list:
        model.search_optimized(q)
    opt_time = (time.perf_counter() - t0) / len(q_list) * 1e6

    reg_data = model.get_regression_data(n_samples=2000)

    return {
        "bundle": bundle,
        "model": model,
        "train_stats": train_stats,
        "bs_time_us": bs_time,
        "opt_time_us": opt_time,
        "speedup": bs_time / opt_time if opt_time > 0 else 0.0,
        "reg_data": reg_data
    }


def render():
    """Render the Distribution Analysis page."""

    page_header(
        title="Key Distribution Analysis",
        subtitle="Empirical Investigation of CDF Non-Linearity, Residual Bounds, and Model Capacity",
        pills=[
            {"text": "Distribution Modeling", "type": "dark"},
            {"text": "Empirical Comparison", "type": "teal"}
        ]
    )

    st.markdown("""
    A fundamental limitation of learned index research is assuming that data is always uniformly distributed.
    Because a single **Linear Regression** model can only represent a straight line ($y = mx + c$), 
    its prediction accuracy depends directly on how closely the data's empirical **Cumulative Distribution Function (CDF)** 
    approximates a linear slope.
    """)

    # Interactive selector to inspect a specific distribution
    st.markdown("### 1. In-Depth Distribution Profiler")
    col_sel, col_sz = st.columns([2, 1])

    with col_sel:
        selected_dist_name = st.selectbox(
            "Select Distribution to Profile",
            [p[0] for p in DISTRIBUTION_PRESETS],
            index=0
        )

    with col_sz:
        sample_size = st.selectbox("Sample Size", [25_000, 50_000, 100_000], index=1)

    selected_tuple = next(p for p in DISTRIBUTION_PRESETS if p[0] == selected_dist_name)
    dist_code = selected_tuple[1]

    with st.spinner("Analyzing distribution and fitting linear regression model..."):
        profile = benchmark_distribution_profile(dist_code, sample_size=sample_size)

    bundle = profile["bundle"]
    train_stats = profile["train_stats"]
    reg_data = profile["reg_data"]

    # Profile Metrics
    metric_row([
        {"label": "Distribution Family", "value": bundle.category, "description": bundle.name},
        {"label": "Model R² Score", "value": f"{train_stats['r2']:.6f}", "description": "Goodness of linear fit"},
        {"label": "Mean Absolute Error", "value": f"{train_stats['mean_error']:.1f} pos", "description": "Average prediction error"},
        {"label": "Max Absolute Error", "value": f"{train_stats['max_error']:,} pos", "description": "Worst-case prediction bound"},
    ])

    st.markdown("")

    metric_row([
        {"label": "Local Search Window", "value": f"{train_stats['search_radius']*2 + 1:,} rows", "description": "Elements scanned in correction"},
        {"label": "Window Fraction", "value": f"{(train_stats['search_radius']*2 + 1)/sample_size*100:.2f}%", "description": "Portion of full table"},
        {"label": "Binary Search Latency", "value": f"{profile['bs_time_us']:.2f} μs", "description": "Whole array O(log N)"},
        {"label": "Learned Index Latency", "value": f"{profile['opt_time_us']:.2f} μs", "description": "Optimized arithmetic + window"},
    ])

    # Side-by-side distribution histogram and regression fit
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_hist = key_distribution_plot(bundle.index_keys, n_bins=70, title=f"Frequency Histogram — {bundle.name}")
        st.plotly_chart(fig_hist, width="stretch")

    with col_chart2:
        fig_fit = regression_plot(
            reg_data["keys"],
            reg_data["actual_positions"],
            reg_data["predicted_positions"],
            max_error=train_stats["max_error"],
            title=f"Linear Model vs. Actual CDF — {bundle.name}"
        )
        st.plotly_chart(fig_fit, width="stretch")

    st.markdown("---")

    # =========================================================================
    # MULTI-DISTRIBUTION COMPARISON MATRIX
    # =========================================================================
    st.markdown("### 2. Multi-Distribution Comparison Matrix")
    st.markdown("""
    The following matrix compares model accuracy, search bounds, and lookup latencies across all 
    major distributions under identical workload configurations (50,000 keys, 5,000 queries):
    """)

    with st.spinner("Generating comparative multi-distribution benchmarks..."):
        profiles = []
        for name, code, desc in DISTRIBUTION_PRESETS:
            prof = benchmark_distribution_profile(code, sample_size=50_000)
            t_s = prof["train_stats"]
            profiles.append({
                "Distribution": name,
                "R² Fit": f"{t_s['r2']:.5f}",
                "Mean Error (pos)": f"{t_s['mean_error']:.1f}",
                "Max Error (pos)": f"{t_s['max_error']:,}",
                "Search Window": f"{t_s['search_radius']*2 + 1:,}",
                "Binary Search (μs)": f"{prof['bs_time_us']:.2f}",
                "Learned Index (μs)": f"{prof['opt_time_us']:.2f}",
                "Speedup vs BS": f"{prof['speedup']:.2f}×"
            })

        matrix_df = pd.DataFrame(profiles)
        st.dataframe(matrix_df, width="stretch", hide_index=True)

    st.markdown("---")

    # =========================================================================
    # THEORETICAL ANALYSIS: WHY A SINGLE LINEAR MODEL FAILS ON SKEWED DATA
    # =========================================================================
    st.markdown("### 3. Theoretical Analysis: Why Single Linear Models Struggle on Skewed Data")

    col_th1, col_th2 = st.columns(2)

    with col_th1:
        st.markdown("""
        #### Uniform Distribution Behavior
        - **Linear CDF:** For a uniform distribution $U(a, b)$, the CDF is perfectly linear:
          $$F(x) = \\frac{x - a}{b - a}$$
        - **Minimal Error:** Linear regression finds the exact slope $m = \\frac{N}{b - a}$. The only errors arise from discrete integer sampling noise.
        - **Tiny Window:** The search radius is typically under 150 elements. Local binary search requires only $\\approx \\log_2(300) \\approx 8$ comparisons.
        - **Speedup:** The learned index easily beats full binary search because $8 < \\log_2(100,000) \\approx 17$.
        """)

    with col_th2:
        st.markdown("""
        #### Skewed / Normal / Real-World Behavior
        - **Non-Linear S-Curve:** For a Normal distribution, the CDF is an error function $\\Phi(x)$; for Log-Normal or power-law data, it is steep at the origin and flat in the tail.
        - **Model Underfitting:** A straight line cannot bend to follow an S-curve or exponential curve. It severely underfits in regions of high curvature.
        - **Exploding Error Bounds:** The worst-case residual $\\epsilon_{\\max}$ balloons from hundreds of positions to **thousands or tens of thousands**.
        - **Enlarged Window:** When the search window spans a large portion of the array, the local binary search must inspect almost as many elements as the baseline binary search, while still paying the prediction arithmetic cost!
        """)

    academic_callout(
        "<strong>Architectural Conclusion:</strong> This demonstrates why modern production learned indexes (such as Google's "
        "<strong>Recursive Model Index (RMI)</strong> or <strong>ALEX</strong>) do not use a single model. Instead, they use a "
        "two-layer hierarchy: a root model that routes the key to one of hundreds of localized leaf linear models, "
        "each approximating a small, nearly linear piecewise segment of the CDF.",
        callout_type="primary",
        title="Academic Insight"
    )


if __name__ == "__main__":
    render()
