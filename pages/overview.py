"""
Page 1: Overview — Landing dashboard for the Learned Index Visualizer.
Academic research aesthetic, clean layout, formal architecture diagram.
"""

import streamlit as st
from components.metrics import page_header, metric_row, academic_callout
from components.diagrams import architecture_diagram


def render():
    """Render the Overview page."""

    page_header(
        title="Learned Index Visualizer",
        subtitle="Machine Learning-Based Database Indexing | Architectural Framework and Empirical Benchmark",
        pills=[
            {"text": "DBMS Mini Project", "type": "dark"},
            {"text": "Group 4", "type": "primary"},
            {"text": "Research Prototype", "type": "teal"}
        ]
    )

    # Summary Cards
    metric_row([
        {"label": "Standard Dataset Size", "value": "100,000", "description": "Index entries per test run"},
        {"label": "Inference Model", "value": "Linear Reg.", "description": "Single-model CDF approximation"},
        {"label": "Evaluated Methods", "value": "3 Algorithms", "description": "Binary, Naive, Optimized"},
        {"label": "Workload Ratio", "value": "50% Present", "description": "50% absent query keys"},
        {"label": "Benchmark Scale", "value": "10,000", "description": "Queries per evaluation cycle"}
    ])

    st.markdown("---")

    # Research Overview
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Conceptual Foundation")
        st.markdown("""
        In traditional Database Management Systems (DBMS), indexing structures such as **B+ Trees** and **Hash Indexes** 
        are standard solutions for efficient record retrieval. These structures consume significant storage overhead (often up to 
        70% of total database footprint) and navigate fixed tree depths regardless of underlying data distribution.

        Introduced in the 2018 landmark paper *"The Case for Learned Index Structures"* by **Kraska et al. (Google / MIT)**, 
        learned indexing posits that:
        
        > **A database index is essentially a function that predicts record position given a key.**

        For sorted records, the position of key $x$ corresponds to its Cumulative Distribution Function:
        $$\\text{Position}(x) = \\text{CDF}(x) \\cdot N$$
        By fitting machine learning models (such as Linear Regression) to approximate this CDF, the system can compute an 
        approximate record location in $O(1)$ time, followed by a localized error-bounded search for exact verification.
        """)

    with col2:
        st.markdown("### Traditional vs. Learned Indexing")
        st.markdown("""
        <table class="academic-table" style="margin-top: 0.2rem;">
            <thead>
                <tr>
                    <th>Dimension</th>
                    <th>Traditional B+ Tree</th>
                    <th>Learned Index</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Lookup Time</strong></td>
                    <td>O(log N) tree traversal</td>
                    <td>O(1) prediction + O(log W) local search</td>
                </tr>
                <tr>
                    <td><strong>Memory Overhead</strong></td>
                    <td>High (internal tree nodes)</td>
                    <td>Minimal (two floats: m, c)</td>
                </tr>
                <tr>
                    <td><strong>Data Adaptability</strong></td>
                    <td>Agnostic to data pattern</td>
                    <td>Directly learns data distribution</td>
                </tr>
                <tr>
                    <td><strong>Complexity</strong></td>
                    <td>Deterministic branching</td>
                    <td>Model inference + error correction</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Architecture Pipeline
    st.markdown("### Pipeline Architecture")
    architecture_diagram()

    st.markdown("---")

    # Project Scope & Academic Disclaimer
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Academic Objectives")
        st.markdown("""
        1. **Preserve Database Semantics:** Maintain proper separation between physical heap storage and logical primary index entries with record pointers.
        2. **Analyze Key Distributions:** Empirically evaluate model accuracy and error bounds across Uniform, Normal, Log-Normal, and real-world SOSD datasets.
        3. **Demonstrate Inference Optimization:** Compare framework-heavy scikit-learn inference with direct compiled arithmetic evaluation.
        4. **Live Verification:** Execute reproducible, freshly measured benchmarks on live queries with 0 tolerance for retrieval mismatches.
        """)

    with col_b:
        st.markdown("### Scope and Disclaimers")
        academic_callout(
            "This application is an educational and empirical research visualizer focusing on a single-stage Linear Regression "
            "model over static sorted key spaces. Production learned indexes (e.g., Recursive Model Indexes, SageDB, ALEX, PGM-index) "
            "employ hierarchical multi-stage model trees, spline approximations, and buffer layers for real-time dynamic inserts/updates. "
            "Timings reflect Python execution environments and illustrate algorithmic concepts rather than production C++ kernel speeds.",
            callout_type="primary",
            title="Educational Prototype Notice"
        )

    st.markdown("---")

    # Team Attribution
    st.markdown("### DBMS Mini Project — Group 4")
    team_df = [
        {"Roll Number": "25124016", "Name": "Arshjeet Singh", "Role": "Full-Stack Development & Architecture"},
        {"Roll Number": "25124014", "Name": "Armaan Bansal", "Role": "Algorithm Implementation & Benchmarking"},
        {"Roll Number": "25124015", "Name": "Arpit Yadav", "Role": "Machine Learning & Distribution Modeling"},
        {"Roll Number": "25124013", "Name": "Anup Kumar", "Role": "Data Engineering & Theoretical Research"},
    ]
    st.dataframe(team_df, use_container_width=True, hide_index=True)
