"""
Page 2: Dataset Explorer — Exploration of synthetic and real-world SOSD benchmark datasets.
Academic research aesthetic, supporting 8 dataset options, duplicate statistics,
physical record view, and reproducible subset sampling.
"""

import streamlit as st
import pandas as pd
import numpy as np

from core.dataset import (
    generate_synthetic_uniform,
    generate_synthetic_normal,
    generate_synthetic_lognormal,
    load_sosd_dataset,
    load_custom_csv,
    SOSD_METADATA
)
from components.metrics import page_header, metric_row, academic_callout
from components.charts import key_distribution_plot


DATASET_OPTIONS = [
    "1. Synthetic Uniform",
    "2. Synthetic Normal (Gaussian)",
    "3. Synthetic Log-Normal (Skewed)",
    "4. Real-world Wikipedia Timestamps (SOSD)",
    "5. Real-world Amazon Books (SOSD)",
    "6. Real-world Facebook User IDs (SOSD)",
    "7. Real-world OpenStreetMap Cell IDs (SOSD)",
    "8. Upload Custom CSV"
]


def load_selected_dataset(dataset_choice: str, sample_size: int, seed: int, uploaded_file=None, key_col=None, val_col=None):
    """Factory loader for selected dataset configuration."""
    if "Uniform" in dataset_choice:
        return generate_synthetic_uniform(size=sample_size, seed=seed)
    elif "Normal" in dataset_choice:
        return generate_synthetic_normal(size=sample_size, seed=seed)
    elif "Log-Normal" in dataset_choice:
        return generate_synthetic_lognormal(size=sample_size, seed=seed)
    elif "Wikipedia" in dataset_choice:
        return load_sosd_dataset("wiki_ts", sample_size=sample_size, seed=seed)
    elif "Books" in dataset_choice:
        return load_sosd_dataset("books", sample_size=sample_size, seed=seed)
    elif "Facebook" in dataset_choice:
        return load_sosd_dataset("fb", sample_size=sample_size, seed=seed)
    elif "OpenStreetMap" in dataset_choice:
        return load_sosd_dataset("osm", sample_size=sample_size, seed=seed)
    elif "Custom CSV" in dataset_choice:
        if uploaded_file is not None and key_col:
            return load_custom_csv(uploaded_file, key_column=key_col, value_column=val_col)
        else:
            return generate_synthetic_uniform(size=sample_size, seed=seed)
    return generate_synthetic_uniform(size=sample_size, seed=seed)


def render():
    """Render the Dataset Explorer page."""

    page_header(
        title="Dataset Explorer",
        subtitle="Key Distribution Characterization, Sampling, and Physical Record Storage",
        pills=[
            {"text": "Data Management", "type": "dark"},
            {"text": "SOSD Benchmarks", "type": "primary"}
        ]
    )

    # --- Dataset Selection Controls ---
    st.markdown("### Configuration and Data Source")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        dataset_choice = st.selectbox(
            "Dataset Source",
            DATASET_OPTIONS,
            index=0,
            help="Select synthetic benchmark distributions or real-world SOSD database workloads."
        )

    with col2:
        sample_size = st.selectbox(
            "Record Sample Size",
            [10_000, 50_000, 100_000, 200_000],
            index=2,
            help="Reproducible subset size for model training and indexing."
        )

    with col3:
        seed = st.number_input("RNG Seed", min_value=0, max_value=99999, value=21)

    # CSV Upload parameters if selected
    uploaded_file = None
    key_col = None
    val_col = None

    if "Custom CSV" in dataset_choice:
        st.markdown("#### Upload Custom Dataset")
        up_col1, up_col2, up_col3 = st.columns([2, 1, 1])
        with up_col1:
            uploaded_file = st.file_uploader("Select CSV File", type=["csv"])
        if uploaded_file is not None:
            preview_df = pd.read_csv(uploaded_file, nrows=5)
            with up_col2:
                key_col = st.selectbox("Primary Key Column (Numeric)", preview_df.columns)
            with up_col3:
                val_cols = ["None"] + list(preview_df.columns)
                sel_val = st.selectbox("Payload Column (Optional)", val_cols)
                val_col = None if sel_val == "None" else sel_val

    # Load / Cache dataset bundle
    cache_key = f"{dataset_choice}_{sample_size}_{seed}_{uploaded_file.name if uploaded_file else 'none'}"
    if "current_bundle" not in st.session_state or st.session_state.get("bundle_cache_key") != cache_key:
        with st.spinner("Preparing dataset and building physical index..."):
            bundle = load_selected_dataset(dataset_choice, sample_size, seed, uploaded_file, key_col, val_col)
            st.session_state.current_bundle = bundle
            st.session_state.bundle_cache_key = cache_key
            # Invalidate trained model and benchmarks when dataset changes
            st.session_state.pop("model", None)
            st.session_state.pop("benchmark_results", None)

    bundle = st.session_state.current_bundle
    summary = bundle.summary()

    # --- Dataset Statistical Metadata ---
    st.markdown("---")
    st.markdown(f"### Dataset Profile: {bundle.name}")

    metric_row([
        {"label": "Classification", "value": bundle.category, "description": "Source origin"},
        {"label": "Total Physical Records", "value": f"{summary['total_records']:,}", "description": "Heap table rows"},
        {"label": "Unique Index Keys", "value": f"{summary['indexed_unique_keys']:,}", "description": "Primary index entries"},
        {"label": "Duplicate Keys", "value": f"{summary['duplicate_count']:,}", "description": "Handled during indexing"},
    ])

    st.markdown("")

    metric_row([
        {"label": "Minimum Key", "value": f"{summary['min_key']:,}", "description": "Index start bound"},
        {"label": "Maximum Key", "value": f"{summary['max_key']:,}", "description": "Index end bound"},
        {"label": "Key Span (Range)", "value": f"{summary['range']:,}", "description": "Max - Min"},
        {"label": "Standard Deviation", "value": f"{summary['std']:,.1f}", "description": "Dispersion measure"},
    ])

    academic_callout(
        f"<strong>Dataset Context:</strong> {bundle.description} "
        f"When sampling subsets from large real-world traces, the sampling methodology draws contiguous or uniform stride slices "
        f"to preserve the authentic CDF density variations and clustering inherent to the original benchmark.",
        callout_type="teal",
        title="Distribution Notes"
    )

    st.markdown("---")

    # --- Key Distribution Visualization ---
    st.markdown("### Key Distribution Analysis")
    fig = key_distribution_plot(bundle.index_keys, n_bins=80, title=f"Key Frequency Distribution — {bundle.name}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Physical Table vs Index Preview ---
    st.markdown("### Database Storage Representation")
    col_heap, col_idx = st.columns(2)

    with col_heap:
        st.markdown("**Physical Table (Heap Storage - Arrival Order)**")
        heap_preview = pd.DataFrame(bundle.physical_records[:8])
        st.dataframe(heap_preview, use_container_width=True, hide_index=True)

    with col_idx:
        st.markdown("**Primary Index Entries (Sorted by Key with Row Pointers)**")
        idx_df = pd.DataFrame({
            "Index Position": range(min(8, len(bundle.index_keys))),
            "Key Value": bundle.index_keys[:8],
            "Physical Row Pointer": [f"Row {r_id}" for r_id in bundle.index_row_ids[:8]],
        })
        st.dataframe(idx_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Interactive Key Lookup with Record Retrieval ---
    st.markdown("### Direct Key Lookup and Record Retrieval")
    search_col1, search_col2 = st.columns([1, 2])

    with search_col1:
        search_target = st.number_input(
            "Enter Search Key",
            value=int(bundle.index_keys[len(bundle.index_keys) // 2]),
            step=1
        )
        execute_search = st.button("Query Index", use_container_width=True)

    with search_col2:
        if execute_search or "last_search_target" in st.session_state:
            st.session_state.last_search_target = search_target
            # Binary search over index keys
            keys = bundle.index_keys
            idx = np.searchsorted(keys, search_target)
            if idx < len(keys) and keys[idx] == search_target:
                row_id = bundle.index_row_ids[idx]
                matched_record = bundle.get_record(row_id)
                st.markdown(f"""
                <div class="academic-box teal" style="margin: 0;">
                    <strong>Record Retrieved Successfully</strong><br>
                    Index Position: <code>{idx:,}</code> | Physical Row ID: <code>{row_id}</code><br>
                    Payload Record: <code>{matched_record}</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="academic-box warning" style="margin: 0;">
                    <strong>Key Not Found:</strong> Key <code>{search_target:,}</code> does not exist in this index.
                </div>
                """, unsafe_allow_html=True)
