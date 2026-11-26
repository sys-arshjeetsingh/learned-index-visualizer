"""Practical CSV-backed learned-index lookup workflow."""

import hashlib
import time
from typing import Optional

import pandas as pd
import streamlit as st

from components.metrics import academic_callout, metric_row, page_header
from core.dataset import load_custom_csv
from core.learned_index_model import LearnedIndexModel


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns that contain at least one numeric key candidate."""
    return [
        column
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def _build_index(uploaded_file, key_column: str, payload_column: Optional[str]):
    """Create a per-session exact learned index from an uploaded CSV snapshot."""
    bundle = load_custom_csv(uploaded_file, key_column, payload_column)
    model = LearnedIndexModel()
    model.train(bundle)
    return bundle, model


def render():
    """Render the practical lookup page."""
    page_header(
        title="Industry key lookup",
        subtitle="Build an exact learned index over a CSV snapshot and retrieve a complete record by numeric key.",
        pills=[
            {"text": "Practical workflow", "type": "dark"},
            {"text": "Exact retrieval", "type": "teal"},
        ],
    )

    academic_callout(
        "Upload a CSV snapshot, choose its numeric primary-key column, build the index once, "
        "then query individual keys. The learned model narrows the candidate range and a bounded "
        "binary search confirms the result, so the returned record is exact.",
        callout_type="teal",
        title="How to use this section",
    )

    uploaded_file = st.file_uploader(
        "CSV dataset",
        type=["csv"],
        key="industry_csv",
        help="The file stays in this app session and is not written back to your source system.",
    )
    if uploaded_file is None:
        st.info("Upload a CSV file to create a searchable learned index.")
        return

    try:
        preview = pd.read_csv(uploaded_file, nrows=20)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"The CSV could not be read: {error}")
        return

    candidates = _numeric_columns(preview)
    if not candidates:
        st.error("No numeric key column was found. Choose a CSV with an integer identifier column.")
        return

    st.dataframe(preview, hide_index=True, alt="Preview of the first 20 uploaded CSV rows")

    config_col, details_col = st.columns([2, 1])
    with config_col:
        key_column = st.selectbox(
            "Primary key column",
            candidates,
            key="industry_key_column",
            help="Each distinct key maps to one record. For duplicate keys, the first row is returned.",
        )


        payload_options = ["None"] + list(preview.columns)
        payload_choice = st.selectbox(
            "Display column (optional)",
            payload_options,
            key="industry_payload_column",
        )
    with details_col:
        st.caption("Index policy")
        st.write("Unique numeric keys, sorted in memory. Rebuild after the source snapshot changes.")

    payload_column = None if payload_choice == "None" else payload_choice
    file_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    index_signature = f"{file_signature}:{key_column}:{payload_column}"

    if st.button("Build learned index", type="primary", icon=":material/account_tree:", key="industry_build"):
        try:
            with st.spinner("Validating records and training the learned index..."):
                bundle, model = _build_index(uploaded_file, key_column, payload_column)
            st.session_state.industry_bundle = bundle
            st.session_state.industry_model = model
            st.session_state.industry_signature = index_signature
            st.session_state.pop("industry_result", None)
            st.success("Index built. You can now run exact key lookups.")
        except (ValueError, OverflowError, TypeError) as error:
            st.error(f"Unable to build an index from this file: {error}")

    if st.session_state.get("industry_signature") != index_signature:
        st.warning("Change detected. Build the index again before querying this configuration.")
        return

    bundle = st.session_state.get("industry_bundle")
    model = st.session_state.get("industry_model")
    if bundle is None or model is None:
        return

    summary = bundle.summary()
    metric_row([
        {"label": "Indexed keys", "value": f"{summary['indexed_unique_keys']:,}", "description": "Unique searchable keys"},
        {"label": "Duplicate keys", "value": f"{summary['duplicate_count']:,}", "description": "First record retained"},
        {"label": "Search window", "value": f"{model.search_radius * 2 + 1:,}", "description": "Maximum candidate range"},
        {"label": "Model fit (R²)", "value": f"{model.r2:.4f}", "description": "Key-to-position fit"},
    ])

    default_key = str(int(bundle.index_keys[len(bundle.index_keys) // 2]))
    with st.form("industry_lookup_form", border=False):
        search_key_text = st.text_input(
            "Key to retrieve",
            value=default_key,
            help="Enter an exact integer key from the selected primary-key column.",
        )
        submitted = st.form_submit_button("Find record", type="primary", icon=":material/search:")

    if submitted:
        try:
            target = int(search_key_text.strip())
        except ValueError:
            st.error("Enter a whole-number key, for example 1024.")
        else:
            start = time.perf_counter()
            result = model.search_optimized(target)
            elapsed_us = (time.perf_counter() - start) * 1_000_000
            st.session_state.industry_result = (target, result, elapsed_us)

    result_state = st.session_state.get("industry_result")
    if result_state is None:
        return

    target, result, elapsed_us = result_state
    if not result["found"]:
        st.warning(f"No record was found for key {target:,}.")
        return

    st.success(f"Exact match found for key {target:,}.")
    metric_row([
        {"label": "Lookup time", "value": f"{elapsed_us:,.1f} μs", "description": "Model prediction plus bounded search"},
        {"label": "Index position", "value": f"{result['index']:,}", "description": "Sorted index location"},
        {"label": "Candidate window", "value": f"{result['search_window']:,}", "description": "Keys examined by the local search"},
        {"label": "Comparisons", "value": str(result['comparisons']), "description": "Bounded binary-search comparisons"},
    ])

    record = pd.DataFrame([result["record"]])
    st.dataframe(record, hide_index=True, alt="Complete record returned for the queried key")

    with st.expander("Operational guidance"):
        st.markdown(
            "This is a CSV-snapshot lookup tool. Rebuild the index after inserts, deletes, or key updates. "
            "For a production database, persist the trained parameters and error bound next to the table's "
            "primary-key index, and retain the bounded-search verification used here."
        )


if __name__ == "__main__":
    render()
