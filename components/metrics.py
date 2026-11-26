"""
Academic Metric card components and text formatters.
Designed cleanly without emoji spam, using crisp borders and clean typography.
"""

import streamlit as st


def metric_card(label: str, value: str, description: str = "", delta: str = "", delta_type: str = "neutral"):
    """Render a minimalist academic metric box."""
    delta_html = ""
    if delta:
        color = "#368A83" if delta_type == "good" else ("#DC2626" if delta_type == "bad" else "#64748B")
        delta_html = f'<div style="color: {color}; font-size: 0.74rem; font-weight: 500; margin-top: 3px;">{delta}</div>'

    desc_html = f'<div class="metric-desc">{description}</div>' if description else ""

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
        {desc_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def metric_row(metrics: list):
    """Render a grid row of academic metric cards."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            metric_card(
                label=m.get("label", ""),
                value=m.get("value", ""),
                description=m.get("description", ""),
                delta=m.get("delta", ""),
                delta_type=m.get("delta_type", "neutral")
            )


def academic_callout(text: str, callout_type: str = "primary", title: str = ""):
    """Render an academic callout box."""
    css_class = "academic-box"
    if callout_type == "teal":
        css_class += " teal"
    elif callout_type == "warning":
        css_class += " warning"

    header_html = f'<div style="font-weight: 600; font-size: 0.84rem; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.03em;">{title}</div>' if title else ""

    st.markdown(f"""
    <div class="{css_class}">
        {header_html}
        <div>{text}</div>
    </div>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", pills: list = None):
    """Render standard academic page header with optional metadata pills."""
    pill_html = ""
    if pills:
        pill_spans = []
        for p in pills:
            cls = "academic-pill"
            if p.get("type") == "teal":
                cls += " teal"
            elif p.get("type") == "dark":
                cls += " dark"
            pill_spans.append(f'<span class="{cls}">{p.get("text", "")}</span>')
        pill_html = f'<div style="margin-top: 6px;">{" ".join(pill_spans)}</div>'

    st.markdown(f"""
    <div class="page-header-academic">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
        {pill_html}
    </div>
    """, unsafe_allow_html=True)
