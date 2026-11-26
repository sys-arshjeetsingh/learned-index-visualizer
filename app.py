"""
Learned Index Visualizer — Main Application Entry Point
Academic Research System Aesthetic | Minimalist Design System

DBMS Mini Project | Group 4
Run with:
    streamlit run app.py
"""

import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Learned Index Visualizer",
    page_icon="None",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Custom Academic CSS ---
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar Header & Branding ---
st.sidebar.markdown("""
<div style="padding: 0.75rem 0.2rem 1rem 0.2rem; border-bottom: 1px solid #1E2E48; margin-bottom: 1rem;">
    <div style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; color: #368A83; text-transform: uppercase;">
        DBMS Research Prototype
    </div>
    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.01em; margin-top: 2px;">
        Learned Index Visualizer
    </div>
    <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 2px;">
        ML-Based Database Indexing
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Options without emojis
PAGES = [
    "Overview",
    "Industry Key Lookup",
    "Dataset Explorer",
    "Learned Index Explained",
    "Algorithm Comparison",
    "Distribution Analysis",
    "Performance Analytics",
    "Technical Explanation",
]

page = st.sidebar.radio(
    "Navigation",
    PAGES,
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Academic Project Metadata in Sidebar Footer
st.sidebar.markdown("""
<div style="font-size: 0.72rem; color: #94A3B8; line-height: 1.45; padding: 0.2rem;">
    <div style="font-weight: 600; color: #E2E8F0; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.4rem;">
        Group 4 &bull; DBMS Mini Project
    </div>
    <div>Arshjeet Singh &bull; 25124016</div>
    <div>Armaan Bansal &bull; 25124014</div>
    <div>Arpit Yadav &bull; 25124015</div>
    <div>Anup Kumar &bull; 25124013</div>
    <div style="margin-top: 0.6rem; color: #64748B; font-size: 0.68rem;">
        University DBMS Submission &bull; 2026
    </div>
</div>
""", unsafe_allow_html=True)

# --- Page Routing ---
if page == "Overview":
    from pages.overview import render
    render()
elif page == "Industry Key Lookup":
    from pages.industry_lookup import render
    render()
elif page == "Dataset Explorer":
    from pages.dataset_explorer import render
    render()
elif page == "Learned Index Explained":
    from pages.learned_index import render
    render()
elif page == "Algorithm Comparison":
    from pages.algorithm_comparison import render
    render()
elif page == "Distribution Analysis":
    from pages.distribution_analysis import render
    render()
elif page == "Performance Analytics":
    from pages.performance_analytics import render
    render()
elif page == "Technical Explanation":
    from pages.technical_explanation import render
    render()
