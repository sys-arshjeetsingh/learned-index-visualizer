"""
Diagrams and memory layout components for the Learned Index Visualizer.
Designed with academic rigor, showing physical record storage, index entries,
and search step histories without decorative emoji clutter.
"""

import streamlit as st


def architecture_diagram():
    """Render the academic pipeline architecture diagram."""
    st.markdown("""
    <div class="architecture-container">
        <div class="arch-node">
            <div class="node-step">Stage 1</div>
            <div class="node-title">Database Records</div>
            <div class="node-desc">Physical Heap Storage</div>
        </div>
        <div class="arch-connector">→</div>
        <div class="arch-node">
            <div class="node-step">Stage 2</div>
            <div class="node-title">Sorted Index</div>
            <div class="node-desc">Sorted (Key, Row_ID)</div>
        </div>
        <div class="arch-connector">→</div>
        <div class="arch-node">
            <div class="node-step">Stage 3</div>
            <div class="node-title">Linear Model</div>
            <div class="node-desc">y = mx + c (CDF Approx)</div>
        </div>
        <div class="arch-connector">→</div>
        <div class="arch-node">
            <div class="node-step">Stage 4</div>
            <div class="node-title">Local Window Search</div>
            <div class="node-desc">Binary Search in [lo, hi]</div>
        </div>
        <div class="arch-connector">→</div>
        <div class="arch-node">
            <div class="node-step">Stage 5</div>
            <div class="node-title">Record Retrieval</div>
            <div class="node-desc">Row Dereference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def physical_vs_index_diagram(unsorted_records: list, sorted_index: list):
    """
    Renders a side-by-side visualization comparing:
    1. Physical Table (Heap File): Unsorted insertion order.
    2. Primary Index: Sorted entries of (Key, Row ID pointer).
    """
    heap_rows = ""
    for r in unsorted_records[:6]:
        heap_rows += f"""
        <tr>
            <td style="font-family: monospace; font-weight: 600; color: #64748B;">Row {r['row_id']}</td>
            <td style="font-family: monospace; font-weight: 600; color: #202936;">{r['key']:,}</td>
            <td style="color: #475569;">{r.get('name', r.get('payload', 'Record'))}</td>
            <td style="font-size: 0.75rem; color: #64748B;">{r.get('department', 'Active')}</td>
        </tr>
        """

    index_rows = ""
    for idx, (k, r_id) in enumerate(sorted_index[:6]):
        index_rows += f"""
        <tr>
            <td style="font-family: monospace; font-weight: 600; color: #368A83;">Pos {idx}</td>
            <td style="font-family: monospace; font-weight: 600; color: #315E91;">{k:,}</td>
            <td style="font-family: monospace; color: #64748B;">→ Row {r_id}</td>
        </tr>
        """

    st.markdown(f"""
    <div class="storage-layout">
        <div class="storage-column">
            <div class="storage-header">Physical Table (Unsorted Heap File)</div>
            <p style="font-size: 0.78rem; color: #64748B; margin-bottom: 0.5rem;">
                Records reside on disk in arrival order. A search requires a full table scan O(N) without an index.
            </p>
            <table class="academic-table" style="margin: 0;">
                <thead>
                    <tr>
                        <th>Storage Row ID</th>
                        <th>Record Key</th>
                        <th>Payload (Name)</th>
                        <th>Department</th>
                    </tr>
                </thead>
                <tbody>
                    {heap_rows}
                </tbody>
            </table>
        </div>
        <div class="storage-column">
            <div class="storage-header">Logical Primary Index (Sorted Entries)</div>
            <p style="font-size: 0.78rem; color: #64748B; margin-bottom: 0.5rem;">
                Sorted by key. Preserves the record association via row pointer. The learned model learns: Key → Index Position.
            </p>
            <table class="academic-table" style="margin: 0;">
                <thead>
                    <tr>
                        <th>Index Pos</th>
                        <th>Key Value</th>
                        <th>Physical Row Pointer</th>
                    </tr>
                </thead>
                <tbody>
                    {index_rows}
                </tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)


def academic_comparison_table():
    """Render formal method comparison table."""
    st.markdown("""
    <table class="academic-table">
        <thead>
            <tr>
                <th>Operational Property</th>
                <th>Traditional Binary Search</th>
                <th>Naive Learned Index</th>
                <th>Optimized Learned Index</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Theoretical Framework</strong></td>
                <td>Classic Divide-and-Conquer</td>
                <td>Model Inference + Bounded Search</td>
                <td>Direct Arithmetic + Bounded Search</td>
            </tr>
            <tr>
                <td><strong>Index Space Scanned</strong></td>
                <td>Full Index [0 ... N-1]</td>
                <td>Local Error Window [lo ... hi]</td>
                <td>Local Error Window [lo ... hi]</td>
            </tr>
            <tr>
                <td><strong>Time Complexity</strong></td>
                <td>O(log N)</td>
                <td>O(1) inference + O(log W)</td>
                <td>O(1) arithmetic + O(log W)</td>
            </tr>
            <tr>
                <td><strong>Inference Mechanism</strong></td>
                <td>None (No Model)</td>
                <td><code>model.predict([[key]])</code> (sklearn)</td>
                <td><code>int(slope * key + intercept)</code></td>
            </tr>
            <tr>
                <td><strong>Per-Query Overhead</strong></td>
                <td>Low (Native Python/C loops)</td>
                <td>Very High (Array boxing & validation)</td>
                <td>Minimal (Single MAC instruction)</td>
            </tr>
            <tr>
                <td><strong>Offline Training</strong></td>
                <td>Zero (Immediate use)</td>
                <td>Required (Linear Regression fit)</td>
                <td>Required (Linear Regression fit)</td>
            </tr>
            <tr>
                <td><strong>Record Dereferencing</strong></td>
                <td>Pointer resolution at match</td>
                <td>Pointer resolution at match</td>
                <td>Pointer resolution at match</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)


def search_steps_display(steps: list, target: int):
    """Render step-by-step trace of comparison operations."""
    if not steps:
        st.markdown("<p style='color: #64748B; font-size: 0.82rem;'>No search steps recorded.</p>", unsafe_allow_html=True)
        return

    for s in steps:
        is_found = (s.get("action") == "FOUND")
        act_class = "found" if is_found else "navigate"
        act_label = "MATCH FOUND" if is_found else s.get("action", "")

        st.markdown(f"""
        <div class="trace-step-row">
            <span class="trace-step-num">Step {s['step']}</span>
            <span class="trace-step-range">Window: [{s['low']:,} ... {s['high']:,}]</span>
            <span class="trace-step-mid">Mid Index {s['mid']:,} &rarr; Key {s['mid_key']:,}</span>
            <span class="trace-step-action {act_class}">{act_label}</span>
        </div>
        """, unsafe_allow_html=True)
