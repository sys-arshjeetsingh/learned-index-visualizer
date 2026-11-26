"""
Page 6: Technical Explanation — Theoretical foundations, mathematical derivations,
and academic critique of learned indexing.
"""

import streamlit as st
from components.metrics import page_header, academic_callout


def render():
    """Render the Technical Explanation page."""

    page_header(
        title="Technical and Theoretical Foundation",
        subtitle="Mathematical Formulations, Database Physical Organization, and Algorithmic Critique",
        pills=[
            {"text": "Database Theory", "type": "dark"},
            {"text": "Mathematical Modeling", "type": "teal"}
        ]
    )

    # 1. Database Indexing Principles
    st.markdown("### 1. Database Indexing Principles and Physical Storage")
    st.markdown("""
    In a relational database management system (RDBMS), data files are typically organized into **Heap Files**:
    - **Physical Record Storage:** Records are appended sequentially into storage blocks on disk or memory pages.
    - **Unsorted Order:** Records arrive in temporal or transaction order, rendering physical key searches an $O(N)$ table scan.
    - **Index Architecture:** A secondary or primary index provides an auxiliary data structure containing pairs of `(search_key, record_pointer)`.
    - **Physical Independence:** The index is ordered logically by key, while the underlying heap file remains unsorted.
    """)

    st.markdown("---")

    # 2. Traditional Index Structures vs. Learned Models
    st.markdown("### 2. Traditional Index Structures vs. Learned Models")
    st.markdown("""
    #### Traditional B+ Tree Indexes
    - **Structure:** Balanced search tree with high fanout (typically 100 to 1,000 pointers per node).
    - **Guarantees:** Deterministic worst-case lookup, insertion, and deletion complexity:
      $$T_{\\text{lookup}} = O(\\log_B N)$$
    - **Overhead:** High storage consumption due to internal index nodes, page fragmentation, and pointer overhead.

    #### The Learned Index Paradigm (Kraska et al., 2018)
    The fundamental insight of learned indexing is recognizing that an index is an approximation of the Cumulative Distribution Function (CDF):
    $$F(x) = P(X \\le x)$$
    For a dataset of $N$ sorted unique keys, the exact position $p(x)$ of key $x$ satisfies:
    $$p(x) = \\lfloor N \\cdot F(x) \\rfloor$$
    Instead of traversing a tree of discrete branch pointers, a machine learning model directly estimates $F(x)$ as a continuous mathematical function:
    $$\\hat{p}(x) = f(x)$$
    """)

    st.markdown("---")

    # 3. Mathematical Model: Linear Regression
    st.markdown("### 3. Mathematical Model: Linear Regression as a 1D CDF Approximator")
    st.markdown("""
    For uniformly distributed keys over range $[x_{\\min}, x_{\\max}]$, the CDF is strictly linear:
    $$F(x) = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}$$
    Therefore, the optimal mapping function is a linear equation:
    $$\\hat{y} = m \\cdot x + c$$
    where:
    - $m = \\frac{N - 1}{x_{\\max} - x_{\\min}}$ (Key density slope)
    - $c = -m \\cdot x_{\\min}$ (Intercept bias)

    Using Ordinary Least Squares (OLS) regression, the coefficients minimize the Residual Sum of Squares (RSS):
    $$\\min_{m, c} \\sum_{i=0}^{N-1} (y_i - (m \\cdot x_i + c))^2$$
    """)

    st.markdown("---")

    # 4. Error Bounds and Correction Search
    st.markdown("### 4. Error Bounds and Correction Search")
    st.markdown("""
    Because a continuous regression model outputs floating-point approximations, discrete integer positions are obtained via truncation:
    $$\\hat{p} = \\lfloor m \\cdot x + c \\rfloor$$

    #### Maximum Absolute Error Bound
    The empirical worst-case training error across all training records is defined as:
    $$\\epsilon_{\\max} = \\max_{0 \\le i < N} |\\hat{p}(x_i) - i|$$

    #### Search Window Definition
    To guarantee that any present key is captured without false negatives, we define the conservative search radius:
    $$r = \\epsilon_{\\max} + 1$$
    The localized search interval $[\\text{lo}, \\text{hi}]$ is clamped to table boundaries:
    $$\\text{lo} = \\max(0, \\hat{p} - r), \\quad \\text{hi} = \\min(N - 1, \\hat{p} + r)$$
    The local search window size is $W = \\text{hi} - \\text{lo} + 1 \\le 2r + 1$.
    A binary search inside this window takes at most:
    $$T_{\\text{local}} = O(\\log_2 W) \\ll O(\\log_2 N)$$
    """)

    st.markdown("---")

    # 5. Framework Overhead: Naive vs. Optimized Inference
    st.markdown("### 5. Framework Overhead: Naive vs. Optimized Inference")
    st.markdown("""
    A major finding in empirical database literature is the **interpreter and framework penalty**:
    
    1. **Naive Inference (`scikit-learn`):**
       Calling `model.predict([[key]])` invokes Python function wrappers, NumPy 2D array allocations, data-type validations, and matrix multiplication kernels. 
       This adds $\\approx 300 - 500 \\; \\mu\\text{s}$ of overhead per query.
    2. **Optimized Inference (Direct Arithmetic):**
       Evaluating `int(m * key + c)` compiles to a single Multiply-Accumulate (MAC) instruction, executing in $< 0.1 \\; \\mu\\text{s}$ in compiled kernels.
       Even in interpreted Python, direct arithmetic reduces latency to $\\approx 5 - 10 \\; \\mu\\text{s}$.
    """)

    st.markdown("---")

    # 6. Realistic Trade-Offs and Limitations
    st.markdown("### 6. Realistic Trade-Offs and Academic Limitations")

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("""
        #### Advantages
        - **Ultra-Compact Memory:** A single linear model replaces millions of tree pointer nodes with just two numbers ($m, c$).
        - **$O(1)$ Direct Addressing:** Avoids multiple CPU cache misses associated with traversing hierarchical pointer trees.
        - **SIMD / GPU Vectorization:** Mathematical dot-products can be vectorized across thousands of parallel queries.
        """)

    with col_l2:
        st.markdown("""
        #### Limitations
        - **Non-Linear Key Distributions:** Fails on heavily skewed, power-law, or multi-modal data without multi-stage hierarchical models.
        - **Static Workloads:** Standard regression models do not easily accommodate real-time inserts, updates, or deletes.
        - **Retraining Cost:** Model training requires upfront compute before queries can be processed.
        - **Out-of-Range Extrapolation:** Keys outside the training range produce inaccurate predictions that require bound clamping.
        """)

    academic_callout(
        "<strong>Summary Conclusion:</strong> Learned indexes do not render traditional indexes obsolete. "
        "Instead, they represent an important evolution toward data-adaptive database structures, where machine learning "
        "complements traditional algorithms to achieve optimal trade-offs between memory footprint and query throughput.",
        callout_type="teal",
        title="Concluding Assessment"
    )


if __name__ == "__main__":
    render()
