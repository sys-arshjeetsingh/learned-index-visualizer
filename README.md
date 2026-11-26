# Learned Index Visualizer

**Machine Learning-Based Database Indexing | Academic Research & Evaluation System**

A functional, academic database research tool that demonstrates the principles of machine learning-based database indexing and empirically compares learned models against traditional binary search.

Developed for university DBMS course evaluation by **DBMS Mini Project — Group 4**.

---

## 1. System Overview

Database indexing structures such as B+ Trees consume substantial memory overhead and navigate fixed-depth pointer trees agnostic of data patterns. In 2018, Kraska et al. (*The Case for Learned Index Structures*, SIGMOD) demonstrated that indexes are fundamentally mathematical functions that map keys to positions:

$$\text{Position}(x) = \lfloor N \cdot F_X(x) \rfloor$$

where $F_X(x)$ is the Cumulative Distribution Function (CDF) of the sorted keys.

This application provides an interactive research platform to:
1. **Model CDFs:** Approximate key-to-position mappings using Linear Regression.
2. **Preserve Database Semantics:** Maintain proper separation between physical heap storage (records) and logical primary index entries (sorted keys with row pointers).
3. **Compare Key Distributions:** Evaluate model behavior across Uniform, Normal, Log-Normal, and real-world SOSD benchmark workloads.
4. **Benchmark Overhead:** Measure the impact of framework evaluation (scikit-learn `predict`) versus direct compiled arithmetic evaluation.

---

## 2. Key Features

- **Academic Minimalist UI/UX:** Styled as a professional database administration and research dashboard using a dedicated design system (Off-White `#F7F8FA`, Dark Navy `#17243A`, Primary Steel Blue `#315E91`, Muted Teal `#368A83`, Charcoal `#202936`). No decorative emoji clutter or generic AI dashboard styling.
- **Real-World SOSD Benchmark Integration:** Built-in loaders for the official SOSD learned-index benchmark suites:
  - Wikipedia Edit Timestamps (`wiki_ts`)
  - Amazon Book Sales Popularity Ranks (`books`)
  - Facebook Social Graph User IDs (`fb`)
  - OpenStreetMap 64-bit Morton Cell IDs (`osm`)
  - User Custom CSV Upload
- **Key-Value Record Handling:** Records are represented as full key-value rows in physical heap storage. The primary index stores `(Key, Row_ID)` pointers. When a search succeeds, the index position resolves to the physical `Row_ID`, returning the full matching record.
- **Visual "Why Must Index Keys Be Sorted?" Explanation:** Dedicated visual comparison contrasting unsorted physical heap files with sorted index structures, explaining why sorting is mathematically necessary to form a monotonic CDF.
- **Cross-Distribution Analysis:** Empirical comparison matrix profiling model $R^2$ fit, MAE, worst-case residual error, local search window size, and query latencies across different key distributions.
- **Live Empirical Benchmarking:** Freshly measured query latencies using `time.perf_counter()`, verified for 0 retrieval mismatches against ground-truth binary search.

---

## 3. Evaluated Search Algorithms

| Method | Conceptual Mechanism | Per-Query Overhead | Memory Complexity |
|---|---|---|---|
| **Binary Search (Baseline)** | Whole-array divide-and-conquer over sorted index | Low | $O(N)$ index keys |
| **Naive Learned Index** | Evaluates `model.predict([[key]])` + local binary search | High (NumPy/sklearn array boxing) | $O(1)$ model + $O(W)$ window |
| **Optimized Learned Index** | Evaluates $\text{round}(m \cdot x + c)$ + local binary search | Minimal (Direct floating-point arithmetic) | $O(1)$ model + $O(W)$ window |

---

## 4. Project Architecture

```
learned-index-visualizer/
├── app.py                          # Main Streamlit dashboard router & layout
├── core/
│   ├── dataset.py                  # Synthetic generators, SOSD benchmark loader, CSV upload, & KV records
│   ├── binary_search.py            # Baseline binary search, range-bounded search, record dereferencing
│   ├── learned_index_model.py      # Linear Regression model, training, 3 search methods, error radius
│   └── benchmark.py                # Reproducible live benchmarking engine with zero-mismatch verification
├── pages/
│   ├── overview.py                 # Academic landing page, architecture diagram, and summary cards
│   ├── dataset_explorer.py         # 8 dataset options, duplicate statistics, sampling, & record lookup
│   ├── learned_index.py            # "Why Must Index Keys Be Sorted?", physical vs index view, 4-step trace
│   ├── algorithm_comparison.py     # Side-by-side search trace returning full key-value records
│   ├── distribution_analysis.py    # Multi-distribution comparison (Uniform, Normal, Skewed, SOSD)
│   ├── performance_analytics.py    # Live benchmarking dashboard with Plotly visualisations
│   └── technical_explanation.py    # Database theory, CDF derivations, error bounds, and limits
├── components/
│   ├── charts.py                   # Plotly charts styled with the academic research palette
│   ├── metrics.py                  # Minimalist academic KPI cards and callout components
│   └── diagrams.py                 # Architecture flow, storage layout, and comparison tables
├── assets/
│   └── style.css                   # Custom academic CSS design system
├── tests/
│   └── test_core.py                # Comprehensive test suite (distributions, SOSD, dereferencing, 0 mismatches)
├── requirements.txt                # Python dependencies
├── README.md                       # Academic project documentation
└── .gitignore                      # Git ignore rules
```

---

## 5. Installation and Execution

### Prerequisites
- Python 3.9 or higher
- Windows, macOS, or Linux

### Installation

1. Navigate to the project directory:
   ```powershell
   cd "learned-index-visualizer"
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```powershell
   streamlit run app.py
   ```

4. The application will open automatically in your browser at `http://localhost:8501`.

---

## 6. Running the Test Suite

Execute the automated test suite covering all distributions, real-world SOSD loaders, record dereferencing, and zero-mismatch verification:

```powershell
python -m pytest tests/test_core.py -v
```

---

## 7. Project Authors

**DBMS Mini Project — Group 4**

- **Arshjeet Singh** — Roll No: `25124016`
- **Armaan Bansal** — Roll No: `25124014`
- **Arpit Yadav** — Roll No: `25124015`
- **Anup Kumar** — Roll No: `25124013`

---

## 8. Academic References

1. Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018). *The Case for Learned Index Structures*. In Proceedings of the 2018 International Conference on Management of Data (SIGMOD '18), pp. 489–504.
2. Marcus, R., Kipf, A., van Renen, A., Stoian, M., Misra, S., Kemper, A., Neumann, T., & Kraska, T. (2020). *Benchmarking Learned Indexes (SOSD)*. Proceedings of the VLDB Endowment, 14(1), 1–13.
3. Ferragina, P., & Vinciguerra, G. (2020). *The PGM-index: a compact and efficient learned index on sorted data*. Proceedings of the VLDB Endowment, 13(8), 1162–1175.
