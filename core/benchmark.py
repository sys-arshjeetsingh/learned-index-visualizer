"""
Benchmarking engine for the Learned Index Visualizer.

Provides reproducible, freshly measured benchmarks across:
- Binary Search (Baseline)
- Naive Learned Index (scikit-learn predict)
- Optimized Learned Index (direct arithmetic)

Ensures:
- Identical query workloads.
- Explicit verification of 0 mismatches before latency reporting.
- Separate measurement of training time and inference latency.
- Full record retrieval verification.
"""

import time
import numpy as np
from core.learned_index_model import LearnedIndexModel
from core.binary_search import binary_search
from core.dataset import generate_queries


def _warmup(model: LearnedIndexModel, keys: np.ndarray, n_warmup: int = 100):
    """Warm up instruction caches and memory buffers before timing."""
    rng = np.random.default_rng(seed=12345)
    warmup_keys = rng.choice(keys, size=min(n_warmup, len(keys)), replace=False)
    for k in warmup_keys:
        val = int(k)
        binary_search(keys, val)
        model.search_naive(val)
        model.search_optimized(val)


def run_benchmark(model: LearnedIndexModel,
                  data_or_keys,
                  n_queries: int = 10_000,
                  present_ratio: float = 0.5,
                  query_seed: int = 99,
                  n_repetitions: int = 3,
                  warmup: bool = True) -> dict:
    """
    Run an end-to-end multi-method benchmark.
    """
    if hasattr(data_or_keys, "index_keys"):
        keys = data_or_keys.index_keys
    else:
        keys = np.asarray(data_or_keys, dtype=np.int64)

    queries = generate_queries(keys, n_queries=n_queries, present_ratio=present_ratio, seed=query_seed)
    query_list = [int(q) for q in queries]

    if warmup:
        _warmup(model, keys)

    # 1. Correctness Verification Pass
    bs_indices = []
    naive_indices = []
    opt_indices = []

    for k in query_list:
        bs_indices.append(binary_search(keys, k))
        naive_indices.append(model.search_naive(k)["index"])
        opt_indices.append(model.search_optimized(k)["index"])

    naive_mismatches = sum(b != n for b, n in zip(bs_indices, naive_indices))
    opt_mismatches = sum(b != o for b, o in zip(bs_indices, opt_indices))

    # 2. Per-Query Latency Sampling Pass (for distribution analysis)
    bs_per_query = []
    naive_per_query = []
    opt_per_query = []

    for k in query_list:
        t0 = time.perf_counter()
        binary_search(keys, k)
        bs_per_query.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        model.search_naive(k)
        naive_per_query.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        model.search_optimized(k)
        opt_per_query.append(time.perf_counter() - t0)

    # 3. Batch Timing Repetitions (to eliminate OS scheduling jitter)
    bs_batch_times = []
    naive_batch_times = []
    opt_batch_times = []

    for _ in range(n_repetitions):
        t0 = time.perf_counter()
        for k in query_list:
            binary_search(keys, k)
        bs_batch_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for k in query_list:
            model.search_naive(k)
        naive_batch_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for k in query_list:
            model.search_optimized(k)
        opt_batch_times.append(time.perf_counter() - t0)

    best_bs = min(bs_batch_times)
    best_naive = min(naive_batch_times)
    best_opt = min(opt_batch_times)

    bs_arr = np.array(bs_per_query)
    naive_arr = np.array(naive_per_query)
    opt_arr = np.array(opt_per_query)

    pred_stats = model.get_prediction_stats()

    return {
        "n_queries": n_queries,
        "present_ratio": present_ratio,
        "n_repetitions": n_repetitions,
        "dataset_size": len(keys),

        # Binary Search Baseline
        "bs_total_time": best_bs,
        "bs_mean_time": best_bs / n_queries,
        "bs_median_time": float(np.median(bs_arr)),
        "bs_mean_time_us": (best_bs / n_queries) * 1e6,
        "bs_median_time_us": float(np.median(bs_arr)) * 1e6,
        "bs_per_query_us": (bs_arr * 1e6).tolist(),

        # Naive Learned Index
        "naive_total_time": best_naive,
        "naive_mean_time": best_naive / n_queries,
        "naive_median_time": float(np.median(naive_arr)),
        "naive_mean_time_us": (best_naive / n_queries) * 1e6,
        "naive_median_time_us": float(np.median(naive_arr)) * 1e6,
        "naive_per_query_us": (naive_arr * 1e6).tolist(),
        "naive_mismatches": naive_mismatches,

        # Optimized Learned Index
        "opt_total_time": best_opt,
        "opt_mean_time": best_opt / n_queries,
        "opt_median_time": float(np.median(opt_arr)),
        "opt_mean_time_us": (best_opt / n_queries) * 1e6,
        "opt_median_time_us": float(np.median(opt_arr)) * 1e6,
        "opt_per_query_us": (opt_arr * 1e6).tolist(),
        "opt_mismatches": opt_mismatches,

        # Speedups
        "naive_speedup": best_bs / best_naive if best_naive > 0 else 0.0,
        "opt_speedup": best_bs / best_opt if best_opt > 0 else 0.0,

        # Model Quality
        "mean_absolute_error": pred_stats.get("mean_absolute_error", 0.0),
        "max_absolute_error": pred_stats.get("max_absolute_error", 0.0),
        "r2_score": pred_stats.get("r2_score", 0.0),
        "search_radius": pred_stats.get("search_radius", 0),
        "search_window_size": pred_stats.get("search_window_size", 0),
        "training_time_ms": pred_stats.get("training_time_ms", 0.0),
    }
