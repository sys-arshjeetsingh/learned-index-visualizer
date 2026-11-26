"""
Learned Index Model implementation for the Learned Index Visualizer.

Implements three distinct search methods:
1. Binary Search (Baseline): Traditional divide-and-conquer over sorted index keys.
2. Naive Learned Index: Evaluates scikit-learn's model.predict() per query, then local binary search.
3. Optimized Learned Index: Evaluates regression parameters directly (slope * key + intercept),
   then local binary search.

Maintains record dereferencing: maps Query Key -> Index Position -> Row ID -> Physical Record.
"""

import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from core.binary_search import (binary_search, binary_search_with_stats,
                                 binary_search_in_range,
                                 binary_search_in_range_with_stats)


class LearnedIndexModel:
    """
    A learned database index that approximates CDF position using Linear Regression.

    Mathematical Model:
        predicted_position = slope * key + intercept
    """

    def __init__(self):
        self.model = LinearRegression()
        self.slope = 0.0
        self.intercept = 0.0
        self.max_error = 0
        self.mean_error = 0.0
        self.r2 = 0.0
        self.bundle = None
        self.keys = None
        self.row_ids = None
        self.training_time = 0.0
        self.errors = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def search_radius(self) -> int:
        """
        Conservative error radius: max_error + 1 to account for integer truncation/rounding.
        Guarantees that any key present in training set is enclosed within [pred - radius, pred + radius].
        """
        return self.max_error + 1

    def train(self, data_or_bundle) -> dict:
        """
        Train the linear regression model on sorted index keys.

        Accepts either a DatasetBundle or raw np.ndarray of keys.
        """
        if hasattr(data_or_bundle, "index_keys"):
            self.bundle = data_or_bundle
            self.keys = data_or_bundle.index_keys
            self.row_ids = data_or_bundle.index_row_ids
        else:
            self.bundle = None
            self.keys = np.asarray(data_or_bundle, dtype=np.int64)
            self.row_ids = np.arange(len(self.keys), dtype=np.int64)

        n = len(self.keys)
        X = self.keys.reshape(-1, 1).astype(np.float64)
        Y = np.arange(n, dtype=np.float64)

        t0 = time.perf_counter()
        self.model.fit(X, Y)
        self.training_time = time.perf_counter() - t0

        self.slope = float(self.model.coef_[0])
        self.intercept = float(self.model.intercept_)

        # Measure empirical prediction error on training keys under exact inference rounding
        predicted_positions = self.slope * self.keys.astype(np.float64) + self.intercept
        predicted_int = predicted_positions.astype(int)
        self.errors = np.abs(predicted_int - Y.astype(int))
        self.max_error = int(np.max(self.errors)) if n > 0 else 0
        self.mean_error = float(np.mean(self.errors)) if n > 0 else 0.0
        self.r2 = float(r2_score(Y, predicted_positions)) if n > 1 else 1.0

        self._is_trained = True

        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "max_error": self.max_error,
            "mean_error": self.mean_error,
            "search_radius": self.search_radius,
            "r2": self.r2,
            "training_time": self.training_time,
            "dataset_size": n,
        }

    def _predict_naive(self, key: int) -> int:
        """Inference via scikit-learn model.predict() with framework overhead."""
        return int(self.model.predict([[key]])[0])

    def _predict_optimized(self, key: int) -> int:
        """Inference via direct floating-point arithmetic (zero framework overhead)."""
        return int(self.slope * key + self.intercept)

    def _clamp_bounds(self, predicted_pos: int) -> tuple:
        """Clamp search window to valid index range [0, N-1]."""
        n = len(self.keys)
        lo = max(0, predicted_pos - self.search_radius)
        hi = min(n - 1, predicted_pos + self.search_radius)
        return lo, hi

    # ------------------------------------------------------------------
    # Search Method 1: Binary Search Baseline
    # ------------------------------------------------------------------
    def search_binary(self, target: int) -> dict:
        """Full binary search over sorted index keys."""
        idx = binary_search(self.keys, target)
        found = (idx != -1)
        row_id = int(self.row_ids[idx]) if found else -1
        record = self.bundle.get_record(row_id) if (found and self.bundle) else None
        return {
            "index": idx,
            "found": found,
            "row_id": row_id,
            "record": record,
            "method": "Binary Search"
        }

    # ------------------------------------------------------------------
    # Search Method 2: Naive Learned Index (model.predict)
    # ------------------------------------------------------------------
    def search_naive(self, target: int) -> dict:
        """Naive learned index: sklearn model.predict() + bounded binary search."""
        pred = self._predict_naive(target)
        lo, hi = self._clamp_bounds(pred)
        idx, comps = binary_search_in_range(self.keys, target, lo, hi)
        found = (idx != -1)
        row_id = int(self.row_ids[idx]) if found else -1
        record = self.bundle.get_record(row_id) if (found and self.bundle) else None
        return {
            "index": idx,
            "found": found,
            "comparisons": comps,
            "predicted_pos": pred,
            "search_lo": lo,
            "search_hi": hi,
            "search_window": hi - lo + 1,
            "row_id": row_id,
            "record": record,
            "method": "Naive Learned Index"
        }

    # ------------------------------------------------------------------
    # Search Method 3: Optimized Learned Index (Direct Arithmetic)
    # ------------------------------------------------------------------
    def search_optimized(self, target: int) -> dict:
        """Optimized learned index: direct arithmetic + bounded binary search."""
        pred = self._predict_optimized(target)
        lo, hi = self._clamp_bounds(pred)
        idx, comps = binary_search_in_range(self.keys, target, lo, hi)
        found = (idx != -1)
        row_id = int(self.row_ids[idx]) if found else -1
        record = self.bundle.get_record(row_id) if (found and self.bundle) else None
        return {
            "index": idx,
            "found": found,
            "comparisons": comps,
            "predicted_pos": pred,
            "search_lo": lo,
            "search_hi": hi,
            "search_window": hi - lo + 1,
            "row_id": row_id,
            "record": record,
            "method": "Optimized Learned Index"
        }

    # ------------------------------------------------------------------
    # Detailed Trace Methods for Interactive UI
    # ------------------------------------------------------------------
    def search_binary_detailed(self, target: int) -> dict:
        """Binary search with step-by-step history and record dereferencing."""
        res = binary_search_with_stats(self.keys, target, self.row_ids, self.bundle)
        res["method"] = "Binary Search"
        res["search_lo"] = 0
        res["search_hi"] = len(self.keys) - 1
        res["search_window"] = len(self.keys)
        return res

    def search_naive_detailed(self, target: int) -> dict:
        """Naive learned index with step-by-step history and record dereferencing."""
        pred = self._predict_naive(target)
        lo, hi = self._clamp_bounds(pred)
        res = binary_search_in_range_with_stats(self.keys, target, lo, hi, self.row_ids, self.bundle)
        res["predicted_pos"] = pred
        res["search_window"] = hi - lo + 1
        res["method"] = "Naive Learned Index"
        res["prediction_error"] = abs(pred - res["index"]) if res["found"] else None
        return res

    def search_optimized_detailed(self, target: int) -> dict:
        """Optimized learned index with step-by-step history and record dereferencing."""
        pred = self._predict_optimized(target)
        lo, hi = self._clamp_bounds(pred)
        res = binary_search_in_range_with_stats(self.keys, target, lo, hi, self.row_ids, self.bundle)
        res["predicted_pos"] = pred
        res["search_window"] = hi - lo + 1
        res["method"] = "Optimized Learned Index"
        res["prediction_error"] = abs(pred - res["index"]) if res["found"] else None
        return res

    # ------------------------------------------------------------------
    # Model Quality Metrics & Sampling
    # ------------------------------------------------------------------
    def get_prediction_stats(self) -> dict:
        """Statistical summary of model fit and error characteristics."""
        if not self._is_trained:
            return {}
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "mean_absolute_error": self.mean_error,
            "max_absolute_error": float(self.max_error),
            "search_radius": self.search_radius,
            "search_window_size": 2 * self.search_radius + 1,
            "r2_score": self.r2,
            "training_time_ms": self.training_time * 1000.0,
            "total_keys": len(self.keys)
        }

    def get_regression_data(self, n_samples: int = 2500) -> dict:
        """Sampled points for Plotly regression & error visualization."""
        if not self._is_trained:
            return {}
        n = len(self.keys)
        sample_idx = np.linspace(0, n - 1, min(n_samples, n), dtype=int)
        sample_keys = self.keys[sample_idx]
        predicted = self.slope * sample_keys.astype(np.float64) + self.intercept
        return {
            "keys": sample_keys,
            "actual_positions": sample_idx,
            "predicted_positions": predicted,
            "errors": predicted - sample_idx.astype(np.float64),
        }
