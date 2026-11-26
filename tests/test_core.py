"""
Comprehensive Test Suite for Learned Index Visualizer.
Tests:
- Key-Value Record Management (Physical Table vs. Primary Index)
- All Dataset Distributions (Uniform, Normal, Log-Normal, SOSD Wikipedia/Books/FB/OSM)
- Binary Search (whole-array, range-bounded, record dereferencing)
- Learned Index Model (training, slope, max error, 3 search methods, 0 mismatches)
- Boundary Queries and Absent Keys
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import (
    generate_synthetic_uniform,
    generate_synthetic_normal,
    generate_synthetic_lognormal,
    load_sosd_dataset,
    generate_queries,
    DatasetBundle
)
from core.binary_search import binary_search, binary_search_with_stats, binary_search_in_range
from core.learned_index_model import LearnedIndexModel


# =========================================================================
# 1. KEY-VALUE RECORD MANAGEMENT & DATASET TESTS
# =========================================================================

class TestDatasetAndRecords:
    def test_synthetic_uniform(self):
        bundle = generate_synthetic_uniform(size=2000, seed=42)
        assert bundle.size == 2000
        assert len(bundle.physical_records) == 2000
        assert np.all(bundle.index_keys[:-1] <= bundle.index_keys[1:]), "Index keys must be sorted"
        assert len(np.unique(bundle.index_keys)) == 2000, "Index keys must be unique"

        # Check key-value dereferencing
        first_row_id = bundle.index_row_ids[0]
        rec = bundle.get_record(first_row_id)
        assert rec is not None
        assert rec["key"] == bundle.index_keys[0]

    def test_synthetic_normal(self):
        bundle = generate_synthetic_normal(size=1500, seed=42)
        assert bundle.size == 1500
        assert np.all(bundle.index_keys[:-1] <= bundle.index_keys[1:])
        summary = bundle.summary()
        assert summary["min_key"] >= 0

    def test_synthetic_lognormal(self):
        bundle = generate_synthetic_lognormal(size=1500, seed=42)
        assert bundle.size == 1500
        assert np.all(bundle.index_keys[:-1] <= bundle.index_keys[1:])

    def test_real_world_sosd_loaders(self):
        for sosd_key in ["wiki_ts", "books", "fb", "osm"]:
            bundle = load_sosd_dataset(sosd_key, sample_size=1000, seed=42)
            assert bundle.size == 1000
            assert bundle.category == "Real-world"
            assert np.all(bundle.index_keys[:-1] <= bundle.index_keys[1:])

    def test_duplicate_tracking(self):
        raw_keys = np.array([100, 200, 100, 300, 200, 400], dtype=np.int64)
        bundle = DatasetBundle(
            name="Test Dups",
            category="Test",
            description="Testing duplicate handling",
            keys=raw_keys
        )
        assert bundle.raw_key_count == 6
        assert bundle.duplicate_count == 2
        assert bundle.size == 4
        assert list(bundle.index_keys) == [100, 200, 300, 400]

    def test_query_generator_ratios(self):
        bundle = generate_synthetic_uniform(size=2000, seed=42)
        queries = generate_queries(bundle.index_keys, n_queries=200, present_ratio=0.5, seed=99)
        assert len(queries) == 200
        key_set = set(bundle.index_keys.tolist())
        present_count = sum(1 for q in queries if int(q) in key_set)
        # Should be exactly 100 present, 100 absent
        assert present_count == 100


# =========================================================================
# 2. BINARY SEARCH TESTS (WITH PHYSICAL RECORD DEREFERENCE)
# =========================================================================

class TestBinarySearchWithDereference:
    def setup_method(self):
        self.bundle = generate_synthetic_uniform(size=1000, seed=21)
        self.keys = self.bundle.index_keys

    def test_find_existing_record(self):
        target_idx = 450
        target_key = int(self.keys[target_idx])
        res = binary_search_with_stats(self.keys, target_key, self.bundle.index_row_ids, self.bundle)
        assert res["found"] is True
        assert res["index"] == target_idx
        assert res["record"] is not None
        assert res["record"]["key"] == target_key

    def test_absent_key(self):
        absent_key = int(self.keys[-1]) + 99999
        res = binary_search_with_stats(self.keys, absent_key, self.bundle.index_row_ids, self.bundle)
        assert res["found"] is False
        assert res["index"] == -1
        assert res["record"] is None

    def test_boundary_keys(self):
        # First key
        first_key = int(self.keys[0])
        res_first = binary_search_with_stats(self.keys, first_key, self.bundle.index_row_ids, self.bundle)
        assert res_first["found"] is True
        assert res_first["index"] == 0

        # Last key
        last_key = int(self.keys[-1])
        res_last = binary_search_with_stats(self.keys, last_key, self.bundle.index_row_ids, self.bundle)
        assert res_last["found"] is True
        assert res_last["index"] == len(self.keys) - 1

    def test_range_search(self):
        target_idx = 500
        target_key = int(self.keys[target_idx])
        idx, comps = binary_search_in_range(self.keys, target_key, 400, 600)
        assert idx == target_idx
        assert comps > 0


# =========================================================================
# 3. LEARNED INDEX MODEL TESTS (3 METHODS & CORRECTNESS)
# =========================================================================

class TestLearnedIndexModel:
    def setup_method(self):
        self.bundle = generate_synthetic_uniform(size=5000, seed=21)
        self.model = LearnedIndexModel()
        self.model.train(self.bundle)

    def test_model_training_parameters(self):
        assert self.model.is_trained
        assert self.model.slope > 0
        assert self.model.max_error >= 0
        assert self.model.r2 > 0.95

    def test_three_methods_agree_on_present_keys(self):
        test_indices = [0, 1, 50, 1250, 2500, 4999]
        for idx in test_indices:
            key = int(self.bundle.index_keys[idx])
            bs_res = self.model.search_binary(key)
            naive_res = self.model.search_naive(key)
            opt_res = self.model.search_optimized(key)

            assert bs_res["found"] and naive_res["found"] and opt_res["found"]
            assert bs_res["index"] == naive_res["index"] == opt_res["index"] == idx
            assert bs_res["row_id"] == naive_res["row_id"] == opt_res["row_id"]
            assert bs_res["record"]["key"] == key

    def test_three_methods_agree_on_absent_keys(self):
        absent_keys = [-500, int(self.bundle.index_keys[-1]) + 5000]
        for k in absent_keys:
            bs_res = self.model.search_binary(k)
            naive_res = self.model.search_naive(k)
            opt_res = self.model.search_optimized(k)

            assert not bs_res["found"]
            assert not naive_res["found"]
            assert not opt_res["found"]
            assert bs_res["index"] == naive_res["index"] == opt_res["index"] == -1
            assert bs_res["record"] is None

    def test_large_scale_zero_mismatch(self):
        """Verify 0 mismatches across 1,000 queries on present and absent keys."""
        queries = generate_queries(self.bundle.index_keys, n_queries=1000, present_ratio=0.5, seed=777)
        mismatches = 0
        for q in queries:
            key = int(q)
            bs_idx = self.model.search_binary(key)["index"]
            opt_idx = self.model.search_optimized(key)["index"]
            if bs_idx != opt_idx:
                mismatches += 1

        assert mismatches == 0, f"Detected {mismatches} mismatches!"

    def test_non_uniform_distributions(self):
        """Ensure learned index operates properly on skewed and real-world distributions."""
        for dist_bundle in [
            generate_synthetic_normal(size=2000, seed=12),
            generate_synthetic_lognormal(size=2000, seed=12),
            load_sosd_dataset("wiki_ts", sample_size=2000, seed=12)
        ]:
            m = LearnedIndexModel()
            m.train(dist_bundle)
            assert m.is_trained
            # Test 100 random keys
            sample_keys = dist_bundle.index_keys[::20]
            for k in sample_keys:
                bs_idx = m.search_binary(int(k))["index"]
                opt_idx = m.search_optimized(int(k))["index"]
                assert bs_idx == opt_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
